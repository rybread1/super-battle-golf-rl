"""Real-time debug overlay showing all detected game state.

Run this while playing Super Battle Golf manually to see what the
detection system is picking up. Press 'q' to quit.

Displays:
- Player state (none / near_ball / stance_no_hit / stance_can_hit / swinging)
- Pin and ball icon detection (position)
- Player progress bar reading (0-1)
- Loading screen detection
- Strokes text detection
- Distance OCR readings (ball/pin meters, optional)
- Ball icon edge warnings (near top/bottom = walked away)
- Reward calculations (step penalty, progress delta)
- Frame rate
"""

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from sbg.capture import ScreenCapture
from sbg.detect import (
    find_icons,
    get_player_progress,
    detect_player_state,
    is_loading_screen,
    read_strokes_text,
)
from sbg.reward import (
    STEP_PENALTY,
    FAILED_SHOT_PENALTY,
    SUCCESSFUL_SHOT_BONUS,
    PROGRESS_SCALE,
    HOLE_BONUS,
    compute_shot_reward,
)
from sbg.window import find_game_window, get_client_region


# Colors (BGR for OpenCV)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
MAGENTA = (255, 0, 255)
DARK_BG = (30, 30, 30)


def draw_text(img, text, pos, color=WHITE, scale=0.5, thickness=1):
    """Draw text with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2), DARK_BG, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return th + baseline + 6  # line height


def draw_panel(img, x, y, width, lines):
    """Draw a panel with multiple lines of colored text. Returns new y."""
    panel_h = len(lines) * 20 + 10
    cv2.rectangle(img, (x - 4, y - 4), (x + width, y + panel_h), DARK_BG, -1)
    cv2.rectangle(img, (x - 4, y - 4), (x + width, y + panel_h), (80, 80, 80), 1)
    cy = y + 14
    for text, color in lines:
        cv2.putText(img, text, (x, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cy += 18
    return y + panel_h + 8


def draw_icon_marker(img, pos, label, color, radius=18):
    """Draw a circle + label at an icon position."""
    cx, cy = pos
    cv2.circle(img, (cx, cy), radius, color, 2)
    cv2.circle(img, (cx, cy), 3, color, -1)
    draw_text(img, label, (cx + radius + 4, cy - 4), color, 0.45)


def draw_exclusion_zones(img):
    """Draw semi-transparent exclusion zones."""
    h, w = img.shape[:2]
    overlay = img.copy()
    # Progress bar area: top 12% & left 40% (only remaining exclusion zone)
    cv2.rectangle(overlay, (0, 0), (int(w * 0.40), int(h * 0.12)), (100, 0, 0), -1)
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)


def draw_progress_bar_region(img):
    """Highlight the progress bar detection region."""
    h, w = img.shape[:2]
    x1, y1 = int(0.02 * w), int(0.02 * h)
    x2, y2 = int(0.33 * w), int(0.08 * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 1)


def draw_power_bar_debug(img, frame_rgb):
    """Draw chunked power bar comparison — splits the bar vertically into
    horizontal chunks, compares outside vs inside for each chunk locally,
    then averages. This cancels out background variation along the bar height.

    >>> ADJUST THESE to tune the two strips and chunk count <<<
    """
    h, w = img.shape[:2]

    # ---- Strip positions (fractional x coords) ----
    A_LEFT, A_RIGHT = 0.244, 0.258   # outside the bar
    B_LEFT, B_RIGHT = 0.258, 0.272   # inside the bar

    # ---- Vertical extent of the bar ----
    BAR_TOP, BAR_BOT = 0.48, 0.84

    # ---- Number of horizontal chunks ----
    NUM_CHUNKS = 12

    # Pixel coords for the full strips
    ax1, ax2 = int(A_LEFT * w), int(A_RIGHT * w)
    bx1, bx2 = int(B_LEFT * w), int(B_RIGHT * w)
    y_top, y_bot = int(BAR_TOP * h), int(BAR_BOT * h)
    chunk_h = (y_bot - y_top) // NUM_CHUNKS

    # Convert full strip region to HSV once
    strip_rgb = frame_rgb[y_top:y_bot, ax1:bx2]
    strip_hsv = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2HSV).astype(float)

    # Per-chunk metrics across multiple color spaces
    chunk_data = []

    for i in range(NUM_CHUNKS):
        cy1 = y_top + i * chunk_h
        cy2 = cy1 + chunk_h
        local_y1 = i * chunk_h
        local_y2 = local_y1 + chunk_h

        # Draw chunk boundaries
        cv2.rectangle(img, (ax1, cy1), (ax2, cy2), CYAN, 1)
        cv2.rectangle(img, (bx1, cy1), (bx2, cy2), MAGENTA, 1)

        # RGB means
        ca_rgb = frame_rgb[cy1:cy2, ax1:ax2].astype(float)
        cb_rgb = frame_rgb[cy1:cy2, bx1:bx2].astype(float)
        ma_rgb = np.mean(ca_rgb, axis=(0, 1))
        mb_rgb = np.mean(cb_rgb, axis=(0, 1))

        # HSV means (from pre-converted strip)
        a_rel_x1, a_rel_x2 = 0, ax2 - ax1
        b_rel_x1, b_rel_x2 = bx1 - ax1, bx2 - ax1
        ca_hsv = strip_hsv[local_y1:local_y2, a_rel_x1:a_rel_x2]
        cb_hsv = strip_hsv[local_y1:local_y2, b_rel_x1:b_rel_x2]
        ma_hsv = np.mean(ca_hsv, axis=(0, 1))  # [H, S, V]
        mb_hsv = np.mean(cb_hsv, axis=(0, 1))

        d = {
            "rgb_diff": float(np.sqrt(np.sum((ma_rgb - mb_rgb) ** 2))),
            "r_diff": mb_rgb[0] - ma_rgb[0],
            "g_diff": mb_rgb[1] - ma_rgb[1],
            "b_diff": mb_rgb[2] - ma_rgb[2],
            "h_a": ma_hsv[0], "s_a": ma_hsv[1], "v_a": ma_hsv[2],
            "h_b": mb_hsv[0], "s_b": mb_hsv[1], "v_b": mb_hsv[2],
            "h_diff": mb_hsv[0] - ma_hsv[0],
            "s_diff": mb_hsv[1] - ma_hsv[1],  # saturation change
            "v_diff": mb_hsv[2] - ma_hsv[2],  # value/brightness change
        }
        chunk_data.append(d)

        # Per-chunk vs_diff (the metric used in detect.py)
        vs = d['v_diff'] - d['s_diff']
        d['vs_diff'] = vs
        lbl_color = GREEN if vs > 40 else RED
        cv2.putText(img, f"{vs:+.0f}", (bx2 + 4, cy1 + chunk_h // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, lbl_color, 1, cv2.LINE_AA)

    # The key metric: avg(v_diff - s_diff) — matches detect.py threshold of 40
    avg_vs = float(np.mean([d["vs_diff"] for d in chunk_data]))
    result = "CAN HIT" if avg_vs > 40 else "NO HIT"
    result_color = GREEN if avg_vs > 40 else RED

    # Labels at top
    draw_text(img, "A", (ax1, y_top - 6), CYAN, 0.4)
    draw_text(img, "B", (bx1, y_top - 6), MAGENTA, 0.4)

    # Panel showing the decision metric
    panel_x = w // 2 - 180
    panel_y = h - 140
    lines = [
        ("=== POWER BAR DEBUG ===", YELLOW),
        (f"avg(V_diff - S_diff): {avg_vs:+.1f}  (threshold: 40)", result_color),
        (f"Result: {result}", result_color),
        ("--- per-chunk V-S diff ---", WHITE),
        (f"  {' '.join(f'{d['vs_diff']:+.0f}' for d in chunk_data)}", WHITE),
    ]
    draw_panel(img, panel_x, panel_y, 370, lines)

    return chunk_data


def main():
    # OCR is slow -- allow disabling it
    use_ocr = "--ocr" in sys.argv
    show_zones = "--zones" in sys.argv
    show_power_debug = "--power" in sys.argv

    print("Finding game window...")
    hwnd = find_game_window(timeout=10)
    region = get_client_region(hwnd)
    print(f"Capture region: {region}")

    cap = ScreenCapture(region=region, fps=30)
    cap.start()
    time.sleep(0.5)

    prev_progress = None
    prev_time = time.time()
    fps_history = []
    frame_count = 0
    ocr_result = {"ball": None, "pin": None}
    ocr_cooldown = 0  # only run OCR every N frames

    # Cached detection results (updated on their own schedules)
    pin_pos = None
    ball_pos = None
    progress = None
    player_state = "none"
    loading = False
    strokes_visible = False
    dbg_mouse_score = 0.0
    dbg_f_key_score = 0.0
    dbg_near_score = 0.0

    # How often to run each detection (every N frames)
    ICON_INTERVAL = 5      # template matching is the bottleneck
    PROGRESS_INTERVAL = 3
    CHEAP_INTERVAL = 2     # stance, loading, text checks

    print("Debug overlay running. Press 'q' in the overlay window to quit.")
    print("Options: --ocr (distance OCR), --zones (exclusion zones), --power (power bar debug)")
    print("Keys: q=quit, r=reset progress, z=zones, o=OCR, p=power bar debug")
    print("Keys: q=quit, r=reset progress, z=zones, o=OCR, p=power bar debug")

    while True:
        frame = cap.grab()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps_history.append(1.0 / dt)
            if len(fps_history) > 30:
                fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

        # --- Run detections (throttled) ---
        h, w = frame.shape[:2]

        # Icon detection (expensive — template matching at 8 scales)
        if frame_count % ICON_INTERVAL == 0:
            pin_pos, ball_pos = find_icons(frame)

        # Progress bar
        if frame_count % PROGRESS_INTERVAL == 0:
            progress = get_player_progress(frame)

        # Cheap detections
        if frame_count % CHEAP_INTERVAL == 0:
            player_state = detect_player_state(frame)
            loading = is_loading_screen(frame)
            strokes_visible = read_strokes_text(frame)
            # Grab intermediate values for debug display
            from sbg.detect import (_match_ui_icon, _mouse_icon_tmpl,
                                    _f_key_icon_tmpl, _near_ball_icon_tmpl)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            _bl = (0.0, 0.85, 0.15, 1.0)
            dbg_mouse_score = _match_ui_icon(frame_gray, _mouse_icon_tmpl, _bl)
            dbg_f_key_score = _match_ui_icon(frame_gray, _f_key_icon_tmpl,
                                             (0.0, 0.82, 0.15, 0.95))
            dbg_near_score = _match_ui_icon(frame_gray, _near_ball_icon_tmpl, _bl)

        # OCR distances (throttled - every 30 frames if enabled)
        if use_ocr:
            ocr_cooldown -= 1
            if ocr_cooldown <= 0:
                from sbg.detect import detect_distances
                ocr_result = detect_distances(frame)
                ocr_cooldown = 30

        # Ball edge warning
        ball_edge_warning = None
        if ball_pos:
            bx, by = ball_pos
            if by < h * 0.15:
                ball_edge_warning = "BALL NEAR TOP EDGE - walked away!"
            elif by > h * 0.85:
                ball_edge_warning = "BALL NEAR BOTTOM EDGE - walked away!"

        # Progress delta (simulated reward)
        progress_delta = None
        shot_reward_est = None
        if progress is not None and prev_progress is not None:
            progress_delta = progress - prev_progress
            shot_reward_est = compute_shot_reward(prev_progress, progress)

        # --- Draw overlay ---
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if show_zones:
            draw_exclusion_zones(display)
            draw_progress_bar_region(display)

        if show_power_debug:
            draw_power_bar_debug(display, frame)

        # Draw icon detections
        if pin_pos:
            draw_icon_marker(display, pin_pos, "PIN", ORANGE)
        if ball_pos:
            color = RED if ball_edge_warning else GREEN
            draw_icon_marker(display, ball_pos, "BALL", color)

        # --- Info panels ---
        panel_x = 10
        panel_y = h - 10  # build panels from bottom up

        # Reward panel
        reward_lines = [
            ("=== REWARDS ===", YELLOW),
            (f"Step penalty:     {STEP_PENALTY:+.3f}", WHITE),
            (f"Failed shot:      {FAILED_SHOT_PENALTY:+.3f}", WHITE),
            (f"Successful shot:  {SUCCESSFUL_SHOT_BONUS:+.3f}", WHITE),
            (f"Progress scale:   {PROGRESS_SCALE:.1f}x", WHITE),
            (f"Hole bonus:       {HOLE_BONUS:+.1f}", WHITE),
        ]
        if shot_reward_est is not None:
            reward_lines.append((f"Est. shot reward: {shot_reward_est:+.3f}", CYAN))
        panel_h = len(reward_lines) * 18 + 10
        panel_y -= panel_h
        draw_panel(display, panel_x, panel_y, 260, reward_lines)

        # State panel (right side)
        state_x = w - 310
        state_colors = {
            "none": WHITE,
            "near_ball": YELLOW,
            "stance_no_hit": ORANGE,
            "stance_can_hit": GREEN,
            "swinging": CYAN,
        }
        state_lines = [
            ("=== GAME STATE ===", YELLOW),
            (f"Player state: {player_state}", state_colors.get(player_state, WHITE)),
            (f"Mouse icon:   {dbg_mouse_score:.3f} (>0.9=stance)", GREEN if dbg_mouse_score > 0.9 else WHITE),
            (f"Near ball:    {dbg_near_score:.3f} (>0.9=near_ball)", GREEN if dbg_near_score > 0.9 else WHITE),
            (f"F key icon:   {dbg_f_key_score:.3f} (>0.9=swinging)", GREEN if dbg_f_key_score > 0.9 else WHITE),
            (f"Loading:      {'YES' if loading else 'no'}", RED if loading else WHITE),
            (f"Strokes vis:  {'YES' if strokes_visible else 'no'}", WHITE),
        ]
        draw_panel(display, state_x, 10, 300, state_lines)

        # Detection panel (right side, below state)
        det_lines = [
            ("=== DETECTIONS ===", YELLOW),
            (f"Pin icon:  {f'({pin_pos[0]}, {pin_pos[1]})' if pin_pos else 'not found'}",
             GREEN if pin_pos else RED),
            (f"Ball icon: {f'({ball_pos[0]}, {ball_pos[1]})' if ball_pos else 'not found'}",
             GREEN if ball_pos else RED),
        ]
        if ball_edge_warning:
            det_lines.append((ball_edge_warning, RED))

        progress_str = f"{progress:.3f}" if progress is not None else "N/A"
        prev_str = f"{prev_progress:.3f}" if prev_progress is not None else "N/A"
        det_lines.append((f"Progress:  {progress_str} (prev: {prev_str})", CYAN))

        if progress_delta is not None:
            delta_color = GREEN if progress_delta > 0 else RED if progress_delta < 0 else WHITE
            det_lines.append((f"Prog delta: {progress_delta:+.4f}", delta_color))

        if use_ocr:
            ball_d = ocr_result.get("ball")
            pin_d = ocr_result.get("pin")
            det_lines.append((f"Ball dist:  {f'{ball_d}m' if ball_d else 'N/A'} (OCR)", MAGENTA))
            det_lines.append((f"Pin dist:   {f'{pin_d}m' if pin_d else 'N/A'} (OCR)", MAGENTA))

        draw_panel(display, state_x, 120, 300, det_lines)

        # FPS + frame info (top left)
        fps_lines = [
            (f"FPS: {avg_fps:.1f}  Frame: {frame_count}", WHITE),
            (f"Resolution: {w}x{h}", WHITE),
        ]
        draw_panel(display, 10, 10, 220, fps_lines)

        # Update prev_progress on a key press or periodically
        # We track it continuously for the delta display
        if frame_count % 60 == 0 and progress is not None:
            prev_progress = progress

        # Show frame
        cv2.imshow("SBG Debug Overlay", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            # Reset prev_progress
            prev_progress = progress
            print(f"Reset prev_progress to {progress}")
        elif key == ord("z"):
            # Toggle exclusion zones
            show_zones = not show_zones
            print(f"Exclusion zones: {'ON' if show_zones else 'OFF'}")
        elif key == ord("o"):
            # Toggle OCR
            use_ocr = not use_ocr
            ocr_cooldown = 0
            print(f"OCR: {'ON' if use_ocr else 'OFF'}")
        elif key == ord("p"):
            # Toggle power bar debug
            show_power_debug = not show_power_debug
            print(f"Power bar debug: {'ON' if show_power_debug else 'OFF'}")

    cap.stop()
    cv2.destroyAllWindows()
    print("Debug overlay closed.")


if __name__ == "__main__":
    main()
