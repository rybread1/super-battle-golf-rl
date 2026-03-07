"""Real-time debug overlay showing all detected game state.

Run this while playing Super Battle Golf manually to see what the
detection system is picking up. Press 'q' to quit.

Displays:
- Pin icon detection (position + match info)
- Ball icon detection (position + match info)
- Player progress bar reading (0-1)
- Stance detection (power bar)
- Loading screen detection
- Bottom text detection
- Strokes text detection
- Distance OCR readings (ball/pin meters)
- Ball icon edge warnings (near top/bottom = bad)
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
    is_in_stance,
    is_loading_screen,
    detect_bottom_text,
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
    # Progress bar area: top 12% & left 40%
    cv2.rectangle(overlay, (0, 0), (int(w * 0.40), int(h * 0.12)), (100, 0, 0), -1)
    # Club selector: right 22%
    cv2.rectangle(overlay, (int(w * 0.78), 0), (w, h), (100, 0, 0), -1)
    # Bottom strip: bottom 5%
    cv2.rectangle(overlay, (0, int(h * 0.95)), (w, h), (100, 0, 0), -1)
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)


def draw_progress_bar_region(img):
    """Highlight the progress bar detection region."""
    h, w = img.shape[:2]
    x1, y1 = int(0.02 * w), int(0.02 * h)
    x2, y2 = int(0.33 * w), int(0.08 * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 1)


def main():
    # OCR is slow -- allow disabling it
    use_ocr = "--ocr" in sys.argv
    show_zones = "--zones" in sys.argv

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
    stance = False
    loading = False
    bottom_text = "none"
    strokes_visible = False

    # How often to run each detection (every N frames)
    ICON_INTERVAL = 5      # template matching is the bottleneck
    PROGRESS_INTERVAL = 3
    CHEAP_INTERVAL = 2     # stance, loading, text checks

    print("Debug overlay running. Press 'q' in the overlay window to quit.")
    print("Options: --ocr (enable distance OCR, slow), --zones (show exclusion zones)")

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
            stance = is_in_stance(frame)
            loading = is_loading_screen(frame)
            bottom_text = detect_bottom_text(frame)
            strokes_visible = read_strokes_text(frame)

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
        state_lines = [
            ("=== GAME STATE ===", YELLOW),
            (f"In stance:    {'YES' if stance else 'no'}", GREEN if stance else WHITE),
            (f"Loading:      {'YES' if loading else 'no'}", RED if loading else WHITE),
            (f"Bottom text:  {bottom_text}", WHITE),
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

    cap.stop()
    cv2.destroyAllWindows()
    print("Debug overlay closed.")


if __name__ == "__main__":
    main()
