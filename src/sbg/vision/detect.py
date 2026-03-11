"""Game state detection from screen frames.

Frames are expected in RGB format (as returned by ScreenCapture).
Regions are relative to actual frame dimensions — we normalize coordinates
as fractions of frame size to handle slight resolution differences.
"""

import pathlib

import cv2
import numpy as np

# --- Progress bar crop region (fractional coordinates) ---
PROGRESS_BAR_LEFT = 0.02
PROGRESS_BAR_TOP = 0.093
PROGRESS_BAR_RIGHT = 0.3
PROGRESS_BAR_BOTTOM = 0.11
PROGRESS_FLAG_FRAC = 0.84       # flag x-position as fraction of bar width
PROGRESS_POLE_FRAC = 0.86       # flag pole x-position (excluded from player search)


def _crop_frac(frame: np.ndarray, left: float, top: float, right: float, bottom: float) -> np.ndarray:
    """Crop a frame using fractional coordinates (0.0-1.0)."""
    h, w = frame.shape[:2]
    return frame[int(top * h):int(bottom * h), int(left * w):int(right * w)]


# UI icon templates for state detection (grayscale, inner-only crops with no
# background pixels so matching is invariant to what's behind the icon)
_TMPL_DIR = pathlib.Path(__file__).parent / "templates"
_mouse_icon_tmpl = cv2.imread(str(_TMPL_DIR / "mouse_arrow.png"), cv2.IMREAD_GRAYSCALE)
_f_key_icon_tmpl = cv2.imread(str(_TMPL_DIR / "f_key_inner.png"), cv2.IMREAD_GRAYSCALE)
_near_ball_icon_tmpl = cv2.imread(str(_TMPL_DIR / "near_ball_inner.png"), cv2.IMREAD_GRAYSCALE)

# Threshold for template matching (TM_CCOEFF_NORMED)
_ICON_MATCH_THRESH = 0.9


def _match_ui_icon(frame_gray: np.ndarray, template: np.ndarray,
                   region: tuple[float, float, float, float]) -> float:
    """Match a grayscale template in a fractional region, return max score."""
    h, w = frame_gray.shape[:2]
    l, t, r, b = region
    crop = frame_gray[int(t * h):int(b * h), int(l * w):int(r * w)]
    if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
        return 0.0
    result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    return float(np.max(result))


def detect_player_state(frame: np.ndarray) -> str:
    """Detect the player's current state from bottom-left UI icons.

    Uses template matching for the mouse button icon (in stance) and
    F key icon (swinging). Much more reliable than white-pixel counting.

    Returns one of:
        'none'          — no stance UI visible
        'near_ball'     — near ball, "Swing Stance [HOLD]" prompt visible
        'stance_no_hit' — in stance but too far to hit (faint power bar)
        'stance_can_hit'— in stance and close enough to hit (solid power bar)
        'swinging'      — actively swinging (F Cancel visible above stance icons)
    """
    # Convert to grayscale for template matching (background-invariant)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _bl_region = (0.0, 0.85, 0.15, 1.0)

    # Check for stance mouse button icon (two split buttons with down-arrow)
    mouse_score = _match_ui_icon(frame_gray, _mouse_icon_tmpl, _bl_region)
    if mouse_score < _ICON_MATCH_THRESH:
        # No stance icon — check for near_ball icon (single mouse, "Swing Stance [HOLD]")
        near_score = _match_ui_icon(frame_gray, _near_ball_icon_tmpl, _bl_region)
        if near_score >= _ICON_MATCH_THRESH:
            return "near_ball"
        return "none"

    # Stance mouse icon found — check for F key icon above it (swinging)
    f_score = _match_ui_icon(frame_gray, _f_key_icon_tmpl,
                             (0.0, 0.82, 0.15, 0.95))
    if f_score >= _ICON_MATCH_THRESH:
        return "swinging"

    # In stance — check power bar to distinguish can_hit vs no_hit.
    # Compare two vertical strips flanking the bar's left edge in HSV.
    # A solid bar brightens (V up) and desaturates (S down), so
    # avg(v_diff - s_diff) is high for can_hit, low for no_hit.
    h, w = frame.shape[:2]
    ax1, ax2 = int(0.254 * w), int(0.262 * w)
    bx1, bx2 = int(0.260 * w), int(0.268 * w)
    y_top, y_bot = int(0.48 * h), int(0.84 * h)
    NUM_CHUNKS = 24
    chunk_h = (y_bot - y_top) // NUM_CHUNKS

    strip_rgb = frame[y_top:y_bot, ax1:bx2]
    strip_hsv = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2HSV).astype(float)
    a_w = ax2 - ax1
    b_off = bx1 - ax1

    vs_diffs = []
    for i in range(NUM_CHUNKS):
        ly = i * chunk_h
        ca = strip_hsv[ly:ly + chunk_h, :a_w]
        cb = strip_hsv[ly:ly + chunk_h, b_off:b_off + (bx2 - bx1)]
        ma = np.mean(ca, axis=(0, 1))
        mb = np.mean(cb, axis=(0, 1))
        vs_diffs.append(abs(mb[2] - ma[2]) + abs(mb[1] - ma[1]))

    median_vs = float(np.median(vs_diffs))
    chunks_above = sum(1 for v in vs_diffs if v > 20)
    consistency = chunks_above / NUM_CHUNKS

    if median_vs > 35 and consistency > 0.6:
        return "stance_can_hit"

    return "stance_no_hit"


def is_in_stance(frame: np.ndarray) -> bool:
    """Detect if the player is in swing stance (any stance state)."""
    return detect_player_state(frame) in ("stance_no_hit", "stance_can_hit", "swinging")


def is_loading_screen(frame: np.ndarray) -> bool:
    """Check if a frame is the loading screen.

    The loading screen is a uniform dark purplish-grey with:
    - Mean RGB close to (50, 46, 50)
    - Very low standard deviation (< 15)
    """
    mean_rgb = np.mean(frame, axis=(0, 1))
    std = np.std(frame.astype(float))
    target = np.array([50, 46, 50])
    color_dist = np.linalg.norm(mean_rgb - target)
    return std < 15 and color_dist < 20


def detect_scoreboard(frame: np.ndarray) -> bool:
    """Detect the between-holes scoreboard screen.

    The scoreboard has a large cream/white panel covering the center of the
    screen with a dark teal banner at the top ("Next hole in X"). We detect
    it by checking for a high density of near-white pixels (V>220, S<40 in
    HSV) in the central region of the frame.
    """
    h, w = frame.shape[:2]
    # Sample the central panel area (roughly where the cream panel sits)
    center = frame[int(h * 0.08):int(h * 0.92), int(w * 0.18):int(w * 0.95)]
    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    # Cream/white pixels: high value, low saturation
    light_mask = (hsv[:, :, 2] > 220) & (hsv[:, :, 1] < 40)
    light_ratio = float(np.sum(light_mask)) / (center.shape[0] * center.shape[1])
    return light_ratio > 0.45


def get_player_progress(frame: np.ndarray) -> float | None:
    """Estimate player progress toward the hole from the top progress bar.

    The bar shows a green terrain strip with the player marked by a small
    white triangle/stem that creates a saturation dip in the green bar.
    The flag is at a fixed position on the right.

    We detect the player by finding the topmost green row (S>120) and
    looking for columns with notably lower saturation — the marker breaks
    the green with a white/light pixel.

    Returns a value 0.0-1.0 (0=tee, 1=hole), or None if undetectable.
    """
    bar = _crop_frac(frame, PROGRESS_BAR_LEFT, PROGRESS_BAR_TOP,
                     PROGRESS_BAR_RIGHT, PROGRESS_BAR_BOTTOM)
    h, w = bar.shape[:2]
    if h < 5 or w < 50:
        return None
    hsv = cv2.cvtColor(bar, cv2.COLOR_RGB2HSV)

    # Find the green bar rows (high saturation, excluding edge columns)
    row_mean_s = np.mean(hsv[:, 10:-20, 1], axis=1)
    green_rows = np.where(row_mean_s > 120)[0]
    if len(green_rows) == 0:
        return None

    # The topmost green row is where the marker is most visible
    top_green = green_rows[0]
    flag_x = int(w * PROGRESS_FLAG_FRAC)
    search_start = 10          # skip left edge/rounded corner
    search_end = flag_x - 5    # stop before flag
    if search_end <= search_start:
        return None

    # Find columns where saturation is notably lower than the green median
    row_s = hsv[top_green, search_start:search_end, 1].astype(float)
    median_s = float(np.median(row_s))
    if median_s < 50:
        return None  # not a green bar (loading screen, etc.)
    threshold = median_s * 0.65

    low_cols = np.where(row_s < threshold)[0] + search_start
    if len(low_cols) == 0:
        return None

    # Cluster contiguous columns, keep narrow ones (marker is 1-3px)
    clusters: list[np.ndarray] = []
    run_start = 0
    for i in range(1, len(low_cols)):
        if low_cols[i] - low_cols[i - 1] > 3:
            clusters.append(low_cols[run_start:i])
            run_start = i
    clusters.append(low_cols[run_start:])

    narrow = [c for c in clusters if len(c) <= 5]
    if not narrow:
        return None

    # Rightmost narrow cluster = furthest progress toward flag
    cluster = narrow[-1]
    player_x = int(np.mean(cluster))

    progress = player_x / flag_x
    return float(np.clip(progress, 0.0, 1.0))


def is_out_of_bounds(frame: np.ndarray) -> bool:
    """Detect the 'Out of bounds / Eliminated in X' red banner.

    The banner is a dark red/maroon rectangle in the upper-center of the
    screen. We detect it by looking for a high density of red pixels
    in a tight region where the banner appears.
    """
    h, w = frame.shape[:2]
    # Banner region: center horizontal, upper-third vertical
    region = frame[int(h * 0.28):int(h * 0.38), int(w * 0.35):int(w * 0.65)]
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    # Dark red/maroon banner: H near 0 or >170 (red wraps), S>150, V 120-200
    red_mask = (((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 170)) &
                (hsv[:, :, 1] > 150) &
                (hsv[:, :, 2] > 120) & (hsv[:, :, 2] < 200))
    red_ratio = float(np.sum(red_mask)) / (region.shape[0] * region.shape[1])
    return red_ratio > 0.05


def read_strokes_text(frame: np.ndarray) -> bool:
    """Detect if the 'Strokes' text is visible (indicates a shot has been taken).

    Returns True if visible, False otherwise.
    The strokes counter appears at top-left after the first shot.
    """
    # Strokes text region (top-left, below progress bar)
    crop = _crop_frac(frame, 0.0, 0.10, 0.13, 0.16)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    white_pixels = np.sum(gray > 200)
    # "Strokes X" text will have a cluster of white pixels
    return white_pixels > 50


def _load_template(name: str) -> np.ndarray:
    """Load a BGRA template from the templates directory."""
    path = pathlib.Path(__file__).parent / "templates" / name
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Template not found: {path}")
    return img


# Templates loaded once at module level
_pin_template = _load_template("pin.png")
_ball_templates = [_load_template("ball_v2.png"), _load_template("ball_v3.png"),
                   _load_template("ball_v4.png")]

_PIN_SCALES = (0.6, 1.0)
_MATCH_THRESHOLD = 0.96
_BALL_MATCH_THRESHOLD = 0.92


def _match_template(frame_bgr: np.ndarray, template_bgra: np.ndarray,
                    threshold: float = _MATCH_THRESHOLD,
                    scales: tuple[float, ...] = (1.0,),
                    ) -> list[tuple[int, int, float, float]]:
    """Match a BGRA template at multiple scales using TM_CCORR_NORMED.

    Returns list of (cx, cy, score, scale) for matches above threshold,
    sorted by score descending.
    """
    t_bgr = template_bgra[:, :, :3]
    t_alpha = template_bgra[:, :, 3] if template_bgra.shape[2] == 4 else None
    fh, fw = frame_bgr.shape[:2]
    matches: list[tuple[int, int, float, float]] = []

    for scale in scales:
        th = int(t_bgr.shape[0] * scale)
        tw = int(t_bgr.shape[1] * scale)
        if th < 5 or tw < 5 or th > fh or tw > fw:
            continue
        s_bgr = cv2.resize(t_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        if t_alpha is not None:
            s_mask = cv2.resize(t_alpha, (tw, th), interpolation=cv2.INTER_NEAREST)
            _, s_mask = cv2.threshold(s_mask, 128, 255, cv2.THRESH_BINARY)
            result = cv2.matchTemplate(frame_bgr, s_bgr, cv2.TM_CCORR_NORMED,
                                       mask=s_mask)
        else:
            result = cv2.matchTemplate(frame_bgr, s_bgr, cv2.TM_CCORR_NORMED)

        for _ in range(3):
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val < threshold:
                break
            mx, my = max_loc
            cx, cy = mx + tw // 2, my + th // 2
            matches.append((cx, cy, float(max_val), scale))
            # Suppress this area
            x1 = max(0, mx - tw // 2)
            y1 = max(0, my - th // 2)
            x2 = min(result.shape[1], mx + tw + tw // 2)
            y2 = min(result.shape[0], my + th + th // 2)
            result[y1:y2, x1:x2] = 0.0

    # Sort by score descending, deduplicate nearby matches
    matches.sort(key=lambda m: -m[2])
    filtered: list[tuple[int, int, float, float]] = []
    for cx, cy, score, scale in matches:
        if all(abs(cx - fx) > 25 or abs(cy - fy) > 25
               for fx, fy, _, _ in filtered):
            filtered.append((cx, cy, score, scale))
        if len(filtered) >= 3:
            break
    return filtered


def _in_exclusion_zone(cx: int, cy: int, h: int, w: int,
                       icon: str = "pin") -> bool:
    """Check if a point is in a UI exclusion zone.

    Pin exclusion: progress bar area (top 12% & left 40%).
    Ball exclusion: top 12% full-width (distance markers), bottom 12%
    (distance markers + prompts).
    """
    if icon == "pin":
        if cx < w * 0.40 and cy < h * 0.12:
            return True  # progress bar flag looks like pin icon
    elif icon == "ball":
        if cy < h * 0.12:
            return True  # distance marker icons across top
        if cy > h * 0.88:
            return True  # distance marker icons at bottom
    return False


def _count_orange(hsv: np.ndarray, cx: int, cy: int,
                  h: int, w: int) -> int:
    """Count orange flag pixels around a point."""
    oy1 = max(0, cy - 20)
    oy2 = min(h, cy + 10)
    ox1 = max(0, cx - 15)
    ox2 = min(w, cx + 15)
    roi = hsv[oy1:oy2, ox1:ox2]
    return int(((roi[:, :, 0] < 25) & (roi[:, :, 1] > 100) &
                (roi[:, :, 2] > 130)).sum())


def _count_green_ring(hsv: np.ndarray, cx: int, cy: int,
                      h: int, w: int, radius: int = 15) -> int:
    """Count icon-green pixels in a region around a point."""
    ry1 = max(0, cy - radius)
    ry2 = min(h, cy + radius)
    rx1 = max(0, cx - radius)
    rx2 = min(w, cx + radius)
    ring = hsv[ry1:ry2, rx1:rx2]
    return int(((ring[:, :, 0] >= 53) & (ring[:, :, 0] <= 72) &
                (ring[:, :, 1] >= 70) &
                (ring[:, :, 2] >= 110) & (ring[:, :, 2] <= 220)).sum())


def _count_white_center(hsv: np.ndarray, cx: int, cy: int,
                        h: int, w: int, radius: int = 10) -> int:
    """Count white pixels in the center of an icon."""
    wy1 = max(0, cy - radius)
    wy2 = min(h, cy + radius)
    wx1 = max(0, cx - radius)
    wx2 = min(w, cx + radius)
    center = hsv[wy1:wy2, wx1:wx2]
    return int(((center[:, :, 1] < 50) & (center[:, :, 2] > 200)).sum())


def _prepare_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert RGB frame to BGR and HSV (cached for reuse)."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)


def find_icons(frame: np.ndarray) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Find both pin and ball icons in a single pass (shared color conversion).

    Returns (pin_pos, ball_pos) — each is (cx, cy) or None.
    """
    h, w = frame.shape[:2]
    frame_bgr, hsv = _prepare_frame(frame)

    # Pin icon
    pin_pos = None
    for cx, cy, score, scale in _match_template(frame_bgr, _pin_template, scales=_PIN_SCALES):
        if _in_exclusion_zone(cx, cy, h, w, icon="pin"):
            continue
        orange = _count_orange(hsv, cx, cy, h, w)
        if orange < 3 or orange > 100:
            continue
        green = _count_green_ring(hsv, cx, cy, h, w)
        if green < 15:
            continue
        pin_pos = (cx, cy)
        break

    # Ball icon — pre-scaled templates, matched at 1.0 only, stop on first hit
    ball_pos = None
    for tmpl in _ball_templates:
        for cx, cy, score, scale in _match_template(
                frame_bgr, tmpl, threshold=_BALL_MATCH_THRESHOLD, scales=(1.0,)):
            if _in_exclusion_zone(cx, cy, h, w, icon="ball"):
                continue
            orange = _count_orange(hsv, cx, cy, h, w)
            if orange >= 5:
                continue
            green = _count_green_ring(hsv, cx, cy, h, w)
            if green < 8:
                continue
            white = _count_white_center(hsv, cx, cy, h, w)
            if white > 200:
                continue  # distance markers have 250-310 white pixels
            ball_pos = (cx, cy)
            break
        if ball_pos is not None:
            break

    return pin_pos, ball_pos


def find_pin_icon(frame: np.ndarray) -> tuple[int, int] | None:
    """Find the pin icon via template matching + color validation.

    Requires orange flag pixels and icon-green ring around the match.
    Frame is RGB. Returns (cx, cy) or None.
    """
    return find_icons(frame)[0]


def find_ball_icon(frame: np.ndarray) -> tuple[int, int] | None:
    """Find the ball icon via template matching + color validation.

    Requires white center, green ring, and no orange flag.
    Frame is RGB. Returns (cx, cy) or None.
    """
    return find_icons(frame)[1]


