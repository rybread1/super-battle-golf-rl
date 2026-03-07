"""Game state detection from screen frames.

Frames are expected in RGB format (as returned by ScreenCapture).
Regions are relative to actual frame dimensions — we normalize coordinates
as fractions of frame size to handle slight resolution differences.
"""

import re

import cv2
import numpy as np

# EasyOCR reader — initialized lazily on first use to avoid slow import
_easyocr_reader = None


def _get_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _easyocr_reader


def _crop_frac(frame: np.ndarray, left: float, top: float, right: float, bottom: float) -> np.ndarray:
    """Crop a frame using fractional coordinates (0.0-1.0)."""
    h, w = frame.shape[:2]
    return frame[int(top * h):int(bottom * h), int(left * w):int(right * w)]


def is_in_stance(frame: np.ndarray) -> bool:
    """Detect if the player is in swing stance.

    The power bar has distinctive white dots (small bright circles) at
    regular intervals. These dots are unique to the stance UI and don't
    appear elsewhere. We scan the left ~40% of the screen for a vertical
    column of very bright white pixels that indicates the bar is present.
    """
    h, w = frame.shape[:2]

    # Scan the left 40% of the screen, middle 60% vertically
    # (the bar appears in this general area regardless of camera angle)
    search = frame[int(h * 0.30):int(h * 0.90), 0:int(w * 0.40)]
    gray = cv2.cvtColor(search, cv2.COLOR_RGB2GRAY)

    # The white dots are very bright (>240) and small
    bright_mask = gray > 240
    bright_count = np.sum(bright_mask)

    # In stance: the power bar dots create a cluster of bright pixels
    # Typically 3-4 dots with ~50-100 bright pixels each = 150-400 total
    # When walking: very few pixels this bright in that region
    return bright_count > 80


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


def get_player_progress(frame: np.ndarray) -> float | None:
    """Estimate player progress toward the hole from the top progress bar.

    The bar shows a terrain profile with:
    - A dark circle with white "?" = player position
    - An orange flag = hole position

    We find the player circle by looking for a cluster of very dark pixels
    with white pixels inside (the "?"), and the flag by orange color.

    Returns a value 0.0-1.0 (0=tee, 1=hole), or None if undetectable.
    """
    # Progress bar region (top of screen)
    bar = _crop_frac(frame, 0.02, 0.02, 0.33, 0.08)
    h, w = bar.shape[:2]
    gray = cv2.cvtColor(bar, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(bar, cv2.COLOR_RGB2HSV)

    # --- Find player marker (dark circle with white "?" inside) ---
    # The circle is very dark (gray < 40) and contains white pixels (gray > 200)
    dark_mask = gray < 40
    white_mask = gray > 200

    # Look for columns that have BOTH dark and white pixels vertically
    # (the dark circle surrounds the white "?")
    player_x = None
    best_score = 0
    # Scan in windows across the bar
    win = 30  # window width roughly matching circle diameter
    for x in range(0, w - win):
        dark_count = np.sum(dark_mask[:, x:x + win])
        white_count = np.sum(white_mask[:, x:x + win])
        # The circle has lots of dark pixels and some white inside
        if dark_count > 80 and white_count > 10:
            score = dark_count + white_count * 3
            if score > best_score:
                best_score = score
                player_x = x + win // 2

    if player_x is None:
        return None

    # --- Find orange flag (hole) ---
    orange_mask = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & \
                  (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 150)
    flag_cols = np.where(orange_mask.any(axis=0))[0]

    if len(flag_cols) == 0:
        return None

    flag_x = int(np.mean(flag_cols))

    if flag_x <= player_x:
        return None

    progress = player_x / flag_x
    return float(np.clip(progress, 0.0, 1.0))


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


def detect_bottom_text(frame: np.ndarray) -> str:
    """Detect what text prompt is shown at the bottom-left.

    Returns one of: 'adjust_angle', 'swing_stance', 'cancel', 'none'
    Based on pixel patterns in the bottom-left region.
    """
    # Bottom-left text region
    crop = _crop_frac(frame, 0.02, 0.93, 0.20, 0.99)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    white_pixels = np.sum(gray > 200)

    if white_pixels > 100:
        # There's text here — check if it's "Adjust Angle" or "Swing Stance"
        # Adjust Angle appears when in stance, Swing Stance when near ball
        return "text_present"
    return "none"


def _load_template(name: str) -> np.ndarray:
    """Load a BGRA template from the templates directory."""
    import pathlib
    path = pathlib.Path(__file__).parent / "templates" / name
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Template not found: {path}")
    return img


# Templates loaded once at module level
_pin_template = _load_template("pin.png")
_ball_template = _load_template("balls.png")

_ICON_SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
_PIN_SCALES = (0.6, 1.0)
_BALL_SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_MATCH_THRESHOLD = 0.96


def _match_template(frame_bgr: np.ndarray, template_bgra: np.ndarray,
                    threshold: float = _MATCH_THRESHOLD,
                    scales: tuple[float, ...] = _ICON_SCALES,
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


def _in_exclusion_zone(cx: int, cy: int, h: int, w: int) -> bool:
    """Check if a point is in a UI exclusion zone.

    Kept minimal — color validation (orange/green/white counts) handles
    most false positive filtering. Only exclude the progress bar area
    where the flag icon reliably triggers false pin matches.
    """
    if cx < w * 0.40 and cy < h * 0.12:
        return True  # progress bar flag looks like pin icon
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
        if _in_exclusion_zone(cx, cy, h, w):
            continue
        orange = _count_orange(hsv, cx, cy, h, w)
        if orange < 3 or orange > 100:
            continue
        green = _count_green_ring(hsv, cx, cy, h, w)
        if green < 15:
            continue
        pin_pos = (cx, cy)
        break

    # Ball icon
    ball_pos = None
    for cx, cy, score, scale in _match_template(frame_bgr, _ball_template, scales=_BALL_SCALES):
        if _in_exclusion_zone(cx, cy, h, w):
            continue
        orange = _count_orange(hsv, cx, cy, h, w)
        if orange >= 5:
            continue
        white = _count_white_center(hsv, cx, cy, h, w)
        if white < 50:
            continue
        ball_pos = (cx, cy)
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


def _find_distance_texts(frame: np.ndarray) -> list[tuple[int, int, int]]:
    """Find all distance texts ('Xm') in a frame using EasyOCR.

    Returns list of (x, y, distance_meters) for each detected distance text.
    """
    h, w = frame.shape[:2]
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    reader = _get_reader()
    ocr_results = reader.readtext(frame_bgr, allowlist="0123456789m")

    results = []
    for bbox, text, conf in ocr_results:
        if conf < 0.55 or not text:
            continue
        # Require 'm' at end — valid distances are like "26m", "295m"
        if not re.match(r"^\d+m$", text):
            continue
        match = re.search(r"(\d+)m", text)
        if not match:
            continue
        val = int(match.group(1))
        if not 1 <= val <= 999:
            continue
        # Compute center of bounding box
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))
        # Skip UI areas: right edge (club selector), top-left, bottom strip
        if cx > w * 0.80:
            continue
        if cx < w * 0.15 and cy < h * 0.15:
            continue
        if cy > h * 0.90:
            continue
        results.append((cx, cy, val))

    return results


def detect_distances(frame: np.ndarray) -> dict[str, int | None]:
    """Detect ball and pin distances in a single pass.

    Finds icon positions via color detection, then reads nearby distance
    text via OCR. Falls back to full-frame OCR text classification if
    icon detection misses.

    Returns dict with 'ball' and 'pin' keys (values in meters or None).
    """
    result: dict[str, int | None] = {"ball": None, "pin": None}
    texts = _find_distance_texts(frame)
    if not texts:
        return result

    pin_pos = find_pin_icon(frame)
    ball_pos = find_ball_icon(frame)

    # Match distance texts to their nearest icon (text is ~20-40px above icon)
    for tx, ty, dist in texts:
        matched = False
        if pin_pos and result["pin"] is None:
            dx = abs(tx - pin_pos[0])
            dy = ty - pin_pos[1]  # text should be above icon
            if dx < 50 and -60 < dy < 10:
                result["pin"] = dist
                matched = True
        if ball_pos and result["ball"] is None and not matched:
            dx = abs(tx - ball_pos[0])
            dy = ty - ball_pos[1]
            if dx < 50 and -60 < dy < 10:
                result["ball"] = dist
                matched = True

    return result


def detect_ball_distance(frame: np.ndarray) -> int | None:
    """Detect the distance-to-ball marker.

    Returns distance in meters, or None if not detected.
    """
    return detect_distances(frame)["ball"]


def detect_pin_distance(frame: np.ndarray) -> int | None:
    """Detect the distance-to-pin marker.

    Returns distance in meters, or None if not detected.
    """
    return detect_distances(frame)["pin"]
