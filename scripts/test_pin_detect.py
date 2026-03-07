"""Test pin/ball detection using strict template matching + color validation."""

import cv2
import numpy as np
import pathlib

TEMPLATES_DIR = pathlib.Path("src/sbg/templates")
SCREENSHOTS = sorted(pathlib.Path("screenshots/gameplay").glob("*.png"))
OUTPUT_DIR = pathlib.Path("screenshots/pin_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_template(name: str) -> np.ndarray:
    path = TEMPLATES_DIR / name
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Template not found: {path}")
    return img


PIN_TMPL = load_template("pin.png")
BALL_TMPL = load_template("balls.png")

SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)


def _match_template(frame_bgr, template_bgra, threshold, scales):
    """Match template at multiple scales using TM_CCORR_NORMED with mask."""
    t_bgr = template_bgra[:, :, :3]
    t_alpha = template_bgra[:, :, 3] if template_bgra.shape[2] == 4 else None
    fh, fw = frame_bgr.shape[:2]
    matches = []

    for scale in scales:
        th = int(t_bgr.shape[0] * scale)
        tw = int(t_bgr.shape[1] * scale)
        if th < 5 or tw < 5 or th > fh or tw > fw:
            continue
        s_bgr = cv2.resize(t_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        if t_alpha is not None:
            s_mask = cv2.resize(t_alpha, (tw, th), interpolation=cv2.INTER_NEAREST)
            _, s_mask = cv2.threshold(s_mask, 128, 255, cv2.THRESH_BINARY)
            result = cv2.matchTemplate(frame_bgr, s_bgr, cv2.TM_CCORR_NORMED, mask=s_mask)
        else:
            result = cv2.matchTemplate(frame_bgr, s_bgr, cv2.TM_CCORR_NORMED)

        for _ in range(3):
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val < threshold:
                break
            mx, my = max_loc
            cx, cy = mx + tw // 2, my + th // 2
            matches.append((cx, cy, float(max_val), scale))
            x1, y1 = max(0, mx - tw // 2), max(0, my - th // 2)
            x2, y2 = min(result.shape[1], mx + tw + tw // 2), min(result.shape[0], my + th + th // 2)
            result[y1:y2, x1:x2] = 0.0

    matches.sort(key=lambda m: -m[2])
    filtered = []
    for cx, cy, score, scale in matches:
        if all(abs(cx - fx) > 25 or abs(cy - fy) > 25 for fx, fy, _, _ in filtered):
            filtered.append((cx, cy, score, scale))
        if len(filtered) >= 3:
            break
    return filtered


def _in_exclusion_zone(cx, cy, h, w):
    if cx < w * 0.40 and cy < h * 0.12:
        return True
    if cx > w * 0.78:
        return True
    if cy > h * 0.95:
        return True
    return False


def _count_orange(hsv, cx, cy, h, w):
    oy1, oy2 = max(0, cy - 20), min(h, cy + 10)
    ox1, ox2 = max(0, cx - 15), min(w, cx + 15)
    roi = hsv[oy1:oy2, ox1:ox2]
    return int(((roi[:, :, 0] < 25) & (roi[:, :, 1] > 100) & (roi[:, :, 2] > 130)).sum())


def _count_green_ring(hsv, cx, cy, h, w, radius=15):
    ry1, ry2 = max(0, cy - radius), min(h, cy + radius)
    rx1, rx2 = max(0, cx - radius), min(w, cx + radius)
    ring = hsv[ry1:ry2, rx1:rx2]
    return int(((ring[:, :, 0] >= 53) & (ring[:, :, 0] <= 72) &
                (ring[:, :, 1] >= 70) &
                (ring[:, :, 2] >= 110) & (ring[:, :, 2] <= 220)).sum())


def _count_white_center(hsv, cx, cy, h, w, radius=10):
    wy1, wy2 = max(0, cy - radius), min(h, cy + radius)
    wx1, wx2 = max(0, cx - radius), min(w, cx + radius)
    center = hsv[wy1:wy2, wx1:wx2]
    return int(((center[:, :, 1] < 50) & (center[:, :, 2] > 200)).sum())


def find_pin(frame_bgr, threshold=0.96):
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    matches = _match_template(frame_bgr, PIN_TMPL, threshold, SCALES)
    for cx, cy, score, scale in matches:
        if _in_exclusion_zone(cx, cy, h, w):
            continue
        orange = _count_orange(hsv, cx, cy, h, w)
        if orange < 3 or orange > 100:
            continue
        green = _count_green_ring(hsv, cx, cy, h, w)
        if green < 15:
            continue
        return (cx, cy, score)
    return None


def find_ball(frame_bgr, threshold=0.96):
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    matches = _match_template(frame_bgr, BALL_TMPL, threshold, SCALES)
    for cx, cy, score, scale in matches:
        if _in_exclusion_zone(cx, cy, h, w):
            continue
        orange = _count_orange(hsv, cx, cy, h, w)
        if orange >= 5:
            continue
        white = _count_white_center(hsv, cx, cy, h, w)
        if white < 50:
            continue
        return (cx, cy, score)
    return None


def main():
    print(f"Pin template: {PIN_TMPL.shape}, Ball template: {BALL_TMPL.shape}")
    print(f"Testing {len(SCREENSHOTS)} screenshots\n")

    pin_found = 0
    ball_found = 0

    for img_path in SCREENSHOTS:
        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            continue

        pin = find_pin(frame_bgr)
        ball = find_ball(frame_bgr)

        if pin:
            pin_found += 1
        if ball:
            ball_found += 1

        annotated = frame_bgr.copy()
        if pin:
            cv2.circle(annotated, (pin[0], pin[1]), 22, (0, 255, 0), 2)
            cv2.putText(annotated, f"PIN {pin[2]:.3f}", (pin[0] - 40, pin[1] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if ball:
            cv2.circle(annotated, (ball[0], ball[1]), 22, (255, 100, 0), 2)
            cv2.putText(annotated, f"BALL {ball[2]:.3f}", (ball[0] - 40, ball[1] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        pin_str = f"({pin[0]},{pin[1]} s={pin[2]:.3f})" if pin else "none"
        ball_str = f"({ball[0]},{ball[1]} s={ball[2]:.3f})" if ball else "none"
        print(f"{img_path.name}: pin={pin_str} ball={ball_str}")

        cv2.imwrite(str(OUTPUT_DIR / img_path.name), annotated)

    print(f"\nPin: {pin_found}/{len(SCREENSHOTS)}, Ball: {ball_found}/{len(SCREENSHOTS)}")
    print(f"Annotated images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
