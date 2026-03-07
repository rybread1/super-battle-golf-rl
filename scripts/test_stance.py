"""Debug stance detection live — test multiple detection methods.

Usage: uv run python scripts/test_stance.py --no-launch

Stand near the ball, enter and exit stance (right-click), and watch the output.
Saves a frame from each state for analysis.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from sbg.window import setup_game_window
from sbg.capture import ScreenCapture


def check_stance_signals(frame):
    """Check multiple possible stance indicators and return their values."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Signal 1: Bottom-right angle buttons (dark squares, only in stance)
    angle_btns = frame[h - 120:h - 20, w - 80:w - 10]
    angle_gray = cv2.cvtColor(angle_btns, cv2.COLOR_RGB2GRAY)
    dark_ratio = np.sum((angle_gray > 30) & (angle_gray < 90)) / angle_gray.size

    # Signal 2: Bottom-left "Adjust Angle" text area
    # In stance: shows icon + "Adjust Angle"
    # Not in stance near ball: "Swing Stance [HOLD]"
    # Not near ball: nothing or "Golf Club"
    bottom_left = frame[h - 35:h - 8, 50:200]
    bl_gray = cv2.cvtColor(bottom_left, cv2.COLOR_RGB2GRAY)
    bl_white = int(np.sum(bl_gray > 200))

    # Signal 3: Orange angle triangle (bottom-right, large orange triangle)
    # This triangle is bigger/more prominent in stance
    orange_region = frame[h - 80:h - 20, w - 140:w - 60]
    orange_hsv = cv2.cvtColor(orange_region, cv2.COLOR_RGB2HSV)
    orange_mask = (orange_hsv[:, :, 0] < 25) & (orange_hsv[:, :, 1] > 150) & (orange_hsv[:, :, 2] > 150)
    orange_ratio = np.mean(orange_mask)

    # Signal 4: Power bar mint-green color in left-center
    # The bar has a specific light green (H=65-85, S=40-120, V=140-220)
    left_region = hsv[int(h * 0.40):int(h * 0.85), int(w * 0.20):int(w * 0.35)]
    mint_mask = (left_region[:, :, 0] > 55) & (left_region[:, :, 0] < 90) & \
                (left_region[:, :, 1] > 30) & (left_region[:, :, 1] < 130) & \
                (left_region[:, :, 2] > 130)
    mint_ratio = np.mean(mint_mask)

    return dark_ratio, bl_white, orange_ratio, mint_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()

    hwnd, region = setup_game_window(launch=not args.no_launch)
    cap = ScreenCapture(region=region, fps=30)
    cap.start()
    time.sleep(1)

    os.makedirs("screenshots", exist_ok=True)

    print("Monitoring 4 stance signals for 20 seconds.")
    print("Enter/exit stance with right-click near ball.\n")
    print(f"{'Time':>5} | {'DarkBtns':>8} | {'BotText':>7} | {'Orange':>6} | {'Mint':>6}")
    print("-" * 48)

    start = time.time()
    frame_num = 0

    while time.time() - start < 20:
        frame = cap.grab()
        if frame is None:
            time.sleep(0.1)
            continue

        elapsed = time.time() - start
        dark, text, orange, mint = check_stance_signals(frame)

        print(f"{elapsed:5.1f} | {dark:8.3f} | {text:7d} | {orange:6.3f} | {mint:6.3f}")

        # Save select frames
        if frame_num % 4 == 0:
            cv2.imwrite(
                f"screenshots/stance_dbg_{frame_num:03d}_{elapsed:.1f}s.png",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )

        frame_num += 1
        time.sleep(0.5)

    cap.stop()
    print(f"\nSaved debug frames to screenshots/")


if __name__ == "__main__":
    main()
