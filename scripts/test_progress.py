"""Diagnostic: test progress bar detection.

Grabs frames every second and prints progress bar readings.
Also saves cropped progress bar images so you can visually verify.

Run with the game on a hole. Walk around to see values change.
"""

import time
import os

import cv2
import numpy as np

from sbg.capture import ScreenCapture
from sbg.detect import get_player_progress, _crop_frac
from sbg.window import find_game_window, position_window, get_client_region


def main():
    print("Finding game window...")
    hwnd = find_game_window(timeout=5)
    position_window(hwnd)
    time.sleep(0.3)
    region = get_client_region(hwnd)

    cap = ScreenCapture(region=region, fps=30)
    cap.start()
    time.sleep(0.5)

    os.makedirs("debug_progress", exist_ok=True)

    print("Reading progress bar every second. Press Ctrl+C to stop.\n")
    print(f"{'Time':>6s}  {'Progress':>10s}  {'Player X':>10s}  {'Flag X':>10s}  Saved")
    print("-" * 65)

    try:
        start = time.time()
        i = 0
        while True:
            frame = cap.grab()
            if frame is None:
                time.sleep(0.1)
                continue

            progress = get_player_progress(frame)

            # Extract debug info manually
            bar = _crop_frac(frame, 0.02, 0.02, 0.33, 0.08)
            h, w = bar.shape[:2]
            gray = cv2.cvtColor(bar, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(bar, cv2.COLOR_RGB2HSV)

            # Find player marker
            dark_mask = gray < 40
            white_mask = gray > 200
            player_x = None
            best_score = 0
            win = 30
            for x in range(0, w - win):
                dark_count = np.sum(dark_mask[:, x:x + win])
                white_count = np.sum(white_mask[:, x:x + win])
                if dark_count > 80 and white_count > 10:
                    score = dark_count + white_count * 3
                    if score > best_score:
                        best_score = score
                        player_x = x + win // 2

            # Find flag
            orange_mask = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & \
                          (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 150)
            flag_cols = np.where(orange_mask.any(axis=0))[0]
            flag_x = int(np.mean(flag_cols)) if len(flag_cols) > 0 else None

            # Save annotated bar image
            bar_bgr = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)
            if player_x is not None:
                cv2.line(bar_bgr, (player_x, 0), (player_x, h), (0, 255, 0), 1)
            if flag_x is not None:
                cv2.line(bar_bgr, (flag_x, 0), (flag_x, h), (0, 0, 255), 1)
            fname = f"debug_progress/bar_{i:03d}.png"
            cv2.imwrite(fname, bar_bgr)

            elapsed = time.time() - start
            p_str = f"{progress:.3f}" if progress is not None else "None"
            px_str = str(player_x) if player_x is not None else "None"
            fx_str = str(flag_x) if flag_x is not None else "None"
            print(f"{elapsed:6.1f}s  {p_str:>10s}  {px_str:>10s}  {fx_str:>10s}  {fname}")

            i += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopped.")

    cap.stop()
    print(f"\nSaved {i} debug images to debug_progress/")


if __name__ == "__main__":
    main()
