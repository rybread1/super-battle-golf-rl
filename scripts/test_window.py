"""Test game window setup — launch, find, position, and capture a frame.

Usage: uv run python scripts/test_window.py [--no-launch]

Use --no-launch if the game is already running.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sbg.window import setup_game_window
from sbg.capture import ScreenCapture
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-launch", action="store_true", help="Skip launching the game")
    args = parser.parse_args()

    hwnd, region = setup_game_window(launch=not args.no_launch)

    print(f"\nCapture region: {region}")
    print("Capturing a test frame...")

    cap = ScreenCapture(region=region, fps=30)
    cap.start()

    import time
    time.sleep(1)  # Let capture stabilize

    frame = cap.grab()
    if frame is not None:
        out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite("test_window_frame.png", out)
        print(f"Saved test_window_frame.png — shape: {frame.shape}")
    else:
        print("ERROR: Could not capture frame")

    cap.stop()
    print("\nWindow is positioned. You can now visually verify the game is at the expected location.")


if __name__ == "__main__":
    main()
