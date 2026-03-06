"""Test the full menu navigation sequence.

Usage: uv run python scripts/test_navigate.py --no-launch

Make sure the game is at the main menu before running.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sbg.window import setup_game_window
from sbg.capture import ScreenCapture
from sbg.navigate import navigate_to_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()

    hwnd, region = setup_game_window(launch=not args.no_launch)

    cap = ScreenCapture(region=region, fps=30)
    cap.start()
    time.sleep(1)

    print(f"\nClient region: {region}")
    print("Starting navigation in 3 seconds — make sure game is at main menu!\n")
    time.sleep(3)

    navigate_to_match(hwnd, region, cap)

    # Take a screenshot of the result
    time.sleep(2)
    frame = cap.grab()
    if frame is not None:
        import cv2
        os.makedirs("screenshots", exist_ok=True)
        cv2.imwrite("screenshots/match_started.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("Saved screenshots/match_started.png")

    cap.stop()


if __name__ == "__main__":
    main()
