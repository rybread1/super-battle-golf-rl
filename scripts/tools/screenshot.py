"""Take a screenshot of the game and save it with a label.

Usage: uv run python scripts/screenshot.py --no-launch --name main_menu
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sbg.game.capture import ScreenCapture
from sbg.game.window import setup_game_window
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--name", default="screenshot", help="Label for the saved file")
    args = parser.parse_args()

    hwnd, region = setup_game_window(launch=not args.no_launch)

    cap = ScreenCapture(region=region, fps=30)
    cap.start()
    time.sleep(1)

    frame = cap.grab()
    if frame is not None:
        path = f"screenshots/{args.name}.png"
        os.makedirs("screenshots", exist_ok=True)
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Saved {path} — shape: {frame.shape}")
        print(f"Open it and note the pixel coordinates of buttons you need to click.")
    else:
        print("ERROR: Could not capture frame")

    cap.stop()


if __name__ == "__main__":
    main()
