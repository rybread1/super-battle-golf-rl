"""Launch Super Battle Golf and navigate to a match.

Usage:
    uv run python scripts/start_game.py             # Launch game and navigate
    uv run python scripts/start_game.py --no-launch  # Game already running
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
    parser = argparse.ArgumentParser(description="Launch and start a Super Battle Golf match")
    parser.add_argument("--no-launch", action="store_true", help="Skip launching — game must already be running")
    args = parser.parse_args()

    hwnd, region = setup_game_window(launch=not args.no_launch)

    cap = ScreenCapture(region=region, fps=30)
    cap.start()
    time.sleep(1)

    print(f"\nClient region: {region}")
    print("Starting navigation in 3 seconds — make sure game is at main menu!\n")
    time.sleep(3)

    navigate_to_match(hwnd, region, cap)

    cap.stop()
    print(f"\nGame is ready. Window handle: {hwnd}, capture region: {region}")


if __name__ == "__main__":
    main()
