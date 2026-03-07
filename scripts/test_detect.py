"""Test detection functions on saved gameplay screenshots.

Usage: uv run python scripts/test_detect.py
"""

import sys
import os
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
from sbg.detect import (
    is_in_stance, is_loading_screen, get_player_progress, read_strokes_text,
    detect_ball_distance, detect_pin_distance,
)


def main():
    frames = sorted(glob.glob("screenshots/gameplay/0*.png"))
    if not frames:
        print("No screenshots found in screenshots/gameplay/")
        return

    sample = frames

    print(f"Testing detection on {len(sample)} frames")
    print(f"{'Frame':>30} | {'Stance':>6} | {'Progress':>8} | {'BallDist':>8} | {'PinDist':>8}")
    print("-" * 80)

    for path in sample:
        name = os.path.basename(path)
        img = cv2.imread(path)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        stance = is_in_stance(frame)
        progress = get_player_progress(frame)
        ball_dist = detect_ball_distance(frame)
        pin_dist = detect_pin_distance(frame)

        prog_str = f"{progress:.3f}" if progress is not None else "None"
        ball_str = f"{ball_dist}m" if ball_dist is not None else "None"
        pin_str = f"{pin_dist}m" if pin_dist is not None else "None"
        print(f"{name:>30} | {str(stance):>6} | {prog_str:>8} | {ball_str:>8} | {pin_str:>8}")


if __name__ == "__main__":
    main()
