"""Record gameplay frames at regular intervals while you play.

Usage: uv run python scripts/record_gameplay.py --no-launch --duration 60

Play the game normally. Frames are saved every 0.5s to screenshots/gameplay/.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between captures")
    parser.add_argument("--dir", type=str, default="gameplay", help="Subdirectory name under screenshots/")
    args = parser.parse_args()

    hwnd, region = setup_game_window(launch=not args.no_launch)

    cap = ScreenCapture(region=region, fps=30)
    cap.start()
    time.sleep(1)

    out_dir = f"screenshots/{args.dir}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Recording for {args.duration}s (every {args.interval}s) — play the game!")
    print(f"Saving to {out_dir}/")
    print("Press Ctrl+C to stop early.\n")

    start = time.time()
    frame_num = 0

    try:
        while time.time() - start < args.duration:
            frame = cap.grab()
            if frame is not None:
                elapsed = time.time() - start
                path = f"{out_dir}/{frame_num:04d}_{elapsed:.1f}s.png"
                cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_num += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped early.")

    cap.stop()
    print(f"Saved {frame_num} frames to {out_dir}/")


if __name__ == "__main__":
    main()
