"""Test screen capture — run this first to verify capture works and identify UI regions.

Usage: uv run python scripts/test_capture.py

This will:
1. Capture screenshots and save them so you can inspect the game layout
2. Help you identify pixel regions for score, game state, etc.
3. Verify frame rate is acceptable
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from sbg.capture import ScreenCapture


def main():
    print("Starting screen capture in 3 seconds...")
    print("Switch to the game window now!")
    time.sleep(3)

    cap = ScreenCapture(monitor=0, fps=30)
    cap.start()

    # Capture a single frame and save it
    frame = cap.grab()
    if frame is not None:
        cv2.imwrite("test_frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Saved test_frame.png — shape: {frame.shape}, dtype: {frame.dtype}")
        print("Open this image to identify score regions, game state areas, etc.")
    else:
        print("ERROR: Got None frame. Try mss fallback.")

    # Benchmark capture FPS
    print("\nBenchmarking capture rate for 5 seconds...")
    count = 0
    start = time.time()
    while time.time() - start < 5:
        f = cap.grab()
        if f is not None:
            count += 1
    elapsed = time.time() - start
    print(f"Captured {count} frames in {elapsed:.1f}s = {count / elapsed:.1f} FPS")

    # Interactive: click to get pixel coordinates
    print("\nShowing live capture. Press 'q' to quit, click to print coordinates.")

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"  Clicked: ({x}, {y}) — pixel value: {param['frame'][y, x]}")

    cv2.namedWindow("Game Capture")
    frame_ref = {"frame": frame}
    cv2.setMouseCallback("Game Capture", on_click, frame_ref)

    while True:
        frame = cap.grab()
        if frame is not None:
            frame_ref["frame"] = frame
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Game Capture", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.stop()


if __name__ == "__main__":
    main()
