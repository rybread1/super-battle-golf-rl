"""Calibrate menu button positions by clicking on them.

Usage: uv run python scripts/calibrate_menu.py --no-launch

Shows the game capture in a window. Click on each button/location as prompted
and the coordinates will be recorded and printed as config at the end.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
from sbg.capture import ScreenCapture
from sbg.window import setup_game_window

STEPS = [
    "Click on the 'Start Game' button",
    "Click on the 'Start Match' button (or wherever you click to begin the match)",
    # Add more steps as needed
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()

    hwnd, region = setup_game_window(launch=not args.no_launch)

    cap = ScreenCapture(region=region, fps=30)
    cap.start()
    time.sleep(1)

    clicks = {}
    step_idx = 0

    def on_click(event, x, y, flags, param):
        nonlocal step_idx
        if event == cv2.EVENT_LBUTTONDOWN and step_idx < len(STEPS):
            step_name = STEPS[step_idx]
            clicks[step_name] = (x, y)
            print(f"  Recorded: ({x}, {y}) for '{step_name}'")
            step_idx += 1
            if step_idx < len(STEPS):
                print(f"\nNext: {STEPS[step_idx]}")
            else:
                print("\nAll steps recorded! Press 'q' to finish.")

    cv2.namedWindow("Calibrate Menu")
    cv2.setMouseCallback("Calibrate Menu", on_click)

    print(f"=== Menu Calibration ===")
    print(f"Window client region: {region}")
    print(f"\nClick on the game capture window (NOT the actual game).")
    print(f"The coordinates are relative to the game's client area.\n")
    print(f"Step: {STEPS[0]}")

    while True:
        frame = cap.grab()
        if frame is not None:
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw crosshairs for recorded clicks
            for name, (cx, cy) in clicks.items():
                cv2.drawMarker(display, (cx, cy), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)
                cv2.putText(display, name[:30], (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Show current instruction
            if step_idx < len(STEPS):
                cv2.putText(display, f"Click: {STEPS[step_idx]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Done! Press 'q' to finish", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Calibrate Menu", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.stop()

    print("\n=== Results ===")
    print("Add these to configs/default.yaml under a 'navigation' section:\n")
    print("navigation:")
    for name, (x, y) in clicks.items():
        key = name.lower().replace("'", "").replace(" ", "_")
        # Shorten the key name
        for prefix in ["click_on_the_"]:
            if key.startswith(prefix):
                key = key[len(prefix):]
        print(f"  {key}:")
        print(f"    x: {x}")
        print(f"    y: {y}")


if __name__ == "__main__":
    main()
