"""Record gameplay frames while you play, with a live preview.

Usage:
    uv run python scripts/record_gameplay.py --dir ball_icons
    uv run python scripts/record_gameplay.py --dir ball_icons --interval 0.2
    uv run python scripts/record_gameplay.py --no-launch --duration 120

Play the game normally. Press SPACE in the preview window to save the
current frame (handy for capturing specific moments). Frames also
auto-save at the configured interval. Press 'q' to stop.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
from sbg.game.window import find_game_window, get_client_region, setup_game_window
from sbg.game.capture import ScreenCapture


def main():
    parser = argparse.ArgumentParser(description="Record gameplay frames with live preview")
    parser.add_argument("--no-launch", action="store_true", help="Don't launch the game (must already be running)")
    parser.add_argument("--duration", type=float, default=300.0, help="Max recording duration in seconds")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between auto-captures (0=manual only)")
    parser.add_argument("--dir", type=str, default="gameplay", help="Subdirectory name under screenshots/")
    args = parser.parse_args()

    if args.no_launch:
        hwnd = find_game_window()
        region = get_client_region(hwnd)
    else:
        hwnd, region = setup_game_window()

    # Verify client region matches expected game resolution
    expected_w, expected_h = 1280, 720
    actual_w, actual_h = region[2], region[3]
    if actual_w != expected_w or actual_h != expected_h:
        print(f"WARNING: Client region is {actual_w}x{actual_h}, expected {expected_w}x{expected_h}")
        print("Frames may not match the vision pipeline. Check game window settings.")

    cap = ScreenCapture(region=region, fps=30)
    cap.start()
    time.sleep(1)

    # Verify first captured frame dimensions match
    test_frame = cap.grab()
    attempts = 0
    while test_frame is None and attempts < 10:
        time.sleep(0.1)
        test_frame = cap.grab()
        attempts += 1
    if test_frame is not None:
        fh, fw = test_frame.shape[:2]
        print(f"Capture resolution: {fw}x{fh} ({test_frame.shape[2]}ch {test_frame.dtype})")
        if fw != expected_w or fh != expected_h:
            print(f"ERROR: Captured frame is {fw}x{fh}, expected {expected_w}x{expected_h}")
            print("Vision pipeline and CNN training expect exact 1280x720 frames.")
            cap.stop()
            return
    else:
        print("WARNING: Could not grab a test frame from capture")

    out_dir = f"screenshots/{args.dir}"
    os.makedirs(out_dir, exist_ok=True)

    # Count existing frames to avoid overwriting
    existing = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    frame_num = len(existing)

    auto_mode = args.interval > 0
    mode_str = f"every {args.interval}s" if auto_mode else "SPACE to capture"
    print(f"Recording to {out_dir}/ ({mode_str})")
    print(f"Max duration: {args.duration}s  |  Starting at frame {frame_num}")
    print("Keys: SPACE=save frame, q=quit")

    start = time.time()
    last_save = 0.0
    saved_count = 0

    try:
        while time.time() - start < args.duration:
            frame = cap.grab()
            if frame is None:
                time.sleep(0.01)
                continue

            elapsed = time.time() - start

            # Draw status bar on preview (don't modify the frame we save)
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = display.shape[:2]
            status = f"Frame: {frame_num} | Saved: {saved_count} | {elapsed:.0f}s / {args.duration:.0f}s"
            cv2.rectangle(display, (0, h - 28), (w, h), (30, 30, 30), -1)
            cv2.putText(display, status, (8, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Record Gameplay", display)
            cv2.moveWindow("Record Gameplay", region[0] + region[2] + 10, region[1])
            key = cv2.waitKey(16) & 0xFF

            should_save = False
            if key == ord("q"):
                break
            elif key == ord(" "):
                should_save = True

            # Auto-save on interval
            if auto_mode and (elapsed - last_save) >= args.interval:
                should_save = True

            if should_save:
                path = os.path.join(out_dir, f"{frame_num:04d}_{elapsed:.1f}s.png")
                cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_num += 1
                saved_count += 1
                last_save = elapsed

    except KeyboardInterrupt:
        print("\nStopped early.")

    cap.stop()
    cv2.destroyAllWindows()
    print(f"Saved {saved_count} frames to {out_dir}/")


if __name__ == "__main__":
    main()
