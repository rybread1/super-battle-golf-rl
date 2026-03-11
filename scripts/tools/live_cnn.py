"""Live CNN overlay — shows model predictions on the game in real time.

Usage:
    uv run python scripts/tools/live_cnn.py --checkpoint runs/cnn_train/v7_best.pt
    uv run python scripts/tools/live_cnn.py --checkpoint runs/cnn_train/v7_best.pt --no-launch --threshold 0.3

Keys: Q=quit, T=cycle threshold (0.3/0.5/0.7)
"""

import argparse
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from sbg.game.capture import ScreenCapture
from sbg.game.window import find_game_window, get_client_region, setup_game_window
from sbg.models.icon_net import IconNet, TARGETS

TARGET_COLORS = {
    "ball_icon": (0, 255, 0),    # green
    "pin_icon": (0, 165, 255),   # orange
    "ball": (255, 255, 0),       # cyan
    "pin": (0, 0, 255),          # red
}

THRESHOLDS = [0.3, 0.5, 0.7]


def main():
    parser = argparse.ArgumentParser(description="Live CNN prediction overlay")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--no-launch", action="store_true", help="Don't launch the game")
    parser.add_argument("--threshold", type=float, default=0.5, help="Initial presence threshold")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = IconNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {args.checkpoint} ({device})")

    # Setup game capture
    if args.no_launch:
        hwnd = find_game_window()
        region = get_client_region(hwnd)
    else:
        hwnd, region = setup_game_window()

    cap = ScreenCapture(region=region, fps=30)
    cap.start()
    time.sleep(1)

    threshold = args.threshold
    fps_history = []
    prev_time = time.time()

    print(f"Live overlay running. Targets: {', '.join(TARGETS)}")
    print(f"Keys: Q=quit, T=cycle threshold ({'/'.join(str(t) for t in THRESHOLDS)})")

    # Precompute input tensor on GPU for speed
    while True:
        frame = cap.grab()
        if frame is None:
            time.sleep(0.01)
            continue

        t0 = time.time()

        # Run inference
        with torch.no_grad():
            preds = model.predict(frame, device=device, threshold=threshold)

        infer_ms = (time.time() - t0) * 1000

        # Draw overlay
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = display.shape[:2]

        for key in TARGETS:
            color = TARGET_COLORS.get(key, (255, 255, 255))
            pred = preds.get(key)
            if pred is not None:
                px, py = pred
                cv2.circle(display, (px, py), 14, color, 2)
                cv2.drawMarker(display, (px, py), color, cv2.MARKER_CROSS, 10, 1)
                cv2.putText(display, key, (px + 16, py + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # FPS tracking
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps_history.append(1.0 / dt)
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps = np.mean(fps_history) if fps_history else 0

        # Info panel
        cv2.rectangle(display, (0, 0), (280, 80), (30, 30, 30), -1)
        cv2.putText(display, f"FPS: {fps:.0f}  Infer: {infer_ms:.0f}ms", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"Threshold: {threshold:.1f}  (T to cycle)", (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Detection summary
        detected = [k for k in TARGETS if preds.get(k) is not None]
        not_detected = [k for k in TARGETS if preds.get(k) is None]
        summary = f"Detected: {', '.join(detected) if detected else 'none'}"
        cv2.putText(display, summary, (8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if detected else (0, 0, 200), 1, cv2.LINE_AA)

        cv2.imshow("CNN Live", display)
        cv2.moveWindow("CNN Live", region[0] + region[2] + 10, region[1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            idx = THRESHOLDS.index(threshold) if threshold in THRESHOLDS else 0
            threshold = THRESHOLDS[(idx + 1) % len(THRESHOLDS)]
            print(f"Threshold: {threshold}")

    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
