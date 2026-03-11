"""Visualize CNN predictions on validation frames.

Shows each frame with ground truth (crosses) and predictions (circles).
Green = correct detection, Red = missed or false positive.

Usage:
    uv run python scripts/tools/visualize_predictions.py --data screenshots/cnn_training/annotations.json --checkpoint runs/cnn_train/v5_spatial_heatmap_best.pt
    uv run python scripts/tools/visualize_predictions.py --data screenshots/cnn_training/annotations.json --checkpoint runs/cnn_train/v5_spatial_heatmap_best.pt --split train
"""

import argparse
import pathlib

import cv2
import torch
from torch.utils.data import random_split

from sbg.models.icon_net import IconNet, TARGETS
from sbg.models.dataset import IconDataset

TARGET_COLORS = {
    "ball_icon": (0, 255, 0),    # green
    "pin_icon": (0, 165, 255),   # orange
    "ball": (255, 255, 0),       # cyan
    "pin": (0, 0, 255),          # red
}


def main():
    parser = argparse.ArgumentParser(description="Visualize CNN predictions")
    parser.add_argument("--data", required=True, help="Path to annotations JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Presence threshold")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val",
                        help="Which split to visualize")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = IconNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Load dataset (no augmentation for visualization)
    dataset = IconDataset(args.data, augment=False)
    base_dir = dataset.base_dir

    if args.split == "all":
        indices = list(range(len(dataset)))
    else:
        n_val = max(1, int(len(dataset) * 0.2))
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val])
        indices = val_set.indices if args.split == "val" else train_set.indices

    print(f"Showing {len(indices)} {args.split} frames. Keys: SPACE=next, Q=quit")

    for idx in indices:
        entry = dataset.frames[idx]
        filename = entry["file"]
        img_path = base_dir / filename
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        # Run prediction
        with torch.no_grad():
            preds = model.predict(img_rgb, device=device, threshold=args.threshold)

        display = img_bgr.copy()

        for key in TARGETS:
            color = TARGET_COLORS.get(key, (255, 255, 255))
            gt = entry.get(key)
            pred = preds.get(key)

            # Ground truth: cross marker
            if gt is not None:
                gx, gy = int(gt[0]), int(gt[1])
                cv2.drawMarker(display, (gx, gy), color, cv2.MARKER_CROSS, 24, 2)
                cv2.putText(display, f"{key} GT", (gx + 14, gy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            # Prediction: circle
            if pred is not None:
                px, py = pred
                cv2.circle(display, (px, py), 12, color, 2)
                prob = torch.sigmoid(torch.tensor(0.0)).item()  # placeholder
                cv2.putText(display, f"{key} pred", (px + 14, py + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

                # Draw line from GT to pred if both exist
                if gt is not None:
                    gx, gy = int(gt[0]), int(gt[1])
                    dist = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
                    line_color = (0, 200, 0) if dist < 50 else (0, 0, 200)
                    cv2.line(display, (gx, gy), (px, py), line_color, 1, cv2.LINE_AA)
                    cv2.putText(display, f"{dist:.0f}px", ((gx + px) // 2, (gy + py) // 2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, line_color, 1, cv2.LINE_AA)

            # Missing: GT exists but no prediction
            if gt is not None and pred is None:
                gx, gy = int(gt[0]), int(gt[1])
                cv2.putText(display, "MISS", (gx + 14, gy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            # False positive: prediction but no GT
            if gt is None and pred is not None:
                px, py = pred
                cv2.putText(display, "FP", (px + 14, py + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # Status bar
        cv2.rectangle(display, (0, h - 28), (w, h), (30, 30, 30), -1)
        cv2.putText(display, f"{filename}  |  SPACE=next  Q=quit", (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Predictions", display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
