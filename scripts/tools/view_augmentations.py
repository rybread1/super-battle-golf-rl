"""Flip through training images with augmentations applied.

Usage:
    uv run python scripts/tools/view_augmentations.py --data screenshots/cnn_training/annotations.json

Keys:
    Right/D  — next image
    Left/A   — previous image
    Space    — re-roll augmentation on current image
    O        — toggle original (no augmentation) vs augmented
    Q/Esc    — quit
"""

import argparse
import json
import pathlib

import cv2
import numpy as np

from sbg.models.icon_net import IconNet, TARGETS
from sbg.models.dataset import IconDataset


# Colors for each target (BGR)
COLORS = {
    "ball_icon": (0, 255, 0),    # green
    "pin_icon": (0, 165, 255),   # orange
    "ball": (255, 255, 0),       # cyan
    "pin": (0, 0, 255),          # red
}

DISPLAY_W, DISPLAY_H = 960, 540


def draw_annotations(img_rgb: np.ndarray, coords: dict, title: str) -> np.ndarray:
    """Draw annotation markers on the image and return BGR for display."""
    display = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # Scale up for display
    display = cv2.resize(display, (DISPLAY_W, DISPLAY_H),
                         interpolation=cv2.INTER_NEAREST)
    h, w = display.shape[:2]

    for key in TARGETS:
        c = coords.get(key)
        if c is not None:
            x_px = int(c[0] * w)
            y_px = int(c[1] * h)
            color = COLORS[key]
            cv2.drawMarker(display, (x_px, y_px), color,
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(display, key, (x_px + 12, y_px - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Title bar
    cv2.putText(display, title, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return display


def load_frame(dataset: IconDataset, idx: int, augment: bool):
    """Load a frame, optionally with augmentation, returning RGB image and coords."""
    entry = dataset.frames[idx]
    img_path = dataset.base_dir / entry["file"]
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    coords = {}
    for key in TARGETS:
        raw = entry.get(key)
        if raw is not None:
            coords[key] = (raw[0] / orig_w, raw[1] / orig_h)
        else:
            coords[key] = None

    img = cv2.resize(img, (IconNet.INPUT_W, IconNet.INPUT_H))

    if augment:
        img, coords = dataset._augment(img, coords)

    return img, coords, entry["file"]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training augmentations")
    parser.add_argument("--data", required=True,
                        help="Path to annotations JSON file")
    args = parser.parse_args()

    dataset = IconDataset(args.data, augment=True)
    n = len(dataset)
    print(f"Loaded {n} frames. Press Right/Left to navigate, "
          f"Space to re-roll, O to toggle original, Q to quit.")

    idx = 0
    show_original = False

    while True:
        img, coords, filename = load_frame(dataset, idx, augment=not show_original)
        mode = "ORIGINAL" if show_original else "AUGMENTED"
        title = f"[{idx+1}/{n}] {filename}  ({mode})"
        display = draw_annotations(img, coords, title)
        cv2.imshow("Augmentation Viewer", display)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):  # q or Esc
            break
        elif key in (ord('d'), 83, 0x27):  # d, right arrow
            idx = (idx + 1) % n
        elif key in (ord('a'), 81, 0x25):  # a, left arrow
            idx = (idx - 1) % n
        elif key == ord(' '):  # space — re-roll augmentation
            pass  # just re-loops with new random augmentation
        elif key == ord('o'):
            show_original = not show_original

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
