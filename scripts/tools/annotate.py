"""Annotate game frames with ball/pin icon and object locations for CNN training.

Usage:
    uv run python scripts/tools/annotate.py --dir cnn_train
    uv run python scripts/tools/annotate.py --dir cnn_train --start 50

For each frame, you annotate four targets in order:
    1. Ball icon (UI indicator)
    2. Pin icon (UI flag indicator)
    3. Ball (actual golf ball in the scene)
    4. Pin (actual flagstick on the green)

Controls:
    Left click  — mark location
    SPACE       — not present (null)
    Z           — undo (go back one frame)
    Q           — save and quit

Annotations are saved to screenshots/<dir>/annotations.json on quit.
Resumes from where you left off if annotations already exist.
"""

import argparse
import json
import os

import cv2

WINDOW_NAME = "Annotate"
TARGET_ORDER = ["ball_icon", "pin_icon", "ball", "pin"]
TARGET_LABELS = {
    "ball_icon": "BALL ICON (UI indicator)",
    "pin_icon": "PIN ICON (UI flag)",
    "ball": "BALL (actual golf ball)",
    "pin": "PIN (actual flagstick)",
}
TARGET_COLORS = {
    "ball_icon": (0, 255, 0),    # green
    "pin_icon": (0, 165, 255),   # orange
    "ball": (255, 255, 0),       # cyan
    "pin": (0, 0, 255),          # red
}


def _delete_files(paths):
    for p in paths:
        try:
            os.remove(p)
            print(f"  Deleted {p}")
        except OSError as e:
            print(f"  Could not delete {p}: {e}")
    if paths:
        print(f"Deleted {len(paths)} junk frames")


def load_annotations(path):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return {entry["file"]: entry for entry in data["frames"]}
    return {}


def save_annotations(path, annotations, all_files):
    frames = []
    for f in all_files:
        if f in annotations:
            frames.append(annotations[f])
    with open(path, "w") as f:
        json.dump({"frames": frames}, f, indent=2)


def draw_frame(img_bgr, filename, target_idx, current_annotations, frame_idx, total):
    display = img_bgr.copy()
    h, w = display.shape[:2]

    # Draw existing annotations for this frame
    ann = current_annotations.get(filename, {})
    for target_name in TARGET_ORDER:
        coords = ann.get(target_name)
        if coords is not None:
            color = TARGET_COLORS[target_name]
            cx, cy = int(coords[0]), int(coords[1])
            cv2.drawMarker(display, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
            cv2.putText(display, target_name, (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Status bar
    if target_idx < len(TARGET_ORDER):
        label = TARGET_LABELS[TARGET_ORDER[target_idx]]
        prompt = f"Click {label} (SPACE=not present)"
    else:
        prompt = "Done"

    cv2.rectangle(display, (0, h - 32), (w, h), (30, 30, 30), -1)
    status = f"[{frame_idx+1}/{total}] {filename}  |  {prompt}"
    cv2.putText(display, status, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Key hints top-right
    hints = "Z=undo  D=delete  Q=save+quit"
    cv2.putText(display, hints, (w - 220, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Legend top-left
    for i, name in enumerate(TARGET_ORDER):
        color = TARGET_COLORS[name]
        done = "x" if ann.get(name) is not None else ("-" if i < target_idx and ann.get(name) is None else " ")
        marker = f"[{done}]" if i < target_idx else "[>]" if i == target_idx else "[ ]"
        cv2.putText(display, f"{marker} {name}", (8, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return display


def new_entry(filename):
    entry = {"file": filename}
    for key in TARGET_ORDER:
        entry[key] = None
    return entry


def main():
    parser = argparse.ArgumentParser(description="Annotate frames for CNN training")
    parser.add_argument("--dir", type=str, required=True, help="Subdirectory under screenshots/")
    parser.add_argument("--start", type=int, default=0, help="Start at frame index N")
    args = parser.parse_args()

    img_dir = f"screenshots/{args.dir}"
    ann_path = os.path.join(img_dir, "annotations.json")

    if not os.path.isdir(img_dir):
        print(f"Directory not found: {img_dir}")
        return

    all_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
    if not all_files:
        print(f"No PNG files found in {img_dir}")
        return

    annotations = load_annotations(ann_path)
    annotated_count = len(annotations)

    start_idx = args.start
    if start_idx == 0 and annotated_count > 0:
        for i, f in enumerate(all_files):
            if f not in annotations:
                start_idx = i
                break
        else:
            start_idx = len(all_files)

    print(f"Found {len(all_files)} frames, {annotated_count} already annotated")
    if start_idx >= len(all_files):
        print("All frames annotated! Use --start N to revisit.")
        return
    print(f"Starting at frame {start_idx}")
    print(f"Controls: click=mark, SPACE=not present, Z=undo, D=delete frame, Q=save+quit")
    deleted = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    click_pos = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    frame_idx = start_idx
    saved_since_last = 0

    while frame_idx < len(all_files):
        filename = all_files[frame_idx]
        img_path = os.path.join(img_dir, filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Could not load {img_path}, skipping")
            frame_idx += 1
            continue

        current = annotations.get(filename, new_entry(filename))
        target_idx = 0

        while target_idx < len(TARGET_ORDER):
            display = draw_frame(img_bgr, filename, target_idx, {filename: current},
                                 frame_idx, len(all_files))
            cv2.imshow(WINDOW_NAME, display)
            click_pos[0] = None

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                annotations[filename] = current
                save_annotations(ann_path, annotations, all_files)
                _delete_files(deleted)
                print(f"Saved {len(annotations)} annotations to {ann_path}")
                cv2.destroyAllWindows()
                return

            elif key == ord("d"):
                # Delete junk frame
                annotations.pop(filename, None)
                deleted.append(img_path)
                print(f"  Marked for deletion: {filename}")
                frame_idx += 1
                all_files = [f for f in all_files if f != filename]
                break

            elif key == ord("z"):
                if frame_idx > 0:
                    frame_idx -= 1
                break

            elif key == ord(" "):
                target_name = TARGET_ORDER[target_idx]
                current[target_name] = None
                target_idx += 1

            elif click_pos[0] is not None:
                target_name = TARGET_ORDER[target_idx]
                x, y = click_pos[0]
                current[target_name] = [x, y]
                click_pos[0] = None
                target_idx += 1

        else:
            annotations[filename] = current
            saved_since_last += 1

            if saved_since_last % 20 == 0:
                save_annotations(ann_path, annotations, all_files)
                print(f"  Auto-saved ({len(annotations)} annotations)")

            display = draw_frame(img_bgr, filename, len(TARGET_ORDER), {filename: current},
                                 frame_idx, len(all_files))
            cv2.imshow(WINDOW_NAME, display)
            cv2.waitKey(300)

            frame_idx += 1

    save_annotations(ann_path, annotations, all_files)
    _delete_files(deleted)
    print(f"Done! Saved {len(annotations)} annotations to {ann_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
