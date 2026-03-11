"""Reorder session 2 frames that have duplicate numbers with session 1.

Session 1 files (high timestamps, already in annotations.json) are left alone.
Session 2 files (low timestamps, duplicating frame numbers) get renumbered
starting after the last session 1 frame.

Usage:
    uv run python scripts/tools/reorder_frames.py --dir cnn_training --dry-run
    uv run python scripts/tools/reorder_frames.py --dir cnn_training
"""

import argparse
import json
import os
import re
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Renumber session 2 frames")
    parser.add_argument("--dir", required=True, help="Subdirectory under screenshots/")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without renaming")
    args = parser.parse_args()

    img_dir = f"screenshots/{args.dir}"
    ann_path = os.path.join(img_dir, "annotations.json")

    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    print(f"Found {len(files)} PNGs in {img_dir}")

    # Parse filenames: NNNN_Ts.png
    parsed = []
    for f in files:
        m = re.match(r"(\d+)_(.+)s\.png", f)
        if m:
            parsed.append((int(m.group(1)), float(m.group(2)), f))

    # Find duplicated frame numbers
    num_counts = Counter(num for num, ts, f in parsed)
    dup_nums = {n for n, c in num_counts.items() if c > 1}

    if not dup_nums:
        print("No duplicates found, nothing to do.")
        return

    # Session 1 = original recording (keep as-is)
    # Session 2 = re-recorded, starting where duplicates begin with low timestamps
    # For duplicated numbers: high timestamp = session 1, low timestamp = session 2
    # For non-duplicated numbers below the dup range: always session 1
    # For non-duplicated numbers above the dup range: session 2
    min_dup = min(dup_nums)
    session1 = []
    session2 = []
    for num, ts, f in parsed:
        if num in dup_nums:
            # For duplicates, session 1 had been recording longer (higher timestamps)
            other_ts = [t for n, t, _ in parsed if n == num and t != ts]
            if other_ts and ts > other_ts[0]:
                session1.append((num, ts, f))
            else:
                session2.append((num, ts, f))
        elif num < min_dup:
            session1.append((num, ts, f))
        else:
            # Non-duplicated number above the dup range = session 2 continuation
            session2.append((num, ts, f))

    session2.sort(key=lambda x: (x[0], x[1]))

    # Find the highest frame number across ALL session 1 files (not just duplicates)
    all_s1_nums = [num for num, ts, f in parsed if f not in {f for _, _, f in session2}]
    next_num = max(all_s1_nums) + 1 if all_s1_nums else 1

    print(f"Session 1: {len(session1)} files (keeping as-is)")
    print(f"Session 2: {len(session2)} files (renumbering from {next_num:04d})")

    rename_map = {}
    for i, (num, ts, old_name) in enumerate(session2):
        new_name = f"{next_num + i:04d}_{ts:.1f}s.png"
        rename_map[old_name] = new_name

    if args.dry_run:
        for old, new in list(rename_map.items())[:15]:
            print(f"  {old} -> {new}")
        if len(rename_map) > 15:
            print(f"  ... and {len(rename_map) - 15} more")
        print(f"\nRun without --dry-run to apply.")
        return

    # Rename via temp names to avoid collisions
    tmp_map = {}
    for old_name in rename_map:
        tmp_name = f"__tmp_{old_name}"
        os.rename(os.path.join(img_dir, old_name), os.path.join(img_dir, tmp_name))
        tmp_map[tmp_name] = rename_map[old_name]

    for tmp_name, new_name in tmp_map.items():
        os.rename(os.path.join(img_dir, tmp_name), os.path.join(img_dir, new_name))

    print(f"Renamed {len(rename_map)} files")

    # Update annotations.json if any session 2 files were already annotated
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            data = json.load(f)

        updated = 0
        for entry in data["frames"]:
            old = entry["file"]
            if old in rename_map:
                entry["file"] = rename_map[old]
                updated += 1

        if updated:
            with open(ann_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Updated {updated} entries in annotations.json")
        else:
            print("No annotation entries needed updating")

    print("Done!")


if __name__ == "__main__":
    main()
