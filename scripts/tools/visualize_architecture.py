"""Generate a visual diagram of the IconNet architecture.

Usage:
    uv run python scripts/tools/visualize_architecture.py
    uv run python scripts/tools/visualize_architecture.py --output docs/architecture.png

Produces a PNG diagram showing the backbone, skip connections, and detection heads.
"""

import argparse
import os

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow is required: uv add pillow")
    raise


# Colors
BG = (25, 25, 30)
CONV_COLOR = (55, 120, 200)
SKIP_COLOR = (200, 140, 50)
FUSE_COLOR = (160, 90, 200)
HEAD_COLOR = (50, 180, 120)
HEATMAP_COLOR = (220, 80, 80)
POOL_COLOR = (100, 100, 120)
TEXT_COLOR = (240, 240, 240)
DIM_TEXT = (160, 160, 170)
ARROW_COLOR = (180, 180, 190)
LINE_COLOR = (80, 80, 90)


def rounded_rect(draw, xy, fill, radius=8):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def draw_arrow(draw, start, end, color=ARROW_COLOR, width=2):
    draw.line([start, end], fill=color, width=width)
    # Arrowhead
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx**2 + dy**2) ** 0.5
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    # Perpendicular
    px, py = -uy, ux
    size = 8
    tip = end
    left = (tip[0] - ux * size + px * size * 0.4, tip[1] - uy * size + py * size * 0.4)
    right = (tip[0] - ux * size - px * size * 0.4, tip[1] - uy * size - py * size * 0.4)
    draw.polygon([tip, left, right], fill=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/architecture.png")
    args = parser.parse_args()

    W, H = 1200, 900
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_sm = ImageFont.truetype("arial.ttf", 11)
        font_lg = ImageFont.truetype("arial.ttf", 18)
        font_title = ImageFont.truetype("arial.ttf", 24)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_sm = font
        font_lg = font
        font_title = font

    # Title
    draw.text((W // 2 - 120, 15), "IconNet Architecture", fill=TEXT_COLOR, font=font_title)

    # =========================================================================
    # Backbone blocks
    # =========================================================================
    blocks = [
        ("Input",    "3ch, 360x640",   None),
        ("Block 1",  "32ch, 180x320",  CONV_COLOR),
        ("Block 2",  "64ch, 90x160",   CONV_COLOR),
        ("Block 3",  "128ch, 45x80",   CONV_COLOR),
        ("Block 4",  "128ch, 22x40",   CONV_COLOR),
        ("Block 5",  "128ch, 11x20",   CONV_COLOR),
    ]

    bx_start = 60
    by_start = 80
    box_w = 150
    box_h = 50
    gap = 30
    block_centers = []

    for i, (name, dims, color) in enumerate(blocks):
        x = bx_start
        y = by_start + i * (box_h + gap)
        cx = x + box_w // 2
        cy = y + box_h // 2
        block_centers.append((cx, cy, x, y))

        if color:
            rounded_rect(draw, (x, y, x + box_w, y + box_h), fill=color)
            # Inner label showing conv details
            draw.text((x + 10, y + 8), name, fill=TEXT_COLOR, font=font)
            draw.text((x + 10, y + 28), dims, fill=DIM_TEXT, font=font_sm)
            # Conv detail on the right
            detail = "Conv3x3 + BN + ReLU + Pool2x2"
            draw.text((x + box_w + 15, cy - 6), detail, fill=DIM_TEXT, font=font_sm)
        else:
            # Input box
            rounded_rect(draw, (x, y, x + box_w, y + box_h), fill=(60, 60, 70))
            draw.text((x + 10, y + 8), name, fill=TEXT_COLOR, font=font)
            draw.text((x + 10, y + 28), "RGB 640x360", fill=DIM_TEXT, font=font_sm)

        # Arrow from previous block
        if i > 0:
            prev_cx, prev_cy, _, prev_y = block_centers[i - 1]
            draw_arrow(draw, (cx, prev_y + box_h), (cx, y))

    # =========================================================================
    # Skip connection (block4 -> fuse)
    # =========================================================================
    b4_cx, b4_cy, b4_x, b4_y = block_centers[4]
    b5_cx, b5_cy, b5_x, b5_y = block_centers[5]

    # Upsample box
    ups_x = 320
    ups_y = b5_y
    ups_w = 130
    ups_h = 50
    rounded_rect(draw, (ups_x, ups_y, ups_x + ups_w, ups_y + ups_h), fill=SKIP_COLOR)
    draw.text((ups_x + 8, ups_y + 8), "Upsample 2x", fill=TEXT_COLOR, font=font)
    draw.text((ups_x + 8, ups_y + 28), "bilinear 11x20->22x40", fill=(40, 40, 40), font=font_sm)

    # Arrow: block5 -> upsample
    draw_arrow(draw, (b5_x + box_w, b5_cy), (ups_x, ups_y + ups_h // 2), color=SKIP_COLOR)

    # Fuse box
    fuse_x = 520
    fuse_y = b4_y + (b5_y - b4_y) // 2 - 10
    fuse_w = 140
    fuse_h = 55
    rounded_rect(draw, (fuse_x, fuse_y, fuse_x + fuse_w, fuse_y + fuse_h), fill=FUSE_COLOR)
    draw.text((fuse_x + 8, fuse_y + 6), "Skip Fuse", fill=TEXT_COLOR, font=font)
    draw.text((fuse_x + 8, fuse_y + 24), "Cat(B4, B5_up)", fill=(60, 40, 60), font=font_sm)
    draw.text((fuse_x + 8, fuse_y + 38), "Conv1x1: 256->128", fill=(60, 40, 60), font=font_sm)

    # Arrow: block4 -> fuse
    draw_arrow(draw, (b4_x + box_w, b4_cy), (fuse_x, fuse_y + 15), color=SKIP_COLOR)
    # Arrow: upsample -> fuse
    draw_arrow(draw, (ups_x + ups_w, ups_y + ups_h // 2), (fuse_x, fuse_y + fuse_h - 15), color=SKIP_COLOR)

    # =========================================================================
    # Global pool
    # =========================================================================
    pool_x = fuse_x + fuse_w + 40
    pool_y = fuse_y + 5
    pool_w = 110
    pool_h = 45
    rounded_rect(draw, (pool_x, pool_y, pool_x + pool_w, pool_y + pool_h), fill=POOL_COLOR)
    draw.text((pool_x + 8, pool_y + 8), "Adaptive", fill=TEXT_COLOR, font=font)
    draw.text((pool_x + 8, pool_y + 26), "AvgPool -> 128", fill=DIM_TEXT, font=font_sm)

    # Arrow: fuse -> pool
    draw_arrow(draw, (fuse_x + fuse_w, fuse_y + fuse_h // 2), (pool_x, pool_y + pool_h // 2))

    # =========================================================================
    # Detection heads
    # =========================================================================
    targets = ["ball_icon", "pin_icon", "ball", "pin"]
    target_colors_rgb = {
        "ball_icon": (0, 220, 100),
        "pin_icon": (255, 165, 0),
        "ball": (0, 220, 220),
        "pin": (220, 60, 60),
    }

    head_start_x = 520
    head_start_y = fuse_y + fuse_h + 80
    head_w = 150
    head_h = 140
    head_gap = 15

    fuse_bottom = fuse_y + fuse_h
    pool_bottom = pool_y + pool_h

    for i, tgt in enumerate(targets):
        hx = head_start_x + i * (head_w + head_gap)
        hy = head_start_y
        tc = target_colors_rgb[tgt]

        # Head container
        draw.rounded_rectangle((hx, hy, hx + head_w, hy + head_h), radius=8,
                               outline=tc, width=2)

        # Title
        draw.text((hx + 10, hy + 6), tgt, fill=tc, font=font)

        # Heatmap conv
        hm_y = hy + 28
        rounded_rect(draw, (hx + 8, hm_y, hx + head_w - 8, hm_y + 38), fill=(60, 50, 50))
        draw.text((hx + 14, hm_y + 4), "Heatmap Conv", fill=HEATMAP_COLOR, font=font_sm)
        draw.text((hx + 14, hm_y + 20), "3x3->3x3->1x1", fill=DIM_TEXT, font=font_sm)

        # Soft-argmax
        sa_y = hm_y + 44
        rounded_rect(draw, (hx + 8, sa_y, hx + head_w - 8, sa_y + 22), fill=(40, 60, 50))
        draw.text((hx + 14, sa_y + 4), "Soft-Argmax -> (x,y)", fill=(150, 230, 150), font=font_sm)

        # Presence FC
        pf_y = sa_y + 28
        rounded_rect(draw, (hx + 8, pf_y, hx + head_w - 8, pf_y + 22), fill=(40, 40, 60))
        draw.text((hx + 14, pf_y + 4), "FC -> presence", fill=(150, 150, 230), font=font_sm)

        # Arrows from fuse/pool to heads
        # Spatial features arrow (from fuse)
        mid_x = hx + head_w // 3
        draw_arrow(draw, (fuse_x + fuse_w // 2, fuse_bottom),
                   (mid_x, hy), color=tc)
        # Pooled features arrow (from pool)
        mid_x2 = hx + 2 * head_w // 3
        draw_arrow(draw, (pool_x + pool_w // 2, pool_bottom),
                   (mid_x2, hy), color=POOL_COLOR)

    # =========================================================================
    # Loss labels
    # =========================================================================
    loss_y = head_start_y + head_h + 30
    draw.text((head_start_x, loss_y), "Loss Components:", fill=TEXT_COLOR, font=font_lg)
    losses = [
        ("BCE", "presence classification", (200, 200, 220)),
        ("Smooth L1 (10x)", "coordinate regression", (150, 230, 150)),
        ("MSE (5x)", "heatmap supervision (gaussian blob at GT)", HEATMAP_COLOR),
    ]
    for i, (name, desc, color) in enumerate(losses):
        ly = loss_y + 28 + i * 22
        draw.text((head_start_x + 10, ly), f"  {name}", fill=color, font=font)
        draw.text((head_start_x + 160, ly), desc, fill=DIM_TEXT, font=font_sm)

    # Gradient clipping note
    draw.text((head_start_x, loss_y + 100), "Gradient clipping: max_norm=1.0",
              fill=DIM_TEXT, font=font)

    # =========================================================================
    # Legend
    # =========================================================================
    leg_x = 60
    leg_y = H - 120
    draw.text((leg_x, leg_y), "Legend:", fill=TEXT_COLOR, font=font)
    legend_items = [
        (CONV_COLOR, "Conv Block"),
        (SKIP_COLOR, "Skip Connection"),
        (FUSE_COLOR, "Feature Fusion"),
        (HEAD_COLOR, "Detection Head"),
        (HEATMAP_COLOR, "Heatmap Supervision"),
    ]
    for i, (color, label) in enumerate(legend_items):
        lx = leg_x + (i % 3) * 180
        ly = leg_y + 25 + (i // 3) * 22
        draw.rectangle((lx, ly + 2, lx + 14, ly + 16), fill=color)
        draw.text((lx + 20, ly), label, fill=TEXT_COLOR, font=font_sm)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    img.save(args.output, "PNG")
    print(f"Saved architecture diagram to {args.output}")


if __name__ == "__main__":
    main()
