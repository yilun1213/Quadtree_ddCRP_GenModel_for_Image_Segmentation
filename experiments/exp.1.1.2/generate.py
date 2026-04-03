from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
import sys
import importlib

import numpy as np
from PIL import Image, ImageDraw

# Ensure imports from workspace root are available when script is run directly.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

QuadTree = importlib.import_module("model.quadtree.depth_dependent_model").QuadTree


# exp. 1.1.2 settings from experiments/exp_plan.md
MAX_DEPTH = 7
BRANCH_PROBS = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]
NUM_TREES = 50
SEED = 20260403


def save_quadtree_image(leaves: list, image_size: int, out_path: Path) -> None:
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    color_map: dict[tuple[int, int], np.ndarray] = {}

    for leaf in leaves:
        key = (leaf.upper_edge, leaf.left_edge)
        if key not in color_map:
            color_map[key] = np.random.randint(40, 255, size=3, dtype=np.uint8)

    for leaf in leaves:
        key = (leaf.upper_edge, leaf.left_edge)
        image[
            leaf.upper_edge:leaf.upper_edge + leaf.size,
            leaf.left_edge:leaf.left_edge + leaf.size,
        ] = color_map[key]

    Image.fromarray(image).save(out_path)


def save_depth_histogram(depth_counter: Counter[int], out_path: Path) -> None:
    depths = list(range(MAX_DEPTH + 1))
    counts = [depth_counter.get(depth, 0) for depth in depths]

    width = 900
    height = 540
    margin_left = 90
    margin_right = 30
    margin_top = 55
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Axes
    x0 = margin_left
    y0 = height - margin_bottom
    x1 = width - margin_right
    y1 = margin_top
    draw.line((x0, y0, x1, y0), fill="black", width=2)
    draw.line((x0, y0, x0, y1), fill="black", width=2)

    max_count = max(counts) if counts else 1
    max_count = max(max_count, 1)
    bar_slot = plot_width / len(depths)
    bar_width = int(bar_slot * 0.65)

    for i, depth in enumerate(depths):
        count = counts[i]
        bar_h = int((count / max_count) * (plot_height * 0.95))
        left = int(x0 + i * bar_slot + (bar_slot - bar_width) / 2)
        top = y0 - bar_h
        right = left + bar_width
        draw.rectangle((left, top, right, y0), fill=(46, 111, 149), outline="black", width=1)

        # X tick labels
        draw.text((left + bar_width // 2 - 4, y0 + 8), str(depth), fill="black")

        # Count labels
        draw.text((left + bar_width // 2 - 10, max(top - 18, y1)), str(count), fill="black")

    # Y helper ticks
    for frac in (0.25, 0.5, 0.75, 1.0):
        y_tick = int(y0 - frac * plot_height)
        val = int(round(frac * max_count))
        draw.line((x0 - 6, y_tick, x0, y_tick), fill="black", width=1)
        draw.text((10, y_tick - 8), str(val), fill="black")

    draw.text((width // 2 - 170, 15), "Leaf Depth Distribution (exp. 1.1.2)", fill="black")
    draw.text((width // 2 - 18, height - 35), "Depth", fill="black")
    draw.text((15, 20), "Number of Leaves", fill="black")
    img.save(out_path)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    root = Path(__file__).resolve().parent
    out_dir = root / "outputs"
    quadtree_dir = out_dir / "quadtree_images"
    quadtree_dir.mkdir(parents=True, exist_ok=True)

    depth_counter: Counter[int] = Counter()

    for idx in range(NUM_TREES):
        qt_seed = SEED + idx
        qt = QuadTree(max_depth=MAX_DEPTH, branch_prob=BRANCH_PROBS, seed=qt_seed)
        leaves = qt.get_leaves()

        for leaf in leaves:
            depth_counter[leaf.depth] += 1

        save_quadtree_image(
            leaves=leaves,
            image_size=2 ** MAX_DEPTH,
            out_path=quadtree_dir / f"quadtree_{idx:03d}.png",
        )

    save_depth_histogram(
        depth_counter=depth_counter,
        out_path=out_dir / "leaf_depth_distribution.png",
    )

    print("exp. 1.1.2 quadtree generation completed")
    print(f"  trees: {NUM_TREES}")
    print(f"  seed: {SEED}")
    print(f"  branch_probs: {BRANCH_PROBS}")
    print(f"  quadtree images: {quadtree_dir}")
    print(f"  depth histogram: {out_dir / 'leaf_depth_distribution.png'}")


if __name__ == "__main__":
    main()
