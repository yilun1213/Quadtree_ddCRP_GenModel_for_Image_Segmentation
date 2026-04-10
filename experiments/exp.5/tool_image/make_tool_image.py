"""
Cuts out annotated regions from nail.png and nut.png using labelme JSON annotations.
Everything outside the annotated polygons is made transparent.
Outputs: tool_image/nail_extracted.png, tool_image/nut_extracted.png
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


SOURCE_DIR = Path(__file__).parent / "source"
OUTPUT_DIR = Path(__file__).parent

TARGETS = [
    ("nail.png", "nail.json", "nail_extracted.png"),
    ("nut.png",  "nut.json",  "nut_extracted.png"),
]


def make_mask(width: int, height: int, shapes: list) -> np.ndarray:
    """Return a boolean mask (H, W) that is True inside any annotated polygon."""
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for shape in shapes:
        if shape.get("shape_type") != "polygon":
            continue
        pts = [(x, y) for x, y in shape["points"]]
        draw.polygon(pts, fill=255)
    return np.array(mask_img) > 0


def extract(image_path: Path, json_path: Path, output_path: Path) -> None:
    with json_path.open(encoding="utf-8") as f:
        ann = json.load(f)

    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    mask = make_mask(w, h, ann["shapes"])

    rgba = np.array(img)
    rgba[~mask, 3] = 0          # transparent outside annotated regions

    result = Image.fromarray(rgba, "RGBA")
    result.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for img_name, json_name, out_name in TARGETS:
        extract(
            SOURCE_DIR / img_name,
            SOURCE_DIR / json_name,
            OUTPUT_DIR / out_name,
        )
