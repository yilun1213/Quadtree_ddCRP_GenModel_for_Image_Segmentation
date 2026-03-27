"""
Generates 512x512 composite images by randomly placing 1-4 nail and 1-4 nut
parts (from nail_extracted.png / nut_extracted.png) without overlap.
Parts are never resized; only random rotation is applied.
Background: Gaussian noise (mean=100, std=50), clipped to [0, 255].

Label image convention
  0 … background
  1 … nail
  2 … nut

Outputs
  nut_nail_data/train_data/images/  (50 images)
  nut_nail_data/train_data/labels/  (50 labels)
  nut_nail_data/test_data/images/   ( 2 images)
  nut_nail_data/test_data/labels/   ( 2 labels)
"""

import random
import numpy as np
from pathlib import Path
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
TOOL_DIR   = BASE_DIR / "tool_image"
NAIL_PATH  = TOOL_DIR / "nail_extracted.png"
NUT_PATH   = TOOL_DIR / "nut_extracted.png"

IMG_SIZE      = 512
PART_MAX_DIM  = 160   # longest dimension of each part after scaling
LABEL_NAIL    = 1
LABEL_NUT     = 2

TRAIN_N    = 50
TEST_N     = 2

# ── helpers ────────────────────────────────────────────────────────────────

def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def resize_part(img: Image.Image, max_dim: int) -> Image.Image:
    """Scale img so its longest dimension equals max_dim, preserving aspect ratio."""
    w, h = img.size
    scale = max_dim / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def rotate_rgba(img: Image.Image, angle: float) -> Image.Image:
    """Rotate RGBA image by angle degrees, expanding canvas, keeping alpha."""
    return img.rotate(angle, expand=True, resample=Image.BICUBIC)


def bbox_of_alpha(img: Image.Image):
    """Return bounding box (left, upper, right, lower) of non-transparent pixels."""
    alpha = np.array(img)[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1)


def crop_to_content(img: Image.Image) -> Image.Image:
    bb = bbox_of_alpha(img)
    if bb is None:
        return img
    return img.crop(bb)


def make_background(size: int) -> np.ndarray:
    """Return (H, W, 3) uint8 grayscale Gaussian noise array (mean=100, std=50)."""
    noise = np.random.normal(100, 50, (size, size))
    gray  = np.clip(noise, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=2)


def masks_overlap(placed: list, x: int, y: int, mask: np.ndarray) -> bool:
    """Return True if the candidate mask at (x, y) overlaps any placed mask."""
    ph, pw = mask.shape
    for (px, py, pmask) in placed:
        qh, qw = pmask.shape
        # check bounding-box intersection first
        if x + pw <= px or px + qw <= x:
            continue
        if y + ph <= py or py + qh <= y:
            continue
        # pixel-level check in the overlapping region
        ox1, oy1 = max(x, px), max(y, py)
        ox2, oy2 = min(x + pw, px + qw), min(y + ph, py + qh)
        c_mask  = mask   [oy1 - y  : oy2 - y,  ox1 - x  : ox2 - x]
        c_pmask = pmask  [oy1 - py : oy2 - py, ox1 - px : ox2 - px]
        if np.any(c_mask & c_pmask):
            return True
    return False


def try_place(canvas_size: int, part: Image.Image, placed: list,
              max_tries: int = 200):
    """
    Try to place `part` at a random position without overlapping placed parts.
    Returns (x, y, rotated_part) or None on failure.
    """
    angle = random.uniform(0, 360)
    rotated = crop_to_content(rotate_rgba(part, angle))
    pw, ph = rotated.size
    if pw > canvas_size or ph > canvas_size:
        return None

    alpha_mask = (np.array(rotated)[:, :, 3] > 0)

    for _ in range(max_tries):
        x = random.randint(0, canvas_size - pw)
        y = random.randint(0, canvas_size - ph)
        if not masks_overlap(placed, x, y, alpha_mask):
            return x, y, rotated, alpha_mask
    return None


def generate_image(nail_src: Image.Image, nut_src: Image.Image, size: int):
    """
    Returns (RGB image as np.ndarray uint8, label image as np.ndarray uint8).
    """
    bg = make_background(size)
    label = np.zeros((size, size), dtype=np.uint8)

    placed = []   # list of (x, y, alpha_mask bool array)

    parts = (
        [(nail_src, LABEL_NAIL)] * random.randint(1, 4) +
        [(nut_src,  LABEL_NUT )] * random.randint(1, 4)
    )
    random.shuffle(parts)

    for part_img, part_label in parts:
        result = try_place(size, part_img, placed)
        if result is None:
            continue   # skip if no room found
        x, y, rotated, alpha_mask = result
        pw, ph = rotated.size

        # composite onto background
        rot_arr = np.array(rotated)
        roi_bg  = bg[y:y+ph, x:x+pw]
        alpha   = rot_arr[:, :, 3:4].astype(np.float32) / 255.0
        roi_bg[:] = (rot_arr[:, :, :3] * alpha + roi_bg * (1 - alpha)).astype(np.uint8)

        # write label
        label[y:y+ph, x:x+pw][alpha_mask] = part_label

        placed.append((x, y, alpha_mask))

    return bg, label


def save_set(nail_src, nut_src, out_dir: Path, n: int, prefix: str):
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        img_arr, lbl_arr = generate_image(nail_src, nut_src, IMG_SIZE)
        name = f"{prefix}_{i:03d}"
        Image.fromarray(img_arr,  "RGB").save(img_dir / f"{name}.png")
        Image.fromarray(lbl_arr,  "L"  ).save(lbl_dir / f"{name}.png")
        print(f"  saved {name}")


if __name__ == "__main__":
    nail_src = resize_part(crop_to_content(load_rgba(NAIL_PATH)), PART_MAX_DIM)
    nut_src  = resize_part(crop_to_content(load_rgba(NUT_PATH)),  PART_MAX_DIM)

    print("Generating train data …")
    save_set(nail_src, nut_src, BASE_DIR / "train_data", TRAIN_N, "train")

    print("Generating test data …")
    save_set(nail_src, nut_src, BASE_DIR / "test_data",  TEST_N,  "test")

    print("Done.")
