from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.pixel import ar_3dmatrix_rgb  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
REGION_PATH = THIS_DIR / "region_000.png"
LABEL_PATH = THIS_DIR / "label_000.png"
OUT_DIR = THIS_DIR / "outputs"
NUM_IMAGES = 3


def _load_region_ids(region_img: np.ndarray) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]]:
    h, w, _ = region_img.shape
    flat = region_img.reshape(-1, 3).astype(np.uint32)
    keys = (flat[:, 0] << 16) + (flat[:, 1] << 8) + flat[:, 2]
    _, inverse = np.unique(keys, return_inverse=True)
    region_id_map = inverse.reshape(h, w).astype(np.int32)

    region_dict: dict[int, set[tuple[int, int]]] = {}
    for i in range(h):
        for j in range(w):
            region_dict.setdefault(int(region_id_map[i, j]), set()).add((i, j))

    return region_id_map, region_dict


def _to_label_index(label_img: np.ndarray) -> np.ndarray:
    mapping = {0: 0, 1: 1, 2: 2, 128: 1, 255: 2}
    out = np.zeros_like(label_img, dtype=np.uint8)
    for src_val, dst_val in mapping.items():
        out[label_img == src_val] = dst_val
    return out


def _majority_label_per_region(region_id_map: np.ndarray, label_idx_img: np.ndarray) -> np.ndarray:
    out = np.zeros_like(label_idx_img, dtype=np.uint8)
    for rid in range(int(region_id_map.max()) + 1):
        mask = region_id_map == rid
        labels = label_idx_img[mask]
        if labels.size == 0:
            continue
        out[mask] = int(np.argmax(np.bincount(labels, minlength=3)))
    return out


def _theta() -> dict:
    # Helper: scalar * I_3
    def sI(s: float) -> list:
        return [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]]

    # Helper: diagonal 3x3
    def d3(a: float, b: float, c: float) -> list:
        return [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]

    # ----- x=0: vertical banding -----
    # Strong positive AR in vertical direction (Δi < 0), weak horizontal
    # (-1,0): diag(0.40, 0.36, 0.32)  det=4.608e-2
    # (-2,0): diag(0.18, 0.16, 0.14)  det=4.032e-3
    # (-3,0): diag(0.07, 0.06, 0.05)  det=2.100e-4
    # (0,-1): 0.03·I_3                det=2.700e-5
    x0_ar = {
        "(-1,0)": d3(0.40, 0.36, 0.32),
        "(-2,0)": d3(0.18, 0.16, 0.14),
        "(-3,0)": d3(0.07, 0.06, 0.05),
        "(0,-1)": sI(0.03),
    }

    # ----- x=1: cross-channel color mixing -----
    # Off-diagonal elements couple RGB channels, creating color-correlated texture
    # (-1,0):   det≈5.55e-3
    # (0,-1):   det≈3.36e-3
    # (-1,-1):  det≈5.37e-4
    # (-2,0):   det≈2.84e-4
    # (0,-2):   det≈1.16e-4
    x1_ar = {
        "(-1,0)":  [[0.20, 0.06, 0.02], [0.01, 0.18, 0.05], [0.02, 0.01, 0.16]],
        "(0,-1)":  [[0.16, 0.01, 0.04], [0.04, 0.15, 0.01], [-0.01, 0.03, 0.14]],
        "(-1,-1)": [[0.10, 0.03, 0.01], [0.01, 0.08, 0.02], [0.01, 0.00, 0.07]],
        "(-2,0)":  [[0.08, 0.02, 0.01], [0.00, 0.06, 0.01], [0.01, 0.00, 0.06]],
        "(0,-2)":  [[0.06, 0.01, 0.02], [0.02, 0.05, 0.00], [0.00, 0.01, 0.04]],
    }

    # ----- x=2: granular / rough texture -----
    # Negative AR with nearest neighbors creates anticorrelation (alternating pattern)
    # (0,-1):   diag(-0.28,-0.24,-0.20)  det=-1.344e-2
    # (-1,0):   diag(-0.24,-0.20,-0.18)  det=-8.640e-3
    # (-1,-1):  diag(0.12, 0.10, 0.08)   det= 9.600e-4  (dampening)
    # (0,-2):   diag(0.06, 0.05, 0.04)   det= 1.200e-4  (dampening)
    x2_ar = {
        "(0,-1)":  d3(-0.28, -0.24, -0.20),
        "(-1,0)":  d3(-0.24, -0.20, -0.18),
        "(-1,-1)": d3(0.12, 0.10, 0.08),
        "(0,-2)":  d3(0.06, 0.05, 0.04),
    }

    return {
        "label_set": [0, 1, 2],
        "channels": 3,
        "mean": [
            [50.0, 50.0, 50.0],
            [150.0, 150.0, 150.0],
            [100.0, 100.0, 100.0],
        ],
        "variance": [
            [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
            [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
            [[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]],
        ],
        "ar_param": [x0_ar, x1_ar, x2_ar],
    }


def main() -> None:
    if not REGION_PATH.exists() or not LABEL_PATH.exists():
        raise FileNotFoundError(f"Required files were not found: {REGION_PATH.name}, {LABEL_PATH.name}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    region_img = np.array(Image.open(REGION_PATH).convert("RGB"), dtype=np.uint8)
    label_img_raw = np.array(Image.open(LABEL_PATH).convert("L"), dtype=np.uint8)

    region_id_map, region_dict = _load_region_ids(region_img)
    label_idx_img = _to_label_index(label_img_raw)
    label_regionwise = _majority_label_per_region(region_id_map, label_idx_img)

    theta = _theta()
    for image_idx in range(NUM_IMAGES):
        rgb = ar_3dmatrix_rgb.generate_rgb_from_labels(
            label_image=label_regionwise,
            region_dict=region_dict,
            theta=theta,
            width=label_regionwise.shape[1],
            height=label_regionwise.shape[0],
            seed=2326 + image_idx,
        )
        out_path = OUT_DIR / f"generated_{image_idx:03d}.png"
        Image.fromarray(rgb).save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()