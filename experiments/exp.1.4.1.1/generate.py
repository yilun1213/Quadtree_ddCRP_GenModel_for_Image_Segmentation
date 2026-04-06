from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image


# Workspace root (.../Quadtree_ddCRP_GenModel_for_Image_Segmentation)
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.pixel import normal_dist  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
REGION_PATH = THIS_DIR / "region_000.png"
LABEL_PATH = THIS_DIR / "label_000.png"
OUT_DIR = THIS_DIR / "outputs"
NUM_IMAGES = 3


def _load_region_ids(region_img: np.ndarray) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]]:
    """Convert region color image to integer region-id map and region_dict."""
    h, w, _ = region_img.shape
    flat = region_img.reshape(-1, 3).astype(np.uint32)
    keys = (flat[:, 0] << 16) + (flat[:, 1] << 8) + flat[:, 2]
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    region_id_map = inverse.reshape(h, w).astype(np.int32)

    region_dict: dict[int, set[tuple[int, int]]] = {
        int(rid): set() for rid in range(len(unique_keys))
    }
    for i in range(h):
        for j in range(w):
            rid = int(region_id_map[i, j])
            region_dict[rid].add((i, j))

    return region_id_map, region_dict


def _to_label_index(label_img: np.ndarray) -> np.ndarray:
    """Map label values to label indices {0,1,2}."""
    mapping = {0: 0, 1: 1, 2: 2, 128: 1, 255: 2}
    out = np.zeros_like(label_img, dtype=np.uint8)
    for src_val, dst_val in mapping.items():
        out[label_img == src_val] = dst_val
    return out


def _majority_label_per_region(region_id_map: np.ndarray, label_idx_img: np.ndarray) -> np.ndarray:
    """Assign each region a single label by majority vote from label image."""
    out = np.zeros_like(label_idx_img, dtype=np.uint8)
    max_region_id = int(region_id_map.max())

    for rid in range(max_region_id + 1):
        mask = region_id_map == rid
        labels = label_idx_img[mask]
        if labels.size == 0:
            continue
        counts = np.bincount(labels, minlength=3)
        region_label = int(np.argmax(counts))
        out[mask] = region_label

    return out


def _exp_1_4_1_1_theta() -> dict:
    # exp_plan.md / exp 1.4.1.1:
    # x=0: mu=(50,50,50), Sigma=diag(20,20,20)
    # x=1: mu=(150,150,150), Sigma=diag(20,20,20)
    # x=2: mu=(200,200,200), Sigma=diag(20,20,20)
    return {
        "label_set": [0, 1, 2],
        "channels": 3,
        "mean": [
            [200.0, 50.0, 50.0],
            [50.0, 200.0, 50.0],
            [50.0, 50.0, 200.0],
        ],
        "variance": [
            [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
            [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
            [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
        ],
    }


def main() -> None:
    if not REGION_PATH.exists() or not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Required files were not found: {REGION_PATH.name}, {LABEL_PATH.name}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    region_img = np.array(Image.open(REGION_PATH).convert("RGB"), dtype=np.uint8)
    label_img_raw = np.array(Image.open(LABEL_PATH).convert("L"), dtype=np.uint8)

    region_id_map, region_dict = _load_region_ids(region_img)
    label_idx_img = _to_label_index(label_img_raw)
    label_regionwise = _majority_label_per_region(region_id_map, label_idx_img)

    theta = _exp_1_4_1_1_theta()

    for i in range(NUM_IMAGES):
        rgb = normal_dist.generate_rgb_from_labels(
            label_image=label_regionwise,
            region_dict=region_dict,
            theta=theta,
            width=label_regionwise.shape[1],
            height=label_regionwise.shape[0],
            seed=2026 + i,
        )
        out_path = OUT_DIR / f"generated_{i:03d}.png"
        Image.fromarray(rgb).save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
