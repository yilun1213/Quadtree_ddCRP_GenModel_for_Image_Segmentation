from __future__ import annotations

import json
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
REFERENCE_THETA_CANDIDATES = [
    THIS_DIR / "ar_param.json",
    ROOT_DIR / "ar_param.json",
]
REFERENCE_LABELS = [0, 1, 2]


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


def _adapt_ar_matrix(matrix_list: list[list[float]], channels: int, label: int, offset: str) -> list[list[float]]:
    matrix = np.asarray(matrix_list, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"AR matrix for label {label} at offset {offset} is not square: {matrix.shape}")
    if matrix.shape[0] < channels:
        raise ValueError(
            f"AR matrix for label {label} at offset {offset} has fewer channels ({matrix.shape[0]}) than requested ({channels})"
        )
    if matrix.shape[0] > channels:
        matrix = matrix[:channels, :channels]
    return matrix.tolist()


def _adapt_mean_vector(mean_list: list[float], channels: int, label: int) -> list[float]:
    mean = np.asarray(mean_list, dtype=np.float64)
    if mean.ndim != 1:
        raise ValueError(f"Mean for label {label} is not a vector: {mean.shape}")
    if mean.shape[0] < channels:
        raise ValueError(f"Mean for label {label} has fewer channels ({mean.shape[0]}) than requested ({channels})")
    if mean.shape[0] > channels:
        mean = mean[:channels]
    return mean.tolist()


def _adapt_cov_matrix(cov_list: list[list[float]], channels: int, label: int) -> list[list[float]]:
    cov = np.asarray(cov_list, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Variance for label {label} is not square: {cov.shape}")
    if cov.shape[0] < channels:
        raise ValueError(
            f"Variance for label {label} has fewer channels ({cov.shape[0]}) than requested ({channels})"
        )
    if cov.shape[0] > channels:
        cov = cov[:channels, :channels]
    return cov.tolist()


def _load_reference_stats(reference_labels: list[int], channels: int) -> tuple[list[list[float]], list[list[list[float]]]]:
    reference_theta_path = next((path for path in REFERENCE_THETA_CANDIDATES if path.exists()), None)
    if reference_theta_path is None:
        searched = ", ".join(str(path) for path in REFERENCE_THETA_CANDIDATES)
        raise FileNotFoundError(f"ar_param.json was not found. Searched: {searched}")

    with reference_theta_path.open("r", encoding="utf-8") as f:
        reference_theta = json.load(f)

    label_to_index = {
        int(label): idx for idx, label in enumerate(reference_theta.get("label_set", []))
    }

    means: list[list[float]] = []
    variances: list[list[list[float]]] = []
    for label in reference_labels:
        if label not in label_to_index:
            raise KeyError(f"Label {label} was not found in {reference_theta_path.name}")
        idx = label_to_index[label]
        means.append(_adapt_mean_vector(reference_theta["mean"][idx], channels=channels, label=label))
        variances.append(_adapt_cov_matrix(reference_theta["variance"][idx], channels=channels, label=label))

    return means, variances


def _load_reference_ar_params(reference_labels: list[int], channels: int) -> list[dict[str, list[list[float]]]]:
    reference_theta_path = next((path for path in REFERENCE_THETA_CANDIDATES if path.exists()), None)
    if reference_theta_path is None:
        searched = ", ".join(str(path) for path in REFERENCE_THETA_CANDIDATES)
        raise FileNotFoundError(f"ar_param.json was not found. Searched: {searched}")

    with reference_theta_path.open("r", encoding="utf-8") as f:
        reference_theta = json.load(f)

    label_to_index = {
        int(label): idx for idx, label in enumerate(reference_theta.get("label_set", []))
    }

    ar_params: list[dict[str, list[list[float]]]] = []
    for label in reference_labels:
        if label not in label_to_index:
            raise KeyError(f"Label {label} was not found in {reference_theta_path.name}")
        raw_ar = reference_theta["ar_param"][label_to_index[label]]
        adapted_ar = {
            offset: _adapt_ar_matrix(matrix_list, channels=channels, label=label, offset=offset)
            for offset, matrix_list in raw_ar.items()
        }
        ar_params.append(adapted_ar)

    return ar_params


def _theta() -> dict:
    channels = 3
    means, variances = _load_reference_stats(REFERENCE_LABELS, channels=channels)
    return {
        "label_set": [0, 1, 2],
        "channels": channels,
        "mean": means,
        "variance": variances,
        "ar_param": _load_reference_ar_params(REFERENCE_LABELS, channels=channels),
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
            seed=2226 + image_idx,
        )
        out_path = OUT_DIR / f"generated_{image_idx:03d}.png"
        Image.fromarray(rgb).save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()