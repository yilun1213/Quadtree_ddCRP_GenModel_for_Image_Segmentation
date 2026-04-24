from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def load_label(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.int32)


def save_label(path: Path, label_map: np.ndarray) -> None:
    Image.fromarray(label_map.astype(np.uint8), mode="L").save(path)


def infer_label_color_map(root: Path, labels: np.ndarray) -> Dict[int, np.ndarray]:
    color_map: Dict[int, np.ndarray] = {}

    for split in ["train_data", "test_data"]:
        lbl_dir = root / split / "labels"
        viz_dir = lbl_dir / "visualize"
        if not lbl_dir.exists() or not viz_dir.exists():
            continue

        for lbl_path in sorted(lbl_dir.glob("*.png")):
            viz_path = viz_dir / f"{lbl_path.stem}.png"
            if not viz_path.exists():
                continue

            lbl = load_label(lbl_path)
            viz = np.asarray(Image.open(viz_path).convert("RGB"), dtype=np.uint8)

            for cls in np.unique(lbl):
                cls_int = int(cls)
                if cls_int in color_map:
                    continue

                pixels = viz[lbl == cls_int]
                if pixels.size == 0:
                    continue
                uniq = np.unique(pixels.reshape(-1, 3), axis=0)
                color_map[cls_int] = uniq[0]

    # Fallback colors if some classes were not observed in visualize folders.
    fallback_palette = np.array(
        [
            [0, 0, 0],
            [255, 255, 255],
            [220, 20, 60],
            [65, 105, 225],
            [60, 179, 113],
            [255, 140, 0],
            [138, 43, 226],
            [255, 215, 0],
            [0, 206, 209],
            [244, 164, 96],
        ],
        dtype=np.uint8,
    )
    for i, cls in enumerate(labels):
        cls_int = int(cls)
        if cls_int not in color_map:
            color_map[cls_int] = fallback_palette[i % len(fallback_palette)]

    return color_map


def label_to_color(label_map: np.ndarray, color_map: Dict[int, np.ndarray]) -> np.ndarray:
    color = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cls_int, rgb in color_map.items():
        color[label_map == cls_int] = rgb
    return color


def estimate_gaussian_params(
    train_image_paths: List[Path],
    train_label_paths: List[Path],
    reg_eps: float,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, float]]:
    all_labels: List[np.ndarray] = []
    for lp in train_label_paths:
        all_labels.append(load_label(lp).reshape(-1))
    labels = np.unique(np.concatenate(all_labels))

    class_pixels: Dict[int, List[np.ndarray]] = {int(c): [] for c in labels}

    for img_path, lbl_path in zip(train_image_paths, train_label_paths):
        img = load_rgb(img_path).reshape(-1, 3)
        lbl = load_label(lbl_path).reshape(-1)

        for c in labels:
            c_int = int(c)
            mask = lbl == c_int
            if np.any(mask):
                class_pixels[c_int].append(img[mask])

    means: Dict[int, np.ndarray] = {}
    covs: Dict[int, np.ndarray] = {}
    inv_covs: Dict[int, np.ndarray] = {}
    log_dets: Dict[int, float] = {}

    for c in labels:
        c_int = int(c)
        data = np.concatenate(class_pixels[c_int], axis=0)
        mu = np.mean(data, axis=0)
        centered = data - mu
        cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
        cov = cov + reg_eps * np.eye(3)

        means[c_int] = mu
        covs[c_int] = cov
        inv_covs[c_int] = np.linalg.inv(cov)
        _, logdet = np.linalg.slogdet(cov)
        log_dets[c_int] = float(logdet)

    return labels.astype(np.int32), means, covs, inv_covs, log_dets


def unary_cost(
    image: np.ndarray,
    labels: np.ndarray,
    means: Dict[int, np.ndarray],
    inv_covs: Dict[int, np.ndarray],
    log_dets: Dict[int, float],
) -> np.ndarray:
    h, w, _ = image.shape
    n_cls = len(labels)
    x = image.reshape(-1, 3)

    costs = np.zeros((x.shape[0], n_cls), dtype=np.float64)
    for i, c in enumerate(labels):
        c_int = int(c)
        diff = x - means[c_int]
        m = np.einsum("bi,ij,bj->b", diff, inv_covs[c_int], diff)
        costs[:, i] = 0.5 * (m + log_dets[c_int])

    return costs.reshape(h, w, n_cls)


def neighbor_mismatch_count(label_map: np.ndarray, cls: int) -> np.ndarray:
    # 4-neighborhood mismatch count if current pixel were assigned cls.
    h, w = label_map.shape
    cnt = np.zeros((h, w), dtype=np.float64)

    up = np.zeros_like(label_map)
    up[1:] = label_map[:-1]
    down = np.zeros_like(label_map)
    down[:-1] = label_map[1:]
    left = np.zeros_like(label_map)
    left[:, 1:] = label_map[:, :-1]
    right = np.zeros_like(label_map)
    right[:, :-1] = label_map[:, 1:]

    valid_up = np.zeros((h, w), dtype=bool)
    valid_up[1:] = True
    valid_down = np.zeros((h, w), dtype=bool)
    valid_down[:-1] = True
    valid_left = np.zeros((h, w), dtype=bool)
    valid_left[:, 1:] = True
    valid_right = np.zeros((h, w), dtype=bool)
    valid_right[:, :-1] = True

    cnt += valid_up & (up != cls)
    cnt += valid_down & (down != cls)
    cnt += valid_left & (left != cls)
    cnt += valid_right & (right != cls)

    return cnt


def run_icm(
    unary: np.ndarray,
    labels: np.ndarray,
    beta: float,
    max_iter: int,
    gt: np.ndarray | None = None,
) -> Tuple[np.ndarray, List[float]]:
    h, w, n_cls = unary.shape
    pred_idx = np.argmin(unary, axis=2)
    pred = labels[pred_idx]

    oa_history: List[float] = []
    if gt is not None:
        oa_history.append(float(np.mean(pred == gt)))

    for _ in range(max_iter):
        prev = pred.copy()
        total_cost = np.zeros((h, w, n_cls), dtype=np.float64)

        for i, c in enumerate(labels):
            smooth = beta * neighbor_mismatch_count(prev, int(c))
            total_cost[:, :, i] = unary[:, :, i] + smooth

        pred = labels[np.argmin(total_cost, axis=2)]

        if gt is not None:
            oa_history.append(float(np.mean(pred == gt)))

        if np.array_equal(pred, prev):
            break

    return pred.astype(np.int32), oa_history


def write_oa_log(path: Path, oa_history: List[float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        if not oa_history:
            f.write("Ground truth not provided. OA log unavailable.\n")
            return

        f.write("iteration\toa\n")
        for i, oa in enumerate(oa_history):
            f.write(f"{i}\t{oa:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Gaussian class models on train_data and run MRF segmentation on test_data."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root path")
    parser.add_argument("--beta", type=float, default=1.2, help="MRF smoothness weight")
    parser.add_argument("--max-iter", type=int, default=20, help="Maximum ICM iterations")
    parser.add_argument("--reg-eps", type=float, default=1e-3, help="Covariance regularization")
    args = parser.parse_args()

    root = args.root.resolve()
    train_img_dir = root / "train_data" / "images"
    train_lbl_dir = root / "train_data" / "labels"
    test_img_dir = root / "test_data" / "images"
    test_lbl_dir = root / "test_data" / "labels"

    pred_dir = root / "test_data" / "pred_labels"
    pred_viz_dir = pred_dir / "visualize"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_viz_dir.mkdir(parents=True, exist_ok=True)

    train_image_paths = sorted(train_img_dir.glob("*.png"))
    train_label_paths = sorted(train_lbl_dir.glob("*.png"))
    test_image_paths = sorted(test_img_dir.glob("*.png"))

    if not train_image_paths or not train_label_paths or not test_image_paths:
        raise FileNotFoundError("Could not find required train/test PNG files.")

    label_lookup = {p.stem: p for p in train_label_paths}
    train_pairs_img: List[Path] = []
    train_pairs_lbl: List[Path] = []
    for ip in train_image_paths:
        if ip.stem in label_lookup:
            train_pairs_img.append(ip)
            train_pairs_lbl.append(label_lookup[ip.stem])

    if not train_pairs_img:
        raise RuntimeError("No matching train image-label file pairs found.")

    labels, means, covs, inv_covs, log_dets = estimate_gaussian_params(
        train_pairs_img,
        train_pairs_lbl,
        reg_eps=args.reg_eps,
    )
    color_map = infer_label_color_map(root, labels)

    # Save learned Gaussian parameters for reproducibility.
    param_path = pred_dir / "gaussian_params.json"
    serializable = {
        "labels": labels.tolist(),
        "means": {str(c): means[int(c)].tolist() for c in labels},
        "covariances": {str(c): covs[int(c)].tolist() for c in labels},
        "visualize_color_map": {str(c): color_map[int(c)].tolist() for c in labels},
    }
    with param_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    test_label_lookup = {p.stem: p for p in sorted(test_lbl_dir.glob("*.png"))}

    for ip in test_image_paths:
        image = load_rgb(ip)
        unary = unary_cost(image, labels, means, inv_covs, log_dets)

        gt = None
        if ip.stem in test_label_lookup:
            gt = load_label(test_label_lookup[ip.stem])

        pred, oa_history = run_icm(
            unary=unary,
            labels=labels,
            beta=args.beta,
            max_iter=args.max_iter,
            gt=gt,
        )

        pred_path = pred_dir / f"{ip.stem}.png"
        pred_viz_path = pred_viz_dir / f"{ip.stem}.png"
        oa_path = pred_dir / f"{ip.stem}_oa.txt"

        save_label(pred_path, pred)
        Image.fromarray(label_to_color(pred, color_map), mode="RGB").save(pred_viz_path)
        write_oa_log(oa_path, oa_history)

        last_oa = oa_history[-1] if oa_history else None
        if last_oa is None:
            print(f"{ip.name}: saved prediction (OA unavailable)")
        else:
            print(
                f"{ip.name}: saved prediction, iterations={len(oa_history)-1}, final_OA={last_oa:.4f}"
            )


if __name__ == "__main__":
    main()
