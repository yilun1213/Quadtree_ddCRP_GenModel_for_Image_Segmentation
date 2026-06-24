from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


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


def meanshift_superpixels(
    image: np.ndarray,
    spatial_radius: float,
    color_radius: float,
    quant_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    filtered_bgr = cv2.pyrMeanShiftFiltering(
        bgr,
        sp=float(spatial_radius),
        sr=float(color_radius),
        maxLevel=1,
    )
    filtered = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)

    # Quantize filtered colors and extract 4-connected components as superpixels.
    q = (filtered // max(1, quant_step)).astype(np.int32)
    code = (q[:, :, 0] << 16) + (q[:, :, 1] << 8) + q[:, :, 2]

    h, w = code.shape
    comp = -np.ones((h, w), dtype=np.int32)
    cid = 0

    for y in range(h):
        for x in range(w):
            if comp[y, x] >= 0:
                continue
            target = code[y, x]
            stack = [(y, x)]
            comp[y, x] = cid

            while stack:
                cy, cx = stack.pop()
                if cy > 0 and comp[cy - 1, cx] < 0 and code[cy - 1, cx] == target:
                    comp[cy - 1, cx] = cid
                    stack.append((cy - 1, cx))
                if cy + 1 < h and comp[cy + 1, cx] < 0 and code[cy + 1, cx] == target:
                    comp[cy + 1, cx] = cid
                    stack.append((cy + 1, cx))
                if cx > 0 and comp[cy, cx - 1] < 0 and code[cy, cx - 1] == target:
                    comp[cy, cx - 1] = cid
                    stack.append((cy, cx - 1))
                if cx + 1 < w and comp[cy, cx + 1] < 0 and code[cy, cx + 1] == target:
                    comp[cy, cx + 1] = cid
                    stack.append((cy, cx + 1))

            cid += 1

    return comp, filtered


def build_region_graph(region_map: np.ndarray) -> List[Set[int]]:
    n_nodes = int(region_map.max()) + 1
    adj: List[Set[int]] = [set() for _ in range(n_nodes)]

    right_a = region_map[:, :-1]
    right_b = region_map[:, 1:]
    mask_right = right_a != right_b
    ys, xs = np.where(mask_right)
    for y, x in zip(ys, xs):
        a = int(right_a[y, x])
        b = int(right_b[y, x])
        adj[a].add(b)
        adj[b].add(a)

    down_a = region_map[:-1, :]
    down_b = region_map[1:, :]
    mask_down = down_a != down_b
    ys, xs = np.where(mask_down)
    for y, x in zip(ys, xs):
        a = int(down_a[y, x])
        b = int(down_b[y, x])
        adj[a].add(b)
        adj[b].add(a)

    return adj


def region_features(image: np.ndarray, region_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_nodes = int(region_map.max()) + 1
    idx = region_map.reshape(-1)
    pix = image.reshape(-1, 3).astype(np.float64)

    area = np.bincount(idx, minlength=n_nodes).astype(np.float64)
    feat = np.zeros((n_nodes, 3), dtype=np.float64)

    for ch in range(3):
        feat[:, ch] = np.bincount(idx, weights=pix[:, ch], minlength=n_nodes)

    feat /= np.maximum(area[:, None], 1.0)
    return feat, area


def region_majority_labels(
    gt_label: np.ndarray,
    region_map: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    n_nodes = int(region_map.max()) + 1
    class_to_idx = {int(c): i for i, c in enumerate(classes)}

    counts = np.zeros((n_nodes, len(classes)), dtype=np.int64)
    ridx = region_map.reshape(-1)
    lflat = gt_label.reshape(-1)

    for r, c in zip(ridx, lflat):
        i = class_to_idx.get(int(c))
        if i is not None:
            counts[int(r), i] += 1

    maj_idx = np.argmax(counts, axis=1)
    return classes[maj_idx].astype(np.int32)


def estimate_superpixel_gaussian_params(
    train_image_paths: List[Path],
    train_label_paths: List[Path],
    reg_eps: float,
    spatial_radius: float,
    color_radius: float,
    quant_step: int,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, float]]:
    all_labels: List[np.ndarray] = []
    for lp in train_label_paths:
        all_labels.append(load_label(lp).reshape(-1))
    classes = np.unique(np.concatenate(all_labels)).astype(np.int32)

    class_feats: Dict[int, List[np.ndarray]] = {int(c): [] for c in classes}

    for img_path, lbl_path in zip(train_image_paths, train_label_paths):
        image = load_rgb(img_path)
        gt = load_label(lbl_path)

        region_map, _ = meanshift_superpixels(
            image=image,
            spatial_radius=spatial_radius,
            color_radius=color_radius,
            quant_step=quant_step,
        )
        feat, _ = region_features(image, region_map)
        maj = region_majority_labels(gt, region_map, classes)

        for c in classes:
            c_int = int(c)
            mask = maj == c_int
            if np.any(mask):
                class_feats[c_int].append(feat[mask])

    means: Dict[int, np.ndarray] = {}
    covs: Dict[int, np.ndarray] = {}
    inv_covs: Dict[int, np.ndarray] = {}
    log_dets: Dict[int, float] = {}

    for c in classes:
        c_int = int(c)
        data = np.concatenate(class_feats[c_int], axis=0)
        mu = np.mean(data, axis=0)

        if data.shape[0] <= 1:
            cov = reg_eps * np.eye(3)
        else:
            centered = data - mu
            cov = (centered.T @ centered) / max(1, data.shape[0] - 1)
            cov = cov + reg_eps * np.eye(3)

        means[c_int] = mu
        covs[c_int] = cov
        inv_covs[c_int] = np.linalg.inv(cov)
        _, logdet = np.linalg.slogdet(cov)
        log_dets[c_int] = float(logdet)

    return classes, means, covs, inv_covs, log_dets


def superpixel_unary_cost(
    region_feat: np.ndarray,
    region_area: np.ndarray,
    classes: np.ndarray,
    means: Dict[int, np.ndarray],
    inv_covs: Dict[int, np.ndarray],
    log_dets: Dict[int, float],
) -> np.ndarray:
    n_nodes = region_feat.shape[0]
    n_cls = len(classes)
    unary = np.zeros((n_nodes, n_cls), dtype=np.float64)

    for i, c in enumerate(classes):
        c_int = int(c)
        diff = region_feat - means[c_int]
        mahal = np.einsum("bi,ij,bj->b", diff, inv_covs[c_int], diff)

        # Area weighting makes region likelihood roughly consistent with pixel-wise energy.
        unary[:, i] = 0.5 * (mahal + log_dets[c_int]) * np.maximum(region_area, 1.0)

    return unary


def run_superpixel_icm(
    unary: np.ndarray,
    adjacency: List[Set[int]],
    classes: np.ndarray,
    beta: float,
    max_iter: int,
    region_map: np.ndarray,
    gt: np.ndarray | None,
) -> Tuple[np.ndarray, List[float]]:
    pred_idx = np.argmin(unary, axis=1)
    pred = classes[pred_idx].astype(np.int32)

    oa_history: List[float] = []
    if gt is not None:
        pixel_pred = pred[region_map]
        oa_history.append(float(np.mean(pixel_pred == gt)))

    for _ in range(max_iter):
        prev = pred.copy()

        for i in range(len(adjacency)):
            costs = unary[i].copy()
            neighbors = adjacency[i]
            if neighbors:
                neigh_labels = prev[np.fromiter(neighbors, dtype=np.int32)]
                for k, c in enumerate(classes):
                    costs[k] += beta * np.sum(neigh_labels != int(c))

            pred[i] = int(classes[int(np.argmin(costs))])

        if gt is not None:
            pixel_pred = pred[region_map]
            oa_history.append(float(np.mean(pixel_pred == gt)))

        if np.array_equal(pred, prev):
            break

    return pred, oa_history


def write_oa_log(path: Path, oa_history: List[float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        if not oa_history:
            f.write("Ground truth not provided. OA log unavailable.\n")
            return

        f.write("iteration\toa\n")
        for i, oa in enumerate(oa_history):
            f.write(f"{i}\t{oa:.6f}\n")


def superpixel_boundary_overlay(image: np.ndarray, region_map: np.ndarray) -> np.ndarray:
    vis = image.copy()
    h, w = region_map.shape
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:-1, :] |= region_map[:-1, :] != region_map[1:, :]
    boundary[:, :-1] |= region_map[:, :-1] != region_map[:, 1:]
    vis[boundary] = np.array([255, 0, 0], dtype=np.uint8)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mean-Shift superpixel segmentation + object-based Potts MRF (ICM) for label estimation."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root path")
    parser.add_argument("--beta", type=float, default=1.2, help="MRF smoothness weight")
    parser.add_argument("--max-iter", type=int, default=20, help="Maximum ICM iterations")
    parser.add_argument("--reg-eps", type=float, default=1e-3, help="Covariance regularization")
    parser.add_argument("--spatial-radius", type=float, default=8.0, help="Mean-Shift spatial radius")
    parser.add_argument("--color-radius", type=float, default=16.0, help="Mean-Shift color radius")
    parser.add_argument(
        "--quant-step",
        type=int,
        default=4,
        help="Color quantization step after Mean-Shift to form connected superpixels",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    train_img_dir = root / "train_data" / "images"
    train_lbl_dir = root / "train_data" / "labels"
    test_img_dir = root / "test_data" / "images"
    test_lbl_dir = root / "test_data" / "labels"

    pred_dir = root / "test_data" / "pred_labels"
    pred_viz_dir = pred_dir / "visualize"
    sp_viz_dir = pred_dir / "superpixel_visualize"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_viz_dir.mkdir(parents=True, exist_ok=True)
    sp_viz_dir.mkdir(parents=True, exist_ok=True)

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

    classes, means, covs, inv_covs, log_dets = estimate_superpixel_gaussian_params(
        train_image_paths=train_pairs_img,
        train_label_paths=train_pairs_lbl,
        reg_eps=args.reg_eps,
        spatial_radius=args.spatial_radius,
        color_radius=args.color_radius,
        quant_step=args.quant_step,
    )

    color_map = infer_label_color_map(root, classes)

    param_path = pred_dir / "superpixel_gaussian_params.json"
    serializable = {
        "labels": classes.tolist(),
        "means": {str(c): means[int(c)].tolist() for c in classes},
        "covariances": {str(c): covs[int(c)].tolist() for c in classes},
        "visualize_color_map": {str(c): color_map[int(c)].tolist() for c in classes},
        "meanshift": {
            "spatial_radius": args.spatial_radius,
            "color_radius": args.color_radius,
            "quant_step": args.quant_step,
        },
    }
    with param_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    test_label_lookup = {p.stem: p for p in sorted(test_lbl_dir.glob("*.png"))}

    for ip in test_image_paths:
        image = load_rgb(ip)
        region_map, _ = meanshift_superpixels(
            image=image,
            spatial_radius=args.spatial_radius,
            color_radius=args.color_radius,
            quant_step=args.quant_step,
        )
        adjacency = build_region_graph(region_map)
        feat, area = region_features(image, region_map)

        unary = superpixel_unary_cost(
            region_feat=feat,
            region_area=area,
            classes=classes,
            means=means,
            inv_covs=inv_covs,
            log_dets=log_dets,
        )

        gt = None
        if ip.stem in test_label_lookup:
            gt = load_label(test_label_lookup[ip.stem])

        pred_region, oa_history = run_superpixel_icm(
            unary=unary,
            adjacency=adjacency,
            classes=classes,
            beta=args.beta,
            max_iter=args.max_iter,
            region_map=region_map,
            gt=gt,
        )

        pred_pixel = pred_region[region_map]

        pred_path = pred_dir / f"{ip.stem}.png"
        pred_viz_path = pred_viz_dir / f"{ip.stem}.png"
        sp_viz_path = sp_viz_dir / f"{ip.stem}.png"
        oa_path = pred_dir / f"{ip.stem}_oa.txt"

        save_label(pred_path, pred_pixel)
        Image.fromarray(label_to_color(pred_pixel, color_map), mode="RGB").save(pred_viz_path)
        Image.fromarray(superpixel_boundary_overlay(image, region_map), mode="RGB").save(sp_viz_path)
        write_oa_log(oa_path, oa_history)

        last_oa = oa_history[-1] if oa_history else None
        n_nodes = int(region_map.max()) + 1
        if last_oa is None:
            print(f"{ip.name}: superpixels={n_nodes}, saved prediction (OA unavailable)")
        else:
            print(
                f"{ip.name}: superpixels={n_nodes}, iterations={len(oa_history)-1}, final_OA={last_oa:.4f}"
            )


if __name__ == "__main__":
    main()
