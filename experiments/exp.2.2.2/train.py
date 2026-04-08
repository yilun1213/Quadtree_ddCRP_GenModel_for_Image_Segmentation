
# train.py — exp.2.2.2 逐次ロジスティック回帰パラメータ推定
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import utils
from model.label.geom_features import compute_geom_features


BASE_DIR = os.path.dirname(__file__)
TRAIN_LABEL_DIR = os.path.join(BASE_DIR, "outputs", "train_data", "labels")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "estimation_results")
EXTRACTED_REGION_DIR = os.path.join(BASE_DIR, "outputs", "train_data", "region_images_extracted")

# ラベル設定
LABEL_SET = [0, 1, 2]
FEATURE_NAMES = ["log_area", "log_perimeter", "circularity"]

# 領域検出の connectivity 設定 (4 or 8)
CONNECTIVITY = 8

# 真値 — exp_plan.md 記載のパラメータ
# weights: shape (K, d): K=クラス数, d=特徴数
TRUE_WEIGHTS = np.array(
    [
        [-1.4, -0.9,  2.8],   # x=0
        [ 0.1,  1.0, -2.5],   # x=1
        [ 2.4,  1.8,  1.5],   # x=2
    ],
    dtype=float,
)
# bias: shape (K,)
TRUE_BIAS = np.array([6.5, -4.5, -31.0], dtype=float)

MIN_REGION_AREA = 32
NUM_CLASSES = len(LABEL_SET)
NUM_FEATURES = len(FEATURE_NAMES)


def _get_connectivity_structure():
    if CONNECTIVITY == 4:
        return ndimage.generate_binary_structure(2, 1)
    else:
        return ndimage.generate_binary_structure(2, 2)


def _normalize_label_array(label_array: np.ndarray) -> np.ndarray:
    if label_array.ndim == 3:
        return label_array[..., 0]
    return label_array


def _extract_label_features_from_image(
    label_img: np.ndarray,
) -> tuple[list[np.ndarray], list[int]]:
    """
    ラベル画像から幾何的特徴量と対応するラベルインデックスを抽出。
    Returns:
        features: list of phi(r) vectors, shape (d,) each
        label_indices: list of class index (0, 1, 2)
    """
    connectivity_struct = _get_connectivity_structure()
    features: list[np.ndarray] = []
    label_indices: list[int] = []

    label_to_index = {label: idx for idx, label in enumerate(LABEL_SET)}

    for label in [int(v) for v in np.unique(label_img)]:
        if label not in label_to_index:
            continue

        mask = (label_img == label)
        labeled_mask, num_regions = ndimage.label(mask, structure=connectivity_struct)
        for region_idx in range(1, num_regions + 1):
            region_mask = (labeled_mask == region_idx)
            if int(np.count_nonzero(region_mask)) < MIN_REGION_AREA:
                continue

            coords = np.argwhere(region_mask)
            region = {(int(i), int(j)) for i, j in coords}
            phi = compute_geom_features(
                region,
                image_size=int(label_img.shape[0]),
                feature_names=FEATURE_NAMES,
            )
            if np.any(np.isnan(phi)) or np.any(np.isinf(phi)):
                continue
            features.append(phi.astype(float))
            label_indices.append(label_to_index[label])

    return features, label_indices


def _visualize_extracted_regions(label_img: np.ndarray, output_path: str) -> None:
    """
    抽出された領域をグラフ着色で可視化。隣接領域・同一ラベル領域が異なる色になるよう着色。
    """
    from PIL import Image

    connectivity_struct = _get_connectivity_structure()
    h, w = label_img.shape

    np.random.seed(None)
    color_palette = [np.random.randint(50, 255, size=3, dtype=np.uint8) for _ in range(256)]

    region_info: dict[int, dict] = {}
    global_region_id = 0

    for label in [int(v) for v in np.unique(label_img)]:
        if label not in LABEL_SET:
            continue
        mask = (label_img == label)
        labeled_mask, num_regions = ndimage.label(mask, structure=connectivity_struct)
        for region_idx in range(1, num_regions + 1):
            region_mask = (labeled_mask == region_idx)
            if int(np.count_nonzero(region_mask)) < MIN_REGION_AREA:
                continue
            region_info[global_region_id] = {"mask": region_mask.copy(), "label": label}
            global_region_id += 1

    adjacency: dict[int, set] = {rid: set() for rid in region_info}

    # 同一ラベルの領域は必ず隣接扱い
    regions_by_label: dict[int, list[int]] = {}
    for rid, info in region_info.items():
        lbl = info["label"]
        regions_by_label.setdefault(lbl, []).append(rid)
    for rids in regions_by_label.values():
        for i, rid1 in enumerate(rids):
            for rid2 in rids[i + 1:]:
                adjacency[rid1].add(rid2)
                adjacency[rid2].add(rid1)

    # 物理的に隣接する領域も隣接扱い
    for rid in region_info:
        dilated = ndimage.binary_dilation(region_info[rid]["mask"], structure=connectivity_struct)
        near_pixels = dilated & ~region_info[rid]["mask"]
        for other_rid in region_info:
            if other_rid <= rid or other_rid in adjacency[rid]:
                continue
            if np.any(near_pixels & region_info[other_rid]["mask"]):
                adjacency[rid].add(other_rid)
                adjacency[other_rid].add(rid)

    # 貪欲グラフ着色
    used_color_index: dict[int, int | None] = {rid: None for rid in region_info}
    region_colors: dict[int, np.ndarray] = {}
    for rid in sorted(region_info):
        neighbor_colors = {used_color_index[nid] for nid in adjacency[rid] if used_color_index[nid] is not None}
        color_idx = 0
        while color_idx in neighbor_colors:
            color_idx += 1
        region_colors[rid] = color_palette[color_idx % len(color_palette)]
        used_color_index[rid] = color_idx

    out_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for rid, color in region_colors.items():
        out_img[region_info[rid]["mask"]] = color

    Image.fromarray(out_img, mode="RGB").save(output_path)


def _fit_logistic(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    l2_reg: float = 1e-3,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    多項ロジスティック回帰を L-BFGS-B で最小化。
    Returns:
        weights: (K, d)
        bias: (K,)
    """
    n, d = x_mat.shape
    k = NUM_CLASSES

    def _unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return theta[: k * d].reshape(k, d), theta[k * d:]

    def _objective_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = _unpack(theta)
        logits = x_mat @ w.T + b        # (N, K)
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        nll = -np.sum(np.log(probs[np.arange(n), y_vec] + 1e-12))
        nll += 0.5 * l2_reg * np.sum(w * w)

        d_logits = probs.copy()
        d_logits[np.arange(n), y_vec] -= 1.0
        grad_w = d_logits.T @ x_mat + l2_reg * w
        grad_b = np.sum(d_logits, axis=0)
        return float(nll), np.concatenate([grad_w.ravel(), grad_b])

    theta0 = np.zeros(k * d + k, dtype=float)
    result = minimize(
        fun=lambda t: _objective_and_grad(t)[0],
        x0=theta0,
        jac=lambda t: _objective_and_grad(t)[1],
        method="L-BFGS-B",
        options={"maxiter": max_iter, "disp": False},
    )
    return _unpack(result.x)


def _plot_error_trajectories(trajectory: list[dict], out_path: str) -> None:
    # 列: bias + 各特徴量 (NUM_FEATURES+1 個)
    num_cols = NUM_FEATURES + 1
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols + 2, 5))

    x_vals = [row["num_samples"] for row in trajectory]

    # bias
    ax = axes[0]
    for label in LABEL_SET:
        y = [row[f"omega_err_bias_x{label}"] for row in trajectory]
        ax.plot(x_vals, y, label=f"x={label}", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("bias")
    ax.set_xlabel("num_samples")
    ax.set_ylabel("error (hat - true)")
    ax.grid(True, alpha=0.3)

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        ax = axes[feat_idx + 1]
        for label in LABEL_SET:
            y = [row[f"omega_err_{feat_name}_x{label}"] for row in trajectory]
            ax.plot(x_vals, y, label=f"x={label}", linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(feat_name)
        ax.set_xlabel("num_samples")
        ax.set_ylabel("error (hat - true)")
        ax.grid(True, alpha=0.3)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=NUM_CLASSES)
    fig.suptitle("Logistic regression parameter error trajectory", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_progressive() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EXTRACTED_REGION_DIR, exist_ok=True)

    label_files = utils.get_image_files(TRAIN_LABEL_DIR)
    if not label_files:
        raise FileNotFoundError(f"ラベル画像が見つかりません: {TRAIN_LABEL_DIR}")

    all_features: list[np.ndarray] = []
    all_label_indices: list[int] = []
    trajectory: list[dict] = []

    print(f"{len(label_files)} 枚のラベル画像で逐次推定を開始")
    for sample_idx, filename in enumerate(label_files, start=1):
        label_img = _normalize_label_array(
            utils.load_image(os.path.join(TRAIN_LABEL_DIR, filename))
        )

        # 抽出済み領域の可視化
        base_name = os.path.splitext(filename)[0]
        _visualize_extracted_regions(
            label_img,
            os.path.join(EXTRACTED_REGION_DIR, f"{base_name}_extracted.png"),
        )

        # 特徴量を抽出して蓄積
        feats, lblidxs = _extract_label_features_from_image(label_img)
        all_features.extend(feats)
        all_label_indices.extend(lblidxs)

        if not all_features:
            print(f"  [{sample_idx:02d}/{len(label_files)}] スキップ: 有効な領域なし")
            continue

        x_mat = np.asarray(all_features, dtype=float)
        y_vec = np.asarray(all_label_indices, dtype=int)

        weights_hat, bias_hat = _fit_logistic(x_mat, y_vec)

        row: dict = {"num_samples": sample_idx}
        for label_idx, label in enumerate(LABEL_SET):
            row[f"omega_hat_bias_x{label}"] = float(bias_hat[label_idx])
            row[f"omega_err_bias_x{label}"] = float(bias_hat[label_idx]) - float(TRUE_BIAS[label_idx])
        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            for label_idx, label in enumerate(LABEL_SET):
                w_hat = float(weights_hat[label_idx, feat_idx])
                row[f"omega_hat_{feat_name}_x{label}"] = w_hat
                row[f"omega_err_{feat_name}_x{label}"] = w_hat - float(TRUE_WEIGHTS[label_idx, feat_idx])

        trajectory.append(row)
        print(f"  [{sample_idx:02d}/{len(label_files)}] 推定完了: {filename}")

    # TSV 出力
    fieldnames = ["num_samples"]
    for label in LABEL_SET:
        fieldnames += [f"omega_hat_bias_x{label}", f"omega_err_bias_x{label}"]
    for feat_name in FEATURE_NAMES:
        for label in LABEL_SET:
            fieldnames += [f"omega_hat_{feat_name}_x{label}", f"omega_err_{feat_name}_x{label}"]

    tsv_path = os.path.join(OUT_DIR, "label_param_trajectory.tsv")
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(trajectory)

    # 最終推定値 JSON 出力
    final = trajectory[-1]
    final_summary = {
        "num_samples": int(final["num_samples"]),
        "feature_names": FEATURE_NAMES,
        "label_set": LABEL_SET,
        "weights_hat": [
            [float(final[f"omega_hat_{feat}_x{label}"]) for feat in FEATURE_NAMES]
            for label in LABEL_SET
        ],
        "bias_hat": [float(final[f"omega_hat_bias_x{label}"]) for label in LABEL_SET],
        "weights_err": [
            [float(final[f"omega_err_{feat}_x{label}"]) for feat in FEATURE_NAMES]
            for label in LABEL_SET
        ],
        "bias_err": [float(final[f"omega_err_bias_x{label}"]) for label in LABEL_SET],
    }
    json_path = os.path.join(OUT_DIR, "label_param_estimate_final.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    # 誤差推移グラフ
    _plot_error_trajectories(
        trajectory,
        out_path=os.path.join(OUT_DIR, "omega_error_trajectory.png"),
    )

    print(f"\n出力完了: {OUT_DIR}")
    print(f"  - TSV: {tsv_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - 図: omega_error_trajectory.png")
    print(f"  - 抽出領域: {EXTRACTED_REGION_DIR}")


if __name__ == "__main__":
    train_progressive()
