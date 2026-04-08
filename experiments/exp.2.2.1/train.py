
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import utils
from model.label.geom_features import compute_geom_features


BASE_DIR = os.path.dirname(__file__)
TRAIN_LABEL_DIR = os.path.join(BASE_DIR, "outputs", "train_data", "labels")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "estimation_results")
EXTRACTED_REGION_DIR = os.path.join(BASE_DIR, "outputs", "train_data", "region_images_extracted")

# exp.2.2.1 で仮定した真値
LABEL_SET = [0, 1, 2]
FEATURE_NAMES = ["log_area", "log_perimeter", "circularity"]

# 領域検出の connectivity 設定 (4 or 8)
CONNECTIVITY = 4
TRUE_MEANS = np.array(
    [
        [4.0, 3.5, 0.45],
        [6.5, 5.0, 0.50],
        [9.0, 6.0, 0.70],
    ],
    dtype=float,
)
TRUE_STDS = np.array(
    [
        [1.0, 0.5, 0.2],
        [1.5, 0.5, 0.1],
        [1.0, 0.5, 0.1],
    ],
    dtype=float,
)

MIN_REGION_AREA = 32


def _get_connectivity_structure():
    """
    指定された connectivity に基づいて、scipy.ndimage で使用する構造を返す。
    
    Returns:
        np.ndarray: connectivity 構造 (4-conectivity または 8-connectivity)
    """
    if CONNECTIVITY == 4:
        return ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    else:
        return ndimage.generate_binary_structure(2, 2)  # 8-connectivity


def _normalize_label_array(label_array: np.ndarray) -> np.ndarray:
    if label_array.ndim == 3:
        return label_array[..., 0]
    return label_array


def _extract_label_features_from_image(label_img: np.ndarray) -> dict[int, list[np.ndarray]]:
    by_label: dict[int, list[np.ndarray]] = {label: [] for label in LABEL_SET}
    connectivity_struct = _get_connectivity_structure()

    labels_in_img = [int(v) for v in np.unique(label_img)]
    for label in labels_in_img:
        if label not in by_label:
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
            by_label[label].append(phi.astype(float))

    return by_label


def _visualize_extracted_regions(label_img: np.ndarray, output_path: str) -> None:
    """
    ラベル画像から抽出された領域を可視化して画像ファイルに保存。
    隣同士の領域では異なる色が割り当てられるようにグラフ着色を行う。
    """
    from PIL import Image
    
    connectivity_struct = _get_connectivity_structure()
    h, w = label_img.shape
    
    # ランダムカラーパレット（可視性のため50-255の範囲）
    np.random.seed(None)  # ランダムな色を毎回生成
    color_palette = [np.random.randint(50, 255, size=3, dtype=np.uint8) for _ in range(256)]
    
    # 全ラベルの全領域を抽出し、グローバルな領域IDを付与
    region_info = {}  # region_id -> {"mask": region_mask, "label": label_value}
    global_region_id = 0
    region_to_label_indices = {}  # region_id -> (label, local_region_idx)
    
    labels_in_img = [int(v) for v in np.unique(label_img)]
    for label in labels_in_img:
        if label not in LABEL_SET:
            continue
        
        mask = (label_img == label)
        labeled_mask, num_regions = ndimage.label(mask, structure=connectivity_struct)
        
        for region_idx in range(1, num_regions + 1):
            region_mask = (labeled_mask == region_idx)
            if int(np.count_nonzero(region_mask)) < MIN_REGION_AREA:
                continue
            
            region_info[global_region_id] = {
                "mask": region_mask.copy(),
                "label": label,
            }
            region_to_label_indices[global_region_id] = (label, region_idx)
            global_region_id += 1
    
    # 隣接グラフを構築（同じラベルの領域同士と、隣接する領域を判定）
    adjacency = {rid: set() for rid in region_info.keys()}
    
    # 同じラベルの領域同士を隣接カウント
    regions_by_label = {}
    for rid, info in region_info.items():
        label = info["label"]
        if label not in regions_by_label:
            regions_by_label[label] = []
        regions_by_label[label].append(rid)
    
    # 同じラベルの領域は互いに隣接扱い（異なる色を割り当てるため）
    for label, region_ids in regions_by_label.items():
        for i, rid1 in enumerate(region_ids):
            for rid2 in region_ids[i+1:]:
                adjacency[rid1].add(rid2)
                adjacency[rid2].add(rid1)
    
    # 隣接する領域を大域で検出
    for rid in region_info.keys():
        mask = region_info[rid]["mask"]
        # 隣接领域を検出するため、1ピクセル膨張させた領域の周辺をチェック
        dilated = ndimage.binary_dilation(mask, structure=connectivity_struct)
        near_pixels = dilated & ~mask
        
        # near_pixels にある他の領域を見つける
        for other_rid in region_info.keys():
            if other_rid <= rid or other_rid in adjacency[rid]:
                continue
            other_mask = region_info[other_rid]["mask"]
            if np.any(near_pixels & other_mask):
                adjacency[rid].add(other_rid)
                adjacency[other_rid].add(rid)
    
    # 貪欲グラフ着色アルゴリズム
    region_colors = {}
    used_colors = {rid: None for rid in region_info.keys()}
    
    for rid in sorted(region_info.keys()):
        # 隣接領域で使用されている色を調べる
        neighbor_colors = {used_colors[nid] for nid in adjacency[rid] if used_colors[nid] is not None}
        
        # 最初の利用可能な色を割り当て
        color_idx = 0
        while color_idx in neighbor_colors:
            color_idx += 1
        
        # パレットを循環させる（256を超えてもループ）
        actual_color_idx = color_idx % len(color_palette)
        region_colors[rid] = color_palette[actual_color_idx]
        used_colors[rid] = color_idx
    
    # 画像を作成
    region_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for rid, color in region_colors.items():
        region_img[region_info[rid]["mask"]] = color
    
    img = Image.fromarray(region_img, mode='RGB')
    img.save(output_path)


def _estimate_from_running_stats(count: np.ndarray, sum_: np.ndarray, sumsq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros_like(sum_)
    stds = np.ones_like(sum_)

    for label_idx in range(len(LABEL_SET)):
        for feat_idx in range(len(FEATURE_NAMES)):
            n = int(count[label_idx, feat_idx])
            if n <= 0:
                means[label_idx, feat_idx] = 0.0
                stds[label_idx, feat_idx] = 1.0
                continue

            mean = sum_[label_idx, feat_idx] / n
            means[label_idx, feat_idx] = mean

            if n == 1:
                std = 1e-6
            else:
                numerator = sumsq[label_idx, feat_idx] - n * (mean ** 2)
                variance = max(numerator / (n - 1), 1e-12)
                std = float(np.sqrt(variance))
            stds[label_idx, feat_idx] = std

    return means, stds


def _plot_error_trajectories(trajectory: list[dict], key: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(18, 5), sharex=True)

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        ax = axes[feat_idx]
        for label in LABEL_SET:
            y = [row[f"{key}_err_{feat_name}_x{label}"] for row in trajectory]
            x = [row["num_samples"] for row in trajectory]
            ax.plot(x, y, label=f"x={label}", linewidth=1.8)

        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(feat_name)
        ax.set_xlabel("num_samples")
        ax.set_ylabel("error")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(LABEL_SET))
    fig.suptitle(f"Estimation error trajectory ({key})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_progressive() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EXTRACTED_REGION_DIR, exist_ok=True)

    label_files = utils.get_image_files(TRAIN_LABEL_DIR)
    if not label_files:
        raise FileNotFoundError(f"No label images found: {TRAIN_LABEL_DIR}")

    count = np.zeros((len(LABEL_SET), len(FEATURE_NAMES)), dtype=np.int64)
    sum_ = np.zeros((len(LABEL_SET), len(FEATURE_NAMES)), dtype=np.float64)
    sumsq = np.zeros((len(LABEL_SET), len(FEATURE_NAMES)), dtype=np.float64)

    trajectory: list[dict] = []

    print(f"{len(label_files)} 枚のラベル画像で逐次推定を開始")
    for sample_idx, filename in enumerate(label_files, start=1):
        label_path = os.path.join(TRAIN_LABEL_DIR, filename)
        label_img = _normalize_label_array(utils.load_image(label_path))
        features_by_label = _extract_label_features_from_image(label_img)
        
        # 抽出された領域を可視化して保存
        base_name = os.path.splitext(filename)[0]
        extracted_region_path = os.path.join(EXTRACTED_REGION_DIR, f"{base_name}_extracted.png")
        _visualize_extracted_regions(label_img, extracted_region_path)

        for label_idx, label in enumerate(LABEL_SET):
            feats = features_by_label[label]
            if not feats:
                continue

            arr = np.asarray(feats, dtype=np.float64)
            count[label_idx, :] += arr.shape[0]
            sum_[label_idx, :] += np.sum(arr, axis=0)
            sumsq[label_idx, :] += np.sum(arr * arr, axis=0)

        means_hat, stds_hat = _estimate_from_running_stats(count, sum_, sumsq)

        row: dict[str, float | int] = {"num_samples": sample_idx}
        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            for label_idx, label in enumerate(LABEL_SET):
                m_hat = float(means_hat[label_idx, feat_idx])
                s_hat = float(stds_hat[label_idx, feat_idx])
                m_err = m_hat - float(TRUE_MEANS[label_idx, feat_idx])
                s_err = s_hat - float(TRUE_STDS[label_idx, feat_idx])

                row[f"m_hat_{feat_name}_x{label}"] = m_hat
                row[f"sigma_hat_{feat_name}_x{label}"] = s_hat
                row[f"m_err_{feat_name}_x{label}"] = m_err
                row[f"sigma_err_{feat_name}_x{label}"] = s_err

        trajectory.append(row)
        print(f"  [{sample_idx:02d}/{len(label_files)}] 推定完了: {filename}")

    trajectory_path = os.path.join(OUT_DIR, "label_param_trajectory.tsv")
    fieldnames = ["num_samples"]
    for feat_name in FEATURE_NAMES:
        for label in LABEL_SET:
            fieldnames.append(f"m_hat_{feat_name}_x{label}")
            fieldnames.append(f"sigma_hat_{feat_name}_x{label}")
            fieldnames.append(f"m_err_{feat_name}_x{label}")
            fieldnames.append(f"sigma_err_{feat_name}_x{label}")

    with open(trajectory_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(trajectory)

    final = trajectory[-1]
    final_summary = {
        "num_samples": int(final["num_samples"]),
        "feature_names": FEATURE_NAMES,
        "label_set": LABEL_SET,
        "means_hat": [
            [float(final[f"m_hat_{feat}_x{label}"]) for feat in FEATURE_NAMES]
            for label in LABEL_SET
        ],
        "stds_hat": [
            [float(final[f"sigma_hat_{feat}_x{label}"]) for feat in FEATURE_NAMES]
            for label in LABEL_SET
        ],
        "means_err": [
            [float(final[f"m_err_{feat}_x{label}"]) for feat in FEATURE_NAMES]
            for label in LABEL_SET
        ],
        "stds_err": [
            [float(final[f"sigma_err_{feat}_x{label}"]) for feat in FEATURE_NAMES]
            for label in LABEL_SET
        ],
    }
    with open(os.path.join(OUT_DIR, "label_param_estimate_final.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    _plot_error_trajectories(
        trajectory,
        key="m",
        out_path=os.path.join(OUT_DIR, "m_error_trajectory.png"),
    )
    _plot_error_trajectories(
        trajectory,
        key="sigma",
        out_path=os.path.join(OUT_DIR, "sigma_error_trajectory.png"),
    )

    print(f"\n出力完了: {OUT_DIR}")
    print(f"- テキスト: {trajectory_path}")
    print("- 画像: m_error_trajectory.png, sigma_error_trajectory.png")
    print(f"- 抽出領域: {EXTRACTED_REGION_DIR}")


if __name__ == "__main__":
    train_progressive()
