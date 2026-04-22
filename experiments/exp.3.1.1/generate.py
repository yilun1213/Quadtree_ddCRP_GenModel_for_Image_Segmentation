# generate.py
# [exp.2.1.1] 真のパラメータを埋め込んで 50 組のデータを生成する
#
# 生成するデータ:
#   outputs/train_data/images/sample_XXXX.png
#   outputs/train_data/labels/sample_XXXX.png
#   outputs/train_data/labels/visualize/sample_XXXX.png
#
# 真のパラメータ (branch_probs):
#   depth 0-6: 0.9, depth 7: 0.0
#
# その他パラメータ (exp. 2.1 共通):
#   ddCRP: alpha=1e-8, beta=8.0, eta=8.0
#   ラベルモデル: 幾何学的特徴量の正規確率モデル
#   ピクセルモデル: 多変量正規分布

import os
import sys
import shutil
import csv
import numpy as np
from PIL import Image
import random
from typing import Callable

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import model.quadtree.depth_dependent_model as quadtree_model
import model.label.geom_features_norm_dist as label_model_module
import model.pixel.normal_dist as pixel_model_module
import model.region.affinity as affinity_module
from model.quadtree.node import Node

# ===== 真のパラメータ =====
MAX_DEPTH = 7
IMAGE_SIZE = 2 ** MAX_DEPTH  # 128

# 四分木の分岐確率 (exp.2.1.2)
TRUE_BRANCH_PROBS = [0.99, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0]

# ddCRP パラメータ (exp. 2.1 共通)
ALPHA = 1e-8
BETA = 8
ETA = 8

# ラベルモデルのパラメータ (幾何学的特徴量の正規確率に基づくモデル, exp. 2.1 共通)
LABEL_SET = [0, 1, 2]
LABEL_VALUE_SET = [0, 128, 255]
LABEL_FEATURE_NAMES = ["log_area", "circularity"]
LABEL_MEANS = [
    [8.0, 0.30],  # x=0
    [5.5, 0.65],  # x=1
    [5.5, 0.35],  # x=2
]
LABEL_STDS = [
    [1.0, 0.05],
    [0.5, 0.05],
    [0.5, 0.05],
]
LABEL_PARAM = {
    "image_size": IMAGE_SIZE,
    "feature_names": LABEL_FEATURE_NAMES,
    "means": LABEL_MEANS,
    "stds": LABEL_STDS,
}

# ピクセル値のパラメータ (exp. 2.1 共通)
PIXEL_PARAM = {
    "label_set": LABEL_SET,
    "mean": [[100, 100, 100], [200, 50, 50], [220, 30, 30]],
    "variance": [
        [[2500, 0, 0], [0, 2500, 0], [0, 0, 2500]],
        [[400, 0, 0], [0, 400, 0], [0, 0, 400]],
        [[400, 0, 0], [0, 400, 0], [0, 0, 400]],
    ],
}

# 生成枚数・出力先
TRAIN_NUM_SAMPLES = 50
TEST_NUM_SAMPLES = 10
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), 'outputs')
TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'train_data')
TEST_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'test_data')
ALL_REGION_FEATURES_CSV = os.path.join(OUTPUT_ROOT, 'all_region_features.csv')
SEED = 40

def ensure_dirs(base_dir: str) -> None:
    for sub in (
        "images",
        "labels",
        "labels/visualize",
        "quadtree_images",
        "region_images",
        "region_features",
        "region_images_extracted",
    ):
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)


def reset_dataset_dir(base_dir: str) -> None:
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    ensure_dirs(base_dir)


def save_quadtree_image(all_leaves: list[Node], max_depth: int, filename: str) -> dict[Node, np.ndarray]:
    size = int(2**max_depth)
    # uint8 は「符号なし8ビット整数（0〜255）」という意味
    image = np.zeros((size, size, 3), dtype=np.uint8)
    color_map: dict[tuple[int, int], np.ndarray] = {}

    for leaf in all_leaves:
        color_key = (leaf.upper_edge, leaf.left_edge)
        if color_key not in color_map:
            random_color = np.random.randint(50, 255, size=3)
            color_map[color_key] = random_color

    for leaf in all_leaves:
        color_key = (leaf.upper_edge, leaf.left_edge)
        color = color_map[color_key]
        image[leaf.upper_edge: leaf.upper_edge+leaf.size,
              leaf.left_edge: leaf.left_edge+leaf.size] = color
        leaf.original_color = color

    Image.fromarray(image).save(filename)

    return {leaf: leaf.original_color for leaf in all_leaves}


def overlap_1d(a1: int, a2: int, b1: int, b2: int) -> bool:
    return not (a2 <= b1 or b2 <= a1)


def precompute_adjacencies(all_leaves: list[Node]) -> dict[Node, list[Node]]:
    """
    事前にすべての葉ノード間の隣接関係を計算し、辞書として返す。
    """
    adjacency_dict = {leaf: [] for leaf in all_leaves}
    # 葉ノードを空間的に効率よく検索できるようにグリッドにマッピングする
    # ここでは簡単のため、総当たりで計算するが、それでも一度だけなので許容範囲
    for i, leaf1 in enumerate(all_leaves):
        for leaf2 in all_leaves[i + 1:]:
            # 水平方向の隣接チェック
            is_horizontally_adjacent = (leaf1.right_edge == leaf2.left_edge or
                                        leaf2.right_edge == leaf1.left_edge)
            y_overlap = (leaf1.upper_edge < leaf2.lower_edge and
                         leaf1.lower_edge > leaf2.upper_edge)

            # 垂直方向の隣接チェック
            is_vertically_adjacent = (leaf1.lower_edge == leaf2.upper_edge or
                                      leaf2.lower_edge == leaf1.upper_edge)
            x_overlap = (leaf1.left_edge < leaf2.right_edge and
                         leaf1.right_edge > leaf2.left_edge)

            if (is_horizontally_adjacent and y_overlap) or (is_vertically_adjacent and x_overlap):
                adjacency_dict[leaf1].append(leaf2)
                adjacency_dict[leaf2].append(leaf1)
    return adjacency_dict


def ddcrp_region_generation(
    all_leaves: list[Node],
    adjacency_dict: dict[Node, list[Node]],
    affinity_func: Callable,
    alpha: float = 1.0,
    **affinity_params
) -> dict[int, set[tuple[int, int]]]:
    """
    論文 2.4 節に基づく ddCRP（距離依存中華料理店過程）による領域生成。
    
    各葉ノード s に対して、結合先 c_s をサンプリングする。
    c_s = s' とは、ノード s がノード s' に結合することを意味する。
    c_s = s とは、ノード s が新しい領域の起点となることを意味する。
    
    サンプリング確率：
    p(c_s = s' | T; α, ...) ∝ f(s, s') / (α + Σ_{s'' ∈ L\{s}} f(s, s''))
    p(c_s = s | T; α, ...)  ∝ α / (α + Σ_{s'' ∈ L\{s}} f(s, s''))
    
    Args:
        all_leaves (list[Node]): 全葉ノード
        adjacency_dict (dict): ノード間の隣接関係
        affinity_func (Callable): 親和度関数 f(s, s', adjacency_dict, **affinity_params) -> float
        alpha (float): 新領域生成パラメータ（デフォルト: 1.0）
        **affinity_params: 親和度関数に渡すパラメータ
    
    Returns:
        dict[int, set[tuple[int, int]]]: 領域ID -> 領域内ピクセル集合
    """
    # ステップ1: 各葉ノードの結合先をサンプリング
    choice_dict: dict[Node, Node] = {}  # c_s を格納

    log_alpha = np.log(alpha) if alpha > 0.0 else -np.inf

    def _logsumexp(log_vals: list) -> float:
        if not log_vals:
            return -np.inf
        m = max(log_vals)
        return m + np.log(sum(np.exp(v - m) for v in log_vals))

    for leaf_s in all_leaves:
        # ノード s の隣接ノード
        neighbors = adjacency_dict.get(leaf_s, [])

        if not neighbors:
            # 隣接ノードがない場合は必ず新領域の起点となる
            choice_dict[leaf_s] = leaf_s
            continue

        # 各隣接ノードに対する対数親和度を計算（affinity_func は log f を返す）
        log_f_map: dict[Node, float] = {}
        for leaf_neighbor in neighbors:
            lf = affinity_func(leaf_s, leaf_neighbor, adjacency_dict, **affinity_params)
            if np.isfinite(lf):
                log_f_map[leaf_neighbor] = lf

        if not log_f_map:
            # 全ての log f が -inf → 自己参照
            choice_dict[leaf_s] = leaf_s
            continue

        log_sum_f = _logsumexp(list(log_f_map.values()))

        # log( α + Σf ) = logsumexp( log_α, log_sum_f )
        log_denom = _logsumexp([log_alpha, log_sum_f])

        # p(c_s = s) = α / denom
        prob_self = np.exp(log_alpha - log_denom)

        # c_s をサンプリング
        r = random.random()
        if r < prob_self:
            choice_dict[leaf_s] = leaf_s
        else:
            cumsum = prob_self
            chosen = leaf_s  # フォールバック（浮動小数点誤差対策）
            for leaf_neighbor, lf in log_f_map.items():
                cumsum += np.exp(lf - log_denom)
                if r < cumsum:
                    chosen = leaf_neighbor
                    break
            choice_dict[leaf_s] = chosen
    
    # ステップ2: 結合グラフから領域を構成（弱連結成分を計算）
    region_dict = _compute_weakly_connected_components(all_leaves, choice_dict)
    
    return region_dict


def _compute_weakly_connected_components(
    all_leaves: list[Node],
    choice_dict: dict[Node, Node]
) -> dict[int, set[tuple[int, int]]]:
    """
    有向グラフから弱連結成分を計算し、領域を構成する。
    
    有向辺は c_s -> c_{c_s} -> ... で構成され、
    全体が形成する無向グラフの連結成分が各領域となる。
    
    Args:
        all_leaves (list[Node]): 全葉ノード
        choice_dict (dict[Node, Node]): 各ノードの結合先
    
    Returns:
        dict[int, set[tuple[int, int]]]: 領域ID -> 領域内ピクセル集合
    """
    # 隣接リストを構成（無向グラフ）
    adj_undirected: dict[Node, set[Node]] = {leaf: set() for leaf in all_leaves}
    
    for leaf_s, leaf_target in choice_dict.items():
        if leaf_s != leaf_target:
            # 無向辺を追加
            adj_undirected[leaf_s].add(leaf_target)
            adj_undirected[leaf_target].add(leaf_s)
    
    # DFS で弱連結成分を計算
    visited = set()
    region_id = 1
    region_dict: dict[int, set[tuple[int, int]]] = {}
    
    for start_leaf in all_leaves:
        if start_leaf in visited:
            continue
        
        # DFS で連結成分を取得
        component = set()
        stack = [start_leaf]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            
            visited.add(node)
            component.add(node)
            
            for neighbor in adj_undirected[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        # 連結成分をピクセル集合に変換
        pixels = set()
        for leaf in component:
            i0, j0, sz = leaf.upper_edge, leaf.left_edge, leaf.size
            for i in range(i0, i0 + sz):
                for j in range(j0, j0 + sz):
                    pixels.add((i, j))
        
        region_dict[region_id] = pixels
        region_id += 1
    
    return region_dict


def save_region_growing_image(max_depth: int, region_dict: dict[int, set[tuple[int, int]]], filename: str) -> None:
    """
    領域画像を保存する。各領域に異なる色を割り当てる。
    
    Args:
        max_depth (int): 四分木の最大深さ
        region_dict (dict): 領域ID -> ピクセル座標集合の辞書
        filename (str): 保存先ファイル名
    """
    size = 2 ** max_depth
    image = np.zeros((size, size, 3), dtype=np.uint8)

    # 各領域に異なる色を割り当てる
    region_colors: dict[int, np.ndarray] = {}
    for region_id in region_dict.keys():
        # ランダムな色を生成（可視性のため 50-255 の範囲）
        region_colors[region_id] = np.random.randint(50, 255, size=3, dtype=np.uint8)
    
    # 各ピクセルに領域の色を設定
    for region_id, pixels in region_dict.items():
        color = region_colors[region_id]
        for (i, j) in pixels:
            image[i, j] = color

    Image.fromarray(image).save(filename)


def _compute_region_perimeter(pixels: set[tuple[int, int]]) -> int:
    perimeter = 0
    for i, j in pixels:
        if (i - 1, j) not in pixels:
            perimeter += 1
        if (i + 1, j) not in pixels:
            perimeter += 1
        if (i, j - 1) not in pixels:
            perimeter += 1
        if (i, j + 1) not in pixels:
            perimeter += 1
    return perimeter


def compute_region_shape_features(region_dict: dict[int, set[tuple[int, int]]]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    for region_id, pixels in region_dict.items():
        if not pixels:
            continue

        area = len(pixels)
        perimeter = _compute_region_perimeter(pixels)

        ys = [p[0] for p in pixels]
        xs = [p[1] for p in pixels]
        min_row, max_row = min(ys), max(ys)
        min_col, max_col = min(xs), max(xs)

        bbox_height = max_row - min_row + 1
        bbox_width = max_col - min_col + 1
        bbox_area = bbox_height * bbox_width

        centroid_row = float(np.mean(ys))
        centroid_col = float(np.mean(xs))

        circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        extent = (area / bbox_area) if bbox_area > 0 else 0.0

        rows.append(
            {
                "region_id": int(region_id),
                "area": int(area),
                "perimeter": int(perimeter),
                "log_area": float(np.log(area)),
                "log_perimeter": float(np.log(perimeter)) if perimeter > 0 else 0.0,
                "circularity": float(circularity),
                "bbox_min_row": int(min_row),
                "bbox_max_row": int(max_row),
                "bbox_min_col": int(min_col),
                "bbox_max_col": int(max_col),
                "bbox_width": int(bbox_width),
                "bbox_height": int(bbox_height),
                "bbox_area": int(bbox_area),
                "extent": float(extent),
                "centroid_row": float(centroid_row),
                "centroid_col": float(centroid_col),
            }
        )

    rows.sort(key=lambda r: int(r["region_id"]))
    return rows


def add_sample_metadata_to_region_features(
    feature_rows: list[dict[str, float | int]],
    dataset_split: str,
    sample_name: str,
) -> list[dict[str, str | float | int]]:
    rows_with_meta: list[dict[str, str | float | int]] = []
    for row in feature_rows:
        row_with_meta: dict[str, str | float | int] = {
            "dataset_split": dataset_split,
            "sample_name": sample_name,
        }
        row_with_meta.update(row)
        rows_with_meta.append(row_with_meta)
    return rows_with_meta


def save_region_shape_features_csv(
    feature_rows: list[dict[str, str | float | int]],
    filename: str,
) -> None:
    fieldnames = [
        "dataset_split",
        "sample_name",
        "region_id",
        "area",
        "perimeter",
        "log_area",
        "log_perimeter",
        "circularity",
        "bbox_min_row",
        "bbox_max_row",
        "bbox_min_col",
        "bbox_max_col",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "extent",
        "centroid_row",
        "centroid_col",
    ]

    if not feature_rows:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)


def sample_label_image(region_dict: dict[int, set[tuple[int, int]]]) -> tuple[np.ndarray, np.ndarray]:
    label_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    label_vis_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    label_value_map = dict(zip(LABEL_SET, LABEL_VALUE_SET))
    for _idx, region in region_dict.items():
        probs = label_model_module.label_prior(region=region, param=LABEL_PARAM)
        chosen_idx = int(np.random.choice(len(LABEL_SET), p=probs))
        label = int(LABEL_SET[chosen_idx])
        vis_value = int(label_value_map.get(label, label))
        for (i, j) in region:
            label_image[i, j] = label
            label_vis_image[i, j] = vis_value
    return label_image, label_vis_image


def save_label_images(
    label_image: np.ndarray,
    label_vis_image: np.ndarray,
    label_filename: str,
    label_vis_filename: str,
) -> None:
    Image.fromarray(label_image).save(label_filename)
    Image.fromarray(label_vis_image).save(label_vis_filename)


def generate_dataset(output_dir: str, num_samples: int, seed_offset: int, dataset_split: str) -> list[dict[str, str | float | int]]:
    print(f"{num_samples} 組のデータを生成します -> {output_dir}\n")
    all_rows: list[dict[str, str | float | int]] = []

    for idx in range(num_samples):
        print(f"[{idx + 1}/{num_samples}] サンプル生成中...")

        qt = quadtree_model.QuadTree(
            max_depth=MAX_DEPTH,
            branch_prob=TRUE_BRANCH_PROBS,
            seed=SEED + seed_offset + idx,
        )
        all_leaves = qt.get_leaves()
        adjacency_dict = precompute_adjacencies(all_leaves)

        region_dict = ddcrp_region_generation(
            all_leaves=all_leaves,
            adjacency_dict=adjacency_dict,
            affinity_func=affinity_module.log_affinity_boundary_and_depth,
            alpha=ALPHA,
            beta=BETA,
            eta=ETA,
        )

        label_array, label_vis_array = sample_label_image(region_dict)

        rgb = pixel_model_module.generate_rgb_from_labels(
            label_image=label_array,
            region_dict=region_dict,
            theta=PIXEL_PARAM,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
            seed=SEED + 1000 + seed_offset + idx,
        )

        stem = f"sample_{idx:04d}"
        save_quadtree_image(
            all_leaves=all_leaves,
            max_depth=MAX_DEPTH,
            filename=os.path.join(output_dir, "quadtree_images", f"{stem}.png"),
        )
        save_region_growing_image(
            max_depth=MAX_DEPTH,
            region_dict=region_dict,
            filename=os.path.join(output_dir, "region_images", f"{stem}.png"),
        )
        region_feature_rows = compute_region_shape_features(region_dict)
        all_rows.extend(
            add_sample_metadata_to_region_features(
                feature_rows=region_feature_rows,
                dataset_split=dataset_split,
                sample_name=stem,
            )
        )
        Image.fromarray(rgb).save(os.path.join(output_dir, "images", f"{stem}.png"))
        Image.fromarray(label_array).save(os.path.join(output_dir, "labels", f"{stem}.png"))
        Image.fromarray(label_vis_array).save(
            os.path.join(output_dir, "labels", "visualize", f"{stem}.png")
        )
        print(f"  -> {stem} 保存完了 (regions={len(region_dict)})")

    print(f"\n生成完了: {num_samples} 組を {output_dir} に保存しました。")
    return all_rows


def main():
    np.random.seed(SEED)
    random.seed(SEED)

    print(f"真の branch_probs: {TRUE_BRANCH_PROBS}")
    print(f"alpha={ALPHA}, beta={BETA}, eta={ETA}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    reset_dataset_dir(TRAIN_OUTPUT_DIR)
    all_region_feature_rows = generate_dataset(
        TRAIN_OUTPUT_DIR,
        TRAIN_NUM_SAMPLES,
        seed_offset=0,
        dataset_split="train",
    )

    reset_dataset_dir(TEST_OUTPUT_DIR)
    all_region_feature_rows.extend(
        generate_dataset(
            TEST_OUTPUT_DIR,
            TEST_NUM_SAMPLES,
            seed_offset=10000,
            dataset_split="test",
        )
    )

    save_region_shape_features_csv(
        feature_rows=all_region_feature_rows,
        filename=ALL_REGION_FEATURES_CSV,
    )
    print(f"全小領域の幾何学的特徴量CSVを保存しました: {ALL_REGION_FEATURES_CSV}")


if __name__ == "__main__":
    main()
