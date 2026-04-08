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
TRUE_BRANCH_PROBS = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]

# ddCRP パラメータ (exp. 2.1 共通)
ALPHA = 1e-8
BETA = 8.0
ETA = 8.0

# ラベルモデルのパラメータ (幾何学的特徴量の正規確率に基づくモデル, exp. 2.1 共通)
LABEL_SET = [0, 1, 2]
LABEL_VALUE_SET = [0, 128, 255]
LABEL_FEATURE_NAMES = ["log_area", "log_perimeter", "circularity"]
LABEL_MEANS = [
    [4.0, 3.5, 0.45],  # x=0
    [6.5, 5.0, 0.50],  # x=1
    [9.0, 6.0, 0.70],  # x=2
]
LABEL_STDS = [
    [1.0, 0.5, 0.2],
    [1.5, 0.5, 0.1],
    [1.0, 0.5, 0.1],
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
    "mean": [[200, 50, 50], [50, 200, 50], [50, 50, 200]],
    "variance": [
        [[20, 0, 0], [0, 20, 0], [0, 0, 20]],
        [[20, 0, 0], [0, 20, 0], [0, 0, 20]],
        [[20, 0, 0], [0, 20, 0], [0, 0, 20]],
    ],
}

# 生成枚数・出力先
NUM_SAMPLES = 50
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'train_data')
SEED = 42

def ensure_dirs(base_dir: str) -> None:
    for sub in ("images", "labels", "labels/visualize", "quadtree_images", "region_images"):
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)


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
    
    # デバッグ情報
    num_self_reference = 0  # 新領域起点となったノード数
    affinity_values = []  # 親和度の値のリスト
    
    for leaf_s in all_leaves:
        # ノード s の隣接ノード
        neighbors = adjacency_dict.get(leaf_s, [])
        
        if not neighbors:
            # 隣接ノードがない場合は必ず新領域の起点となる
            choice_dict[leaf_s] = leaf_s
            num_self_reference += 1
            continue
        
        # 各隣接ノードに対する親和度を計算
        affinities = {}
        affinity_sum = 0.0
        for leaf_neighbor in neighbors:
            aff = affinity_func(leaf_s, leaf_neighbor, adjacency_dict, **affinity_params)
            if aff > 0:
                affinities[leaf_neighbor] = aff
                affinity_sum += aff
                affinity_values.append(aff)
        
        # サンプリング確率を計算
        denominator = alpha + affinity_sum
        
        # 新領域起点となる確率（c_s = s）
        prob_self = alpha / denominator
        
        # 隣接ノードへの結合確率
        probs_neighbors = {leaf_neighbor: aff / denominator for leaf_neighbor, aff in affinities.items()}
        
        # c_s をサンプリング
        r = random.random()
        cumsum = 0.0
        if r < prob_self:
            choice_dict[leaf_s] = leaf_s
            num_self_reference += 1
        else:
            # サンプリング確率の累積分布から選択
            cumsum = prob_self
            chosen = leaf_s  # デフォルト
            for leaf_neighbor, prob_neighbor in probs_neighbors.items():
                cumsum += prob_neighbor
                if r < cumsum:
                    chosen = leaf_neighbor
                    break
            choice_dict[leaf_s] = chosen
    
    # デバッグ情報を出力
    if affinity_values:
        print(f"  [ddCRP Debug] Affinity stats: min={min(affinity_values):.4f}, max={max(affinity_values):.4f}, mean={np.mean(affinity_values):.4f}")
        print(f"  [ddCRP Debug] Self-reference nodes: {num_self_reference}/{len(all_leaves)} ({100*num_self_reference/len(all_leaves):.1f}%)")
        avg_affinity = np.mean(affinity_values)
        expected_prob = alpha / (alpha + avg_affinity * 4) if avg_affinity > 0 else 1.0
        print(f"  [ddCRP Debug] Alpha={alpha}, Expected self-reference prob approx {expected_prob:.4f}")
    
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


def main():
    np.random.seed(SEED)
    random.seed(SEED)

    ensure_dirs(OUTPUT_DIR)

    print(f"真の branch_probs: {TRUE_BRANCH_PROBS}")
    print(f"alpha={ALPHA}, beta={BETA}, eta={ETA}")
    print(f"{NUM_SAMPLES} 組のデータを生成します -> {OUTPUT_DIR}\n")

    for idx in range(NUM_SAMPLES):
        print(f"[{idx + 1}/{NUM_SAMPLES}] サンプル生成中...")

        qt = quadtree_model.QuadTree(
            max_depth=MAX_DEPTH,
            branch_prob=TRUE_BRANCH_PROBS,
            seed=SEED + idx,
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
            seed=SEED + NUM_SAMPLES + idx,
        )

        stem = f"sample_{idx:04d}"
        save_quadtree_image(
            all_leaves=all_leaves,
            max_depth=MAX_DEPTH,
            filename=os.path.join(OUTPUT_DIR, "quadtree_images", f"{stem}.png"),
        )
        save_region_growing_image(
            max_depth=MAX_DEPTH,
            region_dict=region_dict,
            filename=os.path.join(OUTPUT_DIR, "region_images", f"{stem}.png"),
        )
        Image.fromarray(rgb).save(os.path.join(OUTPUT_DIR, "images", f"{stem}.png"))
        Image.fromarray(label_array).save(os.path.join(OUTPUT_DIR, "labels", f"{stem}.png"))
        Image.fromarray(label_vis_array).save(
            os.path.join(OUTPUT_DIR, "labels", "visualize", f"{stem}.png")
        )
        print(f"  -> {stem} 保存完了 (regions={len(region_dict)})")

    print(f"\n生成完了: {NUM_SAMPLES} 組を {OUTPUT_DIR} に保存しました。")


if __name__ == "__main__":
    main()
