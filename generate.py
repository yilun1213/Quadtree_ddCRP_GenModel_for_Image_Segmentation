# generate.py

from config_gen import Config, load_config
import os
import sys
import numpy as np
from PIL import Image
import random
from model.quadtree.node import Node
import traceback
from typing import Callable

def ensure_split_dirs(base_dir: str) -> None:
    for sub in ("images", "labels", "labels/visualize", "regions", "quadtrees"):
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


def label_gen(config: Config, region_dict: dict[int, set[tuple[int, int]]], label_filename: str, label_vis_filename: str):
    size = 2 ** config.quadtree_config.max_depth
    label_image = np.zeros((size, size), dtype=np.uint8)
    label_vis_image = np.zeros((size, size), dtype=np.uint8)
    label_set = config.label_config.label_set
    label_value_map = {
        int(label): int(value)
        for label, value in zip(config.label_config.label_set, config.label_config.label_value_set)
    }
    for idx, region in region_dict.items():
        probs = config.label_config.model.label_prior(
            region=region, param=config.label_config.param)
        chosen_idx = int(np.random.choice(range(config.label_config.label_num), p=probs))
        label = int(label_set[chosen_idx])
        vis_value = int(label_value_map.get(label, label))
        for coords in region:
            i = int(coords[0])
            j = int(coords[1])
            label_image[i][j] = label
            label_vis_image[i][j] = vis_value
    Image.fromarray(label_image).save(label_filename)
    Image.fromarray(label_vis_image).save(label_vis_filename)
    return label_image


def generate_data(config: Config, itr_num: int, out_dir: str):
    seed = config.seed + itr_num if config.seed is not None else None

    print(f"Generating: quadtree {itr_num}")
    qt = config.quadtree_config.model.QuadTree(
        max_depth=config.quadtree_config.max_depth,
        branch_prob=config.quadtree_config.branch_probs,
        seed=seed
    )
    all_leaves = qt.get_leaves()
    quadtree_colors = save_quadtree_image(
        all_leaves=all_leaves, max_depth=config.quadtree_config.max_depth, filename=f"{out_dir}/quadtrees/{itr_num: 04d}.png")

    print(f"Generating: region {itr_num}")
    adjacency_dict = precompute_adjacencies(all_leaves)
    region_dict = ddcrp_region_generation(
        all_leaves=all_leaves,
        adjacency_dict=adjacency_dict,
        affinity_func=config.affinity_func,
        alpha=config.alpha,
        **config.affinity_params
    )
    print(f"  -> Generated {len(region_dict)} regions from {len(all_leaves)} leaves")
    save_region_growing_image(
        max_depth=config.quadtree_config.max_depth,
        region_dict=region_dict,
        filename=f"{out_dir}/regions/{itr_num: 04d}.png"
    )

    print(f"Generating: label {itr_num}")
    label_array = label_gen(config=config, region_dict=region_dict,
                            label_filename=f"{out_dir}/labels/{itr_num: 04d}.png", label_vis_filename=f"{out_dir}/labels/visualize/{itr_num: 04d}.png")

    print(f"Generating: image {itr_num}")
    rgb = config.pixel_config.model.generate_rgb_from_labels(
        label_image=label_array,
        region_dict=region_dict,
        theta=config.pixel_config.param,
        width=2**config.quadtree_config.max_depth,
        height=2**config.quadtree_config.max_depth,
        seed=seed
    )

    Image.fromarray(rgb).save(f"{out_dir}/images/{itr_num: 04d}.png")
    print("-"*20)


def main():
    # 設定の読み込み（config_gen.py 側でパラメータディレクトリ/ファイル名を管理）
    try:
        config_gen = load_config()
    except FileNotFoundError as e:
        print("エラー: パラメータファイルが見つかりません。")
        print(str(e))
        sys.exit(1)

    # パラメータファイルの確認
    param_dir = config_gen.param_dir
    required_files = [
        config_gen.pixel_param_filename,
        config_gen.branch_probs_filename,
        config_gen.label_param_filename,
    ]
    missing_files = []

    if not os.path.exists(param_dir):
        missing_files = required_files
    else:
        for f in required_files:
            if not os.path.exists(os.path.join(param_dir, f)):
                missing_files.append(f)

    if missing_files:
        print("エラー: 以下のパラメータファイルが見つかりません。")
        print(f"検索ディレクトリ: {param_dir}")
        for f in missing_files:
            print(f"- {f}")
        print("\ntrain.py 出力形式に合わせたJSONを配置してください。")
        print("\n[pixel_param.json の例]")
        print('{"label_set": [0, 1], "channels": 3, "mean": [[80, 80, 80], [180, 180, 180]], "variance": [[[400,0,0],[0,400,0],[0,0,400]], [[400,0,0],[0,400,0],[0,0,400]]], "std": [[20, 20, 20], [20, 20, 20]]}')
        print("\n[branch_probs.json の例]")
        print('{"branch_probs": [1.0, 0.67, ...]}')
        print("\n[label_param.json の例]")
        print('{"label_num": 2, "label_set": [0, 1], "label_value_set": [0, 255], "weights": [[...], [...]], "bias": [...], "image_size": 128, "feature_names": ["log_area", "log_perimeter", "circularity"]}')
        sys.exit(1)

    ensure_split_dirs(config_gen.train.dir)
    ensure_split_dirs(config_gen.test.dir)

    for i in range(config_gen.train.num):
        generate_data(config=config_gen, itr_num=i,
                      out_dir=config_gen.train.dir)
    for i in range(config_gen.test.num):
        generate_data(config=config_gen, itr_num=config_gen.train.num +
                      i, out_dir=config_gen.test.dir)

if __name__ == "__main__":
    main()
