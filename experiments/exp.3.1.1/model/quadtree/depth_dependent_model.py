# ./model/depth_dependent_model.py

import os
import numpy as np
from PIL import Image
import random
import json
from model.quadtree.node import Node
import utils



class QuadTree:
    def __init__(self, max_depth: int, branch_prob: list[float], seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.max_depth = max_depth
        self.branch_prob = branch_prob
        self.root = Node(upper_edge=0, left_edge=0, size=2**max_depth, depth=0)
        self._split(self.root)  # Quadtreeが呼び出されたら分割開始

    def _split(self, node: Node):
        d = node.depth
        if d < self.max_depth and random.random() < self.branch_prob[d]:
            node.is_leaf = False
            half = node.size // 2
            
            node.ul_node = Node(node.upper_edge, node.left_edge, half, d + 1)
            node.ur_node = Node(node.upper_edge, node.left_edge + half, half, d + 1)
            node.ll_node = Node(node.upper_edge + half, node.left_edge, half, d + 1)
            node.lr_node = Node(node.upper_edge + half, node.left_edge + half, half, d + 1)

            self._split(node.ul_node)
            self._split(node.ur_node)
            self._split(node.ll_node)
            self._split(node.lr_node)

    def get_leaves(self) -> list[Node]:
        leaves = []
        def _find_leaves(n: Node):
            if n.is_leaf:
                leaves.append(n)
            else:
                _find_leaves(n.ul_node)
                _find_leaves(n.ur_node)
                _find_leaves(n.ll_node)
                _find_leaves(n.lr_node)
        _find_leaves(self.root)
        return leaves


def label_ndarray(node: Node, label_array: np.ndarray) -> np.ndarray:
    """指定されたノードに対応するラベル領域のndarrayを取得する"""
    return label_array[node.upper_edge:node.upper_edge + node.size, node.left_edge:node.left_edge + node.size]


def make_tree(node: Node, max_depth: int):
    """四分木構造を再帰的に事前生成する"""
    if node.depth < max_depth:
        node.is_leaf = False
        half = node.size // 2
        node.ul_node = Node(node.upper_edge, node.left_edge, half, node.depth + 1)
        node.ur_node = Node(node.upper_edge, node.left_edge + half, half, node.depth + 1)
        node.ll_node = Node(node.upper_edge + half, node.left_edge, half, node.depth + 1)
        node.lr_node = Node(node.upper_edge + half, node.left_edge + half, half, node.depth + 1)
        for child in [node.ul_node, node.ur_node, node.ll_node, node.lr_node]:
            make_tree(child, max_depth) # type: ignore


def is_matrix_all_same(matrix: np.ndarray) -> bool:
    """行列内のすべての要素が同じかどうかを判定する"""
    return (matrix == matrix[0, 0]).all()


def recursive_split_for_tree(node: Node, label_array: np.ndarray):
    """ラベル配列に基づき、再帰的に分割/非分割をカウントする"""
    if not is_matrix_all_same(label_ndarray(node, label_array)) and not node.is_leaf:
        node.split_count += 1
        for child in [node.ul_node, node.ur_node, node.ll_node, node.lr_node]:
            recursive_split_for_tree(child, label_array)
    else:
        node.nonsplit_count += 1


def get_split_probs_at_depth(node: Node, depth: int, list_g_per_d: list):
    """指定された深度のノードの分割確率をリストに集める"""
    if node.depth == depth:
        a_s = node.split_count
        b_s = node.nonsplit_count
        # カウントが0のノードは、学習データで一度も評価されていないためスキップ
        if (a_s + b_s) > 0:
            tmp = a_s / (a_s + b_s)
            list_g_per_d.append(tmp)
    if not node.is_leaf:
        for child in [node.ul_node, node.ur_node, node.ll_node, node.lr_node]:
            get_split_probs_at_depth(child, depth, list_g_per_d)


def param_est(train_label_dir: str, out_g_param_path: str):
    """
    学習データから四分木の分岐確率Gを推定し、JSONファイルに保存する。
    """
    label_files = utils.get_image_files(train_label_dir)
    if not label_files:
        print(f"学習用のラベル画像が見つかりません: {train_label_dir}")
        return

    first_label_img = utils.load_image(os.path.join(train_label_dir, label_files[0]))
    height, width = first_label_img.shape[:2]
    max_depth = int(np.log2(width))

    root = Node(upper_edge=0, left_edge=0, size=width, depth=0)
    make_tree(root, max_depth)

    for label_file in label_files:
        label_array = utils.load_image(os.path.join(train_label_dir, label_file))
        recursive_split_for_tree(root, label_array)

    branch_probs = []
    for d in range(max_depth):
        list_g_per_d = []
        get_split_probs_at_depth(root, d, list_g_per_d)
        branch_probs.append(sum(list_g_per_d) / len(list_g_per_d) if list_g_per_d else 0)
    branch_probs.append(0)  # 最大深度では分割しない

    print(f"Estimated branch_probs = {branch_probs}")

    with open(out_g_param_path, 'w') as f:
        json.dump({"branch_probs": branch_probs}, f, indent=4)
    print(f"branch_probsのパラメータを {out_g_param_path} に保存しました。")
