# -*- coding: utf-8 -*-
"""
親和度関数（Affinity Functions）の実装

各葉ノード間の結合のしやすさを表す関数を定義する。
論文 2.4 節の例に基づき、複数のバリエーションを提供可能。
"""

import numpy as np
from model.quadtree.node import Node


def log_affinity_boundary_and_depth(leaf1: Node, leaf2: Node, adjacency_dict: dict, beta: float = 1.0, eta: float = 0.5) -> float:
    """
    論文 2.4 節の例に基づく対数親和度関数。
    
    log f(s, s') = β * B(s, s') + η * (depth(s) - depth(s'))
    
    Returns:
        float: 対数親和度。隣接していない場合は -inf
    """
    if leaf2 not in adjacency_dict.get(leaf1, []):
        return -np.inf
    
    boundary_length = _compute_shared_boundary_length(leaf1, leaf2)
    depth_diff = leaf1.depth - leaf2.depth
    
    return float(beta * boundary_length + eta * depth_diff)


def log_affinity_boundary_depth_and_large_pair(
    leaf1: Node,
    leaf2: Node,
    adjacency_dict: dict,
    beta: float = 1.0,
    eta: float = 0.5,
    gamma: float = 0.0,
) -> float:
    """
    共有境界長 + 深さ差 + 「大ノード同士ボーナス」に基づく対数親和度関数。

    log f(s, s') = beta * B(s, s') + eta * (depth(s) - depth(s')) + gamma * log2(min(size(s), size(s')))

    Returns:
        float: 対数親和度。隣接していない場合は -inf
    """
    if leaf2 not in adjacency_dict.get(leaf1, []):
        return -np.inf

    boundary_length = _compute_shared_boundary_length(leaf1, leaf2)
    depth_diff = leaf1.depth - leaf2.depth
    large_pair_term = _large_pair_score(leaf1, leaf2)

    return float(beta * boundary_length + eta * depth_diff + gamma * large_pair_term)


def log_affinity_boundary_only(leaf1: Node, leaf2: Node, adjacency_dict: dict, beta: float = 1.0) -> float:
    """
    共有境界線の長さのみに基づく対数親和度関数（シンプル版）。
    
    log f(s, s') = β * B(s, s')
    
    Returns:
        float: 対数親和度。隣接していない場合は -inf
    """
    if leaf2 not in adjacency_dict.get(leaf1, []):
        return -np.inf
    
    boundary_length = _compute_shared_boundary_length(leaf1, leaf2)
    return float(beta * boundary_length)


def log_affinity_target_shallow_exp(
    leaf1: Node,
    leaf2: Node,
    adjacency_dict: dict,
    kappa: float = 1.0,
    max_depth: int = 8,
) -> float:
    """
    リンク先ノードの深さだけに依存するシンプルな対数親和度関数。

    log f(s, s') = kappa * (max_depth - depth(s'))

    Returns:
        float: 対数親和度。隣接していない場合は -inf
    """
    if leaf2 not in adjacency_dict.get(leaf1, []):
        return -np.inf

    return float(kappa * (max_depth - leaf2.depth))


def log_affinity_constant(leaf1: Node, leaf2: Node, adjacency_dict: dict) -> float:
    """
    一定の対数親和度を返す関数（テスト用）。
    
    隣接しているすべてのノード対に対して log f = 0 (f = 1) を返す。
    
    Returns:
        float: 対数親和度（隣接している場合は0.0、そうでない場合は -inf）
    """
    if leaf2 not in adjacency_dict.get(leaf1, []):
        return -np.inf
    
    return 0.0


# ============================================================================
# ヘルパー関数
# ============================================================================

def _compute_shared_boundary_length(leaf1: Node, leaf2: Node) -> float:
    """
    2つの葉ノード間の共有境界線の長さを計算する。
    
    2つの矩形が隣接している場合、その共有辺の長さを返す。
    隣接していない場合は0を返す。
    
    Args:
        leaf1 (Node): 第1の葉ノード
        leaf2 (Node): 第2の葉ノード
    
    Returns:
        float: 共有境界線の長さ（ピクセル単位）
    """
    # 水平方向に隣接しているかチェック
    if leaf1.right_edge == leaf2.left_edge or leaf2.right_edge == leaf1.left_edge:
        # y 方向の重複を計算
        y_min = max(leaf1.upper_edge, leaf2.upper_edge)
        y_max = min(leaf1.lower_edge, leaf2.lower_edge)
        if y_min < y_max:
            return float(y_max - y_min)
    
    # 垂直方向に隣接しているかチェック
    if leaf1.lower_edge == leaf2.upper_edge or leaf2.lower_edge == leaf1.upper_edge:
        # x 方向の重複を計算
        x_min = max(leaf1.left_edge, leaf2.left_edge)
        x_max = min(leaf1.right_edge, leaf2.right_edge)
        if x_min < x_max:
            return float(x_max - x_min)
    
    return 0.0


def _large_pair_score(leaf1: Node, leaf2: Node) -> float:
    """
    両ノードが大きいほど大きくなるスコア。

    min(size1, size2) を採用し、片方だけ大きいケースを過大評価しない。
    """
    min_size = max(1, min(leaf1.size, leaf2.size))
    return float(np.log2(min_size))
