# -*- coding: utf-8 -*-
"""
親和度関数（Affinity Functions）の実装

各葉ノード間の結合のしやすさを表す関数を定義する。
論文 2.4 節の例に基づき、複数のバリエーションを提供可能。
"""

import numpy as np
from model.quadtree.node import Node


def affinity_boundary_and_depth(leaf1: Node, leaf2: Node, adjacency_dict: dict, beta: float = 1.0, eta: float = 0.5) -> float:
    """
    論文 2.4 節の例に基づく親和度関数。
    
    隣接する葉ノード s, s' 間の共有境界線の長さ B(s, s') とそれぞれの深さ depth(s), depth(s') に基づいて
    親和度を計算する。
    
    f(s, s') = exp(β * B(s, s') + η * (depth(s) - depth(s')))
    
    Args:
        leaf1 (Node): 第1の葉ノード
        leaf2 (Node): 第2の葉ノード
        adjacency_dict (dict): ノード間の隣接関係を格納した辞書（キー: Node、値: 隣接ノードのリスト）
        beta (float): 共有境界の長さに応じた滑らかさのパラメータ（デフォルト: 1.0）
        eta (float): ノードサイズの階層性を制御するパラメータ（デフォルト: 0.5）
    
    Returns:
        float: 親和度（非負の値）親和度関数の値、隣接していない場合は0
    """
    # leaf1 と leaf2 が隣接しているかチェック
    if leaf2 not in adjacency_dict.get(leaf1, []):
        return 0.0
    
    # 共有境界線の長さ B(s, s') を計算
    boundary_length = _compute_shared_boundary_length(leaf1, leaf2)
    
    # 深さの差を計算
    depth_diff = leaf1.depth - leaf2.depth
    
    # 親和度を計算
    affinity = np.exp(beta * boundary_length + eta * depth_diff)
    
    return float(affinity)


def affinity_boundary_depth_and_large_pair(
    leaf1: Node,
    leaf2: Node,
    adjacency_dict: dict,
    beta: float = 1.0,
    eta: float = 0.5,
    gamma: float = 0.0,
) -> float:
    """
    共有境界長 + 深さ差 + 「大ノード同士ボーナス」に基づく親和度関数。

    f(s, s') = exp( beta * B(s, s')
                    + eta * (depth(s) - depth(s'))
                    + gamma * log2(min(size(s), size(s'))) )

    - depth 差が 0 でも、両者が大きいほど（min size が大きいほど）親和度が上がる。
    - min(size) を使うことで「両方が大きい」場合のみ強く優遇される。

    Args:
        leaf1 (Node): 第1の葉ノード
        leaf2 (Node): 第2の葉ノード
        adjacency_dict (dict): ノード間の隣接関係
        beta (float): 共有境界長の重み
        eta (float): 深さ差の重み
        gamma (float): 大ノード同士ボーナスの重み

    Returns:
        float: 親和度（隣接していない場合は0）
    """
    if leaf2 not in adjacency_dict.get(leaf1, []):
        return 0.0

    boundary_length = _compute_shared_boundary_length(leaf1, leaf2)
    depth_diff = leaf1.depth - leaf2.depth
    large_pair_term = _large_pair_score(leaf1, leaf2)

    affinity = np.exp(beta * boundary_length + eta * depth_diff + gamma * large_pair_term)
    return float(affinity)


def affinity_boundary_only(leaf1: Node, leaf2: Node, adjacency_dict: dict, beta: float = 1.0) -> float:
    """
    共有境界線の長さのみに基づく親和度関数（シンプル版）。
    
    f(s, s') = exp(β * B(s, s'))
    
    Args:
        leaf1 (Node): 第1の葉ノード
        leaf2 (Node): 第2の葉ノード
        adjacency_dict (dict): ノード間の隣接関係を格納した辞書
        beta (float): パラメータ（デフォルト: 1.0）
    
    Returns:
        float: 親和度
    """
    # leaf1 と leaf2 が隣接しているかチェック
    if leaf2 not in adjacency_dict.get(leaf1, []):
        return 0.0
    
    boundary_length = _compute_shared_boundary_length(leaf1, leaf2)
    affinity = np.exp(beta * boundary_length)
    
    return float(affinity)


def affinity_target_shallow_exp(
    leaf1: Node,
    leaf2: Node,
    adjacency_dict: dict,
    kappa: float = 1.0,
    max_depth: int = 8,
) -> float:
    """
    リンク先ノードの深さだけに指数依存するシンプルな親和度関数。

    f(s, s') = exp(kappa * (max_depth - depth(s')))

    - depth(s') が小さい（= ノードが大きい）ほど親和度が高い。
    - max_depth を基準化に使い、親和度の絶対値を確保して alpha に負けにくくする。
    - まずは構造を単純化するため、共有境界長や深さ差項は使わない。

    Args:
        leaf1 (Node): 現在ノード s（未使用）
        leaf2 (Node): リンク先ノード s'
        adjacency_dict (dict): ノード間の隣接関係
        kappa (float): 深さ優遇の強さ（大きいほど浅いノードを強く優遇）
        max_depth (int): 四分木の最大深さ

    Returns:
        float: 親和度（隣接していない場合は0）
    """
    if leaf2 not in adjacency_dict.get(leaf1, []):
        return 0.0

    return float(np.exp(kappa * (max_depth - leaf2.depth)))


def affinity_constant(leaf1: Node, leaf2: Node, adjacency_dict: dict) -> float:
    """
    一定の親和度を返す関数（テスト用）。
    
    隣接しているすべてのノード対に対して同じ親和度を返す。
    
    Args:
        leaf1 (Node): 第1の葉ノード
        leaf2 (Node): 第2の葉ノード
        adjacency_dict (dict): ノード間の隣接関係を格納した辞書
    
    Returns:
        float: 親和度（隣接している場合は1.0、そうでない場合は0.0）
    """
    if leaf2 not in adjacency_dict.get(leaf1, []):
        return 0.0
    
    return 1.0


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
