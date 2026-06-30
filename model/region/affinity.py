# -*- coding: utf-8 -*-
"""
親和度関数（Affinity Functions）の実装

各葉ノード間の結合のしやすさを表す関数を定義する。
"""

import numpy as np
from model.quadtree.node import Node


def log_affinity_depth_only(leaf1: Node, leaf2: Node, adjacency_dict: dict, eta: float = 10.0) -> float:
    """
    深さの差に基づく対数親和度関数。
    
    log f(s, s') = eta * (depth(s) - depth(s'))
    
    Returns:
        float: 対数親和度。隣接していない場合は -inf
    """
    if leaf2 not in adjacency_dict.get(leaf1, []):
        return -np.inf
    
    depth_diff = leaf1.depth - leaf2.depth
    return float(eta * depth_diff)
