# ./model/quadtree/node.py
from __future__ import annotations
import numpy as np

class Node:
    def __init__(self, upper_edge: int, left_edge: int, size: int, depth: int):
        self.upper_edge = upper_edge
        self.left_edge = left_edge
        self.size = size
        self.depth = depth

        # 共通の属性
        self.is_leaf = True
        self.ul_node: Node | None = None
        self.ur_node: Node | None = None
        self.ll_node: Node | None = None
        self.lr_node: Node | None = None

        # generate.py で使用
        self.is_merged = False
        self.is_explored = False
        self.region_id: int | None = None

        # estimate_Tfixed copy.py で使用
        self.logq_Ys = 0.0
        self.logq_YsXs = 0.0
        self.original_color: np.ndarray | None = None

        # depth_dependent_model.py (param_est) で使用
        self.split_count = 0
        self.nonsplit_count = 0

    @property
    def lower_edge(self) -> int:
        return self.upper_edge + self.size

    @property
    def right_edge(self) -> int:
        return self.left_edge + self.size

    def __repr__(self) -> str:
        return (f"Node(upper={self.upper_edge}, lower={self.lower_edge}, "
                f"left={self.left_edge}, right={self.right_edge}, "
                f"size={self.size}, depth={self.depth})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return (self.upper_edge == other.upper_edge and
                self.left_edge == other.left_edge and
                self.size == other.size and
                self.depth == other.depth)

    def __hash__(self) -> int:
        return hash((self.upper_edge, self.left_edge, self.size, self.depth))