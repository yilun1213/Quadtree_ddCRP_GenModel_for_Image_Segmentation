# train.py
# [exp.2.1.1] 四分木のパラメータを，学習枚数を 1 枚ずつ増やしながら推定し，
#              深さごとの推定誤差の推移を折れ線グラフで出力する

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import utils
from model.quadtree.node import Node
from model.quadtree.depth_dependent_model import (
    make_tree,
    recursive_split_for_tree,
    get_split_probs_at_depth,
)

# ===== 真のパラメータ =====
TRUE_BRANCH_PROBS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0]
MAX_DEPTH = len(TRUE_BRANCH_PROBS) - 1  # 7

# ===== パス設定 =====
TRAIN_LABEL_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'train_data', 'labels')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
ESTIMATED_QT_DIR = os.path.join(OUTPUT_DIR, 'estimation_quadtree_images')


def _is_matrix_all_same(matrix: np.ndarray) -> bool:
    return (matrix == matrix[0, 0]).all()


def _build_quadtree_from_label(node: Node, label_array: np.ndarray, max_depth: int) -> None:
    if node.depth >= max_depth:
        return

    region = label_array[
        node.upper_edge: node.upper_edge + node.size,
        node.left_edge: node.left_edge + node.size,
    ]
    if _is_matrix_all_same(region):
        return

    node.is_leaf = False
    half = node.size // 2
    node.ul_node = Node(node.upper_edge, node.left_edge, half, node.depth + 1)
    node.ur_node = Node(node.upper_edge, node.left_edge + half, half, node.depth + 1)
    node.ll_node = Node(node.upper_edge + half, node.left_edge, half, node.depth + 1)
    node.lr_node = Node(node.upper_edge + half, node.left_edge + half, half, node.depth + 1)

    for child in (node.ul_node, node.ur_node, node.ll_node, node.lr_node):
        _build_quadtree_from_label(child, label_array, max_depth)


def _collect_leaves(node: Node, leaves: list[Node]) -> None:
    if node.is_leaf:
        leaves.append(node)
        return
    for child in (node.ul_node, node.ur_node, node.ll_node, node.lr_node):
        if child is not None:
            _collect_leaves(child, leaves)


def save_quadtree_image_from_label(label_array: np.ndarray, out_path: str) -> None:
    root = Node(upper_edge=0, left_edge=0, size=2 ** MAX_DEPTH, depth=0)
    _build_quadtree_from_label(root, label_array, MAX_DEPTH)

    leaves: list[Node] = []
    _collect_leaves(root, leaves)

    image = np.zeros((2 ** MAX_DEPTH, 2 ** MAX_DEPTH, 3), dtype=np.uint8)
    for leaf in leaves:
        seed = (
            (leaf.upper_edge * 73856093)
            ^ (leaf.left_edge * 19349663)
            ^ (leaf.size * 83492791)
        ) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        color = rng.integers(50, 255, size=3, dtype=np.uint8)
        image[
            leaf.upper_edge: leaf.upper_edge + leaf.size,
            leaf.left_edge: leaf.left_edge + leaf.size,
        ] = color

    plt.imsave(out_path, image)


def _clear_estimated_quadtree_images(output_dir: str) -> None:
    for file_name in os.listdir(output_dir):
        if file_name.lower().endswith('.png'):
            os.remove(os.path.join(output_dir, file_name))


def estimate_branch_probs_incremental(label_files: list[str]) -> tuple[list[list[float]], list[str]]:
    """label_files を 1 枚ずつ追加しながら branch_probs を推定する。

    Returns:
        results[i]: i+1 枚目を加えた後の推定 branch_probs (長さ MAX_DEPTH+1)
        log_lines: 逐次推定のログ行
    """
    root = Node(upper_edge=0, left_edge=0, size=2 ** MAX_DEPTH, depth=0)
    make_tree(root, MAX_DEPTH)

    results = []
    log_lines = []
    for i, label_file in enumerate(label_files):
        label_array = utils.load_image(label_file)
        if label_array.ndim == 3:
            label_array = label_array[..., 0]
        recursive_split_for_tree(root, label_array)

        branch_probs = []
        for d in range(MAX_DEPTH):
            list_g = []
            get_split_probs_at_depth(root, d, list_g)
            branch_probs.append(sum(list_g) / len(list_g) if list_g else 0.0)
        branch_probs.append(0.0)  # 最大深度は分割しない

        results.append(branch_probs)
        line = (
            f"  n={i + 1:3d} | "
            + "  ".join(f"d{d}={branch_probs[d]:.4f}" for d in range(MAX_DEPTH))
        )
        print(line)
        log_lines.append(line)

    return results, log_lines


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ESTIMATED_QT_DIR, exist_ok=True)

    label_filenames = utils.get_image_files(TRAIN_LABEL_DIR)
    if not label_filenames:
        print(f"ラベル画像が見つかりません: {TRAIN_LABEL_DIR}")
        print("先に generate.py を実行してください。")
        return

    label_files = [os.path.join(TRAIN_LABEL_DIR, f) for f in sorted(label_filenames)]
    n_total = len(label_files)

    print(f"ラベル画像数: {n_total}")
    print(f"真の branch_probs: {TRUE_BRANCH_PROBS}\n")
    print("=== 四分木パラメータの逐次推定 ===")

    results, sequential_log_lines = estimate_branch_probs_incremental(label_files)

    # 推定時に扱う全ラベル画像から四分木を構築し、train_data と同じ名前で保存
    _clear_estimated_quadtree_images(ESTIMATED_QT_DIR)
    saved_tree_paths = []
    for label_path in label_files:
        label_array = utils.load_image(label_path)
        if label_array.ndim == 3:
            label_array = label_array[..., 0]
        tree_path = os.path.join(ESTIMATED_QT_DIR, os.path.basename(label_path))
        save_quadtree_image_from_label(label_array, tree_path)
        saved_tree_paths.append(tree_path)
        print(f"推定用四分木画像を保存しました: {tree_path}")

    n_values = list(range(1, n_total + 1))
    estimated = np.array(results)           # shape: (n_total, MAX_DEPTH+1)
    true_probs = np.array(TRUE_BRANCH_PROBS)
    abs_errors = np.abs(estimated - true_probs)  # shape: (n_total, MAX_DEPTH+1)

    # --- 最終推定値の表示 ---
    print("\n=== 最終推定結果（全 {} 枚使用時） ===".format(n_total))
    final = results[-1]
    header = f"{'深さ':>6}" + "".join(f"  depth{d:1d}" for d in range(MAX_DEPTH + 1))
    print(header)
    true_line = f"{'真の値':>6}" + "".join(f"  {TRUE_BRANCH_PROBS[d]:6.4f}" for d in range(MAX_DEPTH + 1))
    est_line = f"{'推定値':>6}" + "".join(f"  {final[d]:6.4f}" for d in range(MAX_DEPTH + 1))
    print(true_line)
    print(est_line)
    errors = [abs(final[d] - TRUE_BRANCH_PROBS[d]) for d in range(MAX_DEPTH + 1)]
    err_line = f"{'誤差':>6}" + "".join(f"  {errors[d]:6.4f}" for d in range(MAX_DEPTH + 1))
    print(err_line)

    # 逐次推定ログと最終推定結果をテキスト保存
    report_lines = [
        f"ラベル画像数: {n_total}",
        f"真の branch_probs: {TRUE_BRANCH_PROBS}",
        "",
        "=== 四分木パラメータの逐次推定 ===",
    ]
    report_lines.extend(sequential_log_lines)
    report_lines.extend([
        "",
        "=== 最終推定結果（全 {} 枚使用時） ===".format(n_total),
        header,
        true_line,
        est_line,
        err_line,
        "",
        "=== 推定時に構築した四分木画像（全ラベル画像） ===",
    ])
    for p in saved_tree_paths:
        report_lines.append(p)
    report_path = os.path.join(OUTPUT_DIR, "branch_probs_estimation_log.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"逐次推定ログを保存しました: {report_path}")

    # --- 折れ線グラフ ---
    # depth 7 は常に 0 なので除外（誤差も常に 0）
    fig, ax = plt.subplots(figsize=(10, 6))
    for d in range(MAX_DEPTH):
        ax.plot(
            n_values,
            abs_errors[:, d],
            label=f"depth {d}  (true g={TRUE_BRANCH_PROBS[d]:.1f})",
        )

    ax.set_xlabel("Number of training images n")
    ax.set_ylabel("Absolute estimation error")
    ax.set_title("Transition of estimation errors for quadtree branch probabilities")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(1, n_total)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "branch_probs_error.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n折れ線グラフを保存しました: {out_path}")


if __name__ == "__main__":
    main()
