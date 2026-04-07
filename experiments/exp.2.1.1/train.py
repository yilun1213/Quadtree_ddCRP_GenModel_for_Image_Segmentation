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
TRUE_BRANCH_PROBS = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0]
MAX_DEPTH = len(TRUE_BRANCH_PROBS) - 1  # 7

# ===== パス設定 =====
TRAIN_LABEL_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'train_data', 'labels')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


def estimate_branch_probs_incremental(label_files: list[str]) -> list[list[float]]:
    """label_files を 1 枚ずつ追加しながら branch_probs を推定する。

    Returns:
        results[i]: i+1 枚目を加えた後の推定 branch_probs (長さ MAX_DEPTH+1)
    """
    root = Node(upper_edge=0, left_edge=0, size=2 ** MAX_DEPTH, depth=0)
    make_tree(root, MAX_DEPTH)

    results = []
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
        print(
            f"  n={i + 1:3d} | "
            + "  ".join(f"d{d}={branch_probs[d]:.4f}" for d in range(MAX_DEPTH))
        )

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    results = estimate_branch_probs_incremental(label_files)

    n_values = list(range(1, n_total + 1))
    estimated = np.array(results)           # shape: (n_total, MAX_DEPTH+1)
    true_probs = np.array(TRUE_BRANCH_PROBS)
    abs_errors = np.abs(estimated - true_probs)  # shape: (n_total, MAX_DEPTH+1)

    # --- 最終推定値の表示 ---
    print("\n=== 最終推定結果（全 {} 枚使用時） ===".format(n_total))
    final = results[-1]
    header = f"{'深さ':>6}" + "".join(f"  depth{d:1d}" for d in range(MAX_DEPTH + 1))
    print(header)
    print(f"{'真の値':>6}" + "".join(f"  {TRUE_BRANCH_PROBS[d]:6.4f}" for d in range(MAX_DEPTH + 1)))
    print(f"{'推定値':>6}" + "".join(f"  {final[d]:6.4f}" for d in range(MAX_DEPTH + 1)))
    errors = [abs(final[d] - TRUE_BRANCH_PROBS[d]) for d in range(MAX_DEPTH + 1)]
    print(f"{'誤差':>6}" + "".join(f"  {errors[d]:6.4f}" for d in range(MAX_DEPTH + 1)))

    # --- 折れ線グラフ ---
    # depth 7 は常に 0 なので除外（誤差も常に 0）
    fig, ax = plt.subplots(figsize=(10, 6))
    for d in range(MAX_DEPTH):
        ax.plot(
            n_values,
            abs_errors[:, d],
            label=f"depth {d}  (真値 g={TRUE_BRANCH_PROBS[d]:.1f})",
        )

    ax.set_xlabel("学習枚数 n")
    ax.set_ylabel("推定誤差  |ĝ_d - g_d|")
    ax.set_title("exp.2.1.1: 四分木分岐確率の推定誤差の推移\n(branch_probs=[0.9]*7 + [0])")
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
