# experiments/exp.1.2.1/generate.py
"""
exp.1.2.1 実験スクリプト

exp.1.1.1 で設定したパラメータに従って四分木を 1 つ生成し，
ddCRP モデルによる統合領域を 27 通りのパラメータ（各 10 サンプル）で生成する．
各サンプルの統合領域画像を保存し，幾何学的特徴量（面積・周の長さ・円形度）の
分布を α ごとにグラフ化して出力する．

パラメータ範囲:
    α ∈ {1e-8, 1e-4, 1.0}
    β ∈ {0.0, 8.0, 30.0}
    η ∈ {0.0, 8.0, 30.0}
    → 計 27 通り × 各 10 サンプル

出力先: experiments/exp.1.2.1/outputs/
    quadtree.png
    alpha{α}_beta{β}_eta{η}/
        region_000.png ... region_009.png
    feature_dist_area_alpha{α}.png          (α ごと，β×η の 3×3 グリッド)
    feature_dist_perimeter_alpha{α}.png
    feature_dist_circularity_alpha{α}.png
"""

from __future__ import annotations

import os
import sys
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── パス設定 ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, ROOT_DIR)

from model.quadtree.depth_dependent_model import QuadTree
from model.quadtree.node import Node
from model.region.affinity import log_affinity_boundary_and_depth
from model.label.geom_features import compute_geom_features

# ── 実験定数 ──────────────────────────────────────────────────
SEED        = 40
MAX_DEPTH   = 7
BRANCH_PROBS = [0.99, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0]   # exp.1.1.3
IMAGE_SIZE  = 2 ** MAX_DEPTH   # 128

N_REGIONS_PER_PARAM = 10

ALPHA_VALUES: list[float] = [1e-8, 1e-4, 1.0]
BETA_VALUES:  list[float] = [0.0, 8.0, 30.0]
ETA_VALUES:   list[float] = [0.0, 8.0, 30.0]

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")


# ══════════════════════════════════════════════════════════════
# 四分木ユーティリティ
# ══════════════════════════════════════════════════════════════

def precompute_adjacencies(all_leaves: list[Node]) -> dict[Node, list[Node]]:
    """全葉ノード間の隣接関係を事前計算する．"""
    adj: dict[Node, list[Node]] = {leaf: [] for leaf in all_leaves}
    for i, a in enumerate(all_leaves):
        for b in all_leaves[i + 1:]:
            is_h = (a.right_edge == b.left_edge or b.right_edge == a.left_edge)
            y_ov = (a.upper_edge < b.lower_edge and a.lower_edge > b.upper_edge)
            is_v = (a.lower_edge == b.upper_edge or b.lower_edge == a.upper_edge)
            x_ov = (a.left_edge  < b.right_edge and a.right_edge > b.left_edge)
            if (is_h and y_ov) or (is_v and x_ov):
                adj[a].append(b)
                adj[b].append(a)
    return adj


# ══════════════════════════════════════════════════════════════
# ddCRP 領域生成
# ══════════════════════════════════════════════════════════════

def _logsumexp(log_vals: list[float]) -> float:
    """数値的に安定な log-sum-exp．"""
    if not log_vals:
        return -np.inf
    m = max(log_vals)
    return m + np.log(sum(np.exp(v - m) for v in log_vals))


def ddcrp_region_generation(
    all_leaves: list[Node],
    adj: dict[Node, list[Node]],
    alpha: float,
    beta: float,
    eta: float,
) -> dict[int, set[tuple[int, int]]]:
    """
    ddCRP に従って統合領域を生成する．

    親和度: f(s, s') = exp( β·B(s, s') + η·(depth(s) − depth(s')) )
             隣接していない場合は 0．

    log-sum-exp で数値安定性を確保する（β=30 等でも overflow しない）．
    """
    log_alpha = np.log(alpha) if alpha > 0.0 else -np.inf
    choice_dict: dict[Node, Node] = {}

    for leaf_s in all_leaves:
        neighbors = adj.get(leaf_s, [])

        if not neighbors:
            choice_dict[leaf_s] = leaf_s
            continue

        # 隣接ノードごとに log f を計算
        log_f_map: dict[Node, float] = {}
        for nb in neighbors:
            lf = log_affinity_boundary_and_depth(leaf_s, nb, adj, beta=beta, eta=eta)
            if np.isfinite(lf):
                log_f_map[nb] = lf

        if not log_f_map:
            # 全ての log f が -inf → 自己参照
            choice_dict[leaf_s] = leaf_s
            continue

        log_sum_f = _logsumexp(list(log_f_map.values()))

        # log( α + Σf ) = logsumexp( log_α, log_sum_f )
        log_denom = _logsumexp([log_alpha, log_sum_f])

        # p(c_s = s) = α / denom
        prob_self = np.exp(log_alpha - log_denom)

        r = random.random()
        if r < prob_self:
            choice_dict[leaf_s] = leaf_s
        else:
            cumsum = prob_self
            chosen = leaf_s   # fallback（浮動小数点誤差対策）
            for nb, lf in log_f_map.items():
                cumsum += np.exp(lf - log_denom)
                if r < cumsum:
                    chosen = nb
                    break
            choice_dict[leaf_s] = chosen

    # 有向グラフの弱連結成分 → 領域
    adj_ud: dict[Node, set[Node]] = {leaf: set() for leaf in all_leaves}
    for s, t in choice_dict.items():
        if s is not t:
            adj_ud[s].add(t)
            adj_ud[t].add(s)

    visited: set[Node] = set()
    region_id = 1
    region_dict: dict[int, set[tuple[int, int]]] = {}

    for start in all_leaves:
        if start in visited:
            continue
        component: set[Node] = set()
        stack = [start]
        while stack:
            nd = stack.pop()
            if nd in visited:
                continue
            visited.add(nd)
            component.add(nd)
            for nb in adj_ud[nd]:
                if nb not in visited:
                    stack.append(nb)

        pixels: set[tuple[int, int]] = set()
        for leaf in component:
            i0, j0, sz = leaf.upper_edge, leaf.left_edge, leaf.size
            for ii in range(i0, i0 + sz):
                for jj in range(j0, j0 + sz):
                    pixels.add((ii, jj))

        region_dict[region_id] = pixels
        region_id += 1

    return region_dict


# ══════════════════════════════════════════════════════════════
# 画像保存
# ══════════════════════════════════════════════════════════════

# 色生成用に独立した RNG（メイン乱数状態に影響を与えない）
_color_rng = np.random.default_rng(0)


def save_region_image(
    region_dict: dict[int, set[tuple[int, int]]],
    image_size: int,
    filename: str,
) -> None:
    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for pixels in region_dict.values():
        color = _color_rng.integers(50, 255, size=3, dtype=np.uint8)
        for i, j in pixels:
            img[i, j] = color
    Image.fromarray(img).save(filename)


def save_quadtree_image(
    all_leaves: list[Node],
    image_size: int,
    filename: str,
) -> None:
    """葉ノード単位で色分けした四分木画像を保存する．"""
    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for leaf in all_leaves:
        color = _color_rng.integers(50, 255, size=3, dtype=np.uint8)
        i0, j0, sz = leaf.upper_edge, leaf.left_edge, leaf.size
        img[i0:i0 + sz, j0:j0 + sz] = color
    Image.fromarray(img).save(filename)


# ══════════════════════════════════════════════════════════════
# 幾何学的特徴量の計算
# ══════════════════════════════════════════════════════════════

def compute_all_region_geometrics(
    region_dict: dict[int, set[tuple[int, int]]],
    image_size: int,
) -> list[dict]:
    """
    全領域の幾何学特徴量を算出する．

    model/label/geom_features_norm_dist.py と同じ定義に合わせるため，
    compute_geom_features(..., feature_names=["log_area", "log_perimeter", "circularity"])
    を利用して，対数面積・対数周長・円形度を返す．
    """
    features: list[dict] = []
    for rid in region_dict.keys():
        phi = compute_geom_features(
            region=region_dict[rid],
            image_size=image_size,
            feature_names=["log_area", "log_perimeter", "circularity"],
        )
        features.append(
            {
                "log_area": float(phi[0]),
                "log_perimeter": float(phi[1]),
                "circularity": float(phi[2]),
            }
        )
    return features


# ══════════════════════════════════════════════════════════════
# プロット用 KDE（scipy 不要）
# ══════════════════════════════════════════════════════════════

def _gaussian_kde(data: np.ndarray, x: np.ndarray, bw: float | None = None) -> np.ndarray:
    """numpy のみで実装した Gaussian KDE．"""
    n = len(data)
    if bw is None:
        std = float(np.std(data, ddof=1))
        iqr = float(np.percentile(data, 75) - np.percentile(data, 25))
        scale = min(std, iqr / 1.34) if iqr > 0 else std
        bw = 0.9 * scale * n ** (-0.2)
        if bw <= 0:
            bw = 1e-6
    diff = (x[:, np.newaxis] - data[np.newaxis, :]) / bw   # (X, N)
    y = np.exp(-0.5 * diff ** 2).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))
    return y


# ══════════════════════════════════════════════════════════════
# 分布グラフの出力
# ══════════════════════════════════════════════════════════════

def plot_distributions_for_alpha(
    alpha: float,
    features_by_param: dict[tuple[float, float], list[dict]],
    output_dir: str,
) -> None:
    """
    α を固定して，幾何学的特徴量の分布グラフを 3 枚（特徴量ごと）出力する．
    各グラフは β(行) × η(列) の 3×3 グリッド．
    """
    alpha_label = f"$\\alpha = {alpha:.0e}$".replace("e-0", "e-").replace("e+0", "e+")
    alpha_key   = f"{alpha:.2e}"

    cfg = [
        ("log_area", "log(Area)", False, "area"),
        ("log_perimeter", "log(Perimeter)", False, "perimeter"),
        ("circularity", "Circularity", False, "circularity"),
    ]

    for feat_key, feat_label, use_log, out_key in cfg:
        fig, axes = plt.subplots(3, 3, figsize=(14, 11))
        fig.suptitle(
            f"Distribution of {feat_label} ({alpha_label})",
            fontsize=13,
            y=0.99,
        )

        for row, beta in enumerate(BETA_VALUES):
            for col, eta in enumerate(ETA_VALUES):
                ax = axes[row][col]
                feat_list = features_by_param.get((beta, eta), [])
                raw = [f[feat_key] for f in feat_list if f[feat_key] > 0]

                if len(raw) >= 5:
                    vals = np.array(raw, dtype=float)
                    try:
                        if use_log:
                            lo, hi = vals.min(), vals.max()
                            # 対数軸では対数間隔のビンでヒストグラムを描画する
                            if lo == hi:
                                bins = np.logspace(np.log10(lo * 0.9), np.log10(hi * 1.1), 11)
                            else:
                                bins = np.logspace(np.log10(lo), np.log10(hi), 21)
                            ax.hist(vals, bins=bins, color="steelblue", alpha=0.55,
                                    edgecolor="white", linewidth=0.6)
                            ax.set_xscale("log")
                        else:
                            ax.hist(vals, bins=50, color="steelblue", alpha=0.55,
                                edgecolor="white", linewidth=0.6)
                            ax.set_xlim(left=0.0)

                        n_regions = len(raw)
                        median_val = float(np.median(raw))
                        ax.axvline(median_val, color="tomato", linewidth=1.2,
                                   linestyle="--", label=f"median={median_val:.2g}")
                        ax.legend(fontsize=6, loc="upper right")
                        ax.set_title(
                            f"$\\beta={int(beta)}$,  $\\eta={int(eta)}$\n"
                            f"(N={n_regions} regions)",
                            fontsize=9,
                        )
                    except Exception as exc:
                        ax.text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center",
                                transform=ax.transAxes, fontsize=7, color="red")
                        ax.set_title(
                            f"$\\beta={int(beta)}$,  $\\eta={int(eta)}$", fontsize=9
                        )
                else:
                    ax.text(0.5, 0.5, f"Insufficient data\nN={len(raw)}",
                            ha="center", va="center",
                            transform=ax.transAxes, fontsize=9)
                    ax.set_title(
                        f"$\\beta={int(beta)}$,  $\\eta={int(eta)}$", fontsize=9
                    )

                ax.set_xlabel(feat_label, fontsize=7)
                ax.set_ylabel("Count", fontsize=7)
                ax.tick_params(labelsize=6)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = os.path.join(output_dir, f"feature_dist_{out_key}_alpha{alpha_key}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════
# フォルダ名
# ══════════════════════════════════════════════════════════════

def _param_folder(alpha: float, beta: float, eta: float) -> str:
    return f"alpha{alpha:.2e}_beta{beta:.1f}_eta{eta:.1f}"


# ══════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    # ── 四分木の生成（exp.1.1.1 パラメータ）─────────────────
    print(f"四分木を生成中（exp.1.1.1 パラメータ, seed={SEED}）...")
    qt          = QuadTree(max_depth=MAX_DEPTH, branch_prob=BRANCH_PROBS, seed=SEED)
    all_leaves  = qt.get_leaves()
    print(f"  → 葉ノード数: {len(all_leaves)}")

    print("隣接関係を計算中...")
    adj = precompute_adjacencies(all_leaves)
    print("  → 完了")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_quadtree_image(all_leaves, IMAGE_SIZE, os.path.join(OUTPUT_DIR, "quadtree.png"))

    # α ごとに特徴量を集約
    all_features: dict[float, dict[tuple[float, float], list[dict]]] = {
        a: {} for a in ALPHA_VALUES
    }

    total = len(ALPHA_VALUES) * len(BETA_VALUES) * len(ETA_VALUES)
    count = 0

    # ── 27 通りのパラメータで統合領域を生成 ─────────────────
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            for eta in ETA_VALUES:
                count += 1
                folder   = _param_folder(alpha, beta, eta)
                param_dir = os.path.join(OUTPUT_DIR, folder)
                os.makedirs(param_dir, exist_ok=True)

                print(f"\n[{count:2d}/{total}] α={alpha:.0e}  β={beta:.1f}  η={eta:.1f}")

                param_features: list[dict] = []
                for s_idx in range(N_REGIONS_PER_PARAM):
                    region_dict = ddcrp_region_generation(
                        all_leaves, adj, alpha=alpha, beta=beta, eta=eta
                    )
                    save_region_image(
                        region_dict, IMAGE_SIZE,
                        os.path.join(param_dir, f"region_{s_idx:03d}.png"),
                    )
                    feats = compute_all_region_geometrics(region_dict, IMAGE_SIZE)
                    param_features.extend(feats)
                    print(f"  サンプル {s_idx + 1}/{N_REGIONS_PER_PARAM}: "
                          f"{len(region_dict)} 領域")

                all_features[alpha][(beta, eta)] = param_features

    # ── α ごとに分布グラフを出力 ─────────────────────────────
    print("\n分布グラフを出力中...")
    for alpha in ALPHA_VALUES:
        print(f"  α = {alpha:.0e}")
        plot_distributions_for_alpha(alpha, all_features[alpha], OUTPUT_DIR)

    print("\n完了．")


if __name__ == "__main__":
    main()
