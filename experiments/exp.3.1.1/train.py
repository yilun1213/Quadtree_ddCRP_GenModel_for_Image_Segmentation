# train.py
import os
import sys
import csv
import json
import shutil
import tempfile
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import utils  # type: ignore[reportMissingImports]
from config import Config, config


# ===== 真のパラメータ (generate.py の設定と合わせる) =====
_TRUE_BRANCH_PROBS = [0.99, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0]
_TRUE_LABEL_FEATURE_NAMES = ["log_area", "circularity"]
_TRUE_LABEL_MEANS = [
    [8.0, 0.30],
    [5.5, 0.65],
    [5.5, 0.35],
]
_TRUE_LABEL_STDS = [
    [1.0, 0.05],
    [0.5, 0.05],
    [0.5, 0.05],
]
_TRUE_PIXEL_MEANS = [[100.0, 100.0, 100.0], [200.0, 50.0, 50.0], [220.0, 30.0, 30.0]]
_TRUE_PIXEL_VAR_DIAG = [[2500.0, 2500.0, 2500.0], [400.0, 400.0, 400.0], [400.0, 400.0, 400.0]]
_CHANNEL_NAMES = ["R", "G", "B"]


def _true_label_params_for_features(feature_names: list[str]):
    idx_map = {f: i for i, f in enumerate(_TRUE_LABEL_FEATURE_NAMES)}
    indices = [idx_map.get(f) for f in feature_names]
    means = [[row[i] if i is not None else None for i in indices] for row in _TRUE_LABEL_MEANS]
    stds = [[row[i] if i is not None else None for i in indices] for row in _TRUE_LABEL_STDS]
    return means, stds


def train(config: Config) -> None:
    train_image_dir = config.train_image_dir
    train_label_dir = config.train_label_dir
    os.makedirs(config.out_param_dir, exist_ok=True)
    out_label_param_path = os.path.join(config.out_param_dir, config.label_param_filename)
    out_pixel_param_path = os.path.join(config.out_param_dir, config.pixel_param_filename)
    out_branch_probs_path = os.path.join(config.out_param_dir, config.branch_probs_filename)

    print("Calculating and saving label information")
    label_files = utils.get_image_files(train_label_dir)
    if not label_files:
        raise RuntimeError(f"No label files found in train_label_dir: {train_label_dir}")

    vis_label_dir = config.train_label_vis_dir
    vis_label_files = utils.get_image_files(vis_label_dir) if os.path.exists(vis_label_dir) else []

    if not vis_label_files:
        label_to_color_map = utils.build_label_value_map([train_label_dir, config.test_label_dir])
        utils.generate_visualize_labels(train_label_dir, vis_label_dir, label_to_color_map)
        utils.generate_visualize_labels(config.test_label_dir, config.test_label_vis_dir, label_to_color_map)
        vis_label_files = utils.get_image_files(vis_label_dir)

    filenames = utils.harmonize_lists(label_files, vis_label_files) if vis_label_files else label_files
    label_to_color_map = {}

    if not vis_label_files:
        print("Note: Visualize label images not found. Using raw label values as color mapping.")
    elif not filenames:
        print("Note: Visualize label images have no overlap with labels. Using raw label values as color mapping.")
        filenames = label_files

    for filename in filenames:
        label_path = os.path.join(train_label_dir, filename)
        label_array = utils.load_image(label_path)

        if os.path.exists(os.path.join(vis_label_dir, filename)):
            vis_label_path = os.path.join(vis_label_dir, filename)
            vis_img = Image.fromarray(utils.load_image(vis_label_path))
            vis_label_array = np.array(vis_img.convert("L"))
        else:
            vis_label_array = label_array

        unique_pairs = np.unique(np.stack([label_array, vis_label_array]).reshape(2, -1), axis=1)
        for label_id, color_val in unique_pairs.T:
            if label_id not in label_to_color_map:
                label_to_color_map[int(label_id)] = int(color_val)

    label_set = sorted(label_to_color_map.keys())
    label_value_set = [label_to_color_map[label_id] for label_id in label_set]
    label_num = len(label_set)

    label_info = {
        "label_num": label_num,
        "label_set": label_set,
        "label_value_set": label_value_set,
    }
    with open(out_label_param_path, "w", encoding="utf-8") as f:
        json.dump(label_info, f, indent=4)
    print(f"Saved label info to {out_label_param_path}")

    print("Estimating Parameters of Branching Probability")
    config.quadtree_model.param_est(train_label_dir, out_branch_probs_path)

    print("Estimating Parameters of Probability Function on Label")
    first_label_img = utils.load_image(os.path.join(train_label_dir, label_files[0]))
    image_size = first_label_img.shape[0]
    print(f"Detected image size: {image_size}x{image_size}")

    estimated_label_params = config.label_model.param_est(
        train_label_dir=train_label_dir,
        label_set=label_set,
        label_num=label_num,
        image_size=image_size,
        feature_names=config.label_feature_names,
        min_region_area=config.label_min_region_area,
    )
    label_info.update(estimated_label_params)
    with open(out_label_param_path, "w", encoding="utf-8") as f:
        json.dump(label_info, f, indent=4)

    print("Estimating Parameters of Probability Function on Pixel")
    config.pixel_model.param_est(train_image_dir, train_label_dir, out_pixel_param_path, config.offset)
    config.pixel_model.add_label_set(out_pixel_param_path, out_label_param_path)


def _estimate_params_for_n(
    config: Config,
    sorted_label_files: list[str],
    sorted_image_files: list[str],
    n: int,
    label_set: list[int],
    image_size: int,
    feature_names: list[str],
) -> dict | None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_lbl_dir = os.path.join(tmp_dir, "labels")
        tmp_img_dir = os.path.join(tmp_dir, "images")
        tmp_param_dir = os.path.join(tmp_dir, "params")
        os.makedirs(tmp_lbl_dir)
        os.makedirs(tmp_img_dir)
        os.makedirs(tmp_param_dir)

        for f in sorted_label_files[:n]:
            shutil.copy(os.path.join(config.train_label_dir, f), tmp_lbl_dir)
        for f in sorted_image_files[:n]:
            shutil.copy(os.path.join(config.train_image_dir, f), tmp_img_dir)

        tmp_branch_path = os.path.join(tmp_param_dir, "branch_probs.json")
        tmp_pixel_path = os.path.join(tmp_param_dir, "pixel_param.json")

        config.quadtree_model.param_est(tmp_lbl_dir, tmp_branch_path)
        est_label = config.label_model.param_est(
            train_label_dir=tmp_lbl_dir,
            label_set=label_set,
            label_num=len(label_set),
            image_size=image_size,
            feature_names=feature_names,
            min_region_area=config.label_min_region_area,
        )
        config.pixel_model.param_est(tmp_img_dir, tmp_lbl_dir, tmp_pixel_path, config.offset)

        if not os.path.exists(tmp_branch_path) or not os.path.exists(tmp_pixel_path):
            return None

        with open(tmp_branch_path, encoding="utf-8") as f:
            branch_data = json.load(f)
        with open(tmp_pixel_path, encoding="utf-8") as f:
            pixel_data = json.load(f)

    return {
        "branch_probs": branch_data.get("branch_probs", []),
        "label_means": est_label.get("means", []),
        "label_stds": est_label.get("stds", []),
        "pixel_means": pixel_data.get("mean", []),
        "pixel_var_diag": [
            [pixel_data["variance"][li][ch][ch] for ch in range(3)]
            for li in range(len(pixel_data.get("variance", [])))
        ],
    }


def _run_convergence_analysis(config: Config, label_set: list[int], image_size: int, feature_names: list[str]) -> dict:
    label_files = sorted(utils.get_image_files(config.train_label_dir))
    image_files = sorted(utils.get_image_files(config.train_image_dir))
    n_total = min(len(label_files), len(image_files))

    candidates = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
    sample_counts = sorted({c for c in candidates if c <= n_total} | {n_total})

    history = []
    for n in sample_counts:
        print(f"  [N={n:3d}] estimating ...", end="", flush=True)
        result = _estimate_params_for_n(
            config=config,
            sorted_label_files=label_files,
            sorted_image_files=image_files,
            n=n,
            label_set=label_set,
            image_size=image_size,
            feature_names=feature_names,
        )
        history.append(result)
        print(" done")

    return {"sample_counts": sample_counts, "history": history}


def _safe_get(history_list: list[dict | None], key: str, li=None, fi=None, ch=None) -> list[float]:
    vals = []
    for h in history_list:
        try:
            if h is None:
                raise ValueError("empty")
            v = h[key]
            if li is not None:
                v = v[li]
            if fi is not None:
                v = v[fi]
            if ch is not None:
                v = v[ch]
            vals.append(float(v))
        except Exception:
            vals.append(float("nan"))
    return vals


def _mae(est: list[float], truth: list[float]) -> float:
    arr_e = np.array(est, dtype=float)
    arr_t = np.array(truth, dtype=float)
    if arr_e.shape != arr_t.shape:
        return float("nan")
    return float(np.mean(np.abs(arr_e - arr_t)))


def _err_series(values: list, true_val: float) -> list:
    """Subtract true_val from each element; propagate NaN."""
    return [v - true_val if not (isinstance(v, float) and np.isnan(v)) else np.nan for v in values]


def _plot_convergence_graphs(analysis: dict, feature_names: list[str], label_set: list[int], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sc = analysis["sample_counts"]
    hist = analysis["history"]
    true_lbl_means, true_lbl_stds = _true_label_params_for_features(feature_names)
    n_labels = len(label_set)
    n_feats = len(feature_names)
    colors10 = plt.cm.tab10.colors  # type: ignore[attr-defined]

    # --- Branch probabilities error ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for d, tv in enumerate(_TRUE_BRANCH_PROBS):
        c = colors10[d % 10]
        errs = _err_series(_safe_get(hist, "branch_probs", li=d), tv)
        ax.plot(sc, errs, marker="o", color=c, label=f"Depth {d}")
    ax.axhline(0, linestyle="--", color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Error (estimated - true)")
    ax.set_title("Branch Probabilities Estimation Error")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "convergence_branch_probs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Label feature means error ---
    fig, axes = plt.subplots(n_labels, n_feats, figsize=(4 * n_feats, 3 * n_labels), squeeze=False)
    for li, lbl in enumerate(label_set):
        for fi, feat in enumerate(feature_names):
            ax = axes[li][fi]
            tv = true_lbl_means[li][fi]
            if tv is not None:
                errs = _err_series(_safe_get(hist, "label_means", li=li, fi=fi), tv)
                ax.plot(sc, errs, marker="o", color="steelblue")
                ax.axhline(0, linestyle="--", color="black", linewidth=1, alpha=0.6)
            ax.set_title(f"Label {lbl} | {feat} mean error")
            ax.set_xlabel("N samples")
            ax.set_ylabel("Error (est - true)")
            ax.grid(True, alpha=0.4)
    fig.suptitle("Label Feature Means Estimation Error", fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, "convergence_label_means.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Label feature stds error ---
    fig, axes = plt.subplots(n_labels, n_feats, figsize=(4 * n_feats, 3 * n_labels), squeeze=False)
    for li, lbl in enumerate(label_set):
        for fi, feat in enumerate(feature_names):
            ax = axes[li][fi]
            tv = true_lbl_stds[li][fi]
            if tv is not None:
                errs = _err_series(_safe_get(hist, "label_stds", li=li, fi=fi), tv)
                ax.plot(sc, errs, marker="o", color="darkorange")
                ax.axhline(0, linestyle="--", color="black", linewidth=1, alpha=0.6)
            ax.set_title(f"Label {lbl} | {feat} std error")
            ax.set_xlabel("N samples")
            ax.set_ylabel("Error (est - true)")
            ax.grid(True, alpha=0.4)
    fig.suptitle("Label Feature Stds Estimation Error", fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, "convergence_label_stds.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Pixel means error ---
    fig, axes = plt.subplots(n_labels, 3, figsize=(12, 3 * n_labels), squeeze=False)
    for li, lbl in enumerate(label_set):
        for ch, ch_name in enumerate(_CHANNEL_NAMES):
            ax = axes[li][ch]
            if li < len(_TRUE_PIXEL_MEANS):
                tv = _TRUE_PIXEL_MEANS[li][ch]
                errs = _err_series(_safe_get(hist, "pixel_means", li=li, fi=ch), tv)
                ax.plot(sc, errs, marker="o", color="forestgreen")
                ax.axhline(0, linestyle="--", color="black", linewidth=1, alpha=0.6)
            ax.set_title(f"Label {lbl} | {ch_name} mean error")
            ax.set_xlabel("N samples")
            ax.set_ylabel("Error (est - true)")
            ax.grid(True, alpha=0.4)
    fig.suptitle("Pixel Channel Means Estimation Error", fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, "convergence_pixel_means.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Pixel variance error ---
    fig, axes = plt.subplots(n_labels, 3, figsize=(12, 3 * n_labels), squeeze=False)
    for li, lbl in enumerate(label_set):
        for ch, ch_name in enumerate(_CHANNEL_NAMES):
            ax = axes[li][ch]
            if li < len(_TRUE_PIXEL_VAR_DIAG):
                tv = _TRUE_PIXEL_VAR_DIAG[li][ch]
                errs = _err_series(_safe_get(hist, "pixel_var_diag", li=li, fi=ch), tv)
                ax.plot(sc, errs, marker="o", color="mediumpurple")
                ax.axhline(0, linestyle="--", color="black", linewidth=1, alpha=0.6)
            ax.set_title(f"Label {lbl} | {ch_name} variance error")
            ax.set_xlabel("N samples")
            ax.set_ylabel("Error (est - true)")
            ax.grid(True, alpha=0.4)
    fig.suptitle("Pixel Channel Variance Estimation Error", fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, "convergence_pixel_variance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_and_plot_error_trend(analysis: dict, feature_names: list[str], label_set: list[int], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sc = analysis["sample_counts"]
    hist = analysis["history"]
    true_lbl_means, true_lbl_stds = _true_label_params_for_features(feature_names)

    rows = []
    for i, n in enumerate(sc):
        h = hist[i]
        if h is None:
            rows.append({
                "n_samples": n,
                "branch_mae": np.nan,
                "label_mean_mae": np.nan,
                "label_std_mae": np.nan,
                "pixel_mean_mae": np.nan,
                "pixel_var_mae": np.nan,
            })
            continue

        branch_est = [float(v) for v in h.get("branch_probs", [])[:len(_TRUE_BRANCH_PROBS)]]
        branch_true = _TRUE_BRANCH_PROBS[:len(branch_est)]

        label_mean_est = []
        label_mean_true = []
        label_std_est = []
        label_std_true = []
        for li in range(len(label_set)):
            for fi in range(len(feature_names)):
                if true_lbl_means[li][fi] is not None and li < len(h.get("label_means", [])) and fi < len(h["label_means"][li]):
                    label_mean_est.append(float(h["label_means"][li][fi]))
                    label_mean_true.append(float(true_lbl_means[li][fi]))
                if true_lbl_stds[li][fi] is not None and li < len(h.get("label_stds", [])) and fi < len(h["label_stds"][li]):
                    label_std_est.append(float(h["label_stds"][li][fi]))
                    label_std_true.append(float(true_lbl_stds[li][fi]))

        pixel_mean_est = []
        pixel_mean_true = []
        pixel_var_est = []
        pixel_var_true = []
        for li in range(len(label_set)):
            for ch in range(3):
                if li < len(h.get("pixel_means", [])) and ch < len(h["pixel_means"][li]):
                    pixel_mean_est.append(float(h["pixel_means"][li][ch]))
                    pixel_mean_true.append(float(_TRUE_PIXEL_MEANS[li][ch]))
                if li < len(h.get("pixel_var_diag", [])) and ch < len(h["pixel_var_diag"][li]):
                    pixel_var_est.append(float(h["pixel_var_diag"][li][ch]))
                    pixel_var_true.append(float(_TRUE_PIXEL_VAR_DIAG[li][ch]))

        rows.append({
            "n_samples": n,
            "branch_mae": _mae(branch_est, branch_true),
            "label_mean_mae": _mae(label_mean_est, label_mean_true),
            "label_std_mae": _mae(label_std_est, label_std_true),
            "pixel_mean_mae": _mae(pixel_mean_est, pixel_mean_true),
            "pixel_var_mae": _mae(pixel_var_est, pixel_var_true),
        })

    csv_path = os.path.join(output_dir, "convergence_error_trend.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_samples",
                "branch_mae",
                "label_mean_mae",
                "label_std_mae",
                "pixel_mean_mae",
                "pixel_var_mae",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")

    # Print error trend table to console
    print()
    col_w = 14
    header = (
        f"  {'N':>5}  "
        f"{'Branch MAE':>{col_w}}  "
        f"{'LabelMean MAE':>{col_w}}  "
        f"{'LabelStd MAE':>{col_w}}  "
        f"{'PixelMean MAE':>{col_w}}  "
        f"{'PixelVar MAE':>{col_w}}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print("  Error Trend (MAE by number of training samples)")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        def _fmt(v):
            return f"{v:{col_w}.4f}" if v == v else f"{'N/A':>{col_w}}"
        print(
            f"  {r['n_samples']:>5}  "
            f"{_fmt(r['branch_mae'])}  "
            f"{_fmt(r['label_mean_mae'])}  "
            f"{_fmt(r['label_std_mae'])}  "
            f"{_fmt(r['pixel_mean_mae'])}  "
            f"{_fmt(r['pixel_var_mae'])}"
        )
    print(sep)
    print()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(sc, [r["branch_mae"] for r in rows], marker="o", label="Branch MAE")
    ax.plot(sc, [r["label_mean_mae"] for r in rows], marker="o", label="Label Mean MAE")
    ax.plot(sc, [r["label_std_mae"] for r in rows], marker="o", label="Label Std MAE")
    ax.plot(sc, [r["pixel_mean_mae"] for r in rows], marker="o", label="Pixel Mean MAE")
    ax.plot(sc, [r["pixel_var_mae"] for r in rows], marker="o", label="Pixel Variance MAE")
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Estimation Error Trend")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "convergence_error_trend.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def _print_comparison_table(final_params: dict, feature_names: list[str], label_set: list[int]) -> None:
    true_lbl_means, true_lbl_stds = _true_label_params_for_features(feature_names)
    width = 74

    print()
    print("=" * width)
    print("  Parameter Comparison: True vs Estimated  (all training samples)")
    print("=" * width)

    print()
    print("  Branch Probabilities")
    print(f"  {'Depth':>5}  {'True':>8}  {'Estimated':>10}  {'Error (est-true)':>16}")
    print("  " + "-" * (width - 2))
    bp_est = final_params.get("branch_probs", [])
    for d, tv in enumerate(_TRUE_BRANCH_PROBS):
        ev = bp_est[d] if d < len(bp_est) else None
        ev_s = f"{ev:10.6f}" if ev is not None else f"{'N/A':>10}"
        err_s = f"{ev - tv:+16.6f}" if ev is not None else f"{'N/A':>16}"
        print(f"  {d:>5}  {tv:>8.4f}  {ev_s}  {err_s}")

    print()
    print("  Label Feature Means")
    print(f"  {'Label':>5}  {'Feature':>14}  {'True':>8}  {'Estimated':>10}  {'Error (est-true)':>16}")
    print("  " + "-" * (width - 2))
    lm_est = final_params.get("label_means", [])
    for li, lbl in enumerate(label_set):
        for fi, feat in enumerate(feature_names):
            tv = true_lbl_means[li][fi]
            ev = lm_est[li][fi] if li < len(lm_est) and fi < len(lm_est[li]) else None
            tv_s = f"{tv:8.4f}" if tv is not None else f"{'N/A':>8}"
            ev_s = f"{ev:10.4f}" if ev is not None else f"{'N/A':>10}"
            err_s = f"{ev - tv:+16.4f}" if ev is not None and tv is not None else f"{'N/A':>16}"
            print(f"  {lbl:>5}  {feat:>14}  {tv_s}  {ev_s}  {err_s}")

    print()
    print("  Label Feature Stds")
    print(f"  {'Label':>5}  {'Feature':>14}  {'True':>8}  {'Estimated':>10}  {'Error (est-true)':>16}")
    print("  " + "-" * (width - 2))
    ls_est = final_params.get("label_stds", [])
    for li, lbl in enumerate(label_set):
        for fi, feat in enumerate(feature_names):
            tv = true_lbl_stds[li][fi]
            ev = ls_est[li][fi] if li < len(ls_est) and fi < len(ls_est[li]) else None
            tv_s = f"{tv:8.4f}" if tv is not None else f"{'N/A':>8}"
            ev_s = f"{ev:10.4f}" if ev is not None else f"{'N/A':>10}"
            err_s = f"{ev - tv:+16.4f}" if ev is not None and tv is not None else f"{'N/A':>16}"
            print(f"  {lbl:>5}  {feat:>14}  {tv_s}  {ev_s}  {err_s}")

    print()
    print("  Pixel Channel Means")
    print(f"  {'Label':>5}  {'Ch':>4}  {'True':>8}  {'Estimated':>10}  {'Error (est-true)':>16}")
    print("  " + "-" * (width - 2))
    pm_est = final_params.get("pixel_means", [])
    for li, lbl in enumerate(label_set):
        for ch, ch_name in enumerate(_CHANNEL_NAMES):
            tv = _TRUE_PIXEL_MEANS[li][ch] if li < len(_TRUE_PIXEL_MEANS) else None
            ev = pm_est[li][ch] if li < len(pm_est) and ch < len(pm_est[li]) else None
            tv_s = f"{tv:8.2f}" if tv is not None else f"{'N/A':>8}"
            ev_s = f"{ev:10.4f}" if ev is not None else f"{'N/A':>10}"
            err_s = f"{ev - tv:+16.4f}" if ev is not None and tv is not None else f"{'N/A':>16}"
            print(f"  {lbl:>5}  {ch_name:>4}  {tv_s}  {ev_s}  {err_s}")

    print()
    print("  Pixel Channel Variance (diagonal)")
    print(f"  {'Label':>5}  {'Ch':>4}  {'True':>8}  {'Estimated':>10}  {'Error (est-true)':>16}")
    print("  " + "-" * (width - 2))
    pv_est = final_params.get("pixel_var_diag", [])
    for li, lbl in enumerate(label_set):
        for ch, ch_name in enumerate(_CHANNEL_NAMES):
            tv = _TRUE_PIXEL_VAR_DIAG[li][ch] if li < len(_TRUE_PIXEL_VAR_DIAG) else None
            ev = pv_est[li][ch] if li < len(pv_est) and ch < len(pv_est[li]) else None
            tv_s = f"{tv:8.1f}" if tv is not None else f"{'N/A':>8}"
            ev_s = f"{ev:10.2f}" if ev is not None else f"{'N/A':>10}"
            err_s = f"{ev - tv:+16.2f}" if ev is not None and tv is not None else f"{'N/A':>16}"
            print(f"  {lbl:>5}  {ch_name:>4}  {tv_s}  {ev_s}  {err_s}")

    print()
    print("=" * width)


def _load_final_params(config: Config) -> tuple[dict, list[int], list[str], int]:
    branch_path = os.path.join(config.out_param_dir, config.branch_probs_filename)
    label_path = os.path.join(config.out_param_dir, config.label_param_filename)
    pixel_path = os.path.join(config.out_param_dir, config.pixel_param_filename)

    with open(branch_path, encoding="utf-8") as f:
        branch_data = json.load(f)
    with open(label_path, encoding="utf-8") as f:
        label_data = json.load(f)
    with open(pixel_path, encoding="utf-8") as f:
        pixel_data = json.load(f)

    final_params = {
        "branch_probs": branch_data.get("branch_probs", []),
        "label_means": label_data.get("means", []),
        "label_stds": label_data.get("stds", []),
        "pixel_means": pixel_data.get("mean", []),
        "pixel_var_diag": [
            [pixel_data["variance"][li][ch][ch] for ch in range(3)]
            for li in range(len(pixel_data.get("variance", [])))
        ],
    }
    label_set = label_data.get("label_set", [0, 1, 2])
    feature_names = label_data.get("feature_names", config.label_feature_names)
    image_size = int(label_data.get("image_size", 128))
    return final_params, label_set, feature_names, image_size


if __name__ == "__main__":
    print("[1/3] Train with all training data")
    train(config)

    print("[2/3] Print final parameter comparison table")
    final_params, label_set, feature_names, image_size = _load_final_params(config)
    _print_comparison_table(final_params, feature_names, label_set)

    print("[3/3] Convergence and error trend analysis")
    analysis = _run_convergence_analysis(config, label_set, image_size, feature_names)
    out_root = os.path.join(os.path.dirname(__file__), "outputs")
    _plot_convergence_graphs(analysis, feature_names, label_set, out_root)
    _save_and_plot_error_trend(analysis, feature_names, label_set, out_root)
    print("Done")
