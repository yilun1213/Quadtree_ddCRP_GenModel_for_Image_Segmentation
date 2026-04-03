from __future__ import annotations

import json
import os
import sys

import numpy as np

import utils
from model.quadtree.node import Node


PARAM_FILENAME = "norm_param.json"


def _as_3d(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[..., np.newaxis]
    return img


def _parse_mean_and_cov(theta: dict) -> tuple[np.ndarray, np.ndarray]:
    if "mean" not in theta:
        raise KeyError("theta must contain 'mean'")

    means = np.asarray(theta["mean"], dtype=np.float64)
    if means.ndim != 2:
        raise ValueError("theta['mean'] must be shape (label_num, channels)")

    if "variance" in theta:
        covs = np.asarray(theta["variance"], dtype=np.float64)
        if covs.ndim != 3:
            raise ValueError("theta['variance'] must be shape (label_num, channels, channels)")
    elif "std" in theta:
        stds = np.asarray(theta["std"], dtype=np.float64)
        if stds.ndim != 2:
            raise ValueError("theta['std'] must be shape (label_num, channels)")
        covs = np.array([np.diag(np.maximum(s, 1e-6) ** 2) for s in stds], dtype=np.float64)
    else:
        raise KeyError("theta must contain 'variance' or 'std'")

    if means.shape[0] != covs.shape[0] or means.shape[1] != covs.shape[1]:
        raise ValueError("mean and variance/std shapes are inconsistent")

    return means, covs


def _label_to_index(theta: dict, label: int) -> int:
    if "label_set" in theta:
        try:
            return list(theta["label_set"]).index(label)
        except ValueError:
            return -1
    if 0 <= int(label) < len(theta.get("mean", [])):
        return int(label)
    return -1


def generate_rgb_from_labels(label_image, region_dict, theta, width, height, seed):
    """Generate RGB image from labels using per-label Gaussian parameters."""
    del region_dict
    del width
    del height

    if seed is not None:
        np.random.seed(seed)

    means, covs = _parse_mean_and_cov(theta)
    label_num, channels = means.shape
    if "label_set" in theta and isinstance(theta["label_set"], list):
        label_values = [int(v) for v in theta["label_set"]]
    else:
        label_values = list(range(label_num))

    if len(label_values) != label_num:
        raise ValueError("theta['label_set'] length must match theta['mean'] rows")

    if channels not in (1, 3):
        raise ValueError(f"normal_dist supports channels=1 or 3, got {channels}")

    out = np.zeros((label_image.shape[0], label_image.shape[1], channels), dtype=np.float64)

    for label_idx, label_value in enumerate(label_values):
        mask = label_image == label_value
        n = int(np.count_nonzero(mask))
        if n == 0:
            continue
        cov = covs[label_idx] + 1e-6 * np.eye(channels)
        samples = np.random.multivariate_normal(mean=means[label_idx], cov=cov, size=n)
        out[mask] = samples

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)

    # generate.py saves an RGB image; when parameters are 1ch, replicate to 3ch.
    if channels == 1:
        return np.repeat(out_u8, 3, axis=2)
    return out_u8


def param_est(train_image_dir, train_label_dir, out_param_json, Omega):
    """Estimate Gaussian parameters by MLE for each label and save JSON."""
    del Omega

    image_files = utils.get_image_files(train_image_dir)
    label_files = utils.get_image_files(train_label_dir)
    filename_list = sorted(utils.harmonize_lists(image_files, label_files))

    if not filename_list:
        print("No paired train images/labels found for pixel param estimation.", file=sys.stderr)
        return

    pixel_by_label: dict[int, list[np.ndarray]] = {}
    channels = None

    for filename in filename_list:
        img = _as_3d(utils.load_image(os.path.join(train_image_dir, filename)).astype(np.float64))
        lbl = utils.load_image(os.path.join(train_label_dir, filename)).astype(np.int64)

        if channels is None:
            channels = img.shape[2]
        elif channels != img.shape[2]:
            raise ValueError("All train images must have the same channel count")

        for label in np.unique(lbl):
            label_int = int(label)
            pixels = img[lbl == label_int]
            if pixels.size == 0:
                continue
            pixel_by_label.setdefault(label_int, []).append(pixels)

    if not pixel_by_label:
        print("No labeled pixels found; skipping pixel parameter save.", file=sys.stderr)
        return

    label_set = sorted(pixel_by_label.keys())
    means = []
    variances = []
    stds = []

    for label in label_set:
        samples = np.concatenate(pixel_by_label[label], axis=0)
        if samples.shape[0] < 2:
            mu = np.mean(samples, axis=0)
            cov = np.eye(samples.shape[1], dtype=np.float64)
        else:
            mu = np.mean(samples, axis=0)
            centered = samples - mu
            cov = (centered.T @ centered) / float(samples.shape[0])
            cov = cov + 1e-6 * np.eye(samples.shape[1])

        means.append(mu.tolist())
        variances.append(cov.tolist())
        stds.append(np.sqrt(np.diag(cov)).tolist())

    output = {
        "label_set": label_set,
        "channels": int(channels if channels is not None else 3),
        "mean": means,
        "variance": variances,
        "std": stds,
    }

    with open(out_param_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Saved normal pixel parameters to: {out_param_json}")


def get_pixels_in_raster_order(region_tuple: tuple[Node, ...]) -> list[tuple[int, int]]:
    pixels = []
    for node in region_tuple:
        for r in range(node.upper_edge, node.lower_edge):
            for c in range(node.left_edge, node.right_edge):
                pixels.append((r, c))
    return sorted(pixels)


def log_prob_Y_given_X(region_tuple: tuple[Node, ...], label: int, img_array: np.ndarray, theta: dict) -> float:
    """Compute log p(Y_r | X_r=label) under iid multivariate Gaussian pixels."""
    label_idx = _label_to_index(theta, int(label))
    if label_idx < 0:
        return -np.inf

    means, covs = _parse_mean_and_cov(theta)
    mean_vec = means[label_idx]
    cov = covs[label_idx]
    channels = mean_vec.shape[0]

    if cov.shape != (channels, channels):
        return -np.inf

    cov = cov + 1e-6 * np.eye(channels)
    try:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return -np.inf
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return -np.inf

    pixels = get_pixels_in_raster_order(region_tuple)
    if not pixels:
        return 0.0

    arr = _as_3d(img_array).astype(np.float64)
    const_term = -0.5 * (channels * np.log(2.0 * np.pi) + logdet)

    total = 0.0
    for r, c in pixels:
        diff = arr[r, c] - mean_vec
        maha = float(diff @ inv_cov @ diff.T)
        if not np.isfinite(maha):
            return -np.inf
        total += const_term - 0.5 * maha

    return float(total)


def add_label_set(ar_param_path, label_param_path):
    """Compatibility helper used by train.py; keeps API identical to AR module."""
    with open(ar_param_path, "r", encoding="utf-8") as f:
        pixel_params = json.load(f)
    with open(label_param_path, "r", encoding="utf-8") as f:
        label_params = json.load(f)

    label_set = label_params.get("label_set", pixel_params.get("label_set", []))
    if pixel_params.get("label_set") != label_set:
        pixel_params["label_set"] = label_set
        with open(ar_param_path, "w", encoding="utf-8") as f:
            json.dump(pixel_params, f, ensure_ascii=False, indent=4)