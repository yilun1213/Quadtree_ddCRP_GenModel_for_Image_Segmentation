from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

import utils
from .geom_features import compute_geom_features, DEFAULT_FEATURE_NAMES


def _compute_phi(region: set[tuple[int, int]], param: dict[str, Any]) -> np.ndarray:
    """Compute geometric feature vector phi(r) used by the label model."""
    image_size = int(param.get("image_size", 128))
    feature_names = param.get("feature_names", DEFAULT_FEATURE_NAMES)
    return compute_geom_features(region, image_size=image_size, feature_names=feature_names)


def _parse_weight_bias(param: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse model parameters with backward-compatible keys.

    Expected:
    - weights: (K, d)
    - bias: (K,)
    """
    weights = np.asarray(param["weights"], dtype=float)
    bias_raw = np.asarray(param["bias"], dtype=float)

    if bias_raw.ndim == 2 and bias_raw.shape[1] == 1:
        bias = bias_raw[:, 0]
    elif bias_raw.ndim == 1:
        bias = bias_raw
    else:
        raise ValueError("bias must be shape (K,) or (K,1).")

    if weights.ndim != 2:
        raise ValueError("weights must be shape (K, d).")
    if bias.shape[0] != weights.shape[0]:
        raise ValueError("bias length must match number of classes K.")

    return weights, bias


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax."""
    max_logit = np.max(logits)
    shifted = logits - max_logit
    return shifted - np.log(np.sum(np.exp(shifted)))


def log_label_prior(region: set[tuple[int, int]], param: dict[str, Any]) -> np.ndarray:
    """
    Compute log p(x_r | phi(r)) for multinomial logistic regression.

    log p(x=k | phi) = z_k - log(sum_j exp(z_j)), z_k = w_k^T phi + b_k
    """
    phi = _compute_phi(region, param)
    weights, bias = _parse_weight_bias(param)
    logits = weights @ phi + bias
    return _log_softmax(logits)


def label_prior(region: set[tuple[int, int]], param: dict[str, Any]) -> np.ndarray:
    """Compute p(x_r | phi(r)) as a probability vector with sum=1."""
    logp = log_label_prior(region, param)
    logp_max = np.max(logp)
    p = np.exp(logp - logp_max)
    s = p.sum()
    if s <= 0.0 or not np.isfinite(s):
        k = p.shape[0]
        return np.full(k, 1.0 / k, dtype=float)
    return p / s


def _region_to_feature(
    region_mask: np.ndarray,
    image_size: int,
    feature_names: list[str],
) -> np.ndarray | None:
    """Convert a binary connected-component mask to phi(r)."""
    coords = np.argwhere(region_mask)
    if coords.size == 0:
        return None
    region = {(int(i), int(j)) for i, j in coords}
    phi = compute_geom_features(region, image_size=image_size, feature_names=feature_names)
    if np.any(~np.isfinite(phi)):
        return None
    return phi


def _extract_training_data(
    label_images: list[np.ndarray],
    label_set: list[int],
    image_size: int,
    feature_names: list[str],
    min_region_area: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build training pairs (y_r, phi(r)) from connected regions in label images.

    Returns:
    - X: (N, d)
    - y: (N,) in [0, K-1]
    """
    labels = sorted(int(v) for v in label_set)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    filtered_small_regions = 0
    connectivity_8 = ndimage.generate_binary_structure(2, 2)

    for lbl_img in label_images:
        for label in np.unique(lbl_img):
            label_int = int(label)
            if label_int not in label_to_index:
                continue

            mask = lbl_img == label
            labeled_mask, num_regions = ndimage.label(mask, structure=connectivity_8)
            if num_regions == 0:
                continue

            for rid in range(1, num_regions + 1):
                region_mask = labeled_mask == rid
                if int(np.count_nonzero(region_mask)) < int(min_region_area):
                    filtered_small_regions += 1
                    continue
                phi = _region_to_feature(
                    region_mask,
                    image_size=image_size,
                    feature_names=feature_names,
                )
                if phi is None:
                    continue
                x_list.append(phi)
                y_list.append(label_to_index[label_int])

    if not x_list:
        return (
            np.zeros((0, len(feature_names)), dtype=float),
            np.zeros((0,), dtype=int),
            filtered_small_regions,
        )

    x_mat = np.asarray(x_list, dtype=float)
    y_vec = np.asarray(y_list, dtype=int)
    return x_mat, y_vec, filtered_small_regions


def _fit_multinomial_logistic(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    class_num: int,
    l2_reg: float = 1e-3,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit softmax regression by maximizing log-likelihood (with small L2 regularization).

    Objective minimized:
    NLL(W,b) = -sum_n log p(y_n | x_n) + 0.5 * l2_reg * ||W||^2
    """
    n, d = x_mat.shape
    k = class_num

    if k <= 1:
        return np.zeros((k, d), dtype=float), np.zeros((k,), dtype=float)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = theta[: k * d].reshape(k, d)
        b = theta[k * d :]
        return w, b

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = unpack(theta)
        logits = x_mat @ w.T + b  # (N, K)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        eps = 1e-12
        nll = -np.sum(np.log(probs[np.arange(n), y_vec] + eps))
        nll += 0.5 * l2_reg * np.sum(w * w)

        grad_logits = probs
        grad_logits[np.arange(n), y_vec] -= 1.0

        grad_w = grad_logits.T @ x_mat + l2_reg * w
        grad_b = np.sum(grad_logits, axis=0)
        grad = np.concatenate([grad_w.ravel(), grad_b])
        return float(nll), grad

    theta0 = np.zeros(k * d + k, dtype=float)
    result = minimize(
        fun=lambda t: objective(t)[0],
        x0=theta0,
        jac=lambda t: objective(t)[1],
        method="L-BFGS-B",
        options={"maxiter": max_iter, "disp": False},
    )

    if not result.success:
        print(
            f"Warning: logistic optimization did not fully converge: {result.message}",
            file=sys.stderr,
        )

    w_hat, b_hat = unpack(result.x)
    return w_hat, b_hat


def param_est(
    train_label_dir: str,
    label_set: list[int],
    label_num: int,
    image_size: int = 128,
    feature_names: list[str] | None = None,
    min_region_area: int = 32,
) -> dict[str, Any]:
    """
    Estimate label-model parameters from training labels using logistic regression.

    Interface is intentionally aligned with existing training pipeline.
    """
    filenames = utils.get_image_files(train_label_dir)
    if not filenames:
        print(f"Error: no label images found in {train_label_dir}", file=sys.stderr)
        return {}

    label_images: list[np.ndarray] = []
    for filename in filenames:
        path = os.path.join(train_label_dir, filename)
        try:
            label_images.append(utils.load_image(path))
        except Exception as exc:
            print(f"Warning: failed to read {filename}: {exc}", file=sys.stderr)

    if not label_images:
        print("Error: failed to load any label image.", file=sys.stderr)
        return {}

    selected_features = list(feature_names) if feature_names else list(DEFAULT_FEATURE_NAMES)

    print("Step 1: extracting region-level geometric features for logistic training...")
    x_mat, y_vec, filtered_small_regions = _extract_training_data(
        label_images,
        label_set,
        image_size=image_size,
        feature_names=selected_features,
        min_region_area=min_region_area,
    )

    print(f"  - minimum region area threshold: {min_region_area} px")
    print(f"  - filtered small regions: {filtered_small_regions}")

    if x_mat.shape[0] == 0:
        print("Error: no valid training regions found.", file=sys.stderr)
        return {}

    print(
        f"Extracted {x_mat.shape[0]} regions, feature_dim={x_mat.shape[1]}, class_num={label_num}."
    )

    print("Step 2: maximum-likelihood estimation with multinomial logistic regression...")
    weights, bias = _fit_multinomial_logistic(
        x_mat=x_mat,
        y_vec=y_vec,
        class_num=label_num,
        l2_reg=1e-3,
        max_iter=500,
    )

    print("Logistic parameter estimation finished.")

    return {
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "image_size": int(image_size),
        "feature_names": selected_features,
    }
