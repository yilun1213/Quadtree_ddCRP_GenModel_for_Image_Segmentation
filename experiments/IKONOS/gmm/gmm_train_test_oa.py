from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class GaussianClassModel:
    label: int
    prior: float
    mean: np.ndarray
    covariance: np.ndarray
    inv_covariance: np.ndarray
    log_det_covariance: float


def load_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def list_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    image_files = {p.stem: p for p in images_dir.glob("*.png")}
    label_files = {p.stem: p for p in labels_dir.glob("*.png")}
    common_stems = sorted(image_files.keys() & label_files.keys())
    if not common_stems:
        raise FileNotFoundError(
            f"No matching PNG filename stems found in {images_dir} and {labels_dir}."
        )
    return [(image_files[s], label_files[s]) for s in common_stems]


def build_training_samples(train_root: Path, labels: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    pairs = list_pairs(train_root / "images", train_root / "labels")

    features_all = []
    labels_all = []

    for image_path, label_path in pairs:
        image = load_png(image_path)
        label_img = load_png(label_path)

        if image.ndim != 3:
            raise ValueError(f"Expected RGB image at {image_path}, got shape {image.shape}.")
        if label_img.ndim == 3:
            label_img = label_img[..., 0]

        if image.shape[:2] != label_img.shape[:2]:
            raise ValueError(
                f"Image/label size mismatch: {image_path} {image.shape[:2]} vs {label_path} {label_img.shape[:2]}"
            )

        pixels = image.reshape(-1, image.shape[-1]).astype(np.float64)
        pixel_labels = label_img.reshape(-1).astype(np.int32)

        mask = (pixel_labels == labels[0]) | (pixel_labels == labels[1])
        features_all.append(pixels[mask])
        labels_all.append(pixel_labels[mask])

    x = np.concatenate(features_all, axis=0)
    y = np.concatenate(labels_all, axis=0)
    return x, y


def estimate_models(x: np.ndarray, y: np.ndarray, labels: tuple[int, int], reg: float) -> dict[int, GaussianClassModel]:
    models: dict[int, GaussianClassModel] = {}
    n_total = len(y)

    for cls in labels:
        x_cls = x[y == cls]
        if len(x_cls) == 0:
            raise ValueError(f"No training samples found for label {cls}.")

        prior = len(x_cls) / n_total
        mean = x_cls.mean(axis=0)

        centered = x_cls - mean
        covariance = (centered.T @ centered) / max(len(x_cls) - 1, 1)
        covariance = covariance + np.eye(covariance.shape[0], dtype=np.float64) * reg

        sign, log_det = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError(f"Covariance matrix for label {cls} is not positive definite.")
        inv_cov = np.linalg.inv(covariance)

        models[cls] = GaussianClassModel(
            label=cls,
            prior=prior,
            mean=mean,
            covariance=covariance,
            inv_covariance=inv_cov,
            log_det_covariance=log_det,
        )

    return models


def class_log_likelihood(x: np.ndarray, model: GaussianClassModel) -> np.ndarray:
    d = x.shape[1]
    diff = x - model.mean
    mahal = np.einsum("ij,jk,ik->i", diff, model.inv_covariance, diff)
    return -0.5 * (d * np.log(2.0 * np.pi) + model.log_det_covariance + mahal)


def predict_labels(x: np.ndarray, models: dict[int, GaussianClassModel], labels: tuple[int, int]) -> np.ndarray:
    cls0, cls1 = labels
    score0 = class_log_likelihood(x, models[cls0]) + np.log(models[cls0].prior + 1e-12)
    score1 = class_log_likelihood(x, models[cls1]) + np.log(models[cls1].prior + 1e-12)
    return np.where(score1 > score0, cls1, cls0).astype(np.int32)


def evaluate(
    test_root: Path,
    models: dict[int, GaussianClassModel],
    labels: tuple[int, int],
    pred_dir: Path | None = None,
) -> float:
    pairs = list_pairs(test_root / "images", test_root / "labels")
    visualize_dir: Path | None = None

    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)
        visualize_dir = pred_dir / "visualize"
        visualize_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0

    for image_path, label_path in pairs:
        image = load_png(image_path)
        label_img = load_png(label_path)

        if label_img.ndim == 3:
            label_img = label_img[..., 0]

        x = image.reshape(-1, image.shape[-1]).astype(np.float64)
        y_true = label_img.reshape(-1).astype(np.int32)

        valid = (y_true == labels[0]) | (y_true == labels[1])
        if not np.any(valid):
            continue

        y_pred = predict_labels(x[valid], models, labels)
        y_true_valid = y_true[valid]

        if pred_dir is not None:
            pred_all = np.full(y_true.shape, fill_value=255, dtype=np.uint8)
            pred_all[valid] = y_pred.astype(np.uint8)
            pred_image = pred_all.reshape(label_img.shape)
            out_path = pred_dir / f"{image_path.stem}.png"
            Image.fromarray(pred_image).save(out_path)
            print(f"Saved predicted label: {out_path}")

            # Match existing labels/visualize convention: binary grayscale 0/255.
            pred_vis = np.zeros(y_true.shape, dtype=np.uint8)
            pred_vis[valid] = np.where(y_pred == labels[1], 255, 0).astype(np.uint8)
            vis_image = pred_vis.reshape(label_img.shape)
            vis_path = visualize_dir / f"{image_path.stem}.png"
            Image.fromarray(vis_image).save(vis_path)
            print(f"Saved predicted visualize: {vis_path}")

        total += len(y_true_valid)
        correct += int((y_pred == y_true_valid).sum())

    if total == 0:
        raise ValueError("No valid test pixels found for requested labels.")

    return correct / total


def print_model_summary(models: dict[int, GaussianClassModel], labels: tuple[int, int]) -> None:
    print("Estimated Gaussian parameters:")
    for cls in labels:
        m = models[cls]
        print(f"Label {cls}:")
        print(f"  prior = {m.prior:.6f}")
        print(f"  mean  = {np.array2string(m.mean, precision=4)}")
        print(f"  covariance =\n{np.array2string(m.covariance, precision=4)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate two-class Gaussian mixture parameters from train_data and evaluate OA on test_data."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Dataset root containing train_data and test_data")
    parser.add_argument("--label0", type=int, default=0, help="First label value")
    parser.add_argument("--label1", type=int, default=1, help="Second label value")
    parser.add_argument("--reg", type=float, default=1e-6, help="Covariance regularization")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Directory to save predicted test labels as PNG (default: <root>/test_data/pred_labels)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = (args.label0, args.label1)

    train_root = args.root / "train_data"
    test_root = args.root / "test_data"
    pred_dir = args.pred_dir if args.pred_dir is not None else test_root / "pred_labels"

    x_train, y_train = build_training_samples(train_root, labels)
    models = estimate_models(x_train, y_train, labels, reg=args.reg)
    oa = evaluate(test_root, models, labels, pred_dir=pred_dir)

    print_model_summary(models, labels)
    print(f"Predicted labels output dir: {pred_dir}")
    print(f"\nOverall Accuracy (OA): {oa:.6f} ({oa * 100:.2f}%)")


if __name__ == "__main__":
    main()
