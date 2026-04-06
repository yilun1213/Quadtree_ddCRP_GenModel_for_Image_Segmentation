from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import model.label.geom_features_logistic as label_model
from model.label.geom_features import compute_geom_features


INPUT_REGION_PATH = Path(__file__).with_name("region_000.png")
OUTPUT_DIR = Path(__file__).with_name("outputs")
LABEL_OUTPUT_DIR = OUTPUT_DIR / "labels"
VIS_OUTPUT_DIR = LABEL_OUTPUT_DIR / "visualize"
REPORT_JSON_PATH = OUTPUT_DIR / "region_label_priors.json"
REGION_INDEX_IMAGE_PATH = OUTPUT_DIR / "region_with_indices.png"
FEATURE_TABLE_PATH = OUTPUT_DIR / "region_feature_logistic_prob_table.txt"
NUM_SAMPLES = 5


LABEL_PARAM = {
    "label_num": 3,
    "label_set": [0, 1, 2],
    "label_value_set": [0, 128, 255],
    # Region-shape-driven logistic model:
    # x=0 -> small compact, x=1 -> medium elongated, x=2 -> large and rounded.
    "weights": [
        [-1.4, -0.9, 2.8],
        [0.1, 1.0, -2.5],
        [2.4, 1.8, 1.5],
    ],
    "bias": [6.5, -4.5, -31.0],
    "feature_names": ["log_area", "log_perimeter", "circularity"],
}


def load_region_image(path: Path) -> np.ndarray:
    image = np.array(Image.open(path))
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def extract_regions(region_image: np.ndarray) -> list[set[tuple[int, int]]]:
    if region_image.ndim == 2:
        flat = region_image.reshape(-1, 1)
    else:
        flat = region_image.reshape(-1, region_image.shape[2])

    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    height, width = region_image.shape[:2]
    regions: list[set[tuple[int, int]]] = []

    for color_index in range(len(unique_colors)):
        pixel_indices = np.flatnonzero(inverse == color_index)
        region = {
            (int(pixel_index // width), int(pixel_index % width))
            for pixel_index in pixel_indices
        }
        if region:
            regions.append(region)

    return regions


def build_model_param(image_size: int) -> dict:
    return {
        "weights": LABEL_PARAM["weights"],
        "bias": LABEL_PARAM["bias"],
        "image_size": image_size,
        "feature_names": LABEL_PARAM["feature_names"],
    }


def analyze_regions(
    regions: list[set[tuple[int, int]]],
    image_size: int,
) -> list[dict[str, object]]:
    model_param = build_model_param(image_size)
    label_set = [int(label) for label in LABEL_PARAM["label_set"]]
    feature_names = list(LABEL_PARAM["feature_names"])
    weights = np.asarray(LABEL_PARAM["weights"], dtype=float)
    bias = np.asarray(LABEL_PARAM["bias"], dtype=float)
    region_stats: list[dict[str, object]] = []

    for region_index, region in enumerate(regions):
        features = compute_geom_features(
            region,
            image_size=image_size,
            feature_names=feature_names,
        )
        priors = label_model.label_prior(region=region, param=model_param)
        logits = weights @ np.asarray(features, dtype=float) + bias
        region_stats.append(
            {
                "region_index": region_index,
                "region_no": region_index + 1,
                "region": region,
                "area_pixels": len(region),
                "features": features,
                "logits": logits,
                "label_priors": priors,
            }
        )

    return region_stats


def print_region_analysis(region_stats: list[dict[str, object]]) -> None:
    feature_names = list(LABEL_PARAM["feature_names"])
    label_set = [int(label) for label in LABEL_PARAM["label_set"]]

    print("Region-wise geometric features and label priors:")
    for stat in region_stats:
        region_index = int(stat["region_index"])
        area_pixels = int(stat["area_pixels"])
        features = np.asarray(stat["features"], dtype=float)
        logits = np.asarray(stat["logits"], dtype=float)
        priors = np.asarray(stat["label_priors"], dtype=float)

        print(f"  Region {region_index:03d}: area={area_pixels}")
        for feature_name, feature_value in zip(feature_names, features):
            print(f"    {feature_name}: {feature_value:.6f}")
        for label, logit in zip(label_set, logits):
            print(f"    z(label={label}) = {logit:.6f}")
        for label, prob in zip(label_set, priors):
            print(f"    p(label={label}) = {prob:.6f}")


def save_region_analysis(region_stats: list[dict[str, object]], path: Path) -> None:
    feature_names = list(LABEL_PARAM["feature_names"])
    label_set = [int(label) for label in LABEL_PARAM["label_set"]]
    serializable_stats = []

    for stat in region_stats:
        features = np.asarray(stat["features"], dtype=float)
        logits = np.asarray(stat["logits"], dtype=float)
        priors = np.asarray(stat["label_priors"], dtype=float)
        serializable_stats.append(
            {
                "region_index": int(stat["region_index"]),
                "region_no": int(stat["region_no"]),
                "area_pixels": int(stat["area_pixels"]),
                "features": {
                    feature_name: float(feature_value)
                    for feature_name, feature_value in zip(feature_names, features)
                },
                "logits": {
                    str(label): float(logit)
                    for label, logit in zip(label_set, logits)
                },
                "label_priors": {
                    str(label): float(prob)
                    for label, prob in zip(label_set, priors)
                },
            }
        )

    path.write_text(
        json.dumps(serializable_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_region_index_overlay(
    region_image: np.ndarray,
    region_stats: list[dict[str, object]],
    path: Path,
) -> None:
    pil_image = Image.fromarray(region_image.astype(np.uint8))
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()

    for stat in region_stats:
        region_no = int(stat["region_no"])
        region = stat["region"]
        points = np.asarray(list(region), dtype=float)
        center_i = float(points[:, 0].mean())
        center_j = float(points[:, 1].mean())

        text = str(region_no)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
        text_pos = (center_j - text_w / 2.0, center_i - text_h / 2.0)

        draw.text(
            text_pos,
            text,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    pil_image.save(path)


def save_region_feature_logistic_prob_table(
    region_stats: list[dict[str, object]],
    path: Path,
) -> None:
    feature_names = list(LABEL_PARAM["feature_names"])
    label_set = [int(label) for label in LABEL_PARAM["label_set"]]

    header_cols = ["region_no"]
    for feature_name in feature_names:
        header_cols.append(feature_name)
    for label in label_set:
        header_cols.append(f"z(label={label})")
    for label in label_set:
        header_cols.append(f"p(label={label})")

    lines = []
    lines.append("Region index mapping and feature/logistic-probability table")
    lines.append("- region_no is the index drawn on region_with_indices.png")
    lines.append("- z(label=x) = w^(x)^T phi(r) + bias^(x)")
    lines.append("")
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    for stat in region_stats:
        region_no = int(stat["region_no"])
        features = np.asarray(stat["features"], dtype=float)
        logits = np.asarray(stat["logits"], dtype=float)
        priors = np.asarray(stat["label_priors"], dtype=float)

        row = [str(region_no)]
        for feature_idx, _feature_name in enumerate(feature_names):
            feature_value = float(features[feature_idx])
            row.append(f"{feature_value:.6f}")

        for logit in logits:
            row.append(f"{float(logit):.6f}")

        for prob in priors:
            row.append(f"{float(prob):.6e}")

        lines.append("| " + " | ".join(row) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample_label_image(
    region_stats: list[dict[str, object]],
    image_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    label_set = [int(label) for label in LABEL_PARAM["label_set"]]
    label_value_set = [int(value) for value in LABEL_PARAM["label_value_set"]]
    label_value_map = {
        label: value for label, value in zip(label_set, label_value_set)
    }

    label_image = np.zeros((image_size, image_size), dtype=np.uint8)
    visualize_image = np.zeros((image_size, image_size), dtype=np.uint8)

    for stat in region_stats:
        region = stat["region"]
        probs = np.asarray(stat["label_priors"], dtype=float)
        chosen_index = int(rng.choice(len(label_set), p=probs))
        label = label_set[chosen_index]
        vis_value = label_value_map[label]

        for i, j in region:
            label_image[i, j] = label
            visualize_image[i, j] = vis_value

    return label_image, visualize_image


def save_image(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path)


def ensure_output_dirs() -> None:
    LABEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not INPUT_REGION_PATH.exists():
        raise FileNotFoundError(f"Input region image not found: {INPUT_REGION_PATH}")

    ensure_output_dirs()

    region_image = load_region_image(INPUT_REGION_PATH)
    image_size = int(region_image.shape[0])
    if region_image.shape[0] != region_image.shape[1]:
        raise ValueError("Only square region images are supported.")

    regions = extract_regions(region_image)
    if not regions:
        raise ValueError("No regions were extracted from the input image.")

    print(f"Loaded {INPUT_REGION_PATH.name}: size={image_size}, regions={len(regions)}")

    region_stats = analyze_regions(regions=regions, image_size=image_size)
    print_region_analysis(region_stats)
    save_region_analysis(region_stats, REPORT_JSON_PATH)
    save_region_index_overlay(region_image=region_image, region_stats=region_stats, path=REGION_INDEX_IMAGE_PATH)
    save_region_feature_logistic_prob_table(region_stats=region_stats, path=FEATURE_TABLE_PATH)
    print(f"Saved region analysis to {REPORT_JSON_PATH.name}")
    print(f"Saved indexed region image to {REGION_INDEX_IMAGE_PATH.name}")
    print(f"Saved region feature/logistic-probability table to {FEATURE_TABLE_PATH.name}")

    for sample_index in range(NUM_SAMPLES):
        rng = np.random.default_rng(sample_index)
        label_image, visualize_image = sample_label_image(
            region_stats=region_stats,
            image_size=image_size,
            rng=rng,
        )

        stem = f"label_{sample_index:03d}"
        label_path = LABEL_OUTPUT_DIR / f"{stem}.png"
        vis_path = VIS_OUTPUT_DIR / f"{stem}.png"

        save_image(label_image, label_path)
        save_image(visualize_image, vis_path)
        print(f"Saved {label_path.name} and visualize/{vis_path.name}")


if __name__ == "__main__":
    main()
