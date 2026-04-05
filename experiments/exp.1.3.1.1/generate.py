from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import model.label.geom_features_norm_dist as label_model
from model.label.geom_features import compute_geom_features


INPUT_REGION_PATH = Path(__file__).with_name("region_000.png")
OUTPUT_DIR = Path(__file__).with_name("outputs")
LABEL_OUTPUT_DIR = OUTPUT_DIR / "labels"
VIS_OUTPUT_DIR = LABEL_OUTPUT_DIR / "visualize"
REPORT_JSON_PATH = OUTPUT_DIR / "region_label_priors.json"
NUM_SAMPLES = 5


LABEL_PARAM = {
    "label_num": 3,
    "label_set": [0, 1, 2],
    "label_value_set": [0, 128, 255],
    "means": [
        [9.535675761814312, 7.3526835069212915, 0.0775996551652279],
        [7.602403420240487, 6.90715679601541, 0.026084000913409226],
        [9.519315318835302, 7.591270702553376, 0.043750937130338],
    ],
    "stds": [
        [1.7255673479777514, 1.053401067765219, 0.025070832863695793],
        [0.06332031250252408, 0.1463038346144442, 0.006885707845728361],
        [0.0017857046518637441, 0.03231548185822331, 0.002834923945423524],
    ],
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
        "means": LABEL_PARAM["means"],
        "stds": LABEL_PARAM["stds"],
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
    region_stats: list[dict[str, object]] = []

    for region_index, region in enumerate(regions):
        features = compute_geom_features(
            region,
            image_size=image_size,
            feature_names=feature_names,
        )
        priors = label_model.label_prior(region=region, param=model_param)
        region_stats.append(
            {
                "region_index": region_index,
                "region": region,
                "area_pixels": len(region),
                "features": features,
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
        priors = np.asarray(stat["label_priors"], dtype=float)

        print(f"  Region {region_index:03d}: area={area_pixels}")
        for feature_name, feature_value in zip(feature_names, features):
            print(f"    {feature_name}: {feature_value:.6f}")
        for label, prob in zip(label_set, priors):
            print(f"    p(label={label}) = {prob:.6f}")


def save_region_analysis(region_stats: list[dict[str, object]], path: Path) -> None:
    feature_names = list(LABEL_PARAM["feature_names"])
    label_set = [int(label) for label in LABEL_PARAM["label_set"]]
    serializable_stats = []

    for stat in region_stats:
        features = np.asarray(stat["features"], dtype=float)
        priors = np.asarray(stat["label_priors"], dtype=float)
        serializable_stats.append(
            {
                "region_index": int(stat["region_index"]),
                "area_pixels": int(stat["area_pixels"]),
                "features": {
                    feature_name: float(feature_value)
                    for feature_name, feature_value in zip(feature_names, features)
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
    print(f"Saved region analysis to {REPORT_JSON_PATH.name}")

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
