# config.py
# train.py と estimate_label.py が共有する設定

import os
import sys
from dataclasses import dataclass
from typing import Callable, Any, Dict

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import model.label.geom_features_norm_dist as label_model
import model.pixel.normal_dist as pixel_model
import model.quadtree.depth_dependent_model as quadtree_model
import model.region.affinity as region_model


@dataclass(frozen=True)
class Config:
    train_image_dir: str
    train_label_dir: str
    test_image_dir: str
    test_label_dir: str
    out_param_dir: str
    est_label_folder_path: str
    est_label_dirname: str
    est_label_visualize_dirname: str
    est_region_dirname: str
    est_quadtree_dirname: str
    train_label_vis_dir: str
    test_label_vis_dir: str
    label_param_filename: str
    pixel_param_filename: str
    branch_probs_filename: str
    label_feature_names: list[str]
    label_min_region_area: int
    offset: list
    label_model: Callable
    pixel_model: Callable
    quadtree_model: Callable
    affinity_func: Callable
    alpha: float
    gibbs_num_iterations: int
    affinity_params: Dict[str, Any]
    oa_log_filepath: str
    est_label_diff_dir: str
    enable_logq_cache: bool


EXPERIMENT_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(EXPERIMENT_DIR, "outputs")
TRAIN_DATA_DIR = os.path.join(OUTPUTS_DIR, "train_data")
TEST_DATA_DIR = os.path.join(OUTPUTS_DIR, "test_data")
ESTIMATED_PARAM_DIR = os.path.join(OUTPUTS_DIR, "estimated_param")
ESTIMATION_RESULTS_DIR = os.path.join(OUTPUTS_DIR, "estimation_results")

_BETA = 30.0
_ETA = 30.0
_ALPHA = 1e-8

config = Config(
    train_image_dir=os.path.join(TRAIN_DATA_DIR, "images"),
    train_label_dir=os.path.join(TRAIN_DATA_DIR, "labels"),
    train_label_vis_dir=os.path.join(TRAIN_DATA_DIR, "labels", "visualize"),
    test_image_dir=os.path.join(TEST_DATA_DIR, "images"),
    test_label_dir=os.path.join(TEST_DATA_DIR, "labels"),
    test_label_vis_dir=os.path.join(TEST_DATA_DIR, "labels", "visualize"),
    out_param_dir=ESTIMATED_PARAM_DIR,
    est_label_folder_path=ESTIMATION_RESULTS_DIR,
    est_label_dirname="label",
    est_label_visualize_dirname="visualize",
    est_region_dirname="region",
    est_quadtree_dirname="quadtree",
    label_param_filename="label_param.json",
    pixel_param_filename="pixel_param.json",
    branch_probs_filename="branch_probs.json",
    label_feature_names=["log_area", "circularity"],
    label_min_region_area=32,
    offset=[
        (-2, -2), (-2, -1), (-2, 0),
        (-1, -2), (-1, -1), (-1, 0),
        (0, -2), (0, -1),
    ],
    label_model=label_model,
    pixel_model=pixel_model,
    quadtree_model=quadtree_model,
    affinity_func=region_model.log_affinity_boundary_and_depth,
    alpha=_ALPHA,
    gibbs_num_iterations=10,
    affinity_params={
        "beta": _BETA,
        "eta": _ETA,
    },
    oa_log_filepath=os.path.join(ESTIMATION_RESULTS_DIR, "label", "oa_log.txt"),
    est_label_diff_dir=os.path.join(ESTIMATION_RESULTS_DIR, "label", "diff"),
    enable_logq_cache=True,
)
