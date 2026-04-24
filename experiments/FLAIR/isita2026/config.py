# config.py

from dataclasses import dataclass
from typing import Callable, Any, Dict
import os

# モデルは config 側で import して切替（generate.py と同様の思想）
import model.label.geom_features_norm_dist as label_model
# import model.label.geom_features_norm_dist as label_model
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
    oa_error_csv_path: str
    est_label_diff_dir: str
    enable_logq_cache: bool

# DATASET_DIR = "./generated_data1"
# DATASET_DIR = "./syn_data"
DATASET_DIR = "."

_BETA = 30.0   # 共有境界長の重み
_ETA  = 30.0   # ノードサイズの階層性を制御する重み
_ALPHA = 0.00000001  # ddCRPの新領域生成確率
_PARAM_SUFFIX = f"_beta{_BETA}_eta{_ETA}_alpha{_ALPHA}"

config = Config(
    train_image_dir=os.path.join(DATASET_DIR, "train_data/images"),
    train_label_dir=os.path.join(DATASET_DIR, "train_data/labels"),
    train_label_vis_dir=os.path.join(DATASET_DIR, "train_data/labels/visualize"),
    test_image_dir=os.path.join(DATASET_DIR, "test_data/images"),
    test_label_dir=os.path.join(DATASET_DIR, "test_data/labels"),
    test_label_vis_dir=os.path.join(DATASET_DIR, "test_data/labels/visualize"),
    out_param_dir=os.path.join(DATASET_DIR, "estimated_param" + _PARAM_SUFFIX),
    est_label_folder_path=os.path.join(DATASET_DIR, "estimation_results" + _PARAM_SUFFIX),
    est_label_dirname="label",
    est_label_visualize_dirname="visualize",
    est_region_dirname="region",
    est_quadtree_dirname="quadtree",
    label_param_filename="label_param.json",
    pixel_param_filename="pixel_param.json",
    branch_probs_filename="branch_probs.json",
    label_feature_names=[
        "log_area",
        "log_perimeter",
        "circularity",
    ],
    label_min_region_area=32,
    offset = [
        (-2, -2), (-2, -1), (-2, 0),
        (-1, -2), (-1, -1), (-1, 0),
        (0, -2), (0, -1)
        ],
    label_model=label_model,
    pixel_model=pixel_model,
    quadtree_model=quadtree_model,
    affinity_func=region_model.log_affinity_boundary_and_depth,
    alpha=_ALPHA,  # ddCRPパラメータ（新領域生成確率）
    gibbs_num_iterations=5,
    affinity_params={
        "beta": _BETA,
        "eta": _ETA,
    },
    oa_log_filepath=os.path.join(DATASET_DIR, "estimation_results" + _PARAM_SUFFIX, "label", "oa_log.txt"),
    oa_error_csv_path=os.path.join(DATASET_DIR, "estimation_results" + _PARAM_SUFFIX, "oa_error_trend.csv"),
    est_label_diff_dir=os.path.join(DATASET_DIR, "estimation_results" + _PARAM_SUFFIX, "label", "diff"),
    enable_logq_cache=True,
)
