# config_gen.py
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any, Dict
import json
import os
import model.quadtree.depth_dependent_model as quadtree_model
import model.label.geom_features_logistic as label_model
import model.pixel.normal_dist as pixel_model
import model.region.affinity as affinity_module


@dataclass(frozen=True)
class DataSavingConfig:
    num: int            # 生成枚数
    dir: str            # 出力先ディレクトリ


@dataclass(frozen=True)
class LabelModelConfig:
    label_num: int
    label_set: List[int]
    label_value_set: List[int]
    model: Callable
    param: Any


@dataclass(frozen=True)
class QuadtreeModelConfig:
    model: Callable
    max_depth: int
    branch_probs: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class PixelModelConfig:
    model: Callable
    param: Any


@dataclass(frozen=True)
class Config:
    train: DataSavingConfig
    test: DataSavingConfig
    param_dir: str
    label_param_filename: str
    pixel_param_filename: str
    branch_probs_filename: str
    seed: Optional[int]
    quadtree_config: QuadtreeModelConfig
    affinity_func: Callable  # 親和度関数 f(s, s', adjacency_dict, **affinity_params)
    alpha: float  # ddCRP パラメータ（新領域生成確率を制御）
    affinity_params: Dict[str, Any]  # 親和度関数に渡すパラメータ（beta, eta など）
    label_config: LabelModelConfig
    pixel_config: PixelModelConfig


# 設定
DIR = "./generated_data2"
PARAM_DIRNAME = "true_param"
LABEL_PARAM_FILENAME = "label_param.json"
PIXEL_PARAM_FILENAME = "pixel_param.json"
BRANCH_PROBS_FILENAME = "branch_probs.json"
LABEL_FEATURE_NAMES = [
    "log_area",
    "log_perimeter",
    "circularity",
]

def load_config() -> Config:
    param_dir = os.path.join(DIR, PARAM_DIRNAME)
    
    # Load branch_probs
    with open(os.path.join(param_dir, BRANCH_PROBS_FILENAME), "r") as f:
        branch_data = json.load(f)
        branch_probs = branch_data["branch_probs"]
        
    # Load label_param
    with open(os.path.join(param_dir, LABEL_PARAM_FILENAME), "r") as f:
        label_data = json.load(f)
        
    # Load normal-distribution parameters
    with open(os.path.join(param_dir, PIXEL_PARAM_FILENAME), "r") as f:
        norm_param_data = json.load(f)

    label_num = int(label_data["label_num"])
    label_set = [int(v) for v in label_data.get("label_set", list(range(label_num))) ]
    label_value_set = [int(v) for v in label_data.get("label_value_set", label_set)]

    train_config = DataSavingConfig(
        num=100, dir=os.path.join(DIR, "train_data"))

    test_config = DataSavingConfig(
        num=2,  dir=os.path.join(DIR, "test_data"))

    quadtree_config = QuadtreeModelConfig(
        model=quadtree_model,
        max_depth=len(branch_probs)-1,
        branch_probs=branch_probs,
    )

    label_config = LabelModelConfig(
        label_num=label_num,
        label_set=label_set,
        label_value_set=label_value_set,
        model=label_model,
        param={
            "image_size": int(2**quadtree_config.max_depth),
            "feature_names": label_data.get("feature_names", LABEL_FEATURE_NAMES),
            "weights": label_data["weights"],
            "bias": label_data["bias"],
        },
    )

    pixel_config = PixelModelConfig(
        model=pixel_model,
        param=norm_param_data
    )
    
    # 親和度関数の設定（論文 2.4 節の例 2）
    # 利用可能な親和度関数：
    # - affinity_boundary_and_depth: 共有境界線の長さと深さ差に基づく（推奨）
    # - affinity_boundary_depth_and_large_pair: 上記 + 大ノード同士ボーナス
    # - affinity_target_shallow_exp: リンク先が浅いほど（大きいほど）優遇する単純モデル
    # - affinity_boundary_only: 共有境界線の長さのみに基づく
    # - affinity_constant: 一定の親和度（テスト用）
    affinity_function = affinity_module.affinity_boundary_and_depth
    affinity_function_params = {
        "beta": 0.5,  # 共有境界長 B(s, s') の重み
        "eta": 1.5,    # depth(s) - depth(s') の重み
    }

    return Config(
        train=train_config,
        test=test_config,
        param_dir=param_dir,
        label_param_filename=LABEL_PARAM_FILENAME,
        pixel_param_filename=PIXEL_PARAM_FILENAME,
        branch_probs_filename=BRANCH_PROBS_FILENAME,
        seed=1,
        quadtree_config=quadtree_config,
        affinity_func=affinity_function,
        alpha=0.001,  # ddCRP パラメータ（小さくすると結合しやすく、新領域が減る）
        affinity_params=affinity_function_params,
        label_config=label_config,
        pixel_config=pixel_config,
    )


