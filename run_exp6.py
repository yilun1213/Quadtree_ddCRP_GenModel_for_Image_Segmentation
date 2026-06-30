# run_exp6.py
import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "c:/Users/yitsu/Desktop/Quadtree_ddCRP_GenModel_for_Image_Segmentation/experiments/exp.6"
PYTHON_PATH = r"c:\Users\yitsu\Desktop\Quadtree_ddCRP_GenModel_for_Image_Segmentation\.venv\Scripts\python.exe"

# 1. Parameter settings for experiments
gs_params = {
    1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    2: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.0],
    3: [1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
}

eta_params = {
    1: 0.0,
    2: 5.0,
    3: 10.0,
}

# Label and Pixel parameters (shared)
# True label prior uses 1/|X| = 1/3 for all classes
label_param_data = {
    "label_num": 3,
    "label_set": [0, 1, 2],
    "label_value_set": [0, 128, 255],
    "probs": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
}

pixel_param_data = {
    "label_set": [0, 1, 2],
    "channels": 3,
    "mean": [
        [200.0, 50.0, 50.0],
        [50.0, 200.0, 50.0],
        [50.0, 50.0, 200.0]
    ],
    "variance": [
        [
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 20.0]
        ],
        [
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 20.0]
        ],
        [
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 20.0]
        ]
    ]
}

def create_config_gen_content(dataset_dir, eta):
    return f"""# config_gen.py for exp.6
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any, Dict
import json
import os
import model.quadtree.depth_dependent_model as quadtree_model
import model.label.const_categorical as label_model
import model.pixel.normal_dist as pixel_model
import model.region.affinity as affinity_module

@dataclass(frozen=True)
class DataSavingConfig:
    dir: str
    quadtree_num: int
    regions_per_quadtree: int = 1
    labels_per_region: int = 1
    images_per_label: int = 1

    @property
    def total_images(self) -> int:
        return (
            self.quadtree_num
            * self.regions_per_quadtree
            * self.labels_per_region
            * self.images_per_label
        )

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
    affinity_func: Callable
    alpha: float
    affinity_params: Dict[str, Any]
    label_config: LabelModelConfig
    pixel_config: PixelModelConfig

def load_config() -> Config:
    base_dir = "{dataset_dir}"
    param_dir = os.path.join(base_dir, "true_param")
    
    with open(os.path.join(param_dir, "branch_probs.json"), "r") as f:
        branch_data = json.load(f)
        branch_probs = branch_data["branch_probs"]
        
    with open(os.path.join(param_dir, "label_param.json"), "r") as f:
        label_data = json.load(f)
        
    with open(os.path.join(param_dir, "pixel_param.json"), "r") as f:
        norm_param_data = json.load(f)

    label_num = int(label_data["label_num"])
    label_set = [int(v) for v in label_data.get("label_set", list(range(label_num)))]
    label_value_set = [int(v) for v in label_data.get("label_value_set", label_set)]

    train_config = DataSavingConfig(
        dir=os.path.join(base_dir, "train_data"),
        quadtree_num=50,
        regions_per_quadtree=1,
        labels_per_region=1,
        images_per_label=1,
    )

    test_config = DataSavingConfig(
        dir=os.path.join(base_dir, "test_data"),
        quadtree_num=10,
        regions_per_quadtree=1,
        labels_per_region=1,
        images_per_label=1,
    )

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
        param={{
            "label_num": label_num,
            "probs": label_data["probs"],
        }},
    )

    pixel_config = PixelModelConfig(
        model=pixel_model,
        param=norm_param_data
    )
    
    affinity_function = affinity_module.log_affinity_depth_only
    affinity_function_params = {{
        "eta": {eta},
    }}

    return Config(
        train=train_config,
        test=test_config,
        param_dir=param_dir,
        label_param_filename="label_param.json",
        pixel_param_filename="pixel_param.json",
        branch_probs_filename="branch_probs.json",
        seed=1,
        quadtree_config=quadtree_config,
        affinity_func=affinity_function,
        alpha=1.0,
        affinity_params=affinity_function_params,
        label_config=label_config,
        pixel_config=pixel_config,
    )
"""

def create_config_seg_content(dataset_dir, eta):
    return f"""# config_seg_eta{eta}.py for exp.6
from dataclasses import dataclass
from typing import Callable, Any, Dict
import os
import model.label.const_categorical as label_model
import model.pixel.normal_dist as pixel_model
import model.quadtree.depth_dependent_model as quadtree_model
import model.region.affinity as region_model

@dataclass(frozen=True)
class Config:
    train_image_dir: str
    train_label_dir: str
    train_label_vis_dir: str
    test_image_dir: str
    test_label_dir: str
    test_label_vis_dir: str
    out_param_dir: str
    est_label_folder_path: str
    est_label_dirname: str
    est_label_visualize_dirname: str
    est_region_dirname: str
    est_quadtree_dirname: str
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

DATASET_DIR = "{dataset_dir}"
_ALPHA = 1.0
_ETA = {eta}
_PARAM_SUFFIX = f"_beta0.0_eta{{_ETA}}_alpha{{_ALPHA}}"

config = Config(
    train_image_dir=os.path.join(DATASET_DIR, "train_data/images"),
    train_label_dir=os.path.join(DATASET_DIR, "train_data/labels"),
    train_label_vis_dir=os.path.join(DATASET_DIR, "train_data/labels/visualize"),
    test_image_dir=os.path.join(DATASET_DIR, "test_data/images"),
    test_label_dir=os.path.join(DATASET_DIR, "test_data/labels"),
    test_label_vis_dir=os.path.join(DATASET_DIR, "test_data/labels/visualize"),
    out_param_dir=os.path.join(DATASET_DIR, "estimated_param"),
    est_label_folder_path=os.path.join(DATASET_DIR, "estimation_results_eta" + str(_ETA)),
    est_label_dirname="label",
    est_label_visualize_dirname="visualize",
    est_region_dirname="region",
    est_quadtree_dirname="quadtree",
    label_param_filename="label_param.json",
    pixel_param_filename="pixel_param.json",
    branch_probs_filename="branch_probs.json",
    label_feature_names=["log_area", "log_perimeter", "circularity"],
    label_min_region_area=32,
    offset=[
        (-2, -2), (-2, -1), (-2, 0),
        (-1, -2), (-1, -1), (-1, 0),
        (0, -2), (0, -1)
    ],
    label_model=label_model,
    pixel_model=pixel_model,
    quadtree_model=quadtree_model,
    affinity_func=region_model.log_affinity_depth_only,
    alpha=_ALPHA,
    gibbs_num_iterations=10,
    affinity_params={{
        "eta": _ETA,
    }},
    oa_log_filepath=os.path.join(DATASET_DIR, "estimation_results" + _PARAM_SUFFIX, "label", "oa_log.txt"),
    oa_error_csv_path=os.path.join(DATASET_DIR, "estimation_results_eta" + str(_ETA), "oa_error_trend.csv"),
    est_label_diff_dir=os.path.join(DATASET_DIR, "estimation_results" + _PARAM_SUFFIX, "label", "diff"),
    enable_logq_cache=True,
)
"""

def setup_experiment_folders():
    print("Setting up folders and configs...")
    for i in range(1, 4):
        for j in range(1, 4):
            folder_name = f"exp.6.{i}.{j}"
            exp_dir = os.path.join(BASE_DIR, folder_name)
            true_param_dir = os.path.join(exp_dir, "true_param")
            os.makedirs(true_param_dir, exist_ok=True)
            
            # Save parameter JSONs
            branch_probs = gs_params[i]
            with open(os.path.join(true_param_dir, "branch_probs.json"), "w") as f:
                json.dump({"branch_probs": branch_probs}, f, indent=4)
                
            with open(os.path.join(true_param_dir, "label_param.json"), "w") as f:
                json.dump(label_param_data, f, indent=4)
                
            with open(os.path.join(true_param_dir, "pixel_param.json"), "w") as f:
                json.dump(pixel_param_data, f, indent=4)
                
            # Create configs
            eta = eta_params[j]
            with open(os.path.join(exp_dir, "config_gen.py"), "w", encoding="utf-8") as f:
                f.write(create_config_gen_content(exp_dir.replace("\\", "/"), eta))
                
            for eta_val in [0.0, 5.0, 10.0]:
                with open(os.path.join(exp_dir, f"config_seg_eta{eta_val}.py"), "w", encoding="utf-8") as f:
                    f.write(create_config_seg_content(exp_dir.replace("\\", "/"), eta_val))
                    
    print("Folder and config generation complete.")

def run_experiment(i, j):
    folder_name = f"exp.6.{i}.{j}"
    exp_dir = os.path.join(BASE_DIR, folder_name).replace("\\", "/")
    print(f"\n==================== Starting {folder_name} ====================")
    
    # Ensure all directories exist to avoid cloud-sync deletion race conditions
    for split in ("train_data", "test_data"):
        for sub in ("images", "labels", "labels/visualize", "regions", "quadtrees"):
            os.makedirs(os.path.join(exp_dir, split, sub), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "estimated_param"), exist_ok=True)
    
    # 1. Data generation
    print(f"Running data generation for {folder_name}...")
    config_gen_path = f"{exp_dir}/config_gen.py"
    subprocess.run([PYTHON_PATH, "generate.py", "--config", config_gen_path], check=True)
    
    # 2. Parameter estimation (train)
    print(f"Running train.py for {folder_name}...")
    config_train_path = f"{exp_dir}/config_seg_eta0.0.py" # train doesn't care about eta
    subprocess.run([PYTHON_PATH, "train.py", "--config", config_train_path], check=True)
    
    # 3. Segmentation (Gibbs and ICM)
    for eta_val in [0.0, 5.0, 10.0]:
        print(f"Running predict_gibbs.py for {folder_name} (eta={eta_val})...")
        config_seg_path = f"{exp_dir}/config_seg_eta{eta_val}.py"
        subprocess.run([PYTHON_PATH, "predict_gibbs.py", "--config", config_seg_path], check=True)
        
        print(f"Running predict_icm.py for {folder_name} (eta={eta_val})...")
        subprocess.run([PYTHON_PATH, "predict_icm.py", "--config", config_seg_path], check=True)

def aggregate_and_plot():
    print("\n==================== Aggregating Results and Plotting ====================")
    results = {} # {(i, j): {alg: {eta: [mean_oa_iter1, ...]}}}
    
    for i in range(1, 4):
        for j in range(1, 4):
            folder_name = f"exp.6.{i}.{j}"
            exp_dir = os.path.join(BASE_DIR, folder_name).replace("\\", "/")
            results[(i, j)] = {"gibbs": {}, "icm": {}}
            
            for eta_val in [0.0, 5.0, 10.0]:
                gibbs_csv = f"{exp_dir}/estimation_results_gibbs_eta{eta_val}/oa_error_trend.csv"
                icm_csv = f"{exp_dir}/estimation_results_icm_eta{eta_val}/oa_error_trend.csv"
                
                for alg, csv_path in [("gibbs", gibbs_csv), ("icm", icm_csv)]:
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        iter_cols = [c for c in df.columns if c != 'image']
                        mean_errors = df[iter_cols].mean().values
                        mean_oas = 1.0 - mean_errors
                        results[(i, j)][alg][eta_val] = mean_oas.tolist()
                    else:
                        print(f"Warning: CSV not found: {csv_path}")
                        results[(i, j)][alg][eta_val] = []
                        
    # Plotting for each combination of (i, j)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
    gs_titles = {
        1: "Complete Tree (GS-1)",
        2: "Gradual Split (GS-2)",
        3: "Shallow Tree (GS-3)"
    }
    
    for i in range(1, 4):
        for j in range(1, 4):
            ax = axes[i-1, j-1]
            data = results[(i, j)]
            
            colors = {0.0: 'r', 5.0: 'g', 10.0: 'b'}
            
            for eta_val in [0.0, 5.0, 10.0]:
                oas_gibbs = data["gibbs"].get(eta_val, [])
                if oas_gibbs:
                    iters = list(range(1, len(oas_gibbs) + 1))
                    ax.plot(iters, oas_gibbs, label=f"Gibbs η={eta_val}", linestyle='-', marker='o', color=colors[eta_val])
                    
                oas_icm = data["icm"].get(eta_val, [])
                if oas_icm:
                    iters = list(range(1, len(oas_icm) + 1))
                    ax.plot(iters, oas_icm, label=f"ICM η={eta_val}", linestyle='--', marker='x', color=colors[eta_val])
            
            ax.set_title(f"Gen: {gs_titles[i]}\nTrue η={eta_params[j]}")
            ax.grid(True)
            if i == 3:
                ax.set_xlabel("Iteration")
            if j == 1:
                ax.set_ylabel("Mean Overall Accuracy")
            
            if i == 1 and j == 1:
                ax.legend(loc='lower right', fontsize='small')
                
    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "overall_accuracy_trend.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved accuracy comparison plot to: {plot_path}")
    
    # Save a JSON file with all compiled results
    compiled_results_path = os.path.join(BASE_DIR, "compiled_results.json")
    serialized_results = {f"{k[0]},{k[1]}": v for k, v in results.items()}
    with open(compiled_results_path, "w") as f:
        json.dump(serialized_results, f, indent=4)
    print(f"Saved compiled JSON results to: {compiled_results_path}")

def main():
    setup_experiment_folders()
    
    for i in range(1, 4):
        for j in range(1, 4):
            run_experiment(i, j)
            
    aggregate_and_plot()

if __name__ == "__main__":
    main()
