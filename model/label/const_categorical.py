# ./model/label/const_categorical.py
import os
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
import utils

def label_prior(region: set[tuple[int, int]], param: dict) -> np.ndarray:
    """
    指定されたカテゴリカルパラメータに従うラベル事前分布。
    """
    probs = param.get("probs")
    if probs is None:
        label_num = param.get("label_num", 3)
        return np.full(label_num, 1.0 / label_num, dtype=float)
    return np.asarray(probs, dtype=float)

def param_est(
    train_label_dir: str,
    label_set: list[int],
    label_num: int,
    image_size: int = 128,
    feature_names: list[str] | None = None,
    min_region_area: int = 32,
) -> dict:
    """
    学習データのラベル画像内の連結領域数をラベルごとにカウントし、
    最尤推定（MLE）に基づいてカテゴリカル確率を推定する。
    """
    filenames = utils.get_image_files(train_label_dir)
    counts = {label: 0 for label in label_set}
    connectivity_8 = ndimage.generate_binary_structure(2, 2)
    
    for filename in filenames:
        path = os.path.join(train_label_dir, filename)
        try:
            lbl_img = utils.load_image(path)
            for x in np.unique(lbl_img):
                label_val = int(x)
                if label_val not in counts:
                    continue
                mask = (lbl_img == label_val)
                labeled_mask, num_regions = ndimage.label(mask, structure=connectivity_8)
                for r in range(1, num_regions + 1):
                    region_mask = (labeled_mask == r)
                    if int(np.count_nonzero(region_mask)) >= int(min_region_area):
                        counts[label_val] += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}", file=sys.stderr)
            
    total_regions = sum(counts.values())
    if total_regions == 0:
        print("Warning: No valid regions found for categorical param estimation. Using uniform fallback.", file=sys.stderr)
        probs = [1.0 / label_num] * label_num
    else:
        sorted_labels = sorted(label_set)
        probs = [float(counts[label]) / total_regions for label in sorted_labels]
        
    print(f"Estimated categorical probabilities (MLE): {probs}")
    
    return {
        "label_num": label_num,
        "label_set": label_set,
        "probs": probs,
        "image_size": image_size,
    }
