# ./model/region/area_size.py
import os
import numpy as np
from PIL import Image
from scipy import ndimage
import sys


def label_prior(region: set[tuple[int,int]], param: dict):
    max_depth = param["max_depth"]
    categorical_probs_list = param["categorical_probs_list"]
    region_size = len(region)
    for d in range(max_depth+1):
        min_size = 4 ** (max_depth - d)
        max_size = 4 ** (max_depth - d + 1) - 1
        if min_size <= region_size <= max_size:
            probs = categorical_probs_list[d]
            return probs

def param_est(train_label_dir: str, label_set: list[int], label_num: int):
    """
    学習データから領域面積カテゴリごとのラベル出現確率を推定する。
    """
    label_files = [f for f in os.listdir(train_label_dir) if f.endswith('.png')]
    if not label_files:
        print(f"エラー: {train_label_dir} に学習用のラベル画像が見つかりません。", file=sys.stderr)
        return {}

    print(f"{len(label_files)} 組の学習データを読み込み中...")
    label_data = []
    for filename in label_files:
        try:
            label_path = os.path.join(train_label_dir, filename)
            label_data.append(np.array(Image.open(label_path)))
        except Exception as e:
            print(f"警告: {filename} の読み込みに失敗しました: {e}", file=sys.stderr)

    if not label_data:
        print("エラー: 有効な学習データがありません。", file=sys.stderr)
        return {}

    print("データ読み込み完了。")

    # max_depthを最初の画像から取得
    height, width = label_data[0].shape
    max_depth = int(np.log2(width))

    # 各depthカテゴリごとのラベル出現回数をカウントする配列
    # counts[depth][label_index]
    counts = [[0] * label_num for _ in range(max_depth + 1)]
    label_to_index = {label: i for i, label in enumerate(label_set)}

    print("領域を抽出し、面積カテゴリごとにラベルをカウント中...")
    for lbl_img in label_data:
        for label_val in label_set:
            if label_val not in lbl_img:
                continue

            # 特定のラベルの領域マスクを作成
            mask = (lbl_img == label_val)
            # 連結成分を検出
            labeled_mask, num_regions = ndimage.label(mask, structure=np.ones((3,3)))

            if num_regions > 0:
                # 各領域の面積を計算
                region_areas = ndimage.sum(mask, labeled_mask, range(1, num_regions + 1))
                
                for area in region_areas:
                    # 面積がどのdepthカテゴリに属するか判定
                    for d in range(max_depth + 1):
                        min_size = 4 ** (max_depth - d)
                        max_size = 4 ** (max_depth - d + 1) - 1
                        if min_size <= area <= max_size:
                            label_idx = label_to_index[label_val]
                            counts[d][label_idx] += 1
                            break
    
    print("カテゴリカル確率を計算中...")
    categorical_probs_list = []
    for d in range(max_depth + 1):
        total_count = sum(counts[d])
        if total_count == 0:
            # そのdepthの領域が一つもなければ、一様確率を割り当てる
            probs = [1.0 / label_num] * label_num
        else:
            probs = [count / total_count for count in counts[d]]
        categorical_probs_list.append(probs)

    output_params = {
        "max_depth": max_depth,
        "categorical_probs_list": categorical_probs_list
    }
    
    print("面積ベースのラベル事前確率のパラメータ推定が完了しました。")
    return output_params