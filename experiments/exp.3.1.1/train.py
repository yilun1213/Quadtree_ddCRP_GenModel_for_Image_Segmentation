
# train.py
import os
import sys
import json
import numpy as np
from PIL import Image

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import utils  # type: ignore[reportMissingImports]
from utils import harmonize_lists  # type: ignore[reportMissingImports]
import estimate_label

Config = estimate_label.Config
config = estimate_label.config

def train(config: Config):
    train_image_dir = config.train_image_dir
    train_label_dir = config.train_label_dir
    os.makedirs(config.out_param_dir, exist_ok=True)
    out_label_param_path = os.path.join(config.out_param_dir, config.label_param_filename)
    out_pixel_param_path = os.path.join(config.out_param_dir, config.pixel_param_filename)
    out_branch_probs_path = os.path.join(config.out_param_dir, config.branch_probs_filename)
    offset = config.offset

    # ラベル関連の情報を算出して保存
    print("Calculating and saving label information")
    label_files = utils.get_image_files(train_label_dir)
    if not label_files:
        raise RuntimeError(f"No label files found in train_label_dir: {train_label_dir}")
    vis_label_dir = config.train_label_vis_dir
    vis_label_files = utils.get_image_files(vis_label_dir) if os.path.exists(vis_label_dir) else []

    # 可視化ラベル画像が無ければ自動生成（train/test両方で同一マップを使用）
    if not vis_label_files:
        label_to_color_map = utils.build_label_value_map([train_label_dir, config.test_label_dir])
        utils.generate_visualize_labels(train_label_dir, vis_label_dir, label_to_color_map)
        test_vis_dir = config.test_label_vis_dir
        utils.generate_visualize_labels(config.test_label_dir, test_vis_dir, label_to_color_map)
        vis_label_files = utils.get_image_files(vis_label_dir)
    
    # ファイル名の整合性を確認（可視化画像がある場合のみ）
    if vis_label_files:
        filenames = utils.harmonize_lists(label_files, vis_label_files)
    else:
        filenames = label_files

    # ラベルIDと可視化色の対応マップを作成
    label_to_color_map = {}
    
    # vis_label_filesが無い場合にも対応
    if not vis_label_files:
        print("Note: Visualize label images not found. Using raw label values as color mapping.")
    elif not filenames:
        print("Note: Visualize label images have no overlap with labels. Using raw label values as color mapping.")
        filenames = label_files

    for filename in filenames:
        label_path = os.path.join(train_label_dir, filename)
        label_array = utils.load_image(label_path)
            
        if os.path.exists(os.path.join(vis_label_dir, filename)):
            vis_label_path = os.path.join(vis_label_dir, filename)
            # 可視化画像はRGB(3ch)で保存されている可能性があるため、グレースケール(1ch)に変換して読み込む
            vis_img = Image.fromarray(utils.load_image(vis_label_path))
            vis_label_array = np.array(vis_img.convert('L'))
        else:
            # vis画像が無いなら、labelそのものを値として使う
            vis_label_array = label_array

        # ユニークなラベルとその色を対応付ける
        unique_pairs = np.unique(np.stack([label_array, vis_label_array]).reshape(2, -1), axis=1)
        for label_id, color_val in unique_pairs.T:
            if label_id not in label_to_color_map:
                label_to_color_map[int(label_id)] = int(color_val)
    
    # 対応マップからソート済みのlabel_setと、それに対応するlabel_value_setを生成
    label_set = sorted(label_to_color_map.keys())
    label_value_set = [label_to_color_map[label_id] for label_id in label_set]
    label_num = len(label_set)
    
    label_info = {
        "label_num": label_num,
        "label_set": label_set,
        "label_value_set": label_value_set,
    }
    
    with open(out_label_param_path, 'w') as f:
        json.dump(label_info, f, indent=4)
    print(f"Saved label info to {out_label_param_path}")

    print("Estimating Parameters of Branching Probability")
    config.quadtree_model.param_est(train_label_dir, out_branch_probs_path)

    print("Estimating Parameters of Probability Function on Label")
    # 画像サイズをラベル画像から自動取得
    first_label_file = os.path.join(train_label_dir, label_files[0])
    first_label_img = utils.load_image(first_label_file)
    image_size = first_label_img.shape[0]  # 正方形画像を想定（height）
    print(f"Detected image size: {image_size}x{image_size}")
    
    # geom_glm.param_estを呼び出し、結果を取得
    estimated_label_params = config.label_model.param_est(
        train_label_dir=train_label_dir,
        label_set=label_set,
        label_num=label_num,
        image_size=image_size,  # 自動検出した画像サイズを渡す
        feature_names=config.label_feature_names,
        min_region_area=config.label_min_region_area,
    )
    # 既存のlabel_infoに推定結果をマージ
    label_info.update(estimated_label_params)
    with open(out_label_param_path, 'w') as f:
        json.dump(label_info, f, indent=4)

    print("Estimating Parameters of Probability Function on Pixel")
    config.pixel_model.param_est(
        train_image_dir, train_label_dir, out_pixel_param_path, offset)

    config.pixel_model.add_label_set(out_pixel_param_path, out_label_param_path)


def estimate_test_data(config: Config) -> None:
    estimate_label.process_test_images(config)



if __name__ == '__main__':
    train(config)
    estimate_test_data(config)
