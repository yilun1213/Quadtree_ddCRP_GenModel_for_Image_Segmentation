import os
import numpy as np
from PIL import Image
try:
    import tifffile
except ImportError:
    tifffile = None

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def is_image_file(filename):
    return filename.lower().endswith(SUPPORTED_EXTENSIONS)

def load_image(path):
    """
    画像を読み込み、NumPy配列として返す。
    TIF形式の場合はtifffileを使用し、それ以外はPILを使用する。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff']:
        if tifffile:
            return tifffile.imread(path)
        else:
            # tifffileがない場合のフォールバック
            return np.array(Image.open(path))
    else:
        return np.array(Image.open(path))

def get_image_files(directory):
    """
    指定されたディレクトリ内の画像ファイル名リストを返す（ソート済み）。
    """
    return sorted([f for f in os.listdir(directory) if is_image_file(f)])

def _normalize_label_array(label_array: np.ndarray) -> np.ndarray:
    if label_array.ndim == 3:
        # ラベルが多チャンネルの場合は先頭チャンネルを使用
        return label_array[..., 0]
    return label_array

def build_label_value_map(label_dirs: list[str]) -> dict[int, int]:
    """
    複数フォルダのラベル値から、可視化用の値マップを作成する。
    """
    label_set = set()
    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            continue
        for filename in get_image_files(label_dir):
            label_path = os.path.join(label_dir, filename)
            label_array = _normalize_label_array(load_image(label_path))
            for val in np.unique(label_array):
                label_set.add(int(val))

    sorted_labels = sorted(label_set)
    if not sorted_labels:
        return {}

    if len(sorted_labels) == 1:
        values = [0]
    else:
        values = np.linspace(0, 255, len(sorted_labels)).round().astype(int).tolist()
        if len(set(values)) != len(values):
            step = 255 // max(1, len(sorted_labels) - 1)
            values = [min(255, i * step) for i in range(len(sorted_labels))]

    return {label: value for label, value in zip(sorted_labels, values)}

def _assign_new_values(label_to_color_map: dict[int, int], new_labels: list[int]) -> None:
    used = set(label_to_color_map.values())
    available = [v for v in range(256) if v not in used]
    for label in new_labels:
        if not available:
            label_to_color_map[label] = 255
        else:
            label_to_color_map[label] = available.pop(0)

def generate_visualize_labels(label_dir: str, visualize_dir: str, label_to_color_map: dict[int, int] | None = None) -> dict[int, int]:
    """
    label_dir 内のラベル画像から可視化画像を作成して visualize_dir に保存する。
    label_to_color_map を指定すると、そのマップに従う。
    """
    os.makedirs(visualize_dir, exist_ok=True)
    label_files = get_image_files(label_dir)
    if not label_files:
        return label_to_color_map or {}

    if label_to_color_map is None:
        label_to_color_map = build_label_value_map([label_dir])

    # 未知ラベルがあれば割り当てを追加
    current_labels = set(label_to_color_map.keys())
    found_labels = set()
    for filename in label_files:
        label_path = os.path.join(label_dir, filename)
        label_array = _normalize_label_array(load_image(label_path))
        found_labels.update(int(v) for v in np.unique(label_array))
    new_labels = sorted(list(found_labels - current_labels))
    if new_labels:
        _assign_new_values(label_to_color_map, new_labels)

    # 可視化画像を保存
    for filename in label_files:
        label_path = os.path.join(label_dir, filename)
        label_array = _normalize_label_array(load_image(label_path))
        vis_array = np.zeros(label_array.shape, dtype=np.uint8)
        for label_val, color_val in label_to_color_map.items():
            vis_array[label_array == label_val] = color_val
        out_path = os.path.join(visualize_dir, filename)
        Image.fromarray(vis_array).save(out_path)

    return label_to_color_map

def harmonize_lists(list_a, list_b):
    """
    2つのリストを比較し、異なる要素を警告表示した後、
    共通の要素のみを持つリストを返します。

    :param list_a: 比較するリストA
    :param list_b: 比較するリストB
    :return: 共通要素のみを含むリスト
    """
    # 集合（set）に変換して重複を排除し、集合演算を可能にする
    set_a = set(list_a)
    set_b = set(list_b)

    # 1. 異なる要素（対称差）を検出
    # symmetric_difference(): AにもBにも存在するが、共通ではない要素の集合
    diff_elements = set_a.symmetric_difference(set_b)

    # 2. 異なる要素があれば警告として表示
    if diff_elements:
        print("⚠️ 警告: 画像データフォルダとラベルデータフォルダの間で以下の異なるフォルダが見つかりました。")

        # リストAにしかない要素 (A - B)
        only_in_a = set_a - set_b
        if only_in_a:
            print(f"  - 画像データフォルダにのみ存在するファイル: {list(only_in_a)}")

        # リストBにしかない要素 (B - A)
        only_in_b = set_b - set_a
        if only_in_b:
            print(f"  - ラベルデータフォルダにのみ存在するファイル: {list(only_in_b)}")
    else:
        print("[OK] 学習データに不備無し")

    # 3. AとBの共通要素のみを抽出（共通部分）
    # intersection(): AとBの両方に存在する要素の集合
    common_elements = set_a.intersection(set_b)

    # 共通要素をリストに戻して返す
    return list(common_elements)

def copy_tree_structure(source_node, dest_node):
    """
    source_node の木構造（is_leaf）を dest_node に再帰的にコピーする。
    """
    if source_node is None or dest_node is None:
        return

    dest_node.is_leaf = source_node.is_leaf

    if not source_node.is_leaf:
        copy_tree_structure(source_node.ul_node, dest_node.ul_node)
        copy_tree_structure(source_node.ur_node, dest_node.ur_node)
        copy_tree_structure(source_node.ll_node, dest_node.ll_node)
        copy_tree_structure(source_node.lr_node, dest_node.lr_node)