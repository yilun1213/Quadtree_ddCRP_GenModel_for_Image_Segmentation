# estimate_label.py
"""
論文のセグメンテーションアルゴリズムの実装
Bayes最適解を粗近似するギブスサンプリング手法
"""

import os
import sys
import json
import hashlib
from dataclasses import dataclass
from typing import Callable, Any, Dict
import numpy as np
import time
from PIL import Image
from collections import defaultdict

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import model.label.geom_features_logistic as label_model
import model.pixel.normal_dist as pixel_model
import model.quadtree.depth_dependent_model as quadtree_model
import model.region.affinity as region_model
from model.quadtree.node import Node
from model.quadtree.depth_dependent_model import make_tree, label_ndarray
import utils


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
    label_feature_names=["log_area", "log_perimeter", "circularity"],
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
    gibbs_num_iterations=5,
    affinity_params={
        "beta": _BETA,
        "eta": _ETA,
    },
    oa_log_filepath=os.path.join(ESTIMATION_RESULTS_DIR, "label", "oa_log.txt"),
    est_label_diff_dir=os.path.join(ESTIMATION_RESULTS_DIR, "label", "diff"),
    enable_logq_cache=True,
)


LOG_EPS = 1e-300
MODEL_CFG: Config = config
PIXEL_LIKELIHOOD_WARN_COUNT = 0
PIXEL_LIKELIHOOD_WARN_MAX = 12


def log_prob_Y_given_X(region_tuple, label, img_array, theta):
    # ローカル設定で選択した pixel_model の実装を使う
    return MODEL_CFG.pixel_model.log_prob_Y_given_X(region_tuple, label, img_array, theta)


def label_prior(region, label_param):
    # ローカル設定で選択した label_model の実装を使う
    return MODEL_CFG.label_model.label_prior(region, label_param)


def _safe_log(x: float) -> float:
    return float(np.log(max(float(x), LOG_EPS)))


def _stable_hash_from_jsonable(data: dict) -> str:
    packed = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _hash_image_array(image: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(image.shape).encode("ascii"))
    h.update(str(image.dtype).encode("ascii"))
    h.update(image.tobytes(order="C"))
    return h.hexdigest()


def _module_fingerprint(module_obj) -> dict | None:
    if module_obj is None:
        return None
    fp = {
        "name": getattr(module_obj, "__name__", str(module_obj)),
    }
    module_file = getattr(module_obj, "__file__", None)
    if module_file and os.path.exists(module_file):
        stat = os.stat(module_file)
        fp["file"] = os.path.abspath(module_file).replace("\\", "/")
        fp["mtime_ns"] = int(stat.st_mtime_ns)
        fp["size"] = int(stat.st_size)
    return fp


def _callable_fingerprint(func_obj) -> dict | None:
    if func_obj is None:
        return None
    fp = {
        "module": getattr(func_obj, "__module__", None),
        "qualname": getattr(func_obj, "__qualname__", getattr(func_obj, "__name__", str(func_obj))),
    }
    code_obj = getattr(func_obj, "__code__", None)
    if code_obj is not None:
        fp["firstlineno"] = int(getattr(code_obj, "co_firstlineno", -1))
        fp["argcount"] = int(getattr(code_obj, "co_argcount", -1))
    return fp


def _build_runtime_signature(cfg: Config | None) -> dict:
    if cfg is None:
        return {"cfg": None}
    affinity_params = getattr(cfg, "affinity_params", {})
    return {
        "label_model": _module_fingerprint(getattr(cfg, "label_model", None)),
        "pixel_model": _module_fingerprint(getattr(cfg, "pixel_model", None)),
        "quadtree_model": _module_fingerprint(getattr(cfg, "quadtree_model", None)),
        "affinity_func": _callable_fingerprint(getattr(cfg, "affinity_func", None)),
        "affinity_params_hash": _stable_hash_from_jsonable(affinity_params if isinstance(affinity_params, dict) else {"value": str(affinity_params)}),
    }


def _get_calc_log_dir(cfg: Config) -> str:
    # cfg の test_image_dir を優先し、実験フォルダごとにキャッシュを分離する
    return os.path.join(os.path.dirname(os.path.dirname(cfg.test_image_dir)), ".calc_log")


def _get_step1_cache_path(cfg: Config, image_stem: str | None, image_size: int) -> str:
    stem = image_stem if image_stem else f"image_{image_size}"
    filename = f"{stem}_logp_cache.npz"
    return os.path.join(_get_calc_log_dir(cfg), filename)


def _build_step1_cache_meta(
    image: np.ndarray,
    label_param: dict,
    pixel_param: dict,
    image_size: int,
    num_nodes: int,
    cfg: Config | None = None,
) -> dict:
    return {
        "cache_version": 1,
        "image_size": int(image_size),
        "num_nodes": int(num_nodes),
        "image_shape": [int(v) for v in image.shape],
        "image_hash": _hash_image_array(image),
        "label_param_hash": _stable_hash_from_jsonable(label_param),
        "pixel_param_hash": _stable_hash_from_jsonable(pixel_param),
        "runtime_signature_hash": _stable_hash_from_jsonable(_build_runtime_signature(cfg)),
    }


def _save_step1_logp_cache(cache_path: str, log_p_y_cache: dict, meta: dict) -> None:
    keys = np.array(list(log_p_y_cache.keys()), dtype=np.int32)
    values = np.array([float(log_p_y_cache[k]) for k in log_p_y_cache.keys()], dtype=np.float64)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        node_keys=keys,
        log_p_y=values,
        meta_json=np.array(json.dumps(meta, sort_keys=True)),
    )


def _load_step1_logp_cache_if_valid(cache_path: str, expected_meta: dict) -> tuple[dict | None, str]:
    if not os.path.exists(cache_path):
        return None, "cache file not found"

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            node_keys = data["node_keys"]
            log_p_y = data["log_p_y"]
            meta_json = str(data["meta_json"])
            cached_meta = json.loads(meta_json)
    except Exception as e:
        return None, f"failed to load cache ({type(e).__name__}: {e})"

    required_fields = [
        "cache_version",
        "image_size",
        "num_nodes",
        "image_shape",
        "image_hash",
        "label_param_hash",
        "pixel_param_hash",
        "runtime_signature_hash",
    ]
    for field in required_fields:
        if cached_meta.get(field) != expected_meta.get(field):
            return None, f"metadata mismatch: {field}"

    if node_keys.ndim != 2 or node_keys.shape[1] != 4:
        return None, "invalid node_keys shape"
    if log_p_y.ndim != 1 or node_keys.shape[0] != log_p_y.shape[0]:
        return None, "invalid cache array lengths"

    cache = {}
    for idx in range(node_keys.shape[0]):
        key = tuple(int(v) for v in node_keys[idx].tolist())
        cache[key] = float(log_p_y[idx])

    return cache, "ok"


def _get_node_likelihood_cache_path(cfg: Config, image_stem: str | None, image_size: int) -> str:
    stem = image_stem if image_stem else f"image_{image_size}"
    filename = f"{stem}_node_likelihood_cache.npz"
    return os.path.join(_get_calc_log_dir(cfg), filename)


def _build_node_likelihood_cache_meta(
    image: np.ndarray,
    label_param: dict,
    pixel_param: dict,
    num_leaf_nodes: int,
    num_labels: int,
    cfg: Config | None = None,
) -> dict:
    return {
        "cache_version": 1,
        "num_leaf_nodes": int(num_leaf_nodes),
        "num_labels": int(num_labels),
        "image_shape": [int(v) for v in image.shape],
        "image_hash": _hash_image_array(image),
        "label_param_hash": _stable_hash_from_jsonable(label_param),
        "pixel_param_hash": _stable_hash_from_jsonable(pixel_param),
        "runtime_signature_hash": _stable_hash_from_jsonable(_build_runtime_signature(cfg)),
    }


def _save_node_likelihood_cache(
    cache_path: str,
    node_likelihood_cache: dict,
    leaf_nodes: list,
    num_labels: int,
    meta: dict,
) -> None:
    key_to_idx = {
        (n.upper_edge, n.left_edge, n.size, n.depth): i
        for i, n in enumerate(leaf_nodes)
    }
    node_keys = np.array(
        [[n.upper_edge, n.left_edge, n.size, n.depth] for n in leaf_nodes],
        dtype=np.int32,
    )
    likelihoods = np.full((len(leaf_nodes), num_labels), -1e10, dtype=np.float64)
    for node, label_dict in node_likelihood_cache.items():
        key = (node.upper_edge, node.left_edge, node.size, node.depth)
        idx = key_to_idx.get(key)
        if idx is not None:
            for label_idx, val in label_dict.items():
                likelihoods[idx, int(label_idx)] = float(val)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        node_keys=node_keys,
        likelihoods=likelihoods,
        meta_json=np.array(json.dumps(meta, sort_keys=True)),
    )


def _load_node_likelihood_cache_if_valid(
    cache_path: str,
    expected_meta: dict,
    leaf_nodes: list,
) -> tuple[dict | None, str]:
    if not os.path.exists(cache_path):
        return None, "cache file not found"
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            node_keys = data["node_keys"]
            likelihoods = data["likelihoods"]
            meta_json = str(data["meta_json"])
            cached_meta = json.loads(meta_json)
    except Exception as e:
        return None, f"failed to load cache ({type(e).__name__}: {e})"

    required_fields = [
        "cache_version", "num_leaf_nodes", "num_labels",
        "image_shape", "image_hash", "label_param_hash", "pixel_param_hash", "runtime_signature_hash",
    ]
    for field in required_fields:
        if cached_meta.get(field) != expected_meta.get(field):
            return None, f"metadata mismatch: {field}"

    if node_keys.ndim != 2 or node_keys.shape[1] != 4:
        return None, "invalid node_keys shape"
    if likelihoods.ndim != 2:
        return None, "invalid likelihoods shape"
    if node_keys.shape[0] != len(leaf_nodes):
        return None, f"leaf node count mismatch: expected {len(leaf_nodes)}, got {node_keys.shape[0]}"

    num_labels = likelihoods.shape[1]
    stored_key_to_idx = {
        tuple(int(v) for v in node_keys[i].tolist()): i
        for i in range(node_keys.shape[0])
    }
    cache = {}
    for node in leaf_nodes:
        key = (node.upper_edge, node.left_edge, node.size, node.depth)
        idx = stored_key_to_idx.get(key)
        if idx is None:
            return None, f"node key not found in cache: {key}"
        cache[node] = {l: float(likelihoods[idx, l]) for l in range(num_labels)}
    return cache, "ok"


def build_node_likelihood_cache(
    leaf_nodes: list,
    image: np.ndarray,
    label_param: dict,
    pixel_param: dict,
    cfg: Config | None = None,
    image_stem: str | None = None,
) -> dict:
    """各葉ノードの per-label 対数尤度キャッシュを構築する。
    cfg.enable_logq_cache == True のとき .calc_log フォルダを読み書きする。"""
    global PIXEL_LIKELIHOOD_WARN_COUNT
    num_labels = label_param.get("label_num", 2)
    image_size = int(image.shape[0])

    use_cache = bool(cfg is not None and getattr(cfg, "enable_logq_cache", False))
    cache_path = None
    if use_cache and cfg is not None:
        cache_path = _get_node_likelihood_cache_path(cfg, image_stem, image_size)
        expected_meta = _build_node_likelihood_cache_meta(
            image=image,
            label_param=label_param,
            pixel_param=pixel_param,
            num_leaf_nodes=len(leaf_nodes),
            num_labels=num_labels,
            cfg=cfg,
        )
        loaded, reason = _load_node_likelihood_cache_if_valid(cache_path, expected_meta, leaf_nodes)
        if loaded is not None:
            print(f"  [cache] node_likelihood_cache hit: loaded {len(loaded)} nodes from {cache_path}")
            return loaded
        else:
            print(f"  [cache] node_likelihood_cache miss: {reason}")

    print(f"Pre-computing node likelihood cache for {len(leaf_nodes)} leaf nodes × {num_labels} labels...")
    print(f"  (Total {len(leaf_nodes) * num_labels} likelihood computations)")
    cache_start_time = time.time()
    node_likelihood_cache: dict = {}

    for node_idx, node in enumerate(leaf_nodes):
        node_likelihood_cache[node] = {}
        for label_idx in range(num_labels):
            try:
                log_likelihood = log_prob_Y_given_X((node,), label_idx, image, pixel_param)
                node_likelihood_cache[node][label_idx] = log_likelihood
            except Exception as e:
                if PIXEL_LIKELIHOOD_WARN_COUNT < PIXEL_LIKELIHOOD_WARN_MAX:
                    print(
                        "[WARN] precompute log_prob_Y_given_X failed: "
                        f"node=({node.upper_edge},{node.left_edge},size={node.size},depth={node.depth}) "
                        f"label={label_idx} error={type(e).__name__}: {e}"
                    )
                elif PIXEL_LIKELIHOOD_WARN_COUNT == PIXEL_LIKELIHOOD_WARN_MAX:
                    print("[WARN] Further precompute log_prob_Y_given_X errors suppressed...")
                PIXEL_LIKELIHOOD_WARN_COUNT += 1
                node_likelihood_cache[node][label_idx] = -1e10

        if (node_idx + 1) % max(1, len(leaf_nodes) // 5) == 0:
            elapsed = time.time() - cache_start_time
            print(f"  - Cached {node_idx + 1}/{len(leaf_nodes)} nodes (elapsed: {elapsed:.2f}s)...")

    cache_time = time.time() - cache_start_time
    print(f"Node likelihood cache computed: {len(node_likelihood_cache)} nodes × {num_labels} labels (took {cache_time:.2f}s)")

    if use_cache and cfg is not None and cache_path is not None:
        expected_meta = _build_node_likelihood_cache_meta(
            image=image,
            label_param=label_param,
            pixel_param=pixel_param,
            num_leaf_nodes=len(leaf_nodes),
            num_labels=num_labels,
            cfg=cfg,
        )
        _save_node_likelihood_cache(cache_path, node_likelihood_cache, leaf_nodes, num_labels, expected_meta)
        print(f"  [cache] node_likelihood_cache saved to {cache_path}")

    return node_likelihood_cache


def _logsumexp(log_values: np.ndarray) -> float:
    if log_values.size == 0:
        return -np.inf
    max_log = np.max(log_values)
    if not np.isfinite(max_log):
        return -np.inf
    return float(max_log + np.log(np.sum(np.exp(log_values - max_log))))


def _expected_pixel_channels(pixel_param: dict) -> int:
    """pixel_param から期待チャネル数を推定する。"""
    channels = pixel_param.get("channels")
    if channels is not None:
        return int(channels)

    mean = pixel_param.get("mean", [])
    if isinstance(mean, list) and mean and isinstance(mean[0], list):
        return int(len(mean[0]))
    return 1


def _adapt_image_channels_for_pixel_model(image: np.ndarray, pixel_param: dict) -> np.ndarray:
    """
    画像チャネル数を pixel model の想定チャネル数へ合わせる。

    - expected=1 かつ image が3ch: 先頭チャネルを使用
    - expected>1 かつ image が2D: expected 回複製
    - それ以外の不整合は例外
    """
    expected = _expected_pixel_channels(pixel_param)

    if image.ndim == 2:
        if expected == 1:
            return image
        return np.stack([image] * expected, axis=-1)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image ndim={image.ndim}; expected 2 or 3")

    actual = image.shape[2]
    if actual == expected:
        return image
    if expected == 1:
        # 学習が1chモデルの場合はグレースケールとして扱う
        return image[..., 0]
    if actual == 1 and expected > 1:
        return np.repeat(image, expected, axis=2)

    raise ValueError(
        f"Pixel model channel mismatch: image has {actual} channels, "
        f"but pixel_param expects {expected} channels"
    )


class QuadTreeNode:
    """四分木のノード管理用クラス"""
    def __init__(self, upper_edge, left_edge, size, depth, parent=None):
        self.upper_edge = upper_edge
        self.left_edge = left_edge
        self.size = size
        self.depth = depth
        self.parent = parent
        self.children = []
        self.is_leaf = (depth == 127)  # 最大深度に達したら葉


def _node_key(node: Node) -> tuple[int, int, int, int]:
    return (node.upper_edge, node.left_edge, node.size, node.depth)


def _collect_nodes_by_depth(root: Node):
    """木を1回だけ走査し、深さごとにノードを集約する。"""
    stack = [root]
    nodes_by_depth = defaultdict(list)
    all_nodes = []

    while stack:
        node = stack.pop()
        all_nodes.append(node)
        nodes_by_depth[node.depth].append(node)

        if not node.is_leaf:
            stack.extend([node.ul_node, node.ur_node, node.ll_node, node.lr_node])

    return all_nodes, nodes_by_depth


def _build_leaf_adjacency_from_index_map(leaf_nodes, height, width):
    """v2 と同様に葉インデックス画像を使って隣接葉を高速に抽出する。"""
    leaf_index_map = np.full((height, width), -1, dtype=np.int32)

    for idx, node in enumerate(leaf_nodes):
        leaf_index_map[node.upper_edge:node.lower_edge, node.left_edge:node.right_edge] = idx

    adjacency_dict = defaultdict(list)
    for idx, node in enumerate(leaf_nodes):
        neighbors = set()

        if node.upper_edge > 0:
            top = np.unique(leaf_index_map[node.upper_edge - 1, node.left_edge:node.right_edge])
            neighbors.update(int(v) for v in top if v >= 0 and int(v) != idx)

        if node.lower_edge < height:
            bottom = np.unique(leaf_index_map[node.lower_edge, node.left_edge:node.right_edge])
            neighbors.update(int(v) for v in bottom if v >= 0 and int(v) != idx)

        if node.left_edge > 0:
            left = np.unique(leaf_index_map[node.upper_edge:node.lower_edge, node.left_edge - 1])
            neighbors.update(int(v) for v in left if v >= 0 and int(v) != idx)

        if node.right_edge < width:
            right = np.unique(leaf_index_map[node.upper_edge:node.lower_edge, node.right_edge])
            neighbors.update(int(v) for v in right if v >= 0 and int(v) != idx)

        adjacency_dict[node] = [leaf_nodes[n_idx] for n_idx in neighbors]

    return dict(adjacency_dict)


def _regions_to_region_dict(regions):
    """領域ノード集合を generate.py と同様のピクセル集合辞書に変換する。"""
    region_dict = {}
    for region_id, region_nodes in enumerate(regions, start=1):
        pixels = set()
        for node in region_nodes:
            for i in range(node.upper_edge, node.lower_edge):
                for j in range(node.left_edge, node.right_edge):
                    pixels.add((i, j))
        region_dict[region_id] = pixels
    return region_dict


def save_region_growing_image_from_regions(image_size: int, regions, filename: str) -> None:
    """generate.py と同じ見た目で領域分割図を保存する。"""
    region_dict = _regions_to_region_dict(regions)
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    region_colors = {}
    for region_id in region_dict.keys():
        region_colors[region_id] = np.random.randint(50, 255, size=3, dtype=np.uint8)

    for region_id, pixels in region_dict.items():
        color = region_colors[region_id]
        for i, j in pixels:
            image[i, j] = color

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Image.fromarray(image).save(filename)


def save_quadtree_image_from_leaves(leaf_nodes, image_size: int, filename: str) -> None:
    """generate.py と同じ見た目で四分木葉ノード画像を保存する。"""
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    color_map = {}

    for leaf in leaf_nodes:
        color_key = (leaf.upper_edge, leaf.left_edge)
        if color_key not in color_map:
            color_map[color_key] = np.random.randint(50, 255, size=3, dtype=np.uint8)

    for leaf in leaf_nodes:
        color_key = (leaf.upper_edge, leaf.left_edge)
        color = color_map[color_key]
        image[leaf.upper_edge:leaf.lower_edge, leaf.left_edge:leaf.right_edge] = color

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Image.fromarray(image).save(filename)


def load_label_visualization_map(cfg: Config) -> dict[int, int]:
    """train.py が保存した label_set / label_value_set から可視化値マップを復元する。"""
    label_param_path = os.path.join(cfg.out_param_dir, cfg.label_param_filename)
    if not os.path.exists(label_param_path):
        return {}

    with open(label_param_path, 'r') as f:
        label_info = json.load(f)

    label_set = label_info.get("label_set", [])
    label_value_set = label_info.get("label_value_set", [])
    return {
        int(label): int(value)
        for label, value in zip(label_set, label_value_set)
    }


def build_visualize_label_image(label_array: np.ndarray, label_to_value_map: dict[int, int]) -> np.ndarray:
    """推定ラベル画像を学習済みの可視化値マップに従って grayscale PNG 用画像へ変換する。"""
    vis_array = np.zeros(label_array.shape, dtype=np.uint8)

    if not label_to_value_map:
        unique_labels = sorted(int(v) for v in np.unique(label_array))
        label_to_value_map = utils.build_label_value_map([])
        if unique_labels:
            if len(unique_labels) == 1:
                label_to_value_map = {unique_labels[0]: 0}
            else:
                values = np.linspace(0, 255, len(unique_labels)).round().astype(np.uint8)
                label_to_value_map = {
                    label: int(value)
                    for label, value in zip(unique_labels, values)
                }

    for label_val, color_val in label_to_value_map.items():
        vis_array[label_array == label_val] = color_val

    return vis_array


def compute_overall_accuracy(X_est: np.ndarray, X_true: np.ndarray) -> float:
    """Overall Accuracy (OA) を計算する。OA = 正解ピクセル数 / 全ピクセル数"""
    return float(np.mean(X_est == X_true))


def save_diff_image(X_est: np.ndarray, X_true: np.ndarray, filepath: str) -> None:
    """
    推定ラベルと正解ラベルの差分画像を保存する。
    正解ピクセル: 青 (0, 0, 255)
    不正解ピクセル: 赤 (255, 0, 0)
    """
    H, W = X_est.shape
    diff_image = np.zeros((H, W, 3), dtype=np.uint8)
    correct = (X_est == X_true)
    diff_image[correct] = [0, 0, 255]    # 青: 正解
    diff_image[~correct] = [255, 0, 0]   # 赤: 不正解
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    Image.fromarray(diff_image).save(filepath)


def _estimate_labels_from_regions(
    regions, H: int, W: int, image: np.ndarray,
    label_param_with_size: dict, pixel_param: dict, node_likelihood_cache: dict
) -> np.ndarray:
    """各領域にMAP推定でラベルを割り当て、ラベル画像 (H, W) を返す。"""
    X_est = np.zeros((H, W), dtype=np.int32)
    for region_nodes in regions:
        log_marginal_terms = compute_log_marginal_terms(
            region_nodes, image, label_param_with_size, pixel_param, node_likelihood_cache
        )
        label_est = int(np.argmax(log_marginal_terms))
        for node in region_nodes:
            X_est[node.upper_edge:node.lower_edge, node.left_edge:node.right_edge] = label_est
    return X_est


def _compute_log_p_Y_given_node(node: Node, image: np.ndarray, label_param: dict, pixel_param: dict) -> float:
    """
    ノード s に対する q の葉項を計算する。
    論文の式: (1/|X|) Σ_{x∈X} Π_{(i,j)∈s} p(y_{(i,j)} | x; θ_x)
    対数で返す: log((1/|X|) Σ_x Π p(y|x)) = logsumexp_x(log p(Y_s|x)) - log|X|

    Args:
        node: ノード
        image: RGB画像配列
        label_param: ラベルモデルのパラメータ
        pixel_param: ピクセルモデルのパラメータ

    Returns:
        float: log((1/|X|) Σ_x Π p(y_{(i,j)} | x; θ_x)) の値
    """
    # ラベル数を取得（均一事前確率 1/|X| を使うので label_prior は不要）
    num_labels = label_param.get("label_num", None)
    if num_labels is None:
        num_labels = len(label_param.get("label_set", []))

    region_tuple = (node,)  # log_prob_Y_given_X は tuple[Node, ...] を期待

    # 各ラベルに対する尤度 log p(Y_s | x; θ_x) を計算
    log_likelihoods = np.zeros(num_labels)
    global PIXEL_LIKELIHOOD_WARN_COUNT
    for label_idx in range(num_labels):
        try:
            log_likelihoods[label_idx] = log_prob_Y_given_X(region_tuple, label_idx, image, pixel_param)
        except Exception as e:
            if PIXEL_LIKELIHOOD_WARN_COUNT < PIXEL_LIKELIHOOD_WARN_MAX:
                print(
                    "[WARN] log_prob_Y_given_X failed: "
                    f"node=({node.upper_edge},{node.left_edge},size={node.size},depth={node.depth}) "
                    f"label={label_idx} error={type(e).__name__}: {e}"
                )
            elif PIXEL_LIKELIHOOD_WARN_COUNT == PIXEL_LIKELIHOOD_WARN_MAX:
                print("[WARN] Further log_prob_Y_given_X errors suppressed...")
            PIXEL_LIKELIHOOD_WARN_COUNT += 1
            log_likelihoods[label_idx] = -1e10

    # log((1/|X|) Σ_x Π p(y|x)) = logsumexp_x(log_likelihoods) - log(|X|)
    return _logsumexp(log_likelihoods) - np.log(num_labels)
        

def compute_q_recursive(node, image, branch_probs, label_param, pixel_param):
    """
    論文の再帰関数 q(Y_s | g_s) を計算する
    log q(Y_s | g_s) を返す（対数確率）。
    
    Args:
        node: 現在のノード
        image: RGB画像配列
        branch_probs: 学習済み分岐確率リスト
        label_param: ラベルモデルのパラメータ
        pixel_param: ピクセルモデルのパラメータ
    
    Returns:
        float: log q(Y_s | g_s) の値
    """
    depth = node.depth
    g_s = branch_probs[depth] if depth < len(branch_probs) else 0.0
    
    # 最大深度の場合
    if node.is_leaf or depth >= len(branch_probs) - 1:
        return _compute_log_p_Y_given_node(node, image, label_param, pixel_param)

    log_p_y = _compute_log_p_Y_given_node(node, image, label_param, pixel_param)

    # 子ノードの log q を再帰的に計算
    log_q_children = 0.0
    for child in [node.ul_node, node.ur_node, node.ll_node, node.lr_node]:
        log_q_children += compute_q_recursive(child, image, branch_probs, label_param, pixel_param)

    if g_s >= 1.0:
        return log_q_children
    if g_s <= 0.0:
        return log_p_y

    term_leaf = _safe_log(1.0 - g_s) + log_p_y
    term_split = _safe_log(g_s) + log_q_children
    return float(np.logaddexp(term_leaf, term_split))


def compute_g_given_Y(node, image, branch_probs, q_values):
    """
    論文の g_{s|Y} を計算する
    g_{s|Y} = (g_s * Π_{s'∈Ch(s)} q(Y_{s'} | g_{s'})) / q(Y_s | g_s)
    
    Args:
        node: 現在のノード
        image: RGB画像配列
        branch_probs: 学習済み分岐確率リスト
        q_values: 各ノードの log q 値を格納した辞書
    
    Returns:
        float: g_{s|Y} の値
    """
    depth = node.depth
    g_s = branch_probs[depth] if depth < len(branch_probs) else 0.0
    
    # 最大深度の場合
    if node.is_leaf or depth >= len(branch_probs) - 1:
        return g_s
    
    # log 分子: log g_s + Σ log q(Y_{s'})
    if g_s <= 0.0:
        return 0.0
    log_numerator = _safe_log(g_s)
    for child in [node.ul_node, node.ur_node, node.ll_node, node.lr_node]:
        child_key = (child.upper_edge, child.left_edge, child.size, child.depth)
        log_numerator += q_values.get(child_key, -np.inf)

    # log 分母: log q(Y_s | g_s)
    node_key = (node.upper_edge, node.left_edge, node.size, node.depth)
    log_denominator = q_values.get(node_key, -np.inf)

    if not np.isfinite(log_denominator):
        return g_s

    log_g_given_y = log_numerator - log_denominator
    g_given_y = float(np.exp(log_g_given_y))
    return min(max(g_given_y, 0.0), 1.0)


def compute_map_tree_flags(root, image, branch_probs, label_param, pixel_param, cfg: Config | None = None, image_stem: str | None = None):
    """
    論文のアルゴリズムに基づき、事後確率最大四分木 \hat{T} を計算する
    
    手順:
    1. 全ノードのq値を計算
    2. 全ノードのg_{s|Y}を計算
    3. ψ(s)とf_sを再帰的に計算
    4. f_sに基づいて葉ノードを決定
    
    Args:
        root: 完全四分木のルートノード
        image: RGB画像配列
        branch_probs: 学習済み分岐確率リスト
        label_param: ラベルモデルのパラメータ
        pixel_param: ピクセルモデルのパラメータ
    
    Returns:
        dict: {node: flag} フラグが1なら分岐、0なら葉
    """
    all_nodes, _nodes_by_depth = _collect_nodes_by_depth(root)
    total_nodes = len(all_nodes)
    progress_step = max(1, total_nodes // 10)

    # Step 1: 各ノードの log p(Y_s) と log q(Y_s | g_s) を計算（v2 流儀）
    step1_start = time.time()
    print("  [Step 1/3] Computing log q values (v2 style)...")
    log_p_y_cache = {}

    use_cache = bool(cfg is not None and getattr(cfg, "enable_logq_cache", False))
    cache_path = None
    cache_loaded = False
    if use_cache and cfg is not None:
        cache_path = _get_step1_cache_path(cfg, image_stem, root.size)
        expected_meta = _build_step1_cache_meta(
            image=image,
            label_param=label_param,
            pixel_param=pixel_param,
            image_size=root.size,
            num_nodes=total_nodes,
            cfg=cfg,
        )
        loaded_cache, reason = _load_step1_logp_cache_if_valid(cache_path, expected_meta)
        if loaded_cache is not None:
            log_p_y_cache = loaded_cache
            cache_loaded = True
            print(f"    - [cache] hit: loaded {len(log_p_y_cache)} node log p(Y_s) values from {cache_path}")
        else:
            print(f"    - [cache] miss: {reason}")

    counter = [0]

    def calc_logq(node):
        key = _node_key(node)
        if key not in log_p_y_cache:
            log_p_y_cache[key] = _compute_log_p_Y_given_node(node, image, label_param, pixel_param)

        if node.is_leaf:
            node.logq_Ys = log_p_y_cache[key]
        else:
            calc_logq(node.ul_node)
            calc_logq(node.ur_node)
            calc_logq(node.ll_node)
            calc_logq(node.lr_node)

            g_s = branch_probs[node.depth] if node.depth < len(branch_probs) else 0.0
            log_q_split_children = (
                node.ul_node.logq_Ys
                + node.ur_node.logq_Ys
                + node.ll_node.logq_Ys
                + node.lr_node.logq_Ys
            )
            log_q_leaf_y = log_p_y_cache[key]

            if g_s >= 1.0:
                node.logq_Ys = log_q_split_children
            elif g_s <= 0.0:
                node.logq_Ys = log_q_leaf_y
            else:
                term_leaf = _safe_log(1.0 - g_s) + log_q_leaf_y
                term_split = _safe_log(g_s) + log_q_split_children
                node.logq_Ys = np.logaddexp(term_leaf, term_split)

        counter[0] += 1
        if counter[0] % progress_step == 0 or counter[0] == total_nodes:
            print(f"    - Computed logq for {counter[0]}/{total_nodes} nodes...")

    calc_logq(root)

    if use_cache and cfg is not None and cache_path is not None:
        # キャッシュを読み込んでいなければ保存する（不整合時は上書き）
        if len(log_p_y_cache) == total_nodes:
            expected_meta = _build_step1_cache_meta(
                image=image,
                label_param=label_param,
                pixel_param=pixel_param,
                image_size=root.size,
                num_nodes=total_nodes,
                cfg=cfg,
            )
            if not cache_loaded:
                _save_step1_logp_cache(cache_path, log_p_y_cache, expected_meta)
                print(f"    - [cache] saved Step1 cache to {cache_path}")

    step1_time = time.time() - step1_start
    print(f"    ✓ Step 1 completed: {len(log_p_y_cache)} nodes (took {step1_time:.2f}s)")

    # Step 2: 各ノードで leaf/split の事後確率を比較し MAP 木を決定（v2 流儀）
    step2_start = time.time()
    print("  [Step 2/3] Determining MAP tree structure (v2 style)...")
    counter[0] = 0

    # デバッグ用: 深さごとの統計を収集
    # depth -> {"g_s", "g_given_y_list", "delta_list", "margin_list", "n_leaf", "n_split", "nodes"}
    depth_debug: dict = defaultdict(lambda: {
        "g_s": None,
        "g_given_y_list": [],
        "delta_list": [],          # log_q_split_children - log_q_leaf_y
        "margin_list": [],         # log_psi_split - log_psi_leaf (正なら split 勝ち)
        "n_leaf": 0,
        "n_split": 0,
        "nodes": [],               # 個別ノード情報（浅い深さのみ記録）
    })

    def determine_map_tree(node):
        """
        論文のψ再帰に基づき post-order でMAP木フラグを決定する。
        ψ(s) = max{ 1-g_{s|Y},  g_{s|Y} * Π_{s'∈Ch(s)} ψ(s') }
        f_s = 1 iff 1-g_{s|Y} < g_{s|Y} * Π ψ(s')
        ここで g_{s|Y} = g_s * Π q(Y_{ch}) / q(Y_s)
        """
        key = _node_key(node)
        log_q_leaf_y = log_p_y_cache[key]  # g_{s|Y} 計算のために利用

        if node.is_leaf:
            node.is_leaf_map = True
            # 最大深度では分割不可なので ψ=1（log ψ = 0）
            node.log_psi = 0.0
            counter[0] += 1
            if counter[0] % progress_step == 0 or counter[0] == total_nodes:
                print(f"    - Determined MAP flag for {counter[0]}/{total_nodes} nodes...")
            return

        # Post-order: 子ノードの ψ を先に計算する
        determine_map_tree(node.ul_node)
        determine_map_tree(node.ur_node)
        determine_map_tree(node.ll_node)
        determine_map_tree(node.lr_node)

        log_q_total = node.logq_Ys
        log_q_split_children = (
            node.ul_node.logq_Ys
            + node.ur_node.logq_Ys
            + node.ll_node.logq_Ys
            + node.lr_node.logq_Ys
        )

        # g_{s|Y} = g_s * Π_{ch} q(Y_ch) / q(Y_s) を計算
        g_s = branch_probs[node.depth] if node.depth < len(branch_probs) else 0.0
        if g_s <= 0.0:
            g_given_y = 0.0
        elif g_s >= 1.0:
            g_given_y = 1.0
        else:
            log_g_given_y = _safe_log(g_s) + log_q_split_children - log_q_total
            g_given_y = min(max(float(np.exp(log_g_given_y)), 0.0), 1.0)

        # 子の ψ 値の対数和: Σ log ψ(s')
        log_psi_children = (
            node.ul_node.log_psi
            + node.ur_node.log_psi
            + node.ll_node.log_psi
            + node.lr_node.log_psi
        )

        # ψ(s) = max{ 1-g_{s|Y}, g_{s|Y}*Π ψ(ch) } を比較
        if g_given_y <= 0.0:
            log_psi_leaf = 0.0
            log_psi_split = -np.inf
        elif g_given_y >= 1.0:
            log_psi_leaf = -np.inf
            log_psi_split = log_psi_children
        else:
            log_psi_leaf = _safe_log(1.0 - g_given_y)
            log_psi_split = _safe_log(g_given_y) + log_psi_children

        if log_psi_leaf >= log_psi_split:
            node.is_leaf_map = True
            node.log_psi = log_psi_leaf
        else:
            node.is_leaf_map = False
            node.log_psi = log_psi_split

        # ---- デバッグ統計の収集 ----
        d = node.depth
        dd = depth_debug[d]
        dd["g_s"] = g_s
        delta = log_q_split_children - log_q_leaf_y
        dd["delta_list"].append(delta)
        dd["g_given_y_list"].append(g_given_y)
        margin = (log_psi_split - log_psi_leaf) if np.isfinite(log_psi_split) and np.isfinite(log_psi_leaf) else (
            float('inf') if not np.isfinite(log_psi_leaf) else float('-inf')
        )
        dd["margin_list"].append(margin)
        if node.is_leaf_map:
            dd["n_leaf"] += 1
        else:
            dd["n_split"] += 1
        # 浅い深さ (depth <= 3) は個別ノード情報を記録
        if d <= 3:
            dd["nodes"].append({
                "pos": (node.upper_edge, node.left_edge, node.size),
                "g_s": g_s,
                "g_given_y": g_given_y,
                "log_q_leaf_y": log_q_leaf_y,
                "log_q_split_children": log_q_split_children,
                "delta": delta,
                "log_psi_leaf": log_psi_leaf,
                "log_psi_split": log_psi_split,
                "margin": margin,
                "decision": "LEAF" if node.is_leaf_map else "SPLIT",
            })
        # ----------------------------

        counter[0] += 1
        if counter[0] % progress_step == 0 or counter[0] == total_nodes:
            print(f"    - Determined MAP flag for {counter[0]}/{total_nodes} nodes...")

    determine_map_tree(root)
    step2_time = time.time() - step2_start
    print(f"    ✓ Step 2 completed (took {step2_time:.2f}s)")

    # ---- 深さごとのデバッグサマリーを出力 ----
    print()
    print("  ╔══════════════════════════════════════════════════════════════════════════╗")
    print("  ║                 [DEBUG] MAP Tree Statistics per Depth                  ║")
    print("  ╠════╦═══════╦════════╦════════╦══════════════╦══════════════╦══════╦═════╣")
    print("  ║ d  ║  g_s  ║ g|Y mn ║ g|Y mx ║  delta mean  ║ margin mean  ║ LEAF ║SPLT ║")
    print("  ╠════╬═══════╬════════╬════════╬══════════════╬══════════════╬══════╬═════╣")
    for d in sorted(depth_debug.keys()):
        dd = depth_debug[d]
        g_s_val = dd["g_s"] if dd["g_s"] is not None else float('nan')
        gy_list = dd["g_given_y_list"]
        delta_list = dd["delta_list"]
        margin_list = [m for m in dd["margin_list"] if np.isfinite(m)]
        g_mn = min(gy_list) if gy_list else float('nan')
        g_mx = max(gy_list) if gy_list else float('nan')
        d_mean = float(np.mean(delta_list)) if delta_list else float('nan')
        m_mean = float(np.mean(margin_list)) if margin_list else float('nan')
        n_leaf = dd["n_leaf"]
        n_spl  = dd["n_split"]
        print(f"  ║ {d:2d} ║ {g_s_val:5.3f} ║ {g_mn:6.3f} ║ {g_mx:6.3f} ║ {d_mean:+12.2f} ║ {m_mean:+12.2f} ║ {n_leaf:4d} ║{n_spl:4d} ║")
    print("  ╚════╩═══════╩════════╩════════╩══════════════╩══════════════╩══════╩═════╝")
    print("   ※ delta = log q(children) - log q(leaf_y)  （正＝分割で尤度改善）")
    print("   ※ margin = log ψ_split - log ψ_leaf  （正＝SPLIT 勝ち、負＝LEAF 勝ち）")
    print()

    # 浅い深さの個別ノード詳細
    for d in sorted(k for k in depth_debug.keys() if k <= 3):
        dd = depth_debug[d]
        print(f"  [DEBUG depth={d}] Individual nodes (size={2**(int(np.log2(root.size))-d)}×{2**(int(np.log2(root.size))-d)}):")
        for ni in dd["nodes"]:
            ue, le, sz = ni["pos"]
            print(f"    Node(upper={ue:3d}, left={le:3d}, sz={sz:3d}): "
                  f"g_s={ni['g_s']:.4f}  g|Y={ni['g_given_y']:.4f}  "
                  f"logQ_leaf={ni['log_q_leaf_y']:.1f}  logQ_split={ni['log_q_split_children']:.1f}  "
                  f"Δ={ni['delta']:+.2f}  "
                  f"ψ_leaf={ni['log_psi_leaf']:.1f}  ψ_split={ni['log_psi_split']:.1f}  "
                  f"margin={ni['margin']:+.2f}  → {ni['decision']}")
        print()
    # -----------------------------------------

    # Step 3: 既存の construct_map_tree 互換の flags 辞書へ変換
    step3_start = time.time()
    print("  [Step 3/3] Converting MAP tree to flags...")
    flags = {}
    counter[0] = 0
    depth_flag_counts: dict[int, dict] = defaultdict(lambda: {"leaf": 0, "split": 0})
    stack = [root]
    while stack:
        node = stack.pop()
        key = _node_key(node)

        is_leaf_map = getattr(node, "is_leaf_map", True)
        if node.is_leaf or is_leaf_map:
            flags[key] = 0
            depth_flag_counts[node.depth]["leaf"] += 1
        else:
            flags[key] = 1
            depth_flag_counts[node.depth]["split"] += 1
            stack.extend([node.ul_node, node.ur_node, node.ll_node, node.lr_node])

        counter[0] += 1
        if counter[0] % progress_step == 0 or counter[0] == total_nodes:
            print(f"    - Converted {counter[0]}/{total_nodes} nodes...")

    step3_time = time.time() - step3_start
    n_split_total = sum(v["split"] for v in depth_flag_counts.values())
    n_leaf_total  = sum(v["leaf"]  for v in depth_flag_counts.values())
    print(f"    ✓ Step 3 completed: {len(flags)} flags (split={n_split_total}, leaf={n_leaf_total}) (took {step3_time:.2f}s)")
    print("    [DEBUG] MAP tree node counts per depth (only nodes in MAP tree):")
    for d in sorted(depth_flag_counts.keys()):
        fc = depth_flag_counts[d]
        print(f"      depth={d}: split={fc['split']}  leaf={fc['leaf']}")

    total_map_time = step1_time + step2_time + step3_time
    print(f"  Overall MAP tree construction: {total_map_time:.2f}s")

    return flags


def construct_map_tree(root, flags):
    """
    フラグに基づいて事後確率最大四分木を構築する
    
    Args:
        root: 完全四分木のルートノード
        flags: {node_key: flag} の辞書
    
    Returns:
        tuple: (all_nodes, leaf_nodes, internal_nodes, node_dict)
    """
    all_nodes = []
    leaf_nodes = []
    internal_nodes = []
    node_dict = {}
    
    def traverse(node):
        all_nodes.append(node)
        node_key = (node.upper_edge, node.left_edge, node.size, node.depth)
        node_dict[node_key] = node
        
        flag = flags.get(node_key, 0)
        
        # f_s = 0 の場合、葉ノードとして扱う
        if flag == 0 or node.is_leaf:
            leaf_nodes.append(node)
        else:
            # f_s = 1 の場合、内部ノードとして扱い、子ノードを探索
            internal_nodes.append(node)
            if hasattr(node, 'ul_node'):
                traverse(node.ul_node)
            if hasattr(node, 'ur_node'):
                traverse(node.ur_node)
            if hasattr(node, 'll_node'):
                traverse(node.ll_node)
            if hasattr(node, 'lr_node'):
                traverse(node.lr_node)
    
    traverse(root)
    
    return all_nodes, leaf_nodes, internal_nodes, node_dict


def create_map_quadtree(max_depth, image, branch_probs, label_param, pixel_param, cfg: Config | None = None, image_stem: str | None = None):
    """
    論文のアルゴリズムに基づき、事後確率最大四分木を構築する
    
    Args:
        max_depth: 最大深度
        image: RGB画像配列
        branch_probs: 学習済み分岐確率リスト
        label_param: ラベルモデルのパラメータ
        pixel_param: ピクセルモデルのパラメータ
    
    Returns:
        tuple: (root, all_nodes, leaf_nodes, internal_nodes, node_dict, adjacency_dict)
    """
    # Step 1: 完全四分木を生成
    print(f"  Generating complete quadtree with max_depth={max_depth}...")
    tree_start = time.time()
    root = Node(upper_edge=0, left_edge=0, size=2**max_depth, depth=0)
    make_tree(root, max_depth)
    tree_time = time.time() - tree_start
    print(f"    ✓ Complete quadtree generated (took {tree_time:.2f}s)")
    
    # Step 2: 事後確率最大四分木のフラグを計算
    print(f"  Computing MAP quadtree flags (this may take a while)...")
    flags = compute_map_tree_flags(
        root,
        image,
        branch_probs,
        label_param,
        pixel_param,
        cfg=cfg,
        image_stem=image_stem,
    )
    
    # Step 3: フラグに基づいて事後確率最大四分木を構築
    print(f"  Constructing MAP quadtree...")
    construct_start = time.time()
    all_nodes, leaf_nodes, internal_nodes, node_dict = construct_map_tree(root, flags)
    construct_time = time.time() - construct_start
    print(f"    ✓ MAP quadtree constructed (took {construct_time:.2f}s)")
    
    # Step 4: 隣接辞書を構築（v2相当: 葉インデックス画像で高速化）
    print(f"  Building adjacency dictionary for {len(leaf_nodes)} leaf nodes...")
    adj_start = time.time()
    adjacency_dict = _build_leaf_adjacency_from_index_map(leaf_nodes, root.size, root.size)
    adj_time = time.time() - adj_start
    print(f"    ✓ Adjacency dictionary built (took {adj_time:.2f}s)")
    print(f"    - Total adjacency pairs: {sum(len(v) for v in adjacency_dict.values())}")

    return root, all_nodes, leaf_nodes, internal_nodes, node_dict, adjacency_dict


def _are_adjacent(node1, node2):
    """2つのノードが4-隣接（上下左右）かチェック"""
    size1 = node1.size
    size2 = node2.size
    
    # 上下隣接
    if node1.upper_edge + size1 == node2.upper_edge and node1.left_edge == node2.left_edge:
        if node1.size == node2.size:
            return True
    if node2.upper_edge + size2 == node1.upper_edge and node1.left_edge == node2.left_edge:
        if node1.size == node2.size:
            return True
    
    # 左右隣接
    if node1.left_edge + size1 == node2.left_edge and node1.upper_edge == node2.upper_edge:
        if node1.size == node2.size:
            return True
    if node2.left_edge + size2 == node1.left_edge and node1.upper_edge == node2.upper_edge:
        if node1.size == node2.size:
            return True
    
    return False


def compute_log_marginal_terms(region_nodes, image, label_param, pixel_param, node_likelihood_cache):
    """
    領域 r について、各ラベル x の log M(r, x) を返す（キャッシュ利用版）。
    log M(r, x) = log p_{r,x} + log p(Y_r | x; θ_x)
    
    Args:
        region_nodes: 領域を構成するノード(Node)のリスト
        image: RGB画像配列 (H, W, C)
        label_param: ラベルモデルのパラメータ
        pixel_param: ピクセルモデルのパラメータ
        node_likelihood_cache: ノードごとのピクセル対数尤度キャッシュ {node: {label_idx: log_likelihood}}
    
    Returns:
        np.ndarray: 各ラベルに対する log M(r, x) ベクトル
    """
    # 領域に属するピクセルを集約（幾何特徴量計算用）
    region_set = set()
    
    for node in region_nodes:
        i1, i2 = node.upper_edge, node.upper_edge + node.size
        j1, j2 = node.left_edge, node.left_edge + node.size
        for i in range(i1, i2):
            for j in range(j1, j2):
                region_set.add((i, j))
    
    if not region_set:
        # デフォルト値を返す
        num_labels = label_param.get("label_num", 2)
        return np.full(num_labels, -np.inf, dtype=np.float64)
    
    # 幾何的特徴量に基づくラベル事前確率を計算
    label_probs = label_prior(region_set, label_param)  # shape: (num_labels,)
    log_label_probs = np.log(np.maximum(label_probs, LOG_EPS))
    num_labels = len(label_probs)
    
    # 各ラベルに対するピクセル値の対数尤度を、キャッシュから取得して合計
    log_likelihoods = np.zeros(num_labels)
    
    for label_idx in range(num_labels):
        # 領域内の各ノードについてキャッシュから対数尤度を取得し合計
        for node in region_nodes:
            if node in node_likelihood_cache and label_idx in node_likelihood_cache[node]:
                log_likelihoods[label_idx] += node_likelihood_cache[node][label_idx]
            else:
                # キャッシュに無い場合は非常に小さい値を設定
                log_likelihoods[label_idx] = -1e10
                break
    
    # log M(r, x) = log p_{r,x} + log p(Y_r | x; θ_x)
    return log_label_probs + log_likelihoods


def initialize_connections(leaf_nodes):
    """
    結合変数を初期化する
    初期状態: すべてのノードが自身を参照（各ノードが独立した領域）
    
    Returns:
        dict: {leaf_node: connection_target}
    """
    connections = {node: node for node in leaf_nodes}
    return connections


def get_regions_from_connections(connections):
    """
    結合変数から領域（弱連結成分）を構築する

    O(N) 実装: 一度だけ全エッジを走査して双方向隣接リストを作り、
    その後は BFS で弱連結成分を列挙する。
    以前の実装は DFS の内部で connections.items() を毎回スキャンして
    O(N²) になっていたため大幅に高速化。

    Args:
        connections: {leaf_node: connection_target} 辞書

    Returns:
        list: 領域リスト（各領域は leaf_nodes のリスト）
    """
    # sample_connection 中は「キーには居ないが値には現れる」ノードが一時的に存在する。
    # そのため keys と values の和集合でグラフ頂点を作る。
    all_nodes = set(connections.keys()) | set(connections.values())
    graph: dict = {node: [] for node in all_nodes}
    for src, tgt in connections.items():
        if src is not tgt:  # 自己ループは辺不要
            graph[src].append(tgt)
            graph[tgt].append(src)

    visited = set()
    regions = []

    for start_node in all_nodes:
        if start_node in visited:
            continue
        # BFS（反復実装でスタックオーバーフローを回避）
        region = []
        queue = [start_node]
        visited.add(start_node)
        while queue:
            node = queue.pop()
            region.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        regions.append(region)

    return regions


def compute_log_affinity(leaf1, leaf2, adjacency_dict, cfg: Config):
    """
    2つの葉ノード間の対数親和度を計算
    log f(s, s') = β * B(s, s') + η * (depth(s) - depth(s'))
    
    Returns:
        float: 対数親和度値（隣接していない場合は -inf）
    """
    if cfg.affinity_func is None:
        return -np.inf

    try:
        log_affinity = cfg.affinity_func(
            leaf1,
            leaf2,
            adjacency_dict,
            **cfg.affinity_params,
        )
    except TypeError:
        log_affinity = cfg.affinity_func(leaf1, leaf2, adjacency_dict)

    if np.isnan(log_affinity):
        return -np.inf
    return float(log_affinity)


def sample_connection(leaf_node, connections, leaf_nodes, adjacency_dict, 
                     image, label_param, pixel_param, cfg: Config, node_likelihood_cache, alpha=0.001,
                     verbose=False):
    _t0 = time.time()

    # 1. 現在の接続を一時的に削除（ddCRPの基本操作）
    if leaf_node in connections:
        del connections[leaf_node]
        
    # 2. 削除後のベース領域 (R_{-s}) を計算
    regions_before = get_regions_from_connections(connections)
    
    # 各ノードがどのベース領域に属しているかのマッピングを作成
    node_to_region_base = {}
    for region_nodes in regions_before:
        for n in region_nodes:
            node_to_region_base[n] = region_nodes
            
    # leaf_node のベース領域 r_l を取得（どこにも属していない場合は単独）
    r_leaf_base = node_to_region_base.get(leaf_node, [leaf_node])
    
    # ベース領域 M(r_l) の対数尤度を計算
    log_M_leaf_base_terms = compute_log_marginal_terms(r_leaf_base, image, label_param, pixel_param, node_likelihood_cache)
    log_M_leaf_base_sum = _logsumexp(log_M_leaf_base_terms)

    # 領域ごとの尤度キャッシュ（候補ごとの再計算を防ぐ）
    base_region_log_M_cache = { id(r_leaf_base): log_M_leaf_base_sum }

    candidates = list(adjacency_dict.get(leaf_node, [])) + [leaf_node]
    log_probs = []
    
    for candidate in candidates:
        r_cand_base = node_to_region_base.get(candidate, [candidate])
        
        # 親和度
        log_affinity = compute_log_affinity(leaf_node, candidate, adjacency_dict, cfg)
        prior_prob = _safe_log(alpha) if candidate == leaf_node else log_affinity
        
        # 尤度比 Γ(Y, R) の計算
        if id(r_leaf_base) == id(r_cand_base):
            # 結合しても領域構造が変わらない場合
            log_likelihood_ratio = 0.0
        else:
            # 結合によって2つの領域が統合される場合
            r_merged = r_leaf_base + r_cand_base
            
            # 統合された領域 M(r_k)
            log_M_merged_terms = compute_log_marginal_terms(r_merged, image, label_param, pixel_param, node_likelihood_cache)
            log_M_merged_sum = _logsumexp(log_M_merged_terms)
            
            # 結合先のベース領域 M(r_m)
            cand_id = id(r_cand_base)
            if cand_id not in base_region_log_M_cache:
                terms = compute_log_marginal_terms(r_cand_base, image, label_param, pixel_param, node_likelihood_cache)
                base_region_log_M_cache[cand_id] = _logsumexp(terms)
            log_M_cand_base_sum = base_region_log_M_cache[cand_id]
            
            # log_likelihood_ratio = log M(r_k) - log M(r_l) - log M(r_m)
            log_likelihood_ratio = log_M_merged_sum - log_M_leaf_base_sum - log_M_cand_base_sum
            
        log_probs.append(prior_prob + log_likelihood_ratio)
        
    # Softmaxで確率に変換
    log_probs = np.array(log_probs, dtype=np.float64)
    log_norm = _logsumexp(log_probs)
    probs = np.exp(log_probs - log_norm)
    
    # サンプリング
    new_idx = np.random.choice(len(candidates), p=probs)
    new_target = candidates[new_idx]
    
    connections[leaf_node] = new_target
    return new_target, float(probs[new_idx])


def estimate_label_gibbs_sampling(
    image,
    cfg: Config,
    num_iterations=20,
    burn_in=0,
    region_output_dir=None,
    quadtree_output_path=None,
    image_stem=None,
    label_output_dir=None,
    label_vis_output_dir=None,
    label_diff_output_dir=None,
    oa_log_filepath=None,
    true_label_array=None,
    label_to_value_map=None,
):
    """
    ギブスサンプリングによるセグメンテーション推定
    
    Args:
        image: RGB画像 (H, W, C)
        cfg: Config インスタンス
        num_iterations: サンプリングイテレーション数
        burn_in: バーンイン期間
        region_output_dir: 領域分割図の保存先ディレクトリ
        quadtree_output_path: MAP四分木画像の保存先ファイルパス
        image_stem: 画像ファイルの幹（拡張子なし）
        label_output_dir: 毎イテレーションのラベル推定画像の保存先ディレクトリ
        label_vis_output_dir: 毎イテレーションのラベル可視化画像の保存先ディレクトリ
        label_diff_output_dir: 毎イテレーションの差分画像の保存先ディレクトリ
        oa_log_filepath: Overall Accuracy の推移ログファイルパス
        true_label_array: 正解ラベル画像配列 (OA計算・差分画像に使用)
        label_to_value_map: ラベル可視化値マップ
    
    Returns:
        X_est: 推定ラベル画像 (H, W)
    """
    global MODEL_CFG, PIXEL_LIKELIHOOD_WARN_COUNT
    MODEL_CFG = cfg
    PIXEL_LIKELIHOOD_WARN_COUNT = 0

    H, W = image.shape[:2]
    
    # 画像サイズが2の累乗か確認
    max_depth = int(np.log2(H))
    if 2**max_depth != H:
        # 最も近い2の累乗のサイズにリサイズ
        size = 2**(max_depth)
        print(f"Resizing image from {H}x{W} to {size}x{size}")
        from PIL import Image as PILImage
        img_resized = PILImage.fromarray((image[:,:,0] if len(image.shape) == 3 else image).astype(np.uint8))
        img_resized = img_resized.resize((size, size), PILImage.Resampling.BILINEAR)
        image = np.array(img_resized)
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        H, W = size, size
        max_depth = int(np.log2(H))
    
    # 学習済みパラメータを読み込み
    label_param_path = os.path.join(cfg.out_param_dir, cfg.label_param_filename)
    if os.path.exists(label_param_path):
        with open(label_param_path, 'r') as f:
            label_param = json.load(f)
        print(f"Loaded label parameters: label_num={label_param.get('label_num')}")
    else:
        raise FileNotFoundError(f"Label parameter file not found: {label_param_path}")
    
    # ピクセルモデルのパラメータを読み込み
    pixel_param_path = os.path.join(cfg.out_param_dir, cfg.pixel_param_filename)
    if os.path.exists(pixel_param_path):
        with open(pixel_param_path, 'r') as f:
            pixel_param = json.load(f)
        print(f"Loaded pixel parameters: {len(pixel_param.get('ar_param', []))} labels")
    else:
        raise FileNotFoundError(f"Pixel parameter file not found: {pixel_param_path}")
    
    # 学習済み分岐確率を読み込み
    branch_probs_path = os.path.join(cfg.out_param_dir, cfg.branch_probs_filename)
    if os.path.exists(branch_probs_path):
        with open(branch_probs_path, 'r') as f:
            branch_data = json.load(f)
            branch_probs = branch_data.get("branch_probs", [])
        print(f"Loaded branch probabilities: {branch_probs}")
    else:
        # デフォルト値（すべての深度で0.5）
        branch_probs = [0.5] * (max_depth + 1)
        print(f"Using default branch probabilities: {branch_probs}")
    
    # ピクセルモデルの想定チャネルに合わせて画像を整合
    expected_channels = _expected_pixel_channels(pixel_param)
    image = _adapt_image_channels_for_pixel_model(image, pixel_param).astype(np.float64)
    actual_channels = 1 if image.ndim == 2 else image.shape[2]
    print(f"Pixel model channels expected={expected_channels}, adapted image channels={actual_channels}")

    # 論文のStep 1: 事後確率最大四分木を計算
    print(f"Computing MAP quadtree with max_depth={max_depth}...")
    quadtree_start = time.time()
    root, all_nodes, leaf_nodes, internal_nodes, node_dict, adjacency_dict = create_map_quadtree(
        max_depth, image, branch_probs, label_param, pixel_param, cfg=cfg, image_stem=image_stem
    )
    quadtree_time = time.time() - quadtree_start
    
    print(f"✓ MAP quadtree created in {quadtree_time:.2f}s:")
    print(f"  - Total nodes: {len(all_nodes)}")
    print(f"  - Leaf nodes: {len(leaf_nodes)}")
    print(f"  - Internal nodes: {len(internal_nodes)}")

    if quadtree_output_path is not None:
        save_quadtree_image_from_leaves(leaf_nodes, H, quadtree_output_path)
        print(f"  ✓ Saved MAP quadtree image to {quadtree_output_path}")
    
    # ノードごとのピクセル対数尤度をキャッシュ化（サンプリング前に一度だけ計算）
    print()
    node_likelihood_cache = build_node_likelihood_cache(
        leaf_nodes, image, label_param, pixel_param, cfg=cfg, image_stem=image_stem
    )

    # 初期化
    print(f"\nInitializing Gibbs sampling...")
    connections = initialize_connections(leaf_nodes)

    # label_paramにimage_sizeを追加（ループ前に一度だけ作成）
    label_param_with_size = label_param.copy()
    label_param_with_size['image_size'] = H

    # ラベル推定・OA記録の初期化
    X_est = None
    oa_history = []

    if region_output_dir is not None:
        initial_regions = get_regions_from_connections(connections)
        initial_region_path = os.path.join(region_output_dir, f"{image_stem}_0000.png")
        save_region_growing_image_from_regions(H, initial_regions, initial_region_path)
        print(f"  ✓ Saved initial region partition to {initial_region_path}")
    
    # ギブスサンプリング
    alpha = cfg.alpha
    
    print(f"\n{'='*80}")
    print(f"Starting Gibbs sampling...")
    print(f"  Configuration: num_iterations={num_iterations}, burn_in={burn_in}, alpha={alpha}")
    for iteration in range(num_iterations):
        iter_start_time = time.time()
        print(f"\n  Iteration {iteration+1}/{num_iterations}")
        
        # 全葉ノードについて条件付き分布からサンプリング
        # 最初の3ノードは詳細ログ（verbose=True）でどのステップが遅いか確認
        _verbose_cutoff = 3 if iteration == 0 else 0
        _progress_step = max(1, len(leaf_nodes) // 100)  # 1%刻みで進捗表示
        _node_times = []
        for node_idx, leaf_node in enumerate(leaf_nodes):
            _verbose = (node_idx < _verbose_cutoff)
            if _verbose:
                print(f"    [node {node_idx}] Starting sample_connection (verbose)...")
            _nt0 = time.time()
            sample_connection(leaf_node, connections, leaf_nodes, adjacency_dict, 
                            image, label_param, pixel_param, cfg, node_likelihood_cache, alpha,
                            verbose=_verbose)
            _node_elapsed = time.time() - _nt0
            _node_times.append(_node_elapsed)
            if (node_idx + 1) % _progress_step == 0 or node_idx == 0:
                _elapsed_total = time.time() - iter_start_time
                _avg_per_node = np.mean(_node_times[-_progress_step:]) if _node_times else 0.0
                _remaining_nodes = len(leaf_nodes) - (node_idx + 1)
                _eta = _avg_per_node * _remaining_nodes
                print(f"    - [{node_idx+1:>6}/{len(leaf_nodes)}] elapsed={_elapsed_total:.1f}s  "
                      f"avg/node={_avg_per_node*1000:.1f}ms  ETA={_eta:.0f}s")
        
        # 現在の領域情報を表示
        _t_gr = time.time()
        current_regions = get_regions_from_connections(connections)
        _gr_time = time.time() - _t_gr
        iter_time = time.time() - iter_start_time
        region_sizes = [len(r) for r in current_regions]
        print(f"    ✓ Iteration completed in {iter_time:.2f}s  (get_regions={_gr_time:.2f}s)")
        print(f"    - Regions: {len(current_regions)} regions, avg_size={np.mean(region_sizes):.1f} nodes, " 
              f"min={min(region_sizes)}, max={max(region_sizes)}")

        if region_output_dir is not None and (iteration + 1) % 1 == 0:
            iteration_region_path = os.path.join(region_output_dir, f"{image_stem}_{iteration + 1:04d}.png")
            save_region_growing_image_from_regions(H, current_regions, iteration_region_path)
            print(f"    ✓ Saved region partition snapshot to {iteration_region_path}")

        # 毎イテレーションごとにラベル推定
        label_est_start = time.time()
        X_est = _estimate_labels_from_regions(
            current_regions, H, W, image, label_param_with_size, pixel_param, node_likelihood_cache
        )
        label_est_time = time.time() - label_est_start
        print(f"    ✓ Label estimation completed in {label_est_time:.2f}s")

        # ラベル推定画像を保存
        if label_output_dir is not None:
            os.makedirs(label_output_dir, exist_ok=True)
            iter_label_path = os.path.join(label_output_dir, f"{image_stem}_{iteration + 1:04d}.png")
            Image.fromarray(X_est.astype(np.uint8)).save(iter_label_path)
            print(f"    ✓ Saved label estimate to {iter_label_path}")

            if label_vis_output_dir is not None and label_to_value_map is not None:
                os.makedirs(label_vis_output_dir, exist_ok=True)
                vis_image = build_visualize_label_image(X_est, label_to_value_map)
                iter_vis_path = os.path.join(label_vis_output_dir, f"{image_stem}_{iteration + 1:04d}.png")
                Image.fromarray(vis_image).save(iter_vis_path)
                print(f"    ✓ Saved label visualization to {iter_vis_path}")

        # Overall Accuracy を計算し、差分画像を保存
        if true_label_array is not None:
            oa = compute_overall_accuracy(X_est, true_label_array)
            oa_history.append((iteration + 1, oa))
            print(f"    - Overall Accuracy at iteration {iteration + 1}: {oa:.4f}")

            if label_diff_output_dir is not None:
                os.makedirs(label_diff_output_dir, exist_ok=True)
                diff_path = os.path.join(label_diff_output_dir, f"{image_stem}_{iteration + 1:04d}.png")
                save_diff_image(X_est, true_label_array, diff_path)
                print(f"    ✓ Saved diff image to {diff_path}")

        if iteration >= burn_in:
            # バーンイン後の領域を記録（簡略版：最後の状態を使用）
            pass
    
    # X_est はギブスサンプリングの各イテレーションで更新済み。
    # イテレーション数が0の場合のフォールバック。
    if X_est is None:
        print(f"\n  Label estimation (fallback: num_iterations=0)...")
        fallback_regions = get_regions_from_connections(connections)
        X_est = _estimate_labels_from_regions(
            fallback_regions, H, W, image, label_param_with_size, pixel_param, node_likelihood_cache
        )

    # OA ログをファイルに保存
    if oa_log_filepath is not None and oa_history:
        log_dir = os.path.dirname(oa_log_filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(oa_log_filepath, 'w', encoding='utf-8') as f:
            f.write("iteration,overall_accuracy\n")
            for iter_num, oa in oa_history:
                f.write(f"{iter_num},{oa:.6f}\n")
        print(f"  ✓ OA log saved to {oa_log_filepath}")

    return X_est


def estimate_segmentation(image_path, cfg: Config):
    """
    画像のセグメンテーションを推定する
    
    Args:
        image_path: 入力画像ファイルパス
        cfg: Config インスタンス
    
    Returns:
        X_est: 推定ラベル画像
    """
    # 画像を読み込み
    print(f"  Loading image from {image_path}")
    load_start = time.time()
    image = utils.load_image(image_path)

    image = image.astype(np.float64)
    H, W = image.shape[:2]
    channel_info = 1 if image.ndim == 2 else image.shape[2]
    print(f"    ✓ Image loaded: {H}×{W}×{channel_info}ch (took {time.time() - load_start:.2f}s)")

    image_stem = os.path.splitext(os.path.basename(image_path))[0]

    # 出力ディレクトリの設定
    region_output_dir = os.path.join(cfg.est_label_folder_path, cfg.est_region_dirname)
    quadtree_output_dir = os.path.join(cfg.est_label_folder_path, cfg.est_quadtree_dirname)
    quadtree_output_path = os.path.join(quadtree_output_dir, f"{image_stem}.png")
    label_output_dir = os.path.join(cfg.est_label_folder_path, cfg.est_label_dirname)
    label_vis_output_dir = os.path.join(label_output_dir, cfg.est_label_visualize_dirname)
    label_diff_output_dir = cfg.est_label_diff_dir

    # OA ログファイルパスの設定（画像ごとに独立したファイルを作成）
    oa_log_dir = os.path.dirname(cfg.oa_log_filepath)
    oa_log_basename = os.path.basename(cfg.oa_log_filepath)
    image_oa_log_path = (
        os.path.join(oa_log_dir, f"{image_stem}_{oa_log_basename}") if oa_log_dir
        else f"{image_stem}_{oa_log_basename}"
    )

    os.makedirs(region_output_dir, exist_ok=True)
    os.makedirs(quadtree_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    os.makedirs(label_vis_output_dir, exist_ok=True)
    os.makedirs(label_diff_output_dir, exist_ok=True)
    if oa_log_dir:
        os.makedirs(oa_log_dir, exist_ok=True)

    # 正解ラベルを読み込む（存在する場合のみ）
    true_label_array = None
    true_label_path = os.path.join(cfg.test_label_dir, f"{image_stem}.png")
    if os.path.exists(true_label_path):
        true_label_img = Image.open(true_label_path).convert('L')
        true_label_array = np.array(true_label_img, dtype=np.int32)
        # 画像が2の累乗でない場合のリサイズに合わせて正解ラベルもリサイズ
        max_depth = int(np.log2(H))
        if 2 ** max_depth != H:
            target_size = 2 ** max_depth
            true_label_array = np.array(
                Image.fromarray(true_label_array.astype(np.uint8)).resize(
                    (target_size, target_size), Image.Resampling.NEAREST
                ),
                dtype=np.int32,
            )
        print(f"    - Loaded true label from {true_label_path}")
    else:
        print(f"    - True label not found at {true_label_path}, OA computation skipped")

    # ラベル可視化マップの読み込み
    label_to_value_map = load_label_visualization_map(cfg)

    print(f"    - Region snapshots will be saved to {region_output_dir}")
    print(f"    - MAP quadtree image will be saved to {quadtree_output_path}")
    print(f"    - Label estimates will be saved to {label_output_dir}")
    print(f"    - Diff images will be saved to {label_diff_output_dir}")

    # セグメンテーション推定
    print(f"  Running Gibbs sampling...")
    seg_start = time.time()
    X_est = estimate_label_gibbs_sampling(
        image,
        cfg,
        num_iterations=cfg.gibbs_num_iterations,
        burn_in=50,
        region_output_dir=region_output_dir,
        quadtree_output_path=quadtree_output_path,
        image_stem=image_stem,
        label_output_dir=label_output_dir,
        label_vis_output_dir=label_vis_output_dir,
        label_diff_output_dir=label_diff_output_dir,
        oa_log_filepath=image_oa_log_path,
        true_label_array=true_label_array,
        label_to_value_map=label_to_value_map,
    )
    print(f"    ✓ Segmentation completed (took {time.time() - seg_start:.2f}s)")
    
    return X_est


def save_results(X_est, output_dir, image_name, label_to_value_map, cfg: Config):
    """
    推定結果を保存する
    
    Args:
        X_est: 推定ラベル画像
        output_dir: 出力ディレクトリ
        image_name: 画像名
        label_to_value_map: train.py で保存した可視化値マップ
    """
    os.makedirs(output_dir, exist_ok=True)

    image_stem, _ = os.path.splitext(image_name)
    label_dir = os.path.join(output_dir, cfg.est_label_dirname)
    vis_dir = os.path.join(label_dir, cfg.est_label_visualize_dirname)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    label_path = os.path.join(label_dir, f"{image_stem}.png")
    label_image = X_est.astype(np.uint8)
    Image.fromarray(label_image).save(label_path)
    print(f"    ✓ Saved label image to {label_path}")

    vis_path = os.path.join(vis_dir, f"{image_stem}.png")
    vis_image = build_visualize_label_image(X_est, label_to_value_map)
    Image.fromarray(vis_image).save(vis_path)
    print(f"    ✓ Saved visualization to {vis_path}")


def process_test_images(cfg: Config) -> None:
    image_dir = cfg.test_image_dir

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"image directory not found: {image_dir}")

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    if not image_files:
        raise RuntimeError(f"no image files found in {image_dir}")

    global MODEL_CFG
    MODEL_CFG = cfg

    print(f"Found {len(image_files)} image files in {image_dir}")
    print("=" * 80)

    label_to_value_map = load_label_visualization_map(cfg)
    if label_to_value_map:
        print(f"Loaded label visualization map for {len(label_to_value_map)} labels")
    else:
        print("Label visualization map not found. Falling back to generated grayscale values.")

    total_start_time = time.time()
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        print(f"\n[{idx}/{len(image_files)}] Processing {image_file}...")

        try:
            file_start_time = time.time()
            X_est = estimate_segmentation(image_path, cfg)
            save_results(X_est, cfg.est_label_folder_path, image_file, label_to_value_map, cfg)

            elapsed = time.time() - file_start_time
            print(f"  ✓ Completed {image_file} ({elapsed:.2f}s)")
        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - total_start_time
    print("\n" + "=" * 80)
    print(f"All {len(image_files)} images processed in {total_time:.2f}s!")
    print(f"Average time per image: {total_time / len(image_files):.2f}s")


if __name__ == '__main__':
    process_test_images(config)
