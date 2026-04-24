# predict_icm.py
"""
論文のセグメンテーションアルゴリズムの実装
Bayes最適解を粗近似する領域とラベルの事後確率更新に基づく方法（ICM）
（アルゴリズムB）
"""

import os
import json
import hashlib
import numpy as np
import time
from PIL import Image
from collections import defaultdict
from config import config, Config
import config as config_module
from model.quadtree.node import Node
from model.quadtree.depth_dependent_model import make_tree, label_ndarray
import utils

LOG_EPS = 1e-300
MODEL_CFG: Config = config
PIXEL_LIKELIHOOD_WARN_COUNT = 0
PIXEL_LIKELIHOOD_WARN_MAX = 12


def log_prob_Y_given_X(region_tuple, label, img_array, theta):
    return MODEL_CFG.pixel_model.log_prob_Y_given_X(region_tuple, label, img_array, theta)


def label_prior(region, label_param):
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
    fp = {"name": getattr(module_obj, "__name__", str(module_obj))}
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
    offset = getattr(cfg, "offset", None)
    return {
        "label_model": _module_fingerprint(getattr(cfg, "label_model", None)),
        "pixel_model": _module_fingerprint(getattr(cfg, "pixel_model", None)),
        "quadtree_model": _module_fingerprint(getattr(cfg, "quadtree_model", None)),
        "affinity_func": _callable_fingerprint(getattr(cfg, "affinity_func", None)),
        "affinity_params_hash": _stable_hash_from_jsonable(affinity_params if isinstance(affinity_params, dict) else {"value": str(affinity_params)}),
        "offset_hash": _stable_hash_from_jsonable({"offset": offset}),
    }


def _get_calc_log_dir(cfg: Config) -> str:
    dataset_dir = getattr(config_module, "DATASET_DIR", None)
    if dataset_dir:
        return os.path.join(dataset_dir, ".calc_log")
    return os.path.join(os.path.dirname(os.path.dirname(cfg.test_image_dir)), ".calc_log")


def _get_step1_cache_path(cfg: Config, image_stem: str | None, image_size: int) -> str:
    stem = image_stem if image_stem else f"image_{image_size}"
    filename = f"{stem}_logp_cache.npz"
    return os.path.join(_get_calc_log_dir(cfg), filename)


def _build_step1_cache_meta(
    image: np.ndarray, label_param: dict, pixel_param: dict, image_size: int, num_nodes: int, cfg: Config | None = None,
) -> dict:
    return {
        "cache_version": 2, "image_size": int(image_size), "num_nodes": int(num_nodes),
        "image_shape": [int(v) for v in image.shape], "image_hash": _hash_image_array(image),
        "label_param_hash": _stable_hash_from_jsonable(label_param), "pixel_param_hash": _stable_hash_from_jsonable(pixel_param),
        "runtime_signature_hash": _stable_hash_from_jsonable(_build_runtime_signature(cfg)),
    }


def _save_step1_logp_cache(cache_path: str, log_p_y_cache: dict, meta: dict) -> None:
    keys = np.array(list(log_p_y_cache.keys()), dtype=np.int32)
    values = np.array([float(log_p_y_cache[k]) for k in log_p_y_cache.keys()], dtype=np.float64)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, node_keys=keys, log_p_y=values, meta_json=np.array(json.dumps(meta, sort_keys=True)))


def _load_step1_logp_cache_if_valid(cache_path: str, expected_meta: dict) -> tuple[dict | None, str]:
    if not os.path.exists(cache_path): return None, "cache file not found"
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            node_keys, log_p_y = data["node_keys"], data["log_p_y"]
            cached_meta = json.loads(str(data["meta_json"]))
    except Exception as e:
        return None, f"failed to load cache ({type(e).__name__}: {e})"

    required_fields = ["cache_version", "image_size", "num_nodes", "image_shape", "image_hash", "label_param_hash", "pixel_param_hash", "runtime_signature_hash"]
    for field in required_fields:
        if cached_meta.get(field) != expected_meta.get(field): return None, f"metadata mismatch: {field}"

    if node_keys.ndim != 2 or node_keys.shape[1] != 4 or log_p_y.ndim != 1 or node_keys.shape[0] != log_p_y.shape[0]:
        return None, "invalid cache format"

    cache = {tuple(int(v) for v in node_keys[idx].tolist()): float(log_p_y[idx]) for idx in range(node_keys.shape[0])}
    return cache, "ok"


def _get_node_likelihood_cache_path(cfg: Config, image_stem: str | None, image_size: int) -> str:
    stem = image_stem if image_stem else f"image_{image_size}"
    return os.path.join(_get_calc_log_dir(cfg), f"{stem}_node_likelihood_cache.npz")


def _build_node_likelihood_cache_meta(image: np.ndarray, label_param: dict, pixel_param: dict, num_leaf_nodes: int, num_labels: int, cfg: Config | None = None) -> dict:
    return {
        "cache_version": 2, "num_leaf_nodes": int(num_leaf_nodes), "num_labels": int(num_labels),
        "image_shape": [int(v) for v in image.shape], "image_hash": _hash_image_array(image),
        "label_param_hash": _stable_hash_from_jsonable(label_param), "pixel_param_hash": _stable_hash_from_jsonable(pixel_param),
        "runtime_signature_hash": _stable_hash_from_jsonable(_build_runtime_signature(cfg)),
    }


def _save_node_likelihood_cache(cache_path: str, node_likelihood_cache: dict, leaf_nodes: list, num_labels: int, meta: dict) -> None:
    key_to_idx = {(n.upper_edge, n.left_edge, n.size, n.depth): i for i, n in enumerate(leaf_nodes)}
    node_keys = np.array([[n.upper_edge, n.left_edge, n.size, n.depth] for n in leaf_nodes], dtype=np.int32)
    likelihoods = np.full((len(leaf_nodes), num_labels), -1e10, dtype=np.float64)
    for node, label_dict in node_likelihood_cache.items():
        if (idx := key_to_idx.get((node.upper_edge, node.left_edge, node.size, node.depth))) is not None:
            for label_idx, val in label_dict.items():
                likelihoods[idx, int(label_idx)] = float(val)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, node_keys=node_keys, likelihoods=likelihoods, meta_json=np.array(json.dumps(meta, sort_keys=True)))


def _load_node_likelihood_cache_if_valid(cache_path: str, expected_meta: dict, leaf_nodes: list) -> tuple[dict | None, str]:
    if not os.path.exists(cache_path): return None, "cache file not found"
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            node_keys, likelihoods, cached_meta = data["node_keys"], data["likelihoods"], json.loads(str(data["meta_json"]))
    except Exception as e:
        return None, f"failed to load cache ({type(e).__name__}: {e})"

    for field in ["cache_version", "num_leaf_nodes", "num_labels", "image_shape", "image_hash", "label_param_hash", "pixel_param_hash", "runtime_signature_hash"]:
        if cached_meta.get(field) != expected_meta.get(field): return None, f"metadata mismatch: {field}"

    if node_keys.ndim != 2 or node_keys.shape[1] != 4 or likelihoods.ndim != 2 or node_keys.shape[0] != len(leaf_nodes):
        return None, "invalid format"

    stored_key_to_idx = {tuple(int(v) for v in node_keys[i].tolist()): i for i in range(node_keys.shape[0])}
    cache = {}
    for node in leaf_nodes:
        idx = stored_key_to_idx.get((node.upper_edge, node.left_edge, node.size, node.depth))
        if idx is None: return None, "node key not found"
        cache[node] = {l: float(likelihoods[idx, l]) for l in range(likelihoods.shape[1])}
    return cache, "ok"


def build_node_likelihood_cache(leaf_nodes: list, image: np.ndarray, label_param: dict, pixel_param: dict, pixel_log_likelihood_integrals: np.ndarray, cfg: Config | None = None, image_stem: str | None = None) -> dict:
    num_labels = label_param.get("label_num", 2)
    use_cache = bool(cfg is not None and getattr(cfg, "enable_logq_cache", False))
    cache_path = None
    if use_cache and cfg is not None:
        cache_path = _get_node_likelihood_cache_path(cfg, image_stem, int(image.shape[0]))
        expected_meta = _build_node_likelihood_cache_meta(image, label_param, pixel_param, len(leaf_nodes), num_labels, cfg)
        loaded, reason = _load_node_likelihood_cache_if_valid(cache_path, expected_meta, leaf_nodes)
        if loaded is not None: return loaded

    node_likelihood_cache = {}
    for node in leaf_nodes:
        node_likelihood_cache[node] = {label_idx: _compute_log_likelihood_of_node_given_label_from_integral(node, label_idx, pixel_log_likelihood_integrals) for label_idx in range(num_labels)}

    if use_cache and cfg is not None and cache_path is not None:
        _save_node_likelihood_cache(cache_path, node_likelihood_cache, leaf_nodes, num_labels, expected_meta)
    return node_likelihood_cache


def _get_num_labels(label_param: dict) -> int:
    return int(label_param.get("label_num", len(label_param.get("label_set", []))))


def _compute_pixel_log_likelihood_integrals(image: np.ndarray, pixel_param: dict, num_labels: int, cfg: Config | None = None) -> np.ndarray:
    global PIXEL_LIKELIHOOD_WARN_COUNT
    H, W = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    offsets = _resolve_neighbor_offsets(pixel_param, cfg)
    pixel_log_likelihoods = np.full((num_labels, H, W), -1e10, dtype=np.float64)

    for i in range(H):
        for j in range(W):
            pixel_node = Node(upper_edge=i, left_edge=j, size=1, depth=0)
            center_pixel, neighbor_pixels = _extract_pixel_context(image, i, j, offsets, channels)
            for label_idx in range(num_labels):
                pixel_log_likelihoods[label_idx, i, j] = _compute_log_prob_pixel_given_label(center_pixel, neighbor_pixels, label_idx, pixel_param, offsets, pixel_node, image)

    integrals = np.zeros((num_labels, H + 1, W + 1), dtype=np.float64)
    integrals[:, 1:, 1:] = np.cumsum(np.cumsum(pixel_log_likelihoods, axis=1), axis=2)
    return integrals


def _rect_sum_from_integral(integral_2d: np.ndarray, i1: int, i2: int, j1: int, j2: int) -> float:
    return float(integral_2d[i2, j2] - integral_2d[i1, j2] - integral_2d[i2, j1] + integral_2d[i1, j1])


def _compute_log_likelihood_of_node_given_label_from_integral(node: Node, label_idx: int, pixel_log_likelihood_integrals: np.ndarray) -> float:
    return _rect_sum_from_integral(pixel_log_likelihood_integrals[label_idx], node.upper_edge, node.lower_edge, node.left_edge, node.right_edge)


def _resolve_neighbor_offsets(pixel_param: dict, cfg: Config | None = None) -> list[tuple[int, int]]:
    cfg_offset = getattr(cfg, "offset", None) if cfg is not None else None
    if isinstance(cfg_offset, list) and cfg_offset: return [tuple(int(v) for v in off) for off in cfg_offset]
    ar_param = pixel_param.get("ar_param", [])
    if isinstance(ar_param, list) and ar_param and isinstance(ar_param[0], dict):
        return sorted([tuple(map(int, k.strip("()").split(","))) if isinstance(k, str) else tuple(int(v) for v in k) for k in ar_param[0].keys()])
    return []


def _extract_pixel_context(image: np.ndarray, i: int, j: int, offsets: list[tuple[int, int]], channels: int) -> tuple[np.ndarray, np.ndarray]:
    center = np.array([float(image[i, j])], dtype=np.float64) if image.ndim == 2 else image[i, j].astype(np.float64)
    neighbors = np.full((len(offsets), channels), np.nan, dtype=np.float64)
    H, W = image.shape[:2]
    for idx, (di, dj) in enumerate(offsets):
        if 0 <= (ni := i + di) < H and 0 <= (nj := j + dj) < W:
            neighbors[idx, 0] = float(image[ni, nj]) if image.ndim == 2 else image[ni, nj]
    return center, neighbors


def _compute_log_prob_pixel_given_label(center_pixel: np.ndarray, neighbor_pixels: np.ndarray, label_idx: int, pixel_param: dict, offsets: list[tuple[int, int]], pixel_node: Node, image: np.ndarray) -> float:
    pixel_fn = getattr(MODEL_CFG.pixel_model, "log_prob_pixel_given_label", None)
    if callable(pixel_fn): return float(pixel_fn(center_pixel, neighbor_pixels, label_idx, pixel_param, offsets))
    return float(log_prob_Y_given_X((pixel_node,), label_idx, image, pixel_param))


def _logsumexp(log_values: np.ndarray) -> float:
    if log_values.size == 0: return -np.inf
    max_log = np.max(log_values)
    return float(max_log + np.log(np.sum(np.exp(log_values - max_log)))) if np.isfinite(max_log) else -np.inf


def _expected_pixel_channels(pixel_param: dict) -> int:
    if (channels := pixel_param.get("channels")) is not None: return int(channels)
    mean = pixel_param.get("mean", [])
    return int(len(mean[0])) if isinstance(mean, list) and mean and isinstance(mean[0], list) else 1


def _adapt_image_channels_for_pixel_model(image: np.ndarray, pixel_param: dict) -> np.ndarray:
    expected = _expected_pixel_channels(pixel_param)
    if image.ndim == 2: return image if expected == 1 else np.stack([image] * expected, axis=-1)
    actual = image.shape[2]
    if actual == expected: return image
    if expected == 1: return image[..., 0]
    if actual == 1 and expected > 1: return np.repeat(image, expected, axis=2)
    raise ValueError("Pixel model channel mismatch")


class QuadTreeNode:
    def __init__(self, upper_edge, left_edge, size, depth, parent=None):
        self.upper_edge, self.left_edge, self.size, self.depth, self.parent = upper_edge, left_edge, size, depth, parent
        self.children, self.is_leaf = [], depth == 127


def _node_key(node: Node) -> tuple[int, int, int, int]:
    return (node.upper_edge, node.left_edge, node.size, node.depth)


def _collect_nodes_by_depth(root: Node):
    stack, nodes_by_depth, all_nodes = [root], defaultdict(list), []
    while stack:
        node = stack.pop()
        all_nodes.append(node); nodes_by_depth[node.depth].append(node)
        if not node.is_leaf: stack.extend([node.ul_node, node.ur_node, node.ll_node, node.lr_node])
    return all_nodes, nodes_by_depth


def _build_leaf_adjacency_from_index_map(leaf_nodes, height, width):
    leaf_index_map = np.full((height, width), -1, dtype=np.int32)
    for idx, node in enumerate(leaf_nodes):
        leaf_index_map[node.upper_edge:node.lower_edge, node.left_edge:node.right_edge] = idx

    adjacency_dict = defaultdict(list)
    for idx, node in enumerate(leaf_nodes):
        neighbors = set()
        if node.upper_edge > 0: neighbors.update(int(v) for v in np.unique(leaf_index_map[node.upper_edge - 1, node.left_edge:node.right_edge]) if v >= 0 and int(v) != idx)
        if node.lower_edge < height: neighbors.update(int(v) for v in np.unique(leaf_index_map[node.lower_edge, node.left_edge:node.right_edge]) if v >= 0 and int(v) != idx)
        if node.left_edge > 0: neighbors.update(int(v) for v in np.unique(leaf_index_map[node.upper_edge:node.lower_edge, node.left_edge - 1]) if v >= 0 and int(v) != idx)
        if node.right_edge < width: neighbors.update(int(v) for v in np.unique(leaf_index_map[node.upper_edge:node.lower_edge, node.right_edge]) if v >= 0 and int(v) != idx)
        adjacency_dict[node] = [leaf_nodes[n_idx] for n_idx in neighbors]
    return dict(adjacency_dict)


def _regions_to_region_dict(regions):
    return {region_id: {(i, j) for node in region_nodes for i in range(node.upper_edge, node.lower_edge) for j in range(node.left_edge, node.right_edge)} for region_id, region_nodes in enumerate(regions, start=1)}


def save_region_growing_image_from_regions(image_size: int, regions, filename: str) -> None:
    region_dict = _regions_to_region_dict(regions)
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    region_colors = {rid: np.random.randint(50, 255, size=3, dtype=np.uint8) for rid in region_dict.keys()}
    for rid, pixels in region_dict.items():
        for i, j in pixels: image[i, j] = region_colors[rid]
    os.makedirs(os.path.dirname(filename), exist_ok=True); Image.fromarray(image).save(filename)


def save_quadtree_image_from_leaves(leaf_nodes, image_size: int, filename: str) -> None:
    image, color_map = np.zeros((image_size, image_size, 3), dtype=np.uint8), {}
    for leaf in leaf_nodes:
        if (k := (leaf.upper_edge, leaf.left_edge)) not in color_map: color_map[k] = np.random.randint(50, 255, size=3, dtype=np.uint8)
        image[leaf.upper_edge:leaf.lower_edge, leaf.left_edge:leaf.right_edge] = color_map[k]
    os.makedirs(os.path.dirname(filename), exist_ok=True); Image.fromarray(image).save(filename)


def load_label_visualization_map(cfg: Config) -> dict[int, int]:
    if not os.path.exists(path := os.path.join(cfg.out_param_dir, cfg.label_param_filename)): return {}
    with open(path, 'r') as f: info = json.load(f)
    return {int(l): int(v) for l, v in zip(info.get("label_set", []), info.get("label_value_set", []))}


def build_visualize_label_image(label_array: np.ndarray, label_to_value_map: dict[int, int]) -> np.ndarray:
    vis_array = np.zeros(label_array.shape, dtype=np.uint8)
    if not label_to_value_map and (unique := sorted(int(v) for v in np.unique(label_array))):
        label_to_value_map = {unique[0]: 0} if len(unique) == 1 else {l: int(v) for l, v in zip(unique, np.linspace(0, 255, len(unique)).round().astype(np.uint8))}
    for l_val, c_val in label_to_value_map.items(): vis_array[label_array == l_val] = c_val
    return vis_array


def compute_overall_accuracy(X_est: np.ndarray, X_true: np.ndarray) -> float: return float(np.mean(X_est == X_true))


def _update_oa_error_csv(csv_path: str, image_name: str, iteration: int, error_rate: float) -> None:
    """CSV を更新する。行=画像ファイル名, 列=イテレーション番号, 値=1-OA (誤推定割合)。"""
    import csv as _csv

    data: dict = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                img = row.get('image', '')
                if not img:
                    continue
                data[img] = {}
                for k, v in row.items():
                    if k != 'image' and v != '':
                        try:
                            data[img][int(k)] = float(v)
                        except ValueError:
                            pass

    if image_name not in data:
        data[image_name] = {}
    data[image_name][iteration] = error_rate

    all_iters = sorted({it for iters in data.values() for it in iters})
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['image'] + [str(it) for it in all_iters]
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for img in sorted(data):
            row = {'image': img}
            for it in all_iters:
                row[str(it)] = f"{data[img][it]:.6f}" if it in data[img] else ''
            writer.writerow(row)


def save_diff_image(X_est: np.ndarray, X_true: np.ndarray, filepath: str) -> None:
    diff, correct = np.zeros((*X_est.shape, 3), dtype=np.uint8), X_est == X_true
    diff[correct], diff[~correct] = [0, 0, 255], [255, 0, 0]
    if d := os.path.dirname(filepath): os.makedirs(d, exist_ok=True)
    Image.fromarray(diff).save(filepath)


def _estimate_labels_from_regions(regions, H: int, W: int, image: np.ndarray, label_param_with_size: dict, pixel_param: dict, node_likelihood_cache: dict) -> np.ndarray:
    X_est = np.zeros((H, W), dtype=np.int32)
    for region_nodes in regions:
        label_est = int(np.argmax(compute_log_marginal_terms(region_nodes, image, label_param_with_size, pixel_param, node_likelihood_cache)))
        for node in region_nodes: X_est[node.upper_edge:node.lower_edge, node.left_edge:node.right_edge] = label_est
    return X_est


def _compute_log_p_Y_given_node(node: Node, image: np.ndarray, label_param: dict, pixel_param: dict, pixel_log_likelihood_integrals: np.ndarray) -> float:
    num_labels = _get_num_labels(label_param)
    return _logsumexp(np.array([_compute_log_likelihood_of_node_given_label_from_integral(node, i, pixel_log_likelihood_integrals) for i in range(num_labels)])) - np.log(num_labels)


def compute_map_tree_flags(root, image, branch_probs, label_param, pixel_param, pixel_log_likelihood_integrals: np.ndarray, cfg: Config | None = None, image_stem: str | None = None):
    # (v2流儀による MAP Tree Flags計算処理は predict_gibbs.py と同様のため省略せずそのまま実行)
    all_nodes, _ = _collect_nodes_by_depth(root)
    log_p_y_cache = {}

    def calc_logq(node):
        key = _node_key(node)
        if key not in log_p_y_cache: log_p_y_cache[key] = _compute_log_p_Y_given_node(node, image, label_param, pixel_param, pixel_log_likelihood_integrals)
        if node.is_leaf: node.logq_Ys = log_p_y_cache[key]
        else:
            for c in [node.ul_node, node.ur_node, node.ll_node, node.lr_node]: calc_logq(c)
            g_s = branch_probs[node.depth] if node.depth < len(branch_probs) else 0.0
            split_val = sum(c.logq_Ys for c in [node.ul_node, node.ur_node, node.ll_node, node.lr_node])
            if g_s >= 1.0: node.logq_Ys = split_val
            elif g_s <= 0.0: node.logq_Ys = log_p_y_cache[key]
            else: node.logq_Ys = np.logaddexp(_safe_log(1.0 - g_s) + log_p_y_cache[key], _safe_log(g_s) + split_val)

    calc_logq(root)

    def determine_map_tree(node):
        key = _node_key(node)
        if node.is_leaf: node.is_leaf_map, node.log_psi = True, 0.0; return
        for c in [node.ul_node, node.ur_node, node.ll_node, node.lr_node]: determine_map_tree(c)
        g_s = branch_probs[node.depth] if node.depth < len(branch_probs) else 0.0
        split_q = sum(c.logq_Ys for c in [node.ul_node, node.ur_node, node.ll_node, node.lr_node])
        g_given_y = 0.0 if g_s <= 0.0 else 1.0 if g_s >= 1.0 else min(max(float(np.exp(_safe_log(g_s) + split_q - node.logq_Ys)), 0.0), 1.0)
        split_psi = sum(c.log_psi for c in [node.ul_node, node.ur_node, node.ll_node, node.lr_node])
        log_psi_leaf, log_psi_split = (0.0, -np.inf) if g_given_y <= 0.0 else (-np.inf, split_psi) if g_given_y >= 1.0 else (_safe_log(1.0 - g_given_y), _safe_log(g_given_y) + split_psi)
        node.is_leaf_map, node.log_psi = (True, log_psi_leaf) if log_psi_leaf >= log_psi_split else (False, log_psi_split)

    determine_map_tree(root)

    flags = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf or getattr(node, "is_leaf_map", True): flags[_node_key(node)] = 0
        else: flags[_node_key(node)] = 1; stack.extend([node.ul_node, node.ur_node, node.ll_node, node.lr_node])
    return flags


def construct_map_tree(root, flags):
    all_nodes, leaf_nodes, internal_nodes, node_dict = [], [], [], {}
    def traverse(node):
        all_nodes.append(node); node_dict[_node_key(node)] = node
        if flags.get(_node_key(node), 0) == 0 or node.is_leaf: leaf_nodes.append(node)
        else:
            internal_nodes.append(node)
            for c in [node.ul_node, node.ur_node, node.ll_node, node.lr_node]:
                if hasattr(node, 'ul_node'): traverse(c)
    traverse(root)
    return all_nodes, leaf_nodes, internal_nodes, node_dict


def create_map_quadtree(max_depth, image, branch_probs, label_param, pixel_param, pixel_log_likelihood_integrals: np.ndarray, cfg: Config | None = None, image_stem: str | None = None):
    root = Node(upper_edge=0, left_edge=0, size=2**max_depth, depth=0)
    make_tree(root, max_depth)
    flags = compute_map_tree_flags(root, image, branch_probs, label_param, pixel_param, pixel_log_likelihood_integrals, cfg=cfg, image_stem=image_stem)
    all_nodes, leaf_nodes, internal_nodes, node_dict = construct_map_tree(root, flags)
    adjacency_dict = _build_leaf_adjacency_from_index_map(leaf_nodes, root.size, root.size)
    return root, all_nodes, leaf_nodes, internal_nodes, node_dict, adjacency_dict


def compute_log_marginal_terms(region_nodes, image, label_param, pixel_param, node_likelihood_cache):
    region_set = {(i, j) for node in region_nodes for i in range(node.upper_edge, node.upper_edge + node.size) for j in range(node.left_edge, node.left_edge + node.size)}
    if not region_set: return np.full(label_param.get("label_num", 2), -np.inf, dtype=np.float64)
    
    label_probs = label_prior(region_set, label_param)
    log_label_probs = np.log(np.maximum(label_probs, LOG_EPS))
    log_likelihoods = np.zeros(len(label_probs))
    
    for label_idx in range(len(label_probs)):
        for node in region_nodes:
            log_likelihoods[label_idx] += node_likelihood_cache.get(node, {}).get(label_idx, -1e10)
            
    return log_label_probs + log_likelihoods


def initialize_connections(leaf_nodes):
    return {node: node for node in leaf_nodes}


def get_regions_from_connections(connections):
    all_nodes = set(connections.keys()) | set(connections.values())
    graph: dict = {node: [] for node in all_nodes}
    for src, tgt in connections.items():
        if src is not tgt:
            graph[src].append(tgt); graph[tgt].append(src)

    visited, regions = set(), []
    for start_node in all_nodes:
        if start_node in visited: continue
        region, queue = [], [start_node]
        visited.add(start_node)
        while queue:
            node = queue.pop()
            region.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited: visited.add(neighbor); queue.append(neighbor)
        regions.append(region)
    return regions


def compute_log_affinity(leaf1, leaf2, adjacency_dict, cfg: Config):
    if cfg.affinity_func is None: return -np.inf
    try: val = cfg.affinity_func(leaf1, leaf2, adjacency_dict, **cfg.affinity_params)
    except TypeError: val = cfg.affinity_func(leaf1, leaf2, adjacency_dict)
    return float(val) if not np.isnan(val) else -np.inf


def update_connection_icm(leaf_node, connections, leaf_nodes, adjacency_dict, 
                          image, label_param, pixel_param, cfg: Config, node_likelihood_cache, alpha=0.001):
    """
    【ICM (アルゴリズムB) 用の結合先更新関数】
    事後確率の最大化（Argmax）により、最も確率の高い結合先を決定する。
    また、ラベルについても周辺化（logsumexp）ではなく、その領域での最適ラベルの尤度（np.max）を使用する。
    """
    current_target = connections.get(leaf_node)
    if leaf_node in connections:
        del connections[leaf_node]
        
    regions_before = get_regions_from_connections(connections)
    node_to_region_base = {n: r for r in regions_before for n in r}

    # connections から消えた孤立ノード用に、ノードごとの単一領域を再利用する。
    singleton_region_cache = {}

    def _get_base_region(node):
        region = node_to_region_base.get(node)
        if region is not None:
            return region
        if node not in singleton_region_cache:
            singleton_region_cache[node] = [node]
        return singleton_region_cache[node]

    r_leaf_base = _get_base_region(leaf_node)

    # 周辺化(logsumexp)ではなく、最適ラベルの尤度(max)を使用する
    log_M_leaf_base_max = float(np.max(compute_log_marginal_terms(r_leaf_base, image, label_param, pixel_param, node_likelihood_cache)))
    base_region_log_M_cache = {id(r_leaf_base): log_M_leaf_base_max}

    candidates = list(adjacency_dict.get(leaf_node, [])) + [leaf_node]
    log_probs = []
    
    for candidate in candidates:
        r_cand_base = _get_base_region(candidate)
        prior_prob = _safe_log(alpha) if candidate == leaf_node else compute_log_affinity(leaf_node, candidate, adjacency_dict, cfg)
        
        if id(r_leaf_base) == id(r_cand_base):
            log_likelihood_ratio = 0.0
        else:
            r_merged = r_leaf_base + r_cand_base
            log_M_merged_max = float(np.max(compute_log_marginal_terms(r_merged, image, label_param, pixel_param, node_likelihood_cache)))
            
            cand_id = id(r_cand_base)
            if cand_id not in base_region_log_M_cache:
                base_region_log_M_cache[cand_id] = float(np.max(compute_log_marginal_terms(r_cand_base, image, label_param, pixel_param, node_likelihood_cache)))
            log_M_cand_base_max = base_region_log_M_cache[cand_id]
            
            log_likelihood_ratio = log_M_merged_max - log_M_leaf_base_max - log_M_cand_base_max
            
        log_probs.append(prior_prob + log_likelihood_ratio)
        
    # 確率的サンプリングを行わず、最大のスコアを持つ結合先を選択 (Greedy)
    best_idx = int(np.argmax(np.array(log_probs, dtype=np.float64)))
    new_target = candidates[best_idx]
    
    connections[leaf_node] = new_target
    return new_target, current_target != new_target


def estimate_label_icm(
    image, cfg: Config, num_iterations=20, region_output_dir=None,
    quadtree_output_path=None, image_stem=None, label_output_dir=None,
    label_vis_output_dir=None, label_diff_output_dir=None, oa_log_filepath=None,
    true_label_array=None, label_to_value_map=None, oa_error_csv_path=None,
    num_iterations_for_csv=None,
):
    global MODEL_CFG, PIXEL_LIKELIHOOD_WARN_COUNT
    MODEL_CFG, PIXEL_LIKELIHOOD_WARN_COUNT = cfg, 0
    H, W = image.shape[:2]
    
    max_depth = int(np.log2(H))
    if 2**max_depth != H:
        from PIL import Image as PILImage
        size = 2**max_depth
        img_resized = PILImage.fromarray((image[:,:,0] if len(image.shape) == 3 else image).astype(np.uint8)).resize((size, size), PILImage.Resampling.BILINEAR)
        image = np.stack([np.array(img_resized)]*3, axis=-1) if len(np.array(img_resized).shape) == 2 else np.array(img_resized)
        H, W = size, size
    
    with open(os.path.join(cfg.out_param_dir, cfg.label_param_filename), 'r') as f: label_param = json.load(f)
    with open(os.path.join(cfg.out_param_dir, cfg.pixel_param_filename), 'r') as f: pixel_param = json.load(f)
    
    branch_probs_path = os.path.join(cfg.out_param_dir, cfg.branch_probs_filename)
    branch_probs = json.load(open(branch_probs_path, 'r')).get("branch_probs", []) if os.path.exists(branch_probs_path) else [0.5] * (max_depth + 1)
    
    image = _adapt_image_channels_for_pixel_model(image, pixel_param).astype(np.float64)
    pixel_log_likelihood_integrals = _compute_pixel_log_likelihood_integrals(image, pixel_param, _get_num_labels(label_param), cfg=cfg)

    root, all_nodes, leaf_nodes, internal_nodes, node_dict, adjacency_dict = create_map_quadtree(
        max_depth, image, branch_probs, label_param, pixel_param, pixel_log_likelihood_integrals, cfg=cfg, image_stem=image_stem,
    )
    if quadtree_output_path: save_quadtree_image_from_leaves(leaf_nodes, H, quadtree_output_path)
    
    node_likelihood_cache = build_node_likelihood_cache(leaf_nodes, image, label_param, pixel_param, pixel_log_likelihood_integrals, cfg=cfg, image_stem=image_stem)

    connections = initialize_connections(leaf_nodes)
    label_param_with_size = label_param.copy(); label_param_with_size['image_size'] = H
    X_est, oa_history = None, []

    if region_output_dir: save_region_growing_image_from_regions(H, get_regions_from_connections(connections), os.path.join(region_output_dir, f"{image_stem}_0000.png"))
    
    print(f"\n{'='*80}\nStarting Iterated Conditional Modes (ICM) Algorithm B...")
    
    for iteration in range(num_iterations):
        iter_start_time = time.time()
        print(f"\n  Iteration {iteration+1}/{num_iterations}")
        
        num_changed = 0
        for node_idx, leaf_node in enumerate(leaf_nodes):
            _, changed = update_connection_icm(leaf_node, connections, leaf_nodes, adjacency_dict, image, label_param, pixel_param, cfg, node_likelihood_cache, cfg.alpha)
            if changed: num_changed += 1
            
        current_regions = get_regions_from_connections(connections)
        print(f"    ✓ Iteration completed in {time.time() - iter_start_time:.2f}s  (Nodes changed: {num_changed}/{len(leaf_nodes)})")
        
        if region_output_dir: save_region_growing_image_from_regions(H, current_regions, os.path.join(region_output_dir, f"{image_stem}_{iteration + 1:04d}.png"))
        
        X_est = _estimate_labels_from_regions(current_regions, H, W, image, label_param_with_size, pixel_param, node_likelihood_cache)
        
        if label_output_dir:
            Image.fromarray(X_est.astype(np.uint8)).save(os.path.join(label_output_dir, f"{image_stem}_{iteration + 1:04d}.png"))
            if label_vis_output_dir and label_to_value_map:
                Image.fromarray(build_visualize_label_image(X_est, label_to_value_map)).save(os.path.join(label_vis_output_dir, f"{image_stem}_{iteration + 1:04d}.png"))
                
        if true_label_array is not None:
            oa_history.append((iteration + 1, (oa := compute_overall_accuracy(X_est, true_label_array))))
            print(f"    - Overall Accuracy: {oa:.4f}")
            if oa_error_csv_path is not None and image_stem is not None:
                _update_oa_error_csv(oa_error_csv_path, image_stem, iteration + 1, 1.0 - oa)
            if label_diff_output_dir: save_diff_image(X_est, true_label_array, os.path.join(label_diff_output_dir, f"{image_stem}_{iteration + 1:04d}.png"))
            
        # ICM 特有の早期終了 (どこも変更されなかったら収束)
        if num_changed == 0:
            if oa_error_csv_path is not None and image_stem is not None and oa_history:
                max_iter = num_iterations_for_csv if num_iterations_for_csv is not None else num_iterations
                last_oa = oa_history[-1][1]
                for fill_iter in range(iteration + 2, max_iter + 1):
                    _update_oa_error_csv(oa_error_csv_path, image_stem, fill_iter, 1.0 - last_oa)
            print(f"    ✓ Converged at iteration {iteration+1} (no connections changed)")
            break
            
    if X_est is None: X_est = _estimate_labels_from_regions(get_regions_from_connections(connections), H, W, image, label_param_with_size, pixel_param, node_likelihood_cache)

    if oa_log_filepath and oa_history:
        os.makedirs(os.path.dirname(oa_log_filepath), exist_ok=True)
        with open(oa_log_filepath, 'w', encoding='utf-8') as f:
            f.write("iteration,overall_accuracy\n")
            for i, oa in oa_history: f.write(f"{i},{oa:.6f}\n")

    return X_est

def estimate_segmentation(image_path, cfg: Config, oa_error_csv_path=None):
    image = utils.load_image(image_path).astype(np.float64)
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    
    cfg.est_label_folder_path = cfg.est_label_folder_path.replace("estimation_results", "estimation_results_icm")
    
    region_output_dir = os.path.join(cfg.est_label_folder_path, cfg.est_region_dirname)
    quadtree_output_dir = os.path.join(cfg.est_label_folder_path, cfg.est_quadtree_dirname)
    label_output_dir = os.path.join(cfg.est_label_folder_path, cfg.est_label_dirname)
    label_vis_output_dir = os.path.join(label_output_dir, cfg.est_label_visualize_dirname)
    
    os.makedirs(region_output_dir, exist_ok=True); os.makedirs(quadtree_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True); os.makedirs(label_vis_output_dir, exist_ok=True)
    
    true_label_array = None
    if os.path.exists(p := os.path.join(cfg.test_label_dir, f"{image_stem}.png")):
        true_label_array = np.array(Image.open(p).convert('L'), dtype=np.int32)
        if 2**int(np.log2(image.shape[0])) != image.shape[0]: true_label_array = np.array(Image.fromarray(true_label_array.astype(np.uint8)).resize((2**int(np.log2(image.shape[0])), 2**int(np.log2(image.shape[0]))), Image.Resampling.NEAREST), dtype=np.int32)
        
    if oa_error_csv_path is None:
        oa_error_csv_path = getattr(cfg, 'oa_error_csv_path', None)

    return estimate_label_icm(
        image, cfg, num_iterations=cfg.gibbs_num_iterations, region_output_dir=region_output_dir,
        quadtree_output_path=os.path.join(quadtree_output_dir, f"{image_stem}.png"), image_stem=image_stem,
        label_output_dir=label_output_dir, label_vis_output_dir=label_vis_output_dir,
        true_label_array=true_label_array, label_to_value_map=load_label_visualization_map(cfg),
        oa_error_csv_path=oa_error_csv_path, num_iterations_for_csv=cfg.gibbs_num_iterations,
    )

if __name__ == '__main__':
    for img_file in sorted([f for f in os.listdir(config.test_image_dir) if f.lower().endswith(('.png', '.jpg'))]):
        print(f"Processing {img_file} with ICM (Algorithm B)...")
        estimate_segmentation(os.path.join(config.test_image_dir, img_file), config)