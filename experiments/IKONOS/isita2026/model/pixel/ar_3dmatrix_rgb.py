# ./model/pixel/ar_3dmatrix_rgb.py
from __future__ import annotations
import os
import numpy as np
from PIL import Image
from numpy.linalg import inv, det, solve, eigvals, LinAlgError
import sys
import json
import utils
from model.quadtree.node import Node

# numba のインポートとデコレータを削除しました

### 画像生成 ###

def _generate_pixel_values(height, width, label_image, ar_coeffs_mat_list, means_list, variances_list, sorted_offsets_arr):
    """
    ピクセル生成関数 (Numba非依存の純粋なPython/NumPy実装)。
    チャンネル数に対応。
    """
    channels = means_list.shape[-1]
    rgb_image = np.zeros((height, width, channels), dtype=np.float64)

    for i in range(height):
        for j in range(width):
            label = label_image[i, j]

            ar_coeffs_mat = ar_coeffs_mat_list[label]
            mean = means_list[label]
            var = variances_list[label]

            # ARモデルに基づいて予測値を計算
            prediction = mean.copy()
            centered_neighbor = np.zeros(channels, dtype=np.float64)

            for k in range(len(sorted_offsets_arr)):
                offset = sorted_offsets_arr[k]
                weight_matrix = ar_coeffs_mat[k] # (channels, channels)

                ni, nj = i + offset[0], j + offset[1]

                if 0 <= ni < i or (ni == i and 0 <= nj < j):
                    centered_neighbor[:] = rgb_image[ni, nj] - mean
                    # 行列とベクトルの積
                    prediction += weight_matrix @ centered_neighbor

            try:
                # コレスキー分解を用いて多変量正規分布に従うノイズを生成
                L = np.linalg.cholesky(var)
                standard_normal_noise = np.random.normal(0, 1, size=channels)
                noise = L @ standard_normal_noise

                # ピクセル値を設定
                rgb_image[i, j] = prediction + noise
            except LinAlgError: 
                # 共分散行列が正定値でない場合など
                rgb_image[i, j] = prediction # ノイズなしで設定

    return rgb_image


def generate_rgb_from_labels(label_image, region_dict, theta, width, height, seed):
    """
    ラベル画像からARモデルを用いて画像(RGB等)を生成する。
    """
    if seed is not None:
        np.random.seed(seed)

    # パラメータ変換
    label_num = len(theta["ar_param"])
    means_list = np.array(theta["mean"], dtype=np.float64)
    variances_list = np.array(theta["variance"], dtype=np.float64)
    
    channels = means_list.shape[-1]

    # オフセット処理
    first_param_dict = theta["ar_param"][0]
    sorted_offsets = sorted([
        tuple(map(int, k.strip("()").split(','))) if isinstance(k, str) else k
        for k in first_param_dict.keys()
    ])
    offset_map = {offset: i for i, offset in enumerate(sorted_offsets)}
    num_offsets = len(sorted_offsets)

    # (label_num, num_offsets, channels, channels) の形状で係数行列を格納
    ar_coeffs_mat_list = np.zeros((label_num, num_offsets, channels, channels), dtype=np.float64)

    for label_idx in range(label_num):
        param_dict = theta["ar_param"][label_idx]
        for offset_str, matrix_list in param_dict.items():
            offset_tuple = tuple(map(int, offset_str.strip("()").split(','))) if isinstance(offset_str, str) else offset_str
            if offset_tuple in offset_map:
                idx = offset_map[offset_tuple]
                ar_coeffs_mat_list[label_idx, idx] = np.array(matrix_list, dtype=np.float64)

    sorted_offsets_arr = np.array(sorted_offsets, dtype=np.int64)
    
    # Numbaなしの関数を呼び出し
    rgb_image_float = _generate_pixel_values(
        height,
        width,
        label_image,
        ar_coeffs_mat_list,
        means_list,
        variances_list,
        sorted_offsets_arr
    )

    rgb_image = np.clip(rgb_image_float, 0, 255).astype(np.uint8)

    return rgb_image

### パラメータ推定 ###

def _collect_neighbor_data(image_data_list, label_data_list, Omega_array, label_val, channels):
    """
    近傍データ収集関数 (Numbaなし)。
    """
    y_list = []
    y_omega_list = []

    for img_idx in range(len(image_data_list)):
        img = image_data_list[img_idx]
        lbl = label_data_list[img_idx]
        height, width, _ = img.shape

        # Pythonループでの実装 (速度はNumbaより遅くなります)
        for i in range(height):
            for j in range(width):
                if lbl[i, j] == label_val:
                    neighbor_pixels = np.empty((len(Omega_array), channels), dtype=np.float64)
                    valid_neighbors = True

                    for k in range(len(Omega_array)):
                        di, dj = Omega_array[k]
                        ni, nj = i + di, j + dj

                        if not (0 <= ni < height and 0 <= nj < width and lbl[ni, nj] == label_val):
                            valid_neighbors = False
                            break
                        neighbor_pixels[k] = img[ni, nj]

                    if valid_neighbors:
                        y_list.append(img[i, j])
                        y_omega_list.append(neighbor_pixels)

    return y_list, y_omega_list

def param_est(train_image_dir, train_label_dir, out_param_json, Omega):
    # utilsを使用して画像ファイルを取得
    image_files = utils.get_image_files(train_image_dir)
    label_files = utils.get_image_files(train_label_dir)

    filename_list = utils.harmonize_lists(image_files, label_files)
    filename_list = sorted(filename_list)
    data_num = len(filename_list)
    image_data = []
    label_data = []

    # --- 1. データ読み込み ---
    print(f"{data_num} 組の学習データを読み込み中...")
    for filename in filename_list:
        image_path = os.path.join(train_image_dir, filename)
        label_path = os.path.join(train_label_dir, filename)
        
        # utils.load_imageを使用
        img = utils.load_image(image_path).astype(np.float64)
        lbl = utils.load_image(label_path).astype(np.int64)
        
        # チャンネル数の統一処理 (2次元(グレースケール)の場合は (H, W, 1) に変換)
        if img.ndim == 2:
            img = img[..., np.newaxis]
            
        image_data.append(img)
        label_data.append(lbl)
    print("データ読み込み完了。")

    if not image_data:
        print("有効なデータがありません。終了します。")
        return

    # チャンネル数を判定
    channels = image_data[0].shape[2]
    print(f"検出されたチャンネル数: {channels}")

    unique_labels = np.unique(np.concatenate([lbl.flatten() for lbl in label_data]))
    print(f"対象ラベル: {unique_labels}")
    print("近傍ピクセル集合: ", Omega)

    estimated_params = {}
    Omega_np = np.array(Omega, dtype=np.int64)
    omega_size = len(Omega)

    for x in unique_labels:
        # --- 2. データ収集 ---
        print(f"\n--- ラベル {x} のパラメータ推定開始 (OLS: 最小二乗法) ---")
        print(f"ラベル {x}: 近傍データを収集中...")
        y_list, y_omega_list = _collect_neighbor_data(
            image_data, label_data, Omega_np, int(x), channels
        )
        print("近傍データの収集完了。")

        N = len(y_list)
        # OLSでは正則化がないため、パラメータ数以上のデータが必須
        if N < omega_size * channels: 
            print(f"ラベル {x} の有効なデータ点 ({N}) が不足しているため、スキップします。")
            continue

        Y_raw = np.array(y_list)
        Y_omega_raw = np.array(y_omega_list)

        # --- 3. 平均計算と中心化 ---
        mu_x = np.mean(Y_raw, axis=0)
        print(f"ラベル {x}: データ数 N = {N}, 推定された μ_x = {mu_x}")

        Y_target = Y_raw - mu_x
        Z_feat = (Y_omega_raw - mu_x).reshape(N, -1) # (N, K, C) -> (N, K*C)

        # --- 4. 相関行列の事前計算 ---
        ZtZ = Z_feat.T @ Z_feat
        YtZ = Y_target.T @ Z_feat
        YtY = Y_target.T @ Y_target

        # --- 5. パラメータの確定 (最小二乗法) ---
        try:
            # 正則化項(lambda * I)なしで逆行列を計算
            inv_ZtZ = inv(ZtZ)

            # 係数行列 A_hat
            A_hat_flat = YtZ @ inv_ZtZ

            # 残差二乗和 (RSS) の計算
            rss_matrix = YtY - A_hat_flat @ YtZ.T
            rss = np.trace(rss_matrix)

            # 分散 sigma^2_hat
            sigma2_hat = rss / (channels * N)

            if sigma2_hat <= 1e-9:
                sigma2_hat = 1e-9
            
            # 共分散行列 Sigma_hat
            Sigma_hat = sigma2_hat * np.eye(channels)

            print(f"ラベル {x}: 推定完了。分散 = {sigma2_hat:.6f}")

            # --- 6. 定常性のチェックと係数の調整 ---
            A_hat_x = A_hat_flat.T.reshape(omega_size, channels, channels).transpose(0, 2, 1)

            # コンパニオン行列を作成して固有値をチェック
            dim = channels
            companion_matrix = np.zeros((dim * omega_size, dim * omega_size))
            for i in range(omega_size):
                companion_matrix[0:dim, i*dim:(i+1)*dim] = A_hat_x[i]
            if omega_size > 1:
                companion_matrix[dim:, :-dim] = np.eye(dim * (omega_size - 1))

            eigenvalues = eigvals(companion_matrix)
            max_abs_eigenvalue = np.max(np.abs(eigenvalues))
            print(f"ラベル {x}: コンパニオン行列の最大固有値(絶対値) = {max_abs_eigenvalue}")

            if max_abs_eigenvalue >= 1.0:
                scaling_factor = max_abs_eigenvalue / 0.999
                A_hat_x /= scaling_factor
                print(f"ラベル {x}: 係数が発散的だったため、{scaling_factor:.4f} で縮小しました。")

            print(fr"ラベル {x}: 推定成功")
            estimated_params[x] = {
                'mu_x': mu_x,
                'A_hat_x': A_hat_x,
                'Sigma_hat': Sigma_hat,
                'Omega': Omega
            }

        except (LinAlgError, ValueError) as e:
            print(f"ラベル {x}: 最小二乗法による推定中にエラーが発生しました (特異行列など): {e}。スキップします。", file=sys.stderr)
            continue

    print("\n--- 全ラベルのパラメータ推定が完了しました ---")

    output_data = {
        "ar_param": [],
        "mean": [],
        "variance": [],
        "label_set": sorted(list(estimated_params.keys())),
        "channels": channels
    }

    if not estimated_params:
        print("推定されたパラメータがありません。JSONファイルは保存されません。", file=sys.stderr)
        return

    print("推定結果を指定のJSONフォーマットに変換中...")
    sorted_labels = sorted(estimated_params.keys())

    for label in sorted_labels:
        params = estimated_params[label]
        output_data["mean"].append(params['mu_x'].tolist())
        output_data["variance"].append(params['Sigma_hat'].tolist())

        ar_dict = {}
        omega_list = params['Omega']
        for i, omega_tuple in enumerate(omega_list):
            key = str(omega_tuple)
            ar_dict[key] = params['A_hat_x'][i].tolist()

        output_data["ar_param"].append(ar_dict)
    
    # NumPy型をPython標準型に変換してからJSONに保存
    with open(out_param_json, 'w', encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4, default=int)
    print(f"保存が完了しました: {out_param_json}")

### 確率計算 ###

def get_pixels_in_raster_order(region_tuple: tuple[Node, ...]) -> list[tuple[int, int]]:
    """領域内のピクセル座標をラスタースキャン順で返す"""
    pixels = []
    for node in region_tuple:
        for r in range(node.upper_edge, node.lower_edge):
            for c in range(node.left_edge, node.right_edge):
                pixels.append((r, c))
    return sorted(pixels)

def _log_prob_Y_given_X(
    pixels_in_region_array: np.ndarray,
    img_array: np.ndarray,
    mean_vec: np.ndarray,
    ar_coeffs_mat: np.ndarray, 
    ar_coeffs_off: np.ndarray,
    covariance: np.ndarray
) -> float:
    """
    対数確率計算関数 (汎用チャンネル版, Numbaなし)。
    """
    debug_first_call = not hasattr(_log_prob_Y_given_X, '_debug_printed')
    
    channels = mean_vec.shape[0] # チャンネル数を取得

    if debug_first_call:
        _log_prob_Y_given_X._debug_printed = True
        print(f"[DEBUG _log_prob] First call to _log_prob_Y_given_X")
        print(f"  num_pixels: {pixels_in_region_array.shape[0]}")
        print(f"  img_array shape: {img_array.shape}, dtype: {img_array.dtype}")
        print(f"  mean_vec: {mean_vec}")
        print(f"  covariance shape: {covariance.shape}, dtype: {covariance.dtype}")
        print(f"  initial det(cov): {np.linalg.det(covariance)}")
        print(f"  channels: {channels}")
    
    total_log_prob = 0.0
    
    # 数値安定性のため、共分散行列に微小な正則化を追加
    det_covariance = np.linalg.det(covariance)
    if debug_first_call:
        print(f"[DEBUG _log_prob] Original det_covariance: {det_covariance}")
    
    if det_covariance <= 1e-12:
        # 行列式が極端に小さい場合は正則化を追加
        covariance = covariance + 1e-6 * np.eye(channels)
        det_covariance = np.linalg.det(covariance)
        if debug_first_call:
            print(f"[DEBUG _log_prob] After regularization, det_covariance: {det_covariance}")
    
    if det_covariance <= 0:
        if debug_first_call:
            print(f"[DEBUG _log_prob] det_covariance <= 0, returning -inf")
        return -np.inf

    try:
        inv_covariance = np.linalg.inv(covariance)
        log_det_cov = np.log(det_covariance)
        if debug_first_call:
            print(f"[DEBUG _log_prob] Successfully computed inverse and log_det_cov={log_det_cov}")
    except (LinAlgError, np.linalg.LinAlgError) as e:
        if debug_first_call:
            print(f"[DEBUG _log_prob] Failed to compute inverse: {e}, returning -inf")
        return -np.inf
    
    # チャンネル数に応じた正規化定数
    log_2pi_k = - (channels / 2.0) * np.log(2 * np.pi)

    for r, c in pixels_in_region_array:
        ar_term = np.zeros(channels)
        for i in range(len(ar_coeffs_off)):
            offset = ar_coeffs_off[i]
            neighbor_r, neighbor_c = r + offset[0], c + offset[1]
            # 生成時と同じ条件: ラスタースキャン順で既に処理済みのピクセルを参照
            if 0 <= neighbor_r < r or (neighbor_r == r and 0 <= neighbor_c < c):
                centered_neighbor = np.zeros(channels, dtype=np.float64)
                centered_neighbor[:] = img_array[neighbor_r, neighbor_c] - mean_vec
                ar_term += ar_coeffs_mat[i] @ centered_neighbor

        expected_val = mean_vec + ar_term
        diff = img_array[r, c] - expected_val
        # 二次形式の計算 (数値安定性チェック)
        mahalanobis_dist = diff @ inv_covariance @ diff.T
        
        # 極端に大きな値や無効な値をチェック
        if not np.isfinite(mahalanobis_dist):
            print(f"[DEBUG AR] Non-finite mahalanobis_dist at ({r},{c}): diff={diff}, det_cov={det_covariance}")
            return -np.inf
        if mahalanobis_dist > 1e10:
            print(f"[DEBUG AR] Very large mahalanobis_dist={mahalanobis_dist} at ({r},{c})")
            return -np.inf
            
        log_prob_pixel = log_2pi_k - 0.5 * log_det_cov - 0.5 * mahalanobis_dist
        if not np.isfinite(log_prob_pixel):
            print(f"[DEBUG AR] Non-finite log_prob_pixel={log_prob_pixel} at ({r},{c}): log_det_cov={log_det_cov}, mahal={mahalanobis_dist}")
            return -np.inf
        
        total_log_prob += log_prob_pixel

    return total_log_prob

def log_prob_Y_given_X(
    region_tuple: tuple[Node, ...],
    label: int,
    img_array: np.ndarray,
    theta: dict
) -> float:
    """
    領域rとラベルxが与えられたときの、ARモデルに基づく対数確率計算。
    """
    # デバッグ: 入力をチェック
    debug_first_call = not hasattr(log_prob_Y_given_X, '_debug_printed')
    if debug_first_call:
        log_prob_Y_given_X._debug_printed = True
        print(f"[DEBUG AR] First call to log_prob_Y_given_X")
        print(f"  theta keys: {theta.keys()}")
        print(f"  label_set: {theta.get('label_set', 'NOT FOUND')}")
        print(f"  num of mean vectors: {len(theta.get('mean', []))}")
        print(f"  num of ar_param dicts: {len(theta.get('ar_param', []))}")
    
    try:
        label_idx = theta["label_set"].index(label)
    except (ValueError, KeyError) as e:
        if debug_first_call:
            print(f"[DEBUG AR] Label {label} not found in label_set: {e}")
        return -np.inf
    
    mean_vec = np.array(theta["mean"][label_idx])
    ar_coeffs = {tuple(map(int, k.strip('()').split(','))): np.array(v) for k, v in theta["ar_param"][label_idx].items()}
    covariance = np.array(theta["variance"][label_idx])
    
    if debug_first_call:
        print(f"[DEBUG AR] Label {label} (idx={label_idx}):")
        print(f"  mean_vec shape: {mean_vec.shape}, values: {mean_vec}")
        print(f"  covariance shape: {covariance.shape}")
        print(f"  covariance det: {np.linalg.det(covariance)}")
        print(f"  covariance:\n{covariance}")
        print(f"  num of ar_coeffs: {len(ar_coeffs)}")

    pixels_in_region = get_pixels_in_raster_order(region_tuple)
    if not pixels_in_region:
        return 0.0

    pixels_in_region_array = np.array(pixels_in_region, dtype=np.int64)
    sorted_offsets = sorted(ar_coeffs.keys())
    ar_coeffs_off = np.array(sorted_offsets, dtype=np.int64)
    ar_coeffs_mat = np.array([ar_coeffs[k] for k in sorted_offsets], dtype=np.float64)

    # 入力画像が2次元の場合は3次元(channel=1)に拡張
    if img_array.ndim == 2:
         img_array = img_array[..., np.newaxis]

    result = _log_prob_Y_given_X(
        pixels_in_region_array,
        img_array.astype(np.float64),
        mean_vec.astype(np.float64),
        ar_coeffs_mat,
        ar_coeffs_off,
        covariance.astype(np.float64)
    )
    
    if debug_first_call:
        print(f"[DEBUG AR] _log_prob_Y_given_X returned: {result}")
    
    return result


def log_prob_pixel_given_label(
    center_pixel: np.ndarray,
    neighbor_pixels: np.ndarray,
    label: int,
    theta: dict,
    offsets: list[tuple[int, int]] | None = None,
) -> float:
    """統一インターフェース: 単一画素の AR 対数尤度 log p(y_ij | x, neighbors) を返す。"""
    try:
        label_idx = theta["label_set"].index(label)
    except (ValueError, KeyError):
        return -np.inf

    mean_vec = np.asarray(theta["mean"][label_idx], dtype=np.float64).reshape(-1)
    covariance = np.asarray(theta["variance"][label_idx], dtype=np.float64)
    channels = mean_vec.shape[0]

    center = np.asarray(center_pixel, dtype=np.float64).reshape(-1)
    if center.shape[0] != channels:
        return -np.inf
    if covariance.shape != (channels, channels):
        return -np.inf

    try:
        ar_coeffs = {
            tuple(map(int, k.strip("()").split(","))): np.asarray(v, dtype=np.float64)
            for k, v in theta["ar_param"][label_idx].items()
        }
    except Exception:
        return -np.inf

    if offsets is None:
        offsets = sorted(ar_coeffs.keys())

    neighbors = np.asarray(neighbor_pixels, dtype=np.float64)
    if neighbors.ndim == 1:
        neighbors = neighbors.reshape(-1, 1)

    if neighbors.shape[0] != len(offsets):
        return -np.inf
    if neighbors.shape[1] != channels:
        return -np.inf

    ar_term = np.zeros(channels, dtype=np.float64)
    for idx, off in enumerate(offsets):
        mat = ar_coeffs.get(tuple(off))
        if mat is None:
            continue
        nb = neighbors[idx]
        if not np.all(np.isfinite(nb)):
            continue
        ar_term += mat @ (nb - mean_vec)

    expected_val = mean_vec + ar_term
    diff = center - expected_val

    covariance = covariance + 1e-6 * np.eye(channels)
    try:
        sign, log_det = np.linalg.slogdet(covariance)
        if sign <= 0:
            return -np.inf
        inv_cov = np.linalg.inv(covariance)
    except (LinAlgError, np.linalg.LinAlgError):
        return -np.inf

    mahalanobis_dist = float(diff @ inv_cov @ diff.T)
    if not np.isfinite(mahalanobis_dist):
        return -np.inf

    log_2pi_k = - (channels / 2.0) * np.log(2 * np.pi)
    return float(log_2pi_k - 0.5 * log_det - 0.5 * mahalanobis_dist)

def add_label_set(ar_param_path, label_param_path):
    """
    ar_param.json に label_param.json から読み込んだ label_set を追加する。
    """
    with open(ar_param_path, 'r') as f:
        ar_params = json.load(f)
    
    with open(label_param_path, 'r') as f:
        label_params = json.load(f)

    if ar_params.get("label_set") != label_params["label_set"]:
        ar_params["label_set"] = label_params["label_set"]
        with open(ar_param_path, 'w') as f:
            json.dump(ar_params, f, indent=4)