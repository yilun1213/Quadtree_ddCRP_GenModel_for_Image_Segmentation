# ./model/label/geom_glm.py
from __future__ import annotations
import os
import numpy as np
from PIL import Image
import sys
import json
import utils
from scipy import ndimage
from skimage.measure import regionprops, perimeter
from skimage.morphology import convex_hull_image
from .geom_features import compute_geom_features, DEFAULT_FEATURE_NAMES

def _compute_phi(region: set[tuple[int, int]], param: dict) -> np.ndarray:
    """
    論文の式に基づく特徴量ベクトル φ(r) を計算する。
    
    論文 2.5.1節の Example に記載の通り、領域 r の幾何学的特徴量を抽出。
    φ(r) = (φ_1(r), φ_2(r), ..., φ_d(r))^T
    
    本実装では d=8 (幾何特徴6次元 + 位置情報2次元)
    - φ_1, φ_2: Area(r), Prm(r) の対数
    - φ_3: Cir(r) (円形度)
    - φ_4, φ_5, φ_6: Asp, Rect, Sol (形状特徴)
    - φ_7, φ_8: 正規化重心座標
    
    戻り値: shape (8,) の特徴ベクトル
    """
    # 画像サイズを取得（デフォルトは128）
    image_size = param.get("image_size", 128)
    feature_names = param.get("feature_names", DEFAULT_FEATURE_NAMES)
    phi = compute_geom_features(region, image_size=image_size, feature_names=feature_names)
    return phi  # (8,)

def _compute_log_prob_normal(phi: np.ndarray, param: dict) -> np.ndarray:
    """
    論文の式に基づき、各ラベルxに対する正規分布の対数確率を計算する。
    
    p_x(φ(r)) ∝ Π_i p_x(φ_i(r))
    ここで p_x(φ_i(r)) ~ N(m_i^(x), σ_i^(x)^2)
    
    対数を取ると:
    log p_x(φ(r)) = Σ_i log p_x(φ_i(r))
                  = Σ_i [-0.5 log(2π) - 0.5 log(σ_i^(x)^2) - 0.5((φ_i - m_i^(x))/σ_i^(x))^2]
    
    param: {"means": [[m_1^(0), ..., m_d^(0)], ..., [m_1^(K-1), ..., m_d^(K-1)]],
            "stds": [[σ_1^(0), ..., σ_d^(0)], ..., [σ_1^(K-1), ..., σ_d^(K-1)]]}
    
    戻り値: shape (K,) の対数確率ベクトル（正規化前）
    """
    means = np.asarray(param["means"], dtype=float)  # (K, d)
    stds = np.asarray(param["stds"], dtype=float)    # (K, d)
    
    # 各ラベルxに対して log p_x(φ) を計算
    # phi: (d,), means: (K, d), stds: (K, d)
    # ブロードキャストで (K, d) の差分を計算
    diff = phi - means  # (K, d)
    
    # 各特徴量について正規分布の対数確率密度を計算
    # log N(φ_i; m_i^(x), σ_i^(x)^2) = -0.5 log(2πσ^2) - 0.5((φ_i - m_i)/σ)^2
    log_prob = -0.5 * np.log(2 * np.pi * stds**2) - 0.5 * (diff / stds)**2  # (K, d)
    
    # 各ラベルについて全特徴量の対数確率を合計（独立性の仮定）
    log_prob_sum = np.sum(log_prob, axis=1)  # (K,)
    
    return log_prob_sum

def log_label_prior(region: set[tuple[int, int]], param: dict) -> np.ndarray:
    """
    論文の式に基づくラベルの対数事前確率を計算する。
    
    論文 2.5.1節の Example より:
    p_{r,x} = p_x(φ(r)) / Σ_{x'∈X} p_{x'}(φ(r))
    
    ここで p_x(φ(r)) ∝ Π_i p_x(φ_i(r))
    各特徴量 φ_i(r) はラベルxごとの正規分布 N(m_i^(x), σ_i^(x)^2) に従う。
    
    対数を取ると:
    log p_{r,x} = log p_x(φ(r)) - log(Σ_{x'} p_{x'}(φ(r)))
    
    戻り値: shape (K,) の対数確率ベクトル（正規化済み）
    """
    phi = _compute_phi(region, param)
    log_prob = _compute_log_prob_normal(phi, param)  # (K,)
    
    # log-sum-exp トリックで数値安定性を確保
    log_prob_max = np.max(log_prob)
    log_Z = log_prob_max + np.log(np.sum(np.exp(log_prob - log_prob_max)))
    
    return log_prob - log_Z

def label_prior(region: set[tuple[int, int]], param: dict) -> np.ndarray:
    """
    論文の式に基づくラベルの事前確率を計算する。
    
    論文 2.5.1節、Assumption内の式:
    p(x_r; p) = Categorical(p_r) = Π_{x=0}^{|X|-1} p_{r,x}^{1{x_r = x}}
    
    ここで p_{r,x} は領域 r がラベル x を持つ確率で、
    幾何学的特徴量 φ(r) の正規分布に基づいて決定される:
    
    p_{r,x} = p_x(φ(r)) / Σ_{x'∈X} p_{x'}(φ(r))
    p_x(φ(r)) ∝ Π_i N(φ_i(r); m_i^(x), σ_i^(x)^2)
    
    戻り値: shape (K,) の確率ベクトル (合計1に正規化)
    """
    logp = log_label_prior(region, param) 
    logp_max = np.max(logp)
    p = np.exp(logp - logp_max)
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        # 念のためのフォールバック：一様分布
        K = p.shape[0]
        return np.full(K, 1.0 / K, dtype=float)
    return p / s

    # --- ヘルパー関数 (1): 論文に基づく8次元特徴量を抽出 ---
def extract_features(region_mask, image_size=128, feature_names=None):
    """
    論文 2.5.1節の Example に基づき、領域マスクから幾何的特徴量 φ(r) を抽出する。
    
    特徴量の定義:
    1. log(Area(r)) = log(|r|)  - 面積の対数
    2. log(Prm(r)) = log(|∂r|)  - 周長の対数
    3. Cir(r) = 4πA / P^2       - 円形度
    4. Asp(r)                   - アスペクト比（MBRの長辺/短辺）
    5. Rect(r) = A / MBR面積    - 矩形度
    6. Sol(r) = A / ConvexArea  - 充填率（凸包に対する面積比）
    7. CentroidI / image_size   - 正規化重心i座標
    8. CentroidJ / image_size   - 正規化重心j座標
    
    面積と周長は log1p (log(1+x)) で対数変換し、数値安定性を向上。
    位置情報により、画像内での位置依存のラベル分布を表現可能。
    
    戻り値: shape (8,) の特徴ベクトル、または None (無効な領域の場合)
    """
    
    coords = np.argwhere(region_mask)
    if coords.size == 0:
        return None

    region = {(int(i), int(j)) for i, j in coords}
    names = feature_names or DEFAULT_FEATURE_NAMES
    features = compute_geom_features(region, image_size=image_size, feature_names=names)
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return None
    return features

# --- ヘルパー関数 (2): 正規分布の確率密度関数 ---
def normal_pdf(x, mean, std):
    """
    1次元正規分布の確率密度関数
    N(x; μ, σ^2) = (1/√(2πσ^2)) exp(-(x-μ)^2 / (2σ^2))
    """
    return (1.0 / np.sqrt(2 * np.pi * std**2)) * np.exp(-0.5 * ((x - mean) / std)**2)

# --- メインの推定関数 ---

def param_est(
    train_label_dir: str,
    label_set: list[int],
    label_num: int,
    image_size: int = 128,
    feature_names: list[str] | None = None,
):
    
    # --- ユーザー提供のコード開始 ---
    # (ファイル名リストの定義を修正)
    filename_list = utils.get_image_files(train_label_dir)
    if not filename_list:
        print(f"エラー: {train_label_dir} に画像が見つかりません。", file=sys.stderr)
        return {}
        
    data_num = len(filename_list)
    label_data = []

    print(f"{data_num} 組の学習データを読み込み中...")

    for filename in filename_list:
        try:
            label_path = os.path.join(train_label_dir, filename)
            label_data.append(utils.load_image(label_path))
        except Exception as e:
            print(f"警告: {filename} の読み込みに失敗しました: {e}", file=sys.stderr)

    print("データ読み込み完了。")

    # --- ここからが続き ---

    # --- ステップ1: 全画像から特徴量 (x_r, φ(r)) を抽出 ---
    # 論文 2.5.1節の Example に基づき、各領域 r について
    # 幾何学的特徴量 φ(r) と対応するラベル x_r のペアを抽出
    print(r"ステップ1: 幾何的特徴量 φ(r) を抽出中...")
    
    all_training_data = [] # (true_label, phi_r) のタプルのリスト
    
    selected_features = list(feature_names) if feature_names else list(DEFAULT_FEATURE_NAMES)

    for lbl_img in label_data:
        # 画像内のユニークなラベル（0:背景 は除くことが多いが、今回は含める）
        labels_in_img = np.unique(lbl_img)
        
        for x in labels_in_img:
            # ラベル x のピクセルのみを抽出
            mask = (lbl_img == x)
            
            # 連結成分（領域 R）を検出
            # neighbourhood=1 (4-近傍) or 2 (8-近傍)
            labeled_mask, num_regions = ndimage.label(mask)
            
            if num_regions == 0:
                continue
                
            # 検出された各領域 r について特徴量を抽出
            for r in range(1, num_regions + 1):
                region_mask = (labeled_mask == r)
                
                # φ(r) を計算（論文 2.5.1節の幾何学的特徴量）
                # 8次元ベクトル: [log(Area), log(Prm), Cir, Asp, Rect, Sol, CentI, CentJ]
                phi_r = extract_features(
                    region_mask,
                    image_size=image_size,
                    feature_names=selected_features,
                )
                
                # (x_r, φ(r)) のペアを保存 (None でない場合)
                if phi_r is not None:
                    all_training_data.append((x, phi_r))

    if not all_training_data:
        print("エラー: 有効な訓練データ（領域）が1つも見つかりません。", file=sys.stderr)
        return {}

    # --- ステップ2: 各ラベルごとにデータを分類 ---
    labels = sorted(label_set)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    K = label_num  # クラス数 |X|
    d = len(selected_features)
    
    # 各ラベルごとの特徴量リストを作成
    features_by_label = {label: [] for label in labels}
    for (true_label_val, phi_r) in all_training_data:
        features_by_label[true_label_val].append(phi_r)
    
    print(f"特徴量抽出完了。全 {len(all_training_data)} 領域、{K} クラス、{d}次元特徴量。")
    print("ステップ2: 最尤推定によるパラメータ推定（正規分布の平均と標準偏差）...")

    # --- ステップ3: 最尤推定による正規分布パラメータの計算 ---
    # 論文 2.5.1節の Example に基づき、各ラベル x に対して
    # 各特徴量 φ_i が正規分布 N(m_i^(x), σ_i^(x)^2) に従うと仮定し、
    # 最尤推定（標本平均と標本標準偏差）でパラメータを推定する。
    
    means = np.zeros((K, d))  # 各ラベル、各特徴量の平均 m_i^(x)
    stds = np.zeros((K, d))   # 各ラベル、各特徴量の標準偏差 σ_i^(x)
    
    for label in labels:
        idx = label_to_index[label]
        features = features_by_label[label]
        
        if len(features) == 0:
            # データがない場合はデフォルト値を設定
            print(f"  警告: ラベル {label} のデータが0件です。デフォルト値を使用します。")
            means[idx, :] = 0.0
            stds[idx, :] = 1.0
            continue
        
        # 特徴量を配列に変換 (N_label, d)
        features_array = np.array(features)
        
        # 最尤推定: 標本平均と標本標準偏差
        means[idx, :] = np.mean(features_array, axis=0)
        if len(features) < 2:
            # データ数が1だと不偏推定(ddof=1)でNaNになるため母分散で計算
            stds[idx, :] = np.std(features_array, axis=0, ddof=0)
            print(f"  警告: ラベル {label} のデータが1件のため、ddof=0で標準偏差を計算します。")
        else:
            stds[idx, :] = np.std(features_array, axis=0, ddof=1)  # 不偏推定量
        
        # 標準偏差が0の場合（すべての値が同じ）は小さな値を設定
        stds[idx, :] = np.maximum(stds[idx, :], 1e-6)
        
        print(f"  ラベル {label}: {len(features)} 個の領域")
        print(f"    平均 m^({label}): {means[idx, :]}")
        print(f"    標準偏差 σ^({label}): {stds[idx, :]}")
    
    print("\n正規分布パラメータの推定完了。")
    
    # --- ステップ4: 推定パラメータを指定のJSON形式に変換 ---
    # 正規分布パラメータ（平均と標準偏差）を保存

    output_params = {
        "means": [],   # 各ラベル、各特徴量の平均 m_i^(x)
        "stds": [],    # 各ラベル、各特徴量の標準偏差 σ_i^(x)
        "image_size": image_size,  # 画像サイズを保存（推論時に必要）
        "feature_names": selected_features,
    }

    print("\n推定結果を指定のJSONフォーマットに変換中...")
    
    try:
        # ラベル 0, 1, 2... のインデックス順にソートして処理する
        sorted_labels_by_index = sorted(label_to_index.items(), key=lambda item: item[1])
        
        for label_val, idx in sorted_labels_by_index:
            # 平均と標準偏差を float のリストに変換
            mean_list = [float(m) for m in means[idx, :]]
            std_list = [float(s) for s in stds[idx, :]]
            
            output_params["means"].append(mean_list)
            output_params["stds"].append(std_list)

    except Exception as e:
        print(f"データ変換中に予期せぬエラー: {e}", file=sys.stderr)
        return {} # エラー時はここで中断

    # --- ステップ5: 推定パラメータを辞書として返す ---
    print("パラメータ推定完了。")
    return output_params