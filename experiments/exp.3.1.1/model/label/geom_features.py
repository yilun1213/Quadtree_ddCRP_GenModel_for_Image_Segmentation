# ./model/label/geom_features.py
from __future__ import annotations
import numpy as np

DEFAULT_FEATURE_NAMES = [
    "log_area",
    "log_perimeter",
    "circularity",
    "aspect_ratio",
    "rectangularity",
    "solidity",
    "centroid_i",
    "centroid_j",
]

def _neighbors4(i: int, j: int) -> list[tuple[int, int]]:
    return [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]

def _perimeter_4n(region_set: set[tuple[int, int]]) -> float:
    """4近傍で外部に接する“辺”の総数（離散周長）"""
    perim = 0
    for (i, j) in region_set:
        for nb in _neighbors4(i, j):
            if nb not in region_set:
                perim += 1
    return float(perim)

def _area(region_set: set[tuple[int, int]]) -> float:
    return float(len(region_set))

def _centroid(points: np.ndarray) -> np.ndarray:
    # points.shape = (N, 2) with columns (i, j)
    if points.size == 0:
        return np.array([0.0, 0.0], dtype=float)
    return points.mean(axis=0)

def _pca_project(centered: np.ndarray) -> np.ndarray:
    """
    centered: (N,2) （重心で中心化済みの座標）
    戻り値: 主成分軸への射影座標 uv (N,2)
    """
    if centered.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    cov = (centered.T @ centered) / centered.shape[0]  # 母共分散 1/N
    w, V = np.linalg.eigh(cov)                         # w昇順
    idx = np.argsort(w)[::-1]                          # 降順へ
    V = V[:, idx]
    uv = centered @ V
    return uv  # uv[:,0]=u, uv[:,1]=v

def _mbr_metrics(uv: np.ndarray, area: float) -> tuple[float, float, float]:
    """
    PCA主軸系でのMBR: Asp, MBR面積, Rect(A/MBR) を返す
    """
    eps = 1e-12
    if uv.shape[0] == 0:
        return 1.0, eps, area / eps
    u = uv[:, 0]
    v = uv[:, 1]
    u_len = float(u.max() - u.min()) if u.size else 0.0
    v_len = float(v.max() - v.min()) if v.size else 0.0
    long_side = max(u_len, v_len)
    short_side = max(min(u_len, v_len), eps)
    asp = long_side / short_side
    mbr_area = max(u_len * v_len, eps)
    rect = area / mbr_area
    return asp, mbr_area, rect

def _convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Monotone chain（Andrew法）で凸包（反時計回り）を返す
    points: (N,2) with (i,j)
    """
    pts = np.unique(points, axis=0)
    if pts.shape[0] <= 1:
        return pts.astype(float)

    # sort by i, then j
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])

    lower = []
    for p in pts:
        p = tuple(p)
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        p = tuple(p)
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull

def _polygon_area(poly: np.ndarray) -> float:
    """靴紐公式で多角形面積を返す（poly: (M,2)）。凸包点列は閉路でなくてOK。"""
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]; y = poly[:, 1]
    s = (x * np.roll(y, -1) - y * np.roll(x, -1)).sum()
    return abs(s) * 0.5

def compute_geom_features(
    region: set[tuple[int, int]],
    image_size: int = 128,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """
    論文の式に基づく幾何学的特徴量ベクトル φ(r) を計算する。
    
    φ(r) = [log(Area), log(Prm), Cir, Asp, Rect, Sol, CentroidI, CentroidJ]^T
    
    各特徴量の定義（論文 2.5.1節）:
    - Area(r) = |r|  (領域の面積)
    - Prm(r) = |∂r|  (離散境界辺数、4近傍での周長)
    - Cir(r) = 4πA / P^2  (円形度、コンパクトさの指標)
    - Asp = 長辺/短辺  (PCA主軸MBRのアスペクト比、>=1)
    - Rect = A / MBR面積  (矩形度)
    - Sol = A / ConvexHull(r)の面積  (充填率)
    - CentroidI, CentroidJ: 正規化された重心座標 [0,1]
    
    面積と周長は対数スケール log(1+x) に変換して数値安定性を向上。
    位置情報により、画像内での位置に依存するラベル分布を表現可能。
    （例：空は画像上部、地面は下部に出現しやすい）
    
    戻り値: shape (len(feature_names),) の特徴ベクトル
    """
    # 基本量
    A = _area(region)
    P = _perimeter_4n(region)
    P = max(P, 1.0)                     # 0割回避
    Cir = 4.0 * np.pi * A / (P * P)

    # PCAベース
    pts = np.array(list(region), dtype=float)  # (N,2)
    cen = _centroid(pts)
    centered = pts - cen
    uv = _pca_project(centered)
    Asp, MBR_area, Rect = _mbr_metrics(uv, A)

    # Solidity
    hull = _convex_hull(pts)
    hull_area = max(_polygon_area(hull), 1e-12)
    Sol = A / hull_area
    
    # 位置情報（正規化された重心座標）
    # 画像サイズで正規化することで [0,1] 範囲に収める
    centroid_i = cen[0] / float(image_size)
    centroid_j = cen[1] / float(image_size)
    
    # 面積と周長は対数スケールに変換（勾配降下法の安定性向上）
    log_area = np.log1p(A)  # log(1+A) でゼロ付近も安全
    log_prm = np.log1p(P)

    all_features = {
        "log_area": log_area,
        "log_perimeter": log_prm,
        "circularity": Cir,
        "aspect_ratio": Asp,
        "rectangularity": Rect,
        "solidity": Sol,
        "centroid_i": centroid_i,
        "centroid_j": centroid_j,
    }

    names = feature_names or DEFAULT_FEATURE_NAMES
    unknown = [name for name in names if name not in all_features]
    if unknown:
        raise ValueError(f"Unknown geometric feature names: {unknown}")

    return np.array([all_features[name] for name in names], dtype=float)
