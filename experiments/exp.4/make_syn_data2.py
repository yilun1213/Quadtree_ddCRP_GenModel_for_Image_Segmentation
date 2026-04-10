import os
import cv2
import numpy as np

def create_synthetic_image(size=(256, 256)):
    h, w = size
    
    # 各領域の正規分布パラメータ (平均, 標準偏差)
    # N_1 (矩形), N_2 (円), N_3 (背景)
    mu1, std1 = 150, 20
    mu2, std2 = 150, 20
    mu3, std3 = 100, 70 
    
    # [Step 3] 背景の生成 (N_3)
    img = np.random.normal(mu3, std3, (h, w))
    label = np.zeros((h, w), dtype=np.uint8) # 背景ラベルは0
    
    # [Step 1] 大きい円の配置 (N_2)
    img_area = h * w
    # 面積は画像サイズの1/9 ~ 1/4（大きいオブジェクト）
    target_area = np.random.uniform(img_area / 9, img_area / 4)
    radius = int(np.sqrt(target_area / np.pi))
    radius = np.clip(radius, 20, min(h, w) // 2 - 1)

    # 画像からはみ出ないように中心座標をランダム決定
    cx = np.random.randint(radius, w - radius)
    cy = np.random.randint(radius, h - radius)

    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    big_circle_mask = dist_from_center <= radius

    # 円領域に画素値とラベル(2)を代入
    label[big_circle_mask] = 2
    n_circle_pixels = np.sum(big_circle_mask)
    img[big_circle_mask] = np.random.normal(mu2, std2, n_circle_pixels)
    
    # [Step 2] 小さい矩形の配置 (4~8個) (N_1)
    n_rects = np.random.randint(4, 9)
    
    for _ in range(n_rects):
        # 既存の図形と重ならない小さい矩形を探す（最大100回試行）
        for _ in range(100):
            small_area = np.random.uniform(img_area / 400, img_area / 120)
            aspect_ratio = np.random.uniform(0.7, 1.4)

            rect_h = max(6, int(np.sqrt(small_area / aspect_ratio)))
            rect_w = max(6, int(aspect_ratio * rect_h))

            if rect_h >= h or rect_w >= w:
                continue

            x1 = np.random.randint(0, w - rect_w)
            y1 = np.random.randint(0, h - rect_h)
            x2, y2 = x1 + rect_w, y1 + rect_h

            rect_mask = label[y1:y2, x1:x2] > 0

            # 重なりがなければラベル(1)と画素値を代入してループを抜ける
            if not np.any(rect_mask):
                label[y1:y2, x1:x2] = 1
                img[y1:y2, x1:x2] = np.random.normal(mu1, std1, (rect_h, rect_w))
                break
                
    # 画素値を0-255の範囲に収めてuint8に変換
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img, label

def generate_dataset(n_samples, img_dir, lbl_dir, prefix):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    for i in range(n_samples):
        img, label = create_synthetic_image(size=(256, 256))
        
        # 連番でファイル名を生成
        img_path = os.path.join(img_dir, f"{prefix}_{i:03d}.png")
        lbl_path = os.path.join(lbl_dir, f"{prefix}_{i:03d}.png")
        
        cv2.imwrite(img_path, img)
        cv2.imwrite(lbl_path, label)

if __name__ == "__main__":
    # 保存先ディレクトリの設定
    train_img_dir = "./syn_data/train_data/images"
    train_lbl_dir = "./syn_data/train_data/labels"
    test_img_dir = "./syn_data/test_data/images"
    test_lbl_dir = "./syn_data/test_data/labels"
    
    # 学習データの生成 (100枚)
    print("学習データを生成中...")
    generate_dataset(100, train_img_dir, train_lbl_dir, "train")
    
    # テストデータの生成 (5枚)
    print("テストデータを生成中...")
    generate_dataset(5, test_img_dir, test_lbl_dir, "test")
    
    print("データ生成が完了しました。")