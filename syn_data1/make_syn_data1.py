import os
import cv2
import numpy as np

def create_synthetic_image(size=(256, 256)):
    h, w = size
    
    # 各領域の正規分布パラメータ (平均, 標準偏差)
    # N_1 (矩形), N_2 (円), N_3 (背景)
    mu1, std1 = 200, 20
    mu2, std2 = 200, 20
    mu3, std3 = 100, 70 
    
    # [Step 3] 背景の生成 (N_3)
    img = np.random.normal(mu3, std3, (h, w))
    label = np.zeros((h, w), dtype=np.uint8) # 背景ラベルは0
    
    # [Step 1] 矩形の配置 (N_1)
    img_area = h * w
    # 面積は画像サイズの1/16 ~ 1/9
    target_area = np.random.uniform(img_area / 16, img_area / 9)
    # アスペクト比は0.8 ~ 1.2
    aspect_ratio = np.random.uniform(0.8, 1.2)
    
    rect_h = int(np.sqrt(target_area / aspect_ratio))
    rect_w = int(aspect_ratio * rect_h)
    
    # 画像からはみ出ないように左上座標をランダム決定
    max_x = max(1, w - rect_w)
    max_y = max(1, h - rect_h)
    x1 = np.random.randint(0, max_x)
    y1 = np.random.randint(0, max_y)
    x2, y2 = x1 + rect_w, y1 + rect_h
    
    # 矩形領域に画素値とラベル(1)を代入
    label[y1:y2, x1:x2] = 1
    img[y1:y2, x1:x2] = np.random.normal(mu1, std1, (rect_h, rect_w))
    
    # [Step 2] 円の配置 (2~4個) (N_2)
    n_circles = np.random.randint(4, 9)
    
    for _ in range(n_circles):
        # 矩形や他の円と重ならない座標を探す（最大100回試行）
        for _ in range(100):
            radius = np.random.randint(10, 30) # 円の半径
            cx = np.random.randint(radius, w - radius)
            cy = np.random.randint(radius, h - radius)
            
            # 円のマスクを作成
            y_grid, x_grid = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            circle_mask = dist_from_center <= radius
            
            # 既にラベルが配置されている領域（矩形や他の円）との重なり判定
            if not np.any(label[circle_mask] > 0):
                # 重なりがなければラベル(2)と画素値を代入してループを抜ける
                label[circle_mask] = 2
                n_pixels = np.sum(circle_mask)
                img[circle_mask] = np.random.normal(mu2, std2, n_pixels)
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