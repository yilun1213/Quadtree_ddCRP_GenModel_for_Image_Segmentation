import os
import glob
import numpy as np
from PIL import Image

def convert_images_to_binary_labels(input_dir, output_dir):
    """
    指定フォルダ内の画像を読み込み、255を1に、0を0に変換して保存します。
    """
    # 出力フォルダ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # pngファイルを取得
    files = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not files:
        print(f"画像が見つかりませんでした: {input_dir}")
        return

    print(f"Processing {len(files)} images...")

    for file_path in files:
        filename = os.path.basename(file_path)
        
        # 1. 画像読み込み & グレースケール変換
        img = Image.open(file_path).convert('L')
        img_arr = np.array(img)
        
        # 2. 変換処理 (閾値処理で堅牢に変換: 127より大なら1, それ以外0)
        # これにより 255 -> 1, 0 -> 0 が確実に行われます
        label_arr = (img_arr > 127).astype(np.uint8)
        
        # 3. 確認用（オプショナル）: ユニークな値を出力
        # print(f"{filename}: unique values = {np.unique(label_arr)}")
        
        # 4. 保存
        # 注意: 0と1だけの画像なので、通常のビューアでは真っ黒に見えます
        output_img = Image.fromarray(label_arr, mode='L')
        save_path = os.path.join(output_dir, filename)
        output_img.save(save_path)
        
    print(f"完了しました。保存先: {output_dir}")

# --- 設定 ---

if __name__ == "__main__":
    convert_images_to_binary_labels(r"IKONOS\train_data\labels\visualize", r"IKONOS\train_data\labels")
    convert_images_to_binary_labels(r"IKONOS\test_data\labels\visualize", r"IKONOS\test_data\labels")