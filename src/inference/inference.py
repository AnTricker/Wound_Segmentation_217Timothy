import os
import random
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from typing import cast
import torch
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.unet import UNet


# --- 參數設定 ---
def get_args():
    
    # 1. 建立解析器 (Parser)
    parser = argparse.ArgumentParser(description="Wound Segmentation Inference")
    
    # 2. 必要參數 (Required) - 沒填會報錯
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run (e.g., unet_v1)")
    parser.add_argument("--dataset", type=str, required=True, choices=["WoundSeg", "CO2Wound"], help="Which dataset to use")
    # parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model")
    
    # 3. 選用參數 (Optional) - 沒填就用預設值 (Default)
    parser.add_argument("--img_size", type=int, default=512, help="Input size for the model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for probability map")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # 4. 打包回傳
    return parser.parse_args()


# --- 視覺化工具 ---
def overlap(img, mask, alpha: float=0.1):
    """
    Overlap mask on image.
    
    Args:
        img : (H, W, 3), uint8, RGB image
        mask : (H, W), uint8, values in {0, 255}
        alpha : Darkening factor of masked region (0~1)
    """
    
    # ---------- 1. 半透明黑色覆蓋 ----------
    cpy_img = img.copy() # 複製原圖，以免改到原始資料
    darkened = (cpy_img * (1 - alpha)).astype(np.uint8)
    cpy_img[mask == 255] = darkened[mask == 255] # mask == 255 代表 mask 中白色的區域 (即傷口/目標區域)
    
    # ---------- 2. 找 mask 邊界 ----------
    contours, _ = cv2.findContours(
        mask,                           # 輸入的二值化圖片 (黑白圖)
        mode=cv2.RETR_EXTERNAL,         # 模式：只取「最外層」的輪廓 (如果傷口中間有洞，忽略洞的邊緣)
        method=cv2.CHAIN_APPROX_SIMPLE  # 近似法：只保留輪廓的轉折點 (節省記憶體，畫直線不需要存線上的每個點)
    )
    
    # ---------- 3. 畫白色邊界線 ----------
    if contours:
        cv2.drawContours(
            cpy_img,                # 要畫在哪張圖上 (剛剛已經局部變暗的那張圖)
            contours,               # 輪廓座標列表
            contourIdx=-1,          # -1 代表畫出列表中的「所有」輪廓
            color=(0, 255, 255),    # 線條顏色：白色 (BGR 格式)
            thickness=1             # 線條粗細：1 pixel
        )
    
    return cpy_img


# --- 影像前處理 ---
def preprocess_image(image_path, image_size):
    
    # 1. 讀取圖片
    img_brg = cv2.imread(image_path)
    if img_brg is None:
        return None, None
    
    # 2. 轉 RGB
    img_rgb = cv2.cvtColor(img_brg, cv2.COLOR_BGR2RGB)
    
    # 3. Resize
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    
    # 4. To Tensor
    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()

    # 5. 加 Batch 維度 (1, C, H, W)
    input_tensor = input_tensor.unsqueeze(0)
    
    return img_resized, input_tensor


def main():
    
    # 1. 拿參數 & 設定裝置
    args = get_args()
    device = torch.device(args.device)
    
    # 2. 自動組裝路徑 (這是為了滿足你「自動分類」的需求)
    # 輸入：去 data/raw/inference/<dataset>_test 找圖
    input_dir = os.path.join(project_root, "data", "raw", "inference", f"{args.dataset}_test")
    # 輸出：存到 outputs/inference/<run_name>/<dataset>
    output_dir = os.path.join(project_root, "outputs", "inference", f"{args.run_name}", f"{args.dataset}")
    # 模型權重路徑
    ckpt_path = os.path.join(project_root, f"outputs/checkpoints/{args.run_name}/best.pt")
    
    # 防呆檢查，如果找不到資料夾，直接結束程式
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # 建立輸出資料夾 (如果資料夾不存在，自動幫你創一個)
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 載入模型結構與權重
    model = UNet(in_channels=3, out_channels=1) # 初始化模型架構 # 注意：這裡的參數 (input_channels=3) 必須跟你訓練時一模一樣
    
    # 讀取權重檔 (肉)
    # map_location=device: 確保權重直接載入到正確的裝置 (GPU/CPU)
    checkpoint = torch.load(ckpt_path, map_location=device)
    # 檢查 checkpoint 的結構，找出真正的權重在哪裡
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        print("Loaded weights from key: 'model_state'")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Loaded weights from key: 'state_dict'")
    else:
        state_dict = checkpoint
        print("Loaded weights directly.")

    model.load_state_dict(state_dict)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"Found {len(image_files)} images. Start Inference...")
    
    # for filename in tqdm(image_files, desc="Processing", unit="img"):
    #     image_path = os.path.join(input_dir, filename)
        
    #     # --- A. 前處理 ---
    #     # vis_img 是 RGB Numpy (512, 512, 3), input_tensor 是 CUDA Tensor (1, 3, 512, 512)
    #     vis_img, input_tensor = preprocess_image(image_path, args.img_size)
    #     if input_tensor is None:
    #         continue
    #     input_tensor = input_tensor.to(device)
        
    #     # --- B. 推論 (Inference) ---
    #     with torch.no_grad():
    #         logits = model(input_tensor)                    # 模型輸出 (未經處理的數值)
    #         probs = torch.sigmoid(logits)                   # 轉成機率 (0~1)
    #         probs_mask = (probs > args.threshold).float()   # 大於 0.5 變 1，小於變 0
        
    #     # --- C. 後處理 (Tensor -> Numpy) ---
    #     # squeeze(): 把 (1, 1, 512, 512) 壓扁成 (512, 512)
    #     # cpu().numpy(): 搬回 CPU 並轉成 Numpy
    #     # astype(np.uint8): 轉成整數
    #     pred_mask_np = probs_mask.squeeze().cpu().numpy().astype(np.uint8)
    #     pred_mask_np = pred_mask_np * 255 # 0/1 變成 0/255 (因為 OpenCV 255 才是白色)
        
    #     # --- D. 疊圖 ---
    #     result_img = overlap(img=vis_img, mask=pred_mask_np)
        
    #     # --- E. 存檔 ---
    #     save_path = os.path.join(output_dir, filename)
    #     cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    
    image_files_temp = random.sample(image_files, 10)
    for filename in tqdm(image_files_temp, desc="Processing", unit="img"):
        image_path = os.path.join(input_dir, filename)
        
        # --- A. 前處理 ---
        # vis_img 是 RGB Numpy (512, 512, 3), input_tensor 是 CUDA Tensor (1, 3, 512, 512)
        vis_img, input_tensor = preprocess_image(image_path, args.img_size)
        if input_tensor is None:
            continue
        input_tensor = input_tensor.to(device)
        
        # --- B. 推論 (Inference) ---
        with torch.no_grad():
            logits = model(input_tensor)                    # 模型輸出 (未經處理的數值)
            probs = torch.sigmoid(logits)                   # 轉成機率 (0~1)
            # Debug: 印出最大機率，確認模型有沒有反應
            print(f"[{filename}] Max: {probs.max().item():.4f}")
            pred_mask = (probs > args.threshold).float()    # 大於 0.5 變 1，小於變 0
        
        # --- C. 後處理 (Tensor -> Numpy) ---
        # squeeze(): 把 (1, 1, 512, 512) 壓扁成 (512, 512)
        # cpu().numpy(): 搬回 CPU 並轉成 Numpy
        # astype(np.uint8): 轉成整數
        pred_mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
        pred_mask_np = pred_mask_np * 255 # 0/1 變成 0/255 (因為 OpenCV 255 才是白色)
        
        # --- D. 疊圖 ---
        result_img = overlap(img=vis_img, mask=pred_mask_np)
        
        # --- E. 存檔 ---
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    
    print("Done! All results saved.")


if __name__ == "__main__":
    main()