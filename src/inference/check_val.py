import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.unet import UNet
from datasets import WoundDataset


# --- 視覺化工具 ---
def overlap(img: np.ndarray, mask: np.ndarray, alpha: float=0.1):
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
            color=(255, 255, 255),  # 線條顏色：白色 (BGR 格式)
            thickness=1             # 線條粗細：1 pixel
        )
    
    return cpy_img


def inference_and_save(model: nn.Module, dataset: Dataset, device: torch.device, idx: int, save_dir: str):
    """
    Run inference on an image from the dataset，then concatenate and save the [original img, real mask, predict mask]

    Args:
        model (nn.Module): Trained segmentation model.
        dataset (Dataset): Dataset containing (image, mask) pairs.
        device (torch.device): Device to run inference on (CPU or cuda).
        index (int): Index of the image to be tested.
        save_dir (str): Directory to save the output visualization.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # --- A. 從 Dataset 裡拿資料 (這已經是處理好、Resize 過的 Tensor) ---
    # image: (3, H, W), mask: (1, H, W)
    image, mask = dataset[idx]
    
    # --- B. 模型推論 ---
    model.eval()
    with torch.no_grad():
        # 增加 batch 的維度: (3, H, W) -> (1, 3, H, W)
        # 模型一次是吃一整批 (Batch, Channel, Height, Width)
        input_tensor = image.unsqueeze(0).to(device)
        logits = model(input_tensor) # 把它丟進模型，取得輸出（尚未經果Sigmoid的數值）
        
        # 後處理
        probs = torch.sigmoid(logits) # 把數值壓到 0~1 之間（機率）
        pred_mask = (probs > 0.5).float() # >0.5 我們都算 1（傷口），小於算 0（背景
    
    # --- C. 視覺化準備 ---
    # 1. 處理 Image (C, H, W) -> (H, W, C)
    # 假設 dataset 裡的 image 已經是 0~1 的 float，如果是有 normalize 過 (mean/std)，這裡要先 denormalize 回來
    image_np = image.permute(1, 2, 0).cpu().numpy() # 把 (C, H, W) 維度轉為 (H, W, C) 並把 Tensor 轉為 numpy
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8) # 0/1 -> 0/255
    
    # 2. 處理 Ground Truth Mask (1, H, W) -> (H, W)
    mask_np = mask.squeeze(0).cpu().numpy() # squeeze(0) 是把 channel 維度拿掉，從 (1, H, W) 變回 (H, W)
    mask_np = (mask_np * 255).astype(np.uint8) # 0/1 -> 0/255
    
    # 3. 處理 Predict Mask (1, 1, H, W) -> (H, W)
    pred_np = pred_mask.squeeze().cpu().numpy() # squeeze() 是把 batch 與 channel 維度拿掉，從 (1, 1, H, W) 變回 (H, W)
    pred_np = (pred_np * 255).astype(np.uint8) # 0/1 -> 0/255
    
    # --- D. 疊圖 ---
    # torch.cat 是接合 Tensor
    # dim=2 代表在「寬度」方向拼接 (左圖、中圖、右圖)
    # dim=1 代表在「高度」方向拼接 (上圖、中圖、下圖)
    vis_origin = overlap(image_np, mask_np)
    vis_predict = overlap(image_np, pred_np)
    
    # --- E. 拼接與存檔 ---
    combined = np.hstack([vis_origin, vis_predict])
    save_path = os.path.join(save_dir, f"val_compare_{idx}.png")
    cv2.imwrite(save_path, combined)
    print(f"✅ Saved comparison: {save_path}")


def main():
    # --- A. 設定參數 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = "unet_v1"
    
    # 設定路徑
    image_path = os.path.join(project_root, "data/processed/images")
    mask_path = os.path.join(project_root, "data/processed/masks")
    
    # 模型權重路徑
    ckpt_path = os.path.join(project_root, f"outputs/checkpoints/{run_name}/best.pt")
    
    # 輸出路徑
    output_dir = os.path.join(project_root, f"outputs/inference/{run_name}/check_val")
    
    # --- B. 初始化模型 ---
    # 先建立一個空的 UNet 架構 (還沒有訓練過的腦袋)
    model = UNet(in_channels=3, out_channels=1).to(device=device)
    
    # --- C. 載入權重 (把訓練好的知識灌進去) ---
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # 這裡是用來處理我們之前 code 改版造成的 Key 不一致問題
        # 1. 先看有沒有 'model_state' (新版)
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        # 2. 再看有沒有 'model_state_dict' (舊版)
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        # 3. 如果都沒有，可能整個檔案就是 state_dict
        else:
            state_dict = checkpoint 
            
        model.load_state_dict(state_dict) # 真正的載入動作
        print("Model loaded successfully.")
    else:
        print("❌ Checkpoint not found!")
        return
    
    # --- D. 載入 Dataset ---
    # 這裡很方便，直接用我們之前寫好的 Class 幫忙讀檔
    dataset = WoundDataset(images_dir=image_path, masks_dir=mask_path)
    print(f"Dataset size: {len(dataset)}")
    
    # --- E. 執行迴圈 ---
    # min(10, len(dataset)) 的意思是：
    # 如果資料集有 300 張，我只跑前 10 張。
    # 如果資料集只有 3 張，我就跑 3 張 (避免報錯)。
    num_to_test = min(10, len(dataset))
    
    for i in range(num_to_test):
        # 呼叫上面的大腦，一張一張處理
        inference_and_save(model, dataset, device, i, output_dir)


if __name__ == "__main__":
    main()