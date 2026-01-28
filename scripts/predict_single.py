import os
import sys
import argparse
import glob
import torch
import cv2
import numpy as np
import albumentations as A
import albumentations.pytorch as ToTensorV2

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.models.unet import UNet
from src.engine import infer_one_image
from src.utils import load_checkpoint, make_overlay, make_combine
from data_preprocess.preprocess_split import letterbox_resize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--version", type=str, required=True,
                        help="要使用的模型版本")
    parser.add_argument("--run_name", type=str, required=True,
                        help="第幾次跑")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    checkpoint_path = os.path.join("checkpoints", args.version, args.run_name, "best.pt")
    base_out_dir = "result_single"
    pred_dir = os.path.join(base_out_dir, "predictions")
    viz_dir = os.path.join(base_out_dir, "visualizations")
    overlay_dir = os.path.join(viz_dir, "overlay")
    combine_dir = os.path.join(viz_dir, "combine")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(combine_dir, exist_ok=True)
    
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {DEVICE}")
    
    # 1. 定義影像轉換/預處理
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    ])
    
    # 2. 載入模型
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return
    print("[INFO] Loading model...")
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    load_checkpoint(checkpoint_path, model)
    
    
    # 3. 照片預處理
    user_input = input("Please enter the image name(ex. ncku_001): ")
    img_list = user_input.split(" ")
    img_list = [name + ".jpg" for name in img_list]
    for img_name in img_list:
        img_path = os.path.join("data", "raw", "test", "ncku", img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: 
            print("image not found")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = letterbox_resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        img_float = img.astype(np.float32) / 255.0
        im_transpose = img_float.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(im_transpose)
        
        pred_mask = infer_one_image(model, img_tensor, DEVICE)
        pred_mask = pred_mask.squeeze()
        
        # 存 Predict Mask
        cv2.imwrite(os.path.join(pred_dir, f"{img_name}.png"), pred_mask * 255)
        
        img_vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 存 Visualization 的 Overlay
        overlay = make_overlay(img_vis, pred_mask)
        cv2.imwrite(os.path.join(overlay_dir, f"{img_name}.png"), overlay)
        
        # 存 Visualization 的 Combine
        combine = make_combine(img_vis, pred_mask)
        cv2.imwrite(os.path.join(combine_dir, f"{img_name}.png"), combine)
        print(f"\n✅ {img_name} done! Results: {base_out_dir}")
    print(f"\n✅ All done! Results: {base_out_dir}")


if __name__ == "__main__":
    main()