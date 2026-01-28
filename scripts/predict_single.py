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
    img_path = os.path.join("data")
    img = cv2.imread(img_path)
    letterbox_resize(img, (IMAGE_SIZE, IMAGE_SIZE))


if __name__ == "__main__":
    main()