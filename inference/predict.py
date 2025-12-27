import sys
import os 

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# 設定路徑 hack 以便引用模組
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.unet.unet import UNet
from datasets.wound_dataset import WoundDataset


def save_comparison(model: nn.Module, dataset: Dataset, device: torch.device, index: int, save_dir: str):
    """
    執行推論並儲存對比圖 (Input | Ground Truth | Prediction)。
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get Data
    image, mask = dataset[index]
    # image: (3, H, W), mask: (1, H, W)
    
    model.eval()
    with torch.no_grad():
        input_tensor = image.unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_prob = torch.sigmoid(output)
        pred_mask = (pred_prob > 0.5).float()
    
    # Visualization Prep
    mask_rgb = mask.repeat(3, 1, 1)
    pred_rgb = pred_mask.squeeze(0).cpu().repeat(3, 1, 1)
    
    # Concatenate: [Image, GT, Pred]
    combined = torch.cat([image, mask_rgb, pred_rgb], dim=2)
    
    save_path = os.path.join(save_dir, f"val_compare_{index}.png")
    save_image(combined, save_path)
    print(f"Saved: {save_path}")


def main():
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name: str = "unet_v1"
    
    ckpt_path: str = os.path.join(project_root, f"outputs/checkpoints/{run_name}/best.pt")
    img_dir: str = os.path.join(project_root, "data/processed/images")
    mask_dir: str = os.path.join(project_root, "data/processed/masks")
    save_dir: str = os.path.join(project_root, f"outputs/inference/{run_name}/batch")
    
    # Load Model
    model: nn.Module = UNet(in_channels=3, out_channels=1).to(device)
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found.")
        return
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Handle state dict key mismatch if necessary
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    
    # Dataset
    dataset = WoundDataset(img_dir, mask_dir)
    
    # Test first 5 images
    for i in range(min(5, len(dataset))):
        save_comparison(model, dataset, device, i, save_dir)


if __name__ == "__main__":
    main()