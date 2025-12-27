import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from typing import Tuple, List, cast

# --- 路徑設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.unet import UNet

# 支援的圖片格式
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def load_and_preprocess(image_path: str, device: torch.device, target_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """
    讀取並預處理圖片。
    Args:
        image_path (str): 圖片路徑
        device (torch.device): 裝置
        target_size (Tuple[int, int]): Resize 大小
    Returns:
        torch.Tensor: (1, 3, H, W)
    """
    # 1. 讀取圖片
    pil_image = Image.open(image_path).convert("RGB")
    
    # 2. 定義轉換
    transform_pipeline = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    
    # 3. 執行轉換 (這裡拿掉了 : torch.Tensor 的強制宣告，讓 Python 自動推斷)
    img_tensor = cast(torch.Tensor, transform_pipeline(pil_image))
    
    # 4. 增加 Batch 維度並移至 Device
    # Pylance 可能會在這裡提示 img_tensor 型別不明，但執行時這是 100% 正確的
    return img_tensor.unsqueeze(0).to(device)


def process_single_image(model: nn.Module, image_path: str, output_path: str, device: torch.device) -> None:
    """
    對單張圖片進行推論並存檔。
    Args:
        model (nn.Module): 模型
        image_path (str): 輸入圖片路徑
        output_path (str): 輸出結果路徑
        device (torch.device): 裝置
    """
    # 1. Preprocess
    input_tensor = load_and_preprocess(image_path, device, target_size=(512, 512))
    
    # 2. Inference
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()
        
    # 3. Save (Cat Input + Pred)
    mask_rgb = mask.repeat(1, 3, 1, 1)
    combined = torch.cat([input_tensor, mask_rgb], dim=3)
    
    save_image(combined, output_path)
    print(f"✅ Saved: {os.path.basename(output_path)}")


def process_folder(model: nn.Module, input_dir: str, output_dir: str, device: torch.device) -> None:
    """
    遍歷整個資料夾進行推論。
    Args:
        model (nn.Module): 模型
        input_dir (str): 輸入資料夾路徑
        output_dir (str): 輸出資料夾路徑
        device (torch.device): 裝置
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 取得所有圖片檔案
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
    
    print(f"📂 Found {len(files)} images in {input_dir}")
    
    for filename in files:
        img_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        
        process_single_image(model, img_path, out_path, device)


def main() -> None:
    # --- 1. 設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = "unet_v1"
    
    ckpt_path = os.path.join(project_root, f"outputs/checkpoints/{run_name}/best.pt")
    
    # 輸入資料夾 (WoundSeg)
    input_folder_name = "WoundSeg" 
    input_root_dir = os.path.join(project_root, "data/raw/inference_only", input_folder_name)
    
    # 輸出資料夾
    output_root_dir = os.path.join(project_root, f"outputs/inference/{run_name}", input_folder_name)
    
    # --- 2. 載入模型 ---
    print(f"Using Device: {device}")
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    print("✅ Model loaded successfully.")
    
    # --- 3. 執行資料夾推論 ---
    if os.path.exists(input_root_dir):
        process_folder(model, input_root_dir, output_root_dir, device)
        print(f"\n🎉 All Done! Results saved to: {output_root_dir}")
    else:
        print(f"❌ Input folder not found: {input_root_dir}")


if __name__ == "__main__":
    main()