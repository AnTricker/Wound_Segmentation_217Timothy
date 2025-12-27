import os
import torch
from typing import Tuple, Optional, Union

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metrics: Optional[float] = None) -> None:
    """
    儲存檢查點 (Checkpoint)。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), # 統一 Key 名稱
        "best_metric": best_metrics,
    }, path)


def load_checkpoint(
    path: str, 
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    map_location: Union[str, torch.device] = "cpu"
) -> Tuple[int, Optional[float]]:
    """
    載入檢查點。回傳 (start_epoch, best_metric)。
    """
    print(f"[INFO] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=map_location)
    
    # 1. 載入模型權重
    model.load_state_dict(ckpt["model_state"])
    
    # 2. 載入優化器狀態 (修正 Key Mismatch 的 Bug)
    if optimizer is not None:
        # 為了相容性，我們先檢查新舊兩種 key
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        elif "optim_state" in ckpt: # 相容舊版存檔
            optimizer.load_state_dict(ckpt["optim_state"])
        elif "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print("[WARNING] Optimizer state not found in checkpoint. Training will resume with fresh optimizer.")

    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", None)

    return epoch, best_metric