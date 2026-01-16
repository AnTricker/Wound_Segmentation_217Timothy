import torch

def dice_coeff(logits, target, threshold=0.5):
    """
    計算 Dice Coefficient
    Args:
        logits: 模型輸出的原始數值 (B, 1, H, W)，尚未經過 Sigmoid
        target: 真實標籤 (B, 1, H, W)，值為 0 或 1
        smooth: 平滑項，避免除以 0
    Returns:
        dice: 計算出的 Dice 分數
    """
    
    # 1. 轉成機率 (0~1)
    probs = torch.sigmoid(logits)
    
    # 2. 【關鍵差異】這裡要做硬切割 (Hard Thresholding)
    # 大於 0.5 變 1 (傷口)，小於 0.5 變 0 (背景)
    preds = (probs > threshold).float()
    
    # 3. Flatten
    preds_flat = preds.view(-1)
    target_flat = target.view(-1)
    
    # 4. Intersection
    intersection = (preds_flat * target_flat).sum()
    
    # 5. Computing Dice Coeff: (2 * 交集) / (預測總和 + 真實總和)
    dice_score = (2. * intersection) / (preds_flat.sum() + target_flat.sum() + 1e-6)
    
    # Return float (用 .item() 把 tensor 轉成數字)
    return dice_score.item()