import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        """
        Args:
            alpha: 控制 FP 的權重
            beta:  控制 FN 的權重
            smooth: 平滑項
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits, target):
        """
        Args:
            logits (_type_): _description_
            target (_type_): _description_
        """
        
        # 1. 轉成機率
        probs = torch.sigmoid(logits)
        
        # 2. 攤平 (Flatten)
        # view(-1) 會把所有維度拉成一條長長的向量
        probs_flat = probs.view(-1)
        target_flat = target.view(-1)
        
        # 3. 計算 TP, FP, FN
        TP = (probs_flat * target_flat).sum()
        FP = (probs_flat * (1 - target_flat)).sum()
        FN = ((1 - probs_flat) * target_flat).sum()
        
        tversky_score = (TP + self.smooth) / (TP + (self.alpha * FP) + (self.beta * FN) + self.smooth)
        
        return 1 - tversky_score


class BCETverskyLoss(nn.Module):
    """
    結合 BCE (像素級分類精準度) + Tversky ()
    """
    
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
    
    def forward(self, logits, target):
        loss_bce = self.bce(logits, target)
        loss_tversky = self.tversky(logits, target)
        
        return self.bce_weight * loss_bce + (1 - self.bce_weight) * loss_tversky