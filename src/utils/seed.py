import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    固定所有隨機種子，確保實驗可重現 (Reproducibility)
    """
    
    # 1. Python 內建亂數
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy 亂數
    np.random.seed(seed)
    
    # 3. PyTorch 亂數 (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 4. 確保捲積運算結果固定 (會犧牲一點點速度，但為了可重現性是值得的)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[INFO] Random seed set to {seed}")