import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class WoundDataset(Dataset):
    """
    Wound segmentation dataset.

    Expects:
      images_dir/
        ├─ W_0001.png
        ├─ C_0001.png
        └─ ...
      masks_dir/
        ├─ W_0001.png
        ├─ C_0001.png
        └─ ...
    """
    
    def __init__(
        self, 
        images_dir: str, 
        masks_dir: str, 
        transform=None
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.files = sorted(os.listdir(images_dir))
        
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {images_dir}")
    
    
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        
        img_path = os.path.join(self.images_dir, fname)
        mask_path = os.path.join(self.masks_dir, fname)
        
        # --- read image ---
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        
        # --- read mask ---
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        
        # --- optional augmentation ---
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        # --- to tensor ---
        img = img.astype(np.float32) / 255.0       # (H, W, 3)
        img = np.transpose(img, (2, 0, 1))         # (3, H, W)

        mask = (mask > 0).astype(np.float32)       # (H, W), binary 0/1
        mask = np.expand_dims(mask, axis=0)        # (1, H, W)

        return (
            torch.from_numpy(img),
            torch.from_numpy(mask)
        )