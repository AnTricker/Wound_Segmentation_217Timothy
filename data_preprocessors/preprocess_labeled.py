import os
import random
import cv2
import numpy as np
from tqdm import tqdm
import shutil

CLEAR_PROCESSED = True

RAW_ROOT = "data/raw/labeled"
OUT_IMG_DIR = "data/processed/images"
OUT_MASK_DIR = "data/processed/masks"

TARGET_SIZE = (512, 512) # (W, H) for cv2.resize


if CLEAR_PROCESSED:
    print("[INFO] Clearing processed directory...")
    shutil.rmtree("data/processed", ignore_errors=True)
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

def to_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a Gary/RGB Mask into single-channel Binary Mask (0/255)."""
    
    if mask is None:
        raise ValueError("mask is None")
    
    # if mask is RGB/BGR, convert it to Gray
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Ensure uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Resize with NEAREST
    mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    
    # Binarize(二元化): any non-zero -> 255
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def read_and_resize_image(img_path: str) -> np.ndarray:
    """Read images as RGB and resize it to TARGET_SIZE(512 x 512)."""
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    return img


def save_pair(out_name: str, img_rgb: np.ndarray, mask_bin: np.ndarray) -> None:
    # Save image as PNG (BGR for OpenCV write)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT_IMG_DIR, out_name), img_bgr)
    cv2.imwrite(os.path.join(OUT_MASK_DIR, out_name), mask_bin)


def process_woundseg():
    """
    WoundSeg:
      images: <id>.png
      masks : <id>.png (RGB but actually black/white)
    """
    
    name = "WoundSeg"
    prefix = "W"
    img_dir = os.path.join(RAW_ROOT, name, "images")
    mask_dir = os.path.join(RAW_ROOT, name, "masks")
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
    
    count = 1
    skipped = 0
    for f in tqdm(img_files, desc=f"Processing {name}"):
        img_path = os.path.join(img_dir, f)
        mask_path = os.path.join(mask_dir, f)
        
        if not os.path.exists(mask_path):
            skipped += 1
            continue
        
        img = read_and_resize_image(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = to_binary_mask(mask)

        out_name = f"{prefix}_{count:04d}.png"
        save_pair(out_name, img, mask)
        count += 1

    print(f"[{name}] saved: {count - 1}, skipped(no mask): {skipped}")


def find_mask_for_co2(mask_dir: str, base: str):
    """
    CO2Wound images: IMG<id>.jpg
    masks often: IMG<id>.png (grayscale)
    This function tries common extensions for safety.
    """
    
    candidates = [
        os.path.join(mask_dir, base + ".png"),
        os.path.join(mask_dir, base + ".PNG"),
        os.path.join(mask_dir, base + ".jpg"),
        os.path.join(mask_dir, base + ".JPG"),
        os.path.join(mask_dir, base + ".jpeg"),
        os.path.join(mask_dir, base + ".JPEG"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None



def process_co2wound():
    """
    CO2Wound:
      images: IMG<id>.jpg
      masks : IMG<id>.png (grayscale), size varies (4:3)
    """
    
    name = "CO2Wound"
    prefix = "C"
    img_dir = os.path.join(RAW_ROOT, name, "images")
    mask_dir = os.path.join(RAW_ROOT, name, "masks")
    
    # accept jpg/jpeg
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")
    ])
    
    count = 1
    skipped = 0
    for f in tqdm(img_files, desc=f"Processing {name}"):
        img_path = os.path.join(img_dir, f)
        base, _ = os.path.splitext(f)
        mask_path = find_mask_for_co2(mask_dir, base)
        if mask_path is None:
            skipped += 1
            continue
        
        img = read_and_resize_image(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = to_binary_mask(mask)
        
        out_name = f"{prefix}_{count:04d}.png"
        save_pair(out_name, img, mask)
        count += 1
    
    print(f"[{name}] saved: {count - 1}, skipped(no mask): {skipped}")


import random

def quick_sanity_check(n=8, alpha=0.4):
    """
    Sanity check visualization:
    - Randomly sample n images
    - Darken wound area with semi-transparent black
    - Draw white contour on wound boundary
    """
    
    out_dir = "outputs/sanity_check"
    os.makedirs(out_dir, exist_ok=True)

    all_files = os.listdir(OUT_IMG_DIR)
    if len(all_files) == 0:
        print("[sanity_check] No images found.")
        return

    sample_files = random.sample(all_files, min(n, len(all_files)))

    for f in sample_files:
        img = cv2.imread(os.path.join(OUT_IMG_DIR, f))
        mask = cv2.imread(os.path.join(OUT_MASK_DIR, f), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        # ---------- 1. 半透明黑色覆蓋 ----------
        overlay = img.copy()
        darkened = (overlay * (1 - alpha)).astype(np.uint8)
        overlay[mask == 255] = darkened[mask == 255]

        # ---------- 2. 找 mask 邊界 ----------
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ---------- 3. 畫白色邊界線 ----------
        cv2.drawContours(
            overlay,
            contours,
            contourIdx=-1,
            color=(255, 255, 255),  # white (BGR)
            thickness=2
        )

        # ---------- 4. 存檔 ----------
        cv2.imwrite(
            os.path.join(out_dir, f"overlay_{f}"),
            overlay
        )

    print(f"[sanity_check] saved {len(sample_files)} overlays to {out_dir}/")


if __name__ == "__main__":
    process_woundseg()
    process_co2wound()
    quick_sanity_check(n=8)
    