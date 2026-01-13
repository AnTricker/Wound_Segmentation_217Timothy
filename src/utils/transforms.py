import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=(512, 512)):
    return A.Compose([
        # 1. 隨機翻轉
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # 2. 隨機旋轉與縮放 (增加難度)
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        
        # 3. 亮度對比調整 (模擬不同光照)
        A.RandomBrightnessContrast(p=0.2),
        
        # 注意：我們已經在前處理 resize 好了，所以這裡通常不需要再 Resize
        # 但為了保險起見，可以加一個 Ensure Size
        A.Resize(height=img_size[1], width=img_size[0]),
        
        # 4. 轉 Tensor (這步會自動除以 255 並轉 CHW，如果在 Dataset 手寫了就不用這行)
        # 為了配合上面 Dataset 手寫的邏輯，這裡我先註解掉，讓 Dataset 自己處理 Tensor 轉換
        # ToTensorV2(),
    ])


# 驗證集通常只做 Resize (確保尺寸對)，不做翻轉等破壞性增強
def get_val_transforms(img_size=(512, 512)):
    return A.Compose([
        A.Resize(height=img_size[1], width=img_size[0]),
    ])