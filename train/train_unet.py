import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.wound_dataset import WoundDataset
from models.unet.unet import UNet


def main():
    # -----------------------------
    # 基本設定
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    img_dir = "data/processed/images"
    mask_dir = "data/processed/masks"
    
    batch_size = 2
    num_epochs = 2
    lr = 1e-4
    
    # -----------------------------
    # Dataset / DataLoader
    # -----------------------------
    dataset = WoundDataset(
        images_dir=img_dir,
        masks_dir=mask_dir,
        transform=None
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Dataset Size: {len(dataset)}")
    
    # -----------------------------
    # Model
    # -----------------------------
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(device)
    
    # -----------------------------
    # Loss / Optimizer
    # -----------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # -----------------------------
    # Training loop
    # -----------------------------
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for step, (images, masks) in enumerate(dataloader):
            images = images.to(device)   # (B, 3, H, W)
            masks = masks.to(device)     # (B, 1, H, W)

            # forward
            outputs = model(images)      # (B, 1, H, W)
            loss = criterion(outputs, masks)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step [{step}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}] Average Loss: {avg_loss:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    main()