import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from your repository (now in the same directory)
from UNetmodel import UNet
from dataset import SkinLesionDataset
from loss import BCEDiceLoss, calculate_dice

def get_transforms():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ])
    
    val_transforms = A.Compose([
        ToTensorV2()
    ])
    return train_transforms, val_transforms

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Paths pointing to Google Drive
    train_img_dir = "/content/drive/MyDrive/ISIC/training_input"
    train_msk_dir = "/content/drive/MyDrive/ISIC/mask_input"
    # Note: Using training_input for both train and val since validation_input has no masks
    
    train_tf, val_tf = get_transforms()
    
    # Split training data into 80% train, 20% validation
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    all_images = sorted([f for f in os.listdir(train_img_dir) if os.path.splitext(f.lower())[1] in valid_extensions])
    
    np.random.seed(42)
    np.random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    train_dataset = SkinLesionDataset(train_img_dir, train_msk_dir, transform=train_tf, image_files=train_images)
    val_dataset = SkinLesionDataset(train_img_dir, train_msk_dir, transform=val_tf, image_files=val_images)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    num_epochs = 100
    patience = 15
    patience_counter = 0
    best_val_dice = 0.0

    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    # Save models back to Google Drive
    save_dir = "/content/drive/MyDrive/medical-Image-Segmentation/models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        model.eval()
        val_loss = 0.0
        val_dice_scores = []
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, masks in val_loop:
                images, masks = images.to(device), masks.to(device)
                
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
                dice = calculate_dice(logits, masks)
                val_dice_scores.append(dice.item())
                
                val_loop.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        epoch_val_dice = np.mean(val_dice_scores)
        
        history['val_loss'].append(val_loss)
        history['val_dice'].append(epoch_val_dice)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {epoch_val_dice:.4f}")

        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_unet.pth"))
            print(f"-> Saved new best model to {save_dir}/best_unet.pth")
        else:
            patience_counter += 1
            print(f"-> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break
            
        scheduler.step()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice', color='green')
    plt.legend()
    plt.title('Validation Dice Score')
    
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    print(f"Training complete! Curves saved to {save_dir}/training_curves.png")

if __name__ == "__main__":
    train()
