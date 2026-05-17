import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow

from src.UNetmodel import UNet
from src.dataset import SkinLesionDataset
from src.loss import calculate_dice

def calculate_iou(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def evaluate():
    
    mlflow.set_experiment("Skin_Lesion_Segmentation_UNet")
    
    with mlflow.start_run(run_name="Evaluation_Only") as run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Evaluating on: {device}")

       
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "best_unet.pth")
        image_dir = os.path.join(base_dir, "data", "ISIC_Training_Input")
        mask_dir = os.path.join(base_dir, "data", "ISIC_Training_GroundTruth")

        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}. Please update model_path.")
            return
            
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print(f"Error: Data not found at {image_dir} or {mask_dir}. Please update image_dir/mask_dir.")
            return

        print("Loading model...")
        model = UNet(in_channels=3, out_channels=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        val_transforms = A.Compose([ToTensorV2()])

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        all_images = sorted([f for f in os.listdir(image_dir) if os.path.splitext(f.lower())[1] in valid_extensions])
        
        # Taking the same 20% validation split as during training
        split_idx = int(0.8 * len(all_images))
        val_images = all_images[split_idx:]

        val_dataset = SkinLesionDataset(image_dir, mask_dir, transform=val_transforms, image_files=val_images)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        dice_scores = []
        iou_scores = []

        from tqdm import tqdm
        print("Calculating metrics on Validation set...")
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Evaluating Images"):
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                
                dice = calculate_dice(logits, masks)
                iou = calculate_iou(logits, masks)
                
                dice_scores.append(dice.item())
                iou_scores.append(iou.item())

        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)
        
        print(f"Mean Validation Dice: {mean_dice:.4f}")
        print(f"Mean Validation IoU: {mean_iou:.4f}")

        # Log these final metrics to MLflow
        mlflow.log_metrics({
            "eval_mean_dice": mean_dice,
            "eval_mean_iou": mean_iou
        })
        mlflow.log_param("model_evaluated", model_path)

        # Generate Visualizations
        print("Generating Prediction Plots...")
        num_samples = min(5, len(val_dataset))
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                if i >= num_samples:
                    break
                    
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                img_np = images[0].cpu().numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np, 0, 1) 
                mask_np = masks[0].cpu().numpy().squeeze()
                pred_np = preds[0].cpu().numpy().squeeze()

                axes[i, 0].imshow(img_np)
                axes[i, 0].set_title("Input Image")
                axes[i, 0].axis('off')

                axes[i, 1].imshow(mask_np, cmap='gray')
                axes[i, 1].set_title("Ground Truth Mask")
                axes[i, 1].axis('off')

                axes[i, 2].imshow(pred_np, cmap='gray')
                axes[i, 2].set_title("Predicted Mask")
                axes[i, 2].axis('off')

        plt.tight_layout()
        os.makedirs("models", exist_ok=True)
        plot_path = "models/prediction_visualizations.png"
        plt.savefig(plot_path)
        print(f"Visualizations saved to {plot_path}")
        
        # Log the plot as an artifact in MLflow
        mlflow.log_artifact(plot_path, "evaluation_plots")
        print("Evaluation complete. All results logged to MLflow!")

if __name__ == "__main__":
    evaluate()
