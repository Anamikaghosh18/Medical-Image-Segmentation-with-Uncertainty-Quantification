import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from urllib.request import urlretrieve
from src.UNetmodel import UNet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
MODEL_URL = (
    "https://github.com/Anamikaghosh18/"
    "Medical-Image-Segmentation-with-Uncertainty-Quantification/"
    "releases/download/ai-model/best_unet.pth"
)

MODEL_PATH = Path("models/best_unet.pth")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    print("Downloading model from GitHub Release...")
    urlretrieve(MODEL_URL, MODEL_PATH)

print("Loading model weights...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully.")

def check_image_suitability(image_np, mask_np):
    """
    Classifies image suitability for dermoscopic analysis.
    Returns: (status_category, detail_message)
    Categories:
      - "valid": Confirmed valid skin scan.
      - "atypical": Valid skin scan but with potential artifacts (hair, markers, scale).
      - "unrelated": Completely unrelated image (landscape, animal, food, facial selfie, room scene).
    """
    import cv2
    import numpy as np

    # 1. Convert to HSV and Grayscale
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # 2. Extract core features
    # Non-skin warm-tones range: Hue in [22, 158] with saturated color
    non_skin_mask = (h >= 22) & (h <= 158) & (s > 25) & (v > 25)
    non_skin_percentage = (non_skin_mask.sum() / (image_np.shape[0] * image_np.shape[1])) * 100

    # Color dispersion (Fixed circular wrapping bug by shifting Hue by 90 degrees)
    h_shifted = (h.astype(np.float32) + 90.0) % 180.0
    std_hue = float(np.std(h_shifted))

    # Textural/Edge density
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    edge_score = float(np.abs(laplacian).mean())

    # Prediction fragmentation
    num_islands = 0
    if mask_np.sum() > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np.astype(np.uint8))
        num_islands = num_labels - 1

    # 3. Categorization logic
    
    # CASE A: Completely Unrelated/Wrong Image (Selfie, animal, food, room, text, highly colorful graphics)
    # Calibrated to be highly conservative to eliminate false positives on valid skin scans.
    if edge_score > 26.0 or non_skin_percentage > 35.0 or std_hue > 42.0 or num_islands > 22:
        reasons = []
        if edge_score > 26.0: reasons.append("too many complex background details or text")
        if non_skin_percentage > 35.0: reasons.append("colors that do not look like skin (like clothing, bright objects, or scenery)")
        if std_hue > 42.0: reasons.append("too many different colors in the photo")
        if num_islands > 22: reasons.append("an unusually chaotic shape layout")
        
        reason_str = " and ".join(reasons[:2]) if len(reasons) > 1 else reasons[0]
        return "unrelated", f"We couldn't recognize this as a skin photo because we detected {reason_str}. Please upload a clear, close-up picture of a skin mole or spot."

    # CASE B: Atypical Skin Scan (Clinical artifacts, hair, marker ink, ruler markings, but still a skin scan)
    elif edge_score > 12.0 or non_skin_percentage > 12.0 or std_hue > 26.0 or num_islands > 7:
        reasons = []
        if edge_score > 12.0: reasons.append("fine hairs or folds on the skin")
        if non_skin_percentage > 12.0: reasons.append("dark shadows or marker/pen lines")
        if std_hue > 26.0: reasons.append("uneven shading in the spot")
        if num_islands > 7: reasons.append("multiple small separate spots")
        
        reason_str = ", ".join(reasons)
        return "atypical", f"We found some scan factors that might affect accuracy ({reason_str}). The AI still outlined the spot, but for the best result, try clearing away hair or improving the lighting."

    # CASE C: Confirmed clean skin scan
    return "valid", ""



@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form("unet")):
    print(f"[ENGINE LOGGER] Executing inference sequence with active engine: {model_name.upper()}")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    
    # ==========================================
    # EXPERIMENTAL MODEL REGISTRY REGISTRY
    # ==========================================
    # To run experiments with future neural network models (e.g. TransUNet, Attention U-Net, UNet++):
    # 1. Load your model checkpoints globally:
    #    transunet = TransUNet(...).to(device)
    #    transunet.load_state_dict(torch.load("models/best_transunet.pth"))
    # 2. Map the active model instance using the chosen model_name:
    #    models_dict = {
    #        "unet": model, # standard U-Net
    #        "transunet": transunet,
    #        "attention_unet": attention_unet,
    #        "unet_plusplus": unet_plusplus
    #    }
    #    active_model = models_dict.get(model_name, model)
    # ==========================================
    active_model = model
    
    transform = A.Compose([A.Resize(256, 256), ToTensorV2()])
    transformed = transform(image=image_np)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    if img_tensor.max() > 1.0: 
        img_tensor = img_tensor.float() / 255.0
        
    with torch.no_grad():
        logits = active_model(img_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
    mask_np = preds[0].cpu().numpy().squeeze()
    probs_np = probs[0].cpu().numpy().squeeze()
    
    total_pixels = mask_np.size
    lesion_pixels = mask_np.sum()
    area_percentage = (lesion_pixels / total_pixels) * 100
    
    import cv2
    mask_resized_for_check = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    status_cat, detail_msg = check_image_suitability(image_np, mask_resized_for_check)
    
    if status_cat == "unrelated":
        # Reject wrong/unrelated uploads immediately, bypassing mask calculations
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        orig_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return JSONResponse({
            "status": "unrelated",
            "message": detail_msg,
            "area_percentage": "0.0%",
            "confidence": "0.0%",
            "irregularity": "1.00",
            "diameter": "0.0 mm",
            "asymmetry": "0.0%",
            "overlay_image": f"data:image/jpeg;base64,{orig_b64}",
            "heatmap_image": f"data:image/jpeg;base64,{orig_b64}"
        })
        
    # Quantitative Pathology Metrics Calculations
    import cv2
    asymmetry_val = "0.0%"
    irregularity_val = "1.00"
    diameter_val = "0.0 mm"
    
    if lesion_pixels > 0:
        mask_original_size = cv2.resize((mask_np * 255).astype(np.uint8), (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_original_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if area > 0:
                # 1. Border Irregularity Compactness ratio (P^2 / (4 * pi * A))
                irregularity = (perimeter ** 2) / (4 * np.pi * area)
                irregularity_val = f"{irregularity:.2f}"
                
                # 2. Asymmetry Index (horizontal mirror overlap)
                x, y, w, h = cv2.boundingRect(cnt)
                roi = mask_original_size[y:y+h, x:x+w]
                h_flip = cv2.flip(roi, 1)
                union_mask = cv2.bitwise_or(roi, h_flip)
                intersect_mask = cv2.bitwise_and(roi, h_flip)
                union_sum = union_mask.sum()
                if union_sum > 0:
                    asymmetry = 1.0 - (intersect_mask.sum() / union_sum)
                    asymmetry_val = f"{asymmetry * 100:.1f}%"
                else:
                    asymmetry_val = "0.0%"
                
                # 3. Physical Diameter (Assuming 15 pixels = 1mm)
                # equivalent circle diameter d = 2 * sqrt(area / pi)
                pixel_to_mm_ratio = 15.0
                diameter_pixels = 2 * np.sqrt(area / np.pi)
                diameter_mm = diameter_pixels / pixel_to_mm_ratio
                diameter_val = f"{diameter_mm:.1f} mm"

    if status_cat == "atypical":
        if lesion_pixels > 0:
            confidence = probs_np[mask_np == 1].mean() * 100
            status = "warning"
        else:
            confidence = 0.0
            status = "safe"
        message = detail_msg
    elif lesion_pixels > 0:
        confidence = probs_np[mask_np == 1].mean() * 100
        status = "anomaly"
        message = "Skin Spot Detected: The AI has successfully outlined the boundaries of the spot."
    else:
        confidence = 0.0
        status = "safe"
        message = "No Spot Detected: The skin surface appears clear, with no distinct spots detected."
        
    overlay = image_np.copy()
    mask_resized = np.array(Image.fromarray(mask_np).resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST))
    overlay[mask_resized == 1] = overlay[mask_resized == 1] * 0.6 + np.array([220, 20, 60]) * 0.4
    
    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    buffered = io.BytesIO()
    overlay_img.save(buffered, format="JPEG")
    overlay_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    import matplotlib.cm as cm
    probs_resized = cv2.resize(probs_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cm.jet(probs_resized)[:, :, :3] * 255
    alpha = 0.5
    overlay_heatmap = image_np.copy()
    mask_soft = probs_resized > 0.1
    for c in range(3):
        overlay_heatmap[:,:,c] = np.where(mask_soft, overlay_heatmap[:,:,c] * (1 - alpha) + heatmap[:,:,c] * alpha, overlay_heatmap[:,:,c])
    
    heatmap_img = Image.fromarray(overlay_heatmap.astype(np.uint8))
    buffered_hm = io.BytesIO()
    heatmap_img.save(buffered_hm, format="JPEG")
    heatmap_b64 = base64.b64encode(buffered_hm.getvalue()).decode("utf-8")
    
    return JSONResponse({
        "status": status,
        "message": message,
        "area_percentage": f"{area_percentage:.1f}%",
        "confidence": f"{confidence:.1f}%",
        "irregularity": irregularity_val,
        "diameter": diameter_val,
        "asymmetry": asymmetry_val,
        "overlay_image": f"data:image/jpeg;base64,{overlay_b64}",
        "heatmap_image": f"data:image/jpeg;base64,{heatmap_b64}"
    })

from pydantic import BaseModel
class EvalRequest(BaseModel):
    batches: int = 10
    samples: int = 3

@app.post("/api/evaluate")
async def evaluate(req: EvalRequest):
    image_dir = "data/ISIC_Training_Input"
    mask_dir = "data/ISIC_Training_GroundTruth"
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        return JSONResponse({"error": "Dataset volumes not mounted."})
        
    val_transforms = A.Compose([ToTensorV2()])
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    all_images = sorted([f for f in os.listdir(image_dir) if os.path.splitext(f.lower())[1] in valid_extensions])
    
    split_idx = int(0.8 * len(all_images))
    val_images = all_images[split_idx:]
    
    from src.dataset import SkinLesionDataset
    from torch.utils.data import DataLoader
    from src.loss import calculate_dice
    
    def calculate_iou(preds, targets, smooth=1e-6):
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        intersection = (preds * targets).sum(dim=(2,3))
        union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()
        
    val_dataset = SkinLesionDataset(image_dir, mask_dir, transform=val_transforms, image_files=val_images)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    dice_scores = []
    iou_scores = []
    batch_probs = []
    batch_targets = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= req.batches: break
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            dice = calculate_dice(logits, masks)
            iou = calculate_iou(logits, masks)
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
            
            batch_probs.append(torch.sigmoid(logits).cpu().numpy().flatten())
            batch_targets.append(masks.cpu().numpy().flatten())

    mean_dice = np.mean(dice_scores) * 100 if len(dice_scores) > 0 else 0
    mean_iou = np.mean(iou_scores) * 100 if len(iou_scores) > 0 else 0
    total_eval = len(dice_scores) * 8
    
    from sklearn.metrics import roc_curve, auc
    if len(batch_probs) > 0:
        all_probs = np.concatenate(batch_probs)
        all_targets = np.concatenate(batch_targets)
        if len(all_probs) > 100000:
            indices = np.random.choice(len(all_probs), 100000, replace=False)
            all_probs = all_probs[indices]
            all_targets = all_targets[indices]

        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        roc_auc = auc(fpr, tpr)
        
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(fpr, tpr, color='#0f766e', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax2.plot([0, 1], [0, 1], color='#94a3b8', lw=2, linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic')
        ax2.legend(loc="lower right")
        plt.tight_layout()
        buffered2 = io.BytesIO()
        fig2.savefig(buffered2, format="PNG")
        plt.close(fig2)
        roc_b64 = base64.b64encode(buffered2.getvalue()).decode("utf-8")
    else:
        roc_b64 = ""
    
    fig, axes = plt.subplots(req.samples, 3, figsize=(12, 4 * req.samples))
    if req.samples == 1: axes = [axes]
    
    val_loader_viz = DataLoader(val_dataset, batch_size=1, shuffle=True)
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader_viz):
            if i >= req.samples: break
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

            img_np = np.clip(images[0].cpu().numpy().transpose(1, 2, 0), 0, 1) 
            mask_np = masks[0].cpu().numpy().squeeze()
            pred_np = preds[0].cpu().numpy().squeeze()

            axes[i][0].imshow(img_np)
            axes[i][0].set_title("Input Image", fontweight='bold')
            axes[i][0].axis('off')

            axes[i][1].imshow(mask_np, cmap='gray')
            axes[i][1].set_title("Ground Truth", fontweight='bold')
            axes[i][1].axis('off')

            axes[i][2].imshow(pred_np, cmap='gray')
            axes[i][2].set_title("UNet Prediction", fontweight='bold')
            axes[i][2].axis('off')

    plt.tight_layout()
    buffered = io.BytesIO()
    plt.savefig(buffered, format="PNG")
    plt.close(fig)
    plot_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return JSONResponse({
        "mean_dice": f"{mean_dice:.1f}%",
        "mean_iou": f"{mean_iou:.1f}%",
        "total_eval": total_eval,
        "gallery_image": f"data:image/png;base64,{plot_b64}",
        "roc_image": f"data:image/png;base64,{roc_b64}"
    })

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

