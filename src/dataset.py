import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(256, 256), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.transform = transform

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.images = sorted(
            [f for f in os.listdir(image_dir)
             if os.path.splitext(f.lower())[1] in valid_extensions]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_id = os.path.splitext(image_name)[0]

        # Construct paths
        image_path = os.path.join(self.image_dir, image_name)
        mask_name = f"{image_id}_segmentation.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        # Resize
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # Binarize mask
        mask = (mask > 127).astype(np.float32)

        # Apply augmentations if provided
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Albumentations might return the mask as 2D without the channel dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            # Fallback to manual tensor conversion if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask