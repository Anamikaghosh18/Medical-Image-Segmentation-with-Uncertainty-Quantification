from dataset import SkinLesionDataset
from torch.utils.data import DataLoader

def create_dataloaders(
    train_image_dir="../data/ISIC_Training_Input",
    train_mask_dir="../data/ISIC_Training_GroundTruth",
    val_image_dir="../data/ISIC_Validation_Input",
    val_mask_dir="../data/ISIC_Validation_GroundTruth",
    batch_size=8,
    image_size=(256, 256),
    num_workers=2,
):
     # Training dataset
    train_dataset = SkinLesionDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    

    # Validation dataset
    val_dataset = SkinLesionDataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        size=image_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader