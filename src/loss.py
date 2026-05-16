import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE Loss expects logits
        bce_loss = self.bce(logits, targets)
        
        # Dice Loss needs probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten predictions and targets
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        dice_loss = 1.0 - dice_score
        
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)


def calculate_dice(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice_score = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    
    return dice_score
