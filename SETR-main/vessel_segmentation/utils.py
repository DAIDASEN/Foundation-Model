# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, dice, path, scheduler=None, **kwargs):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': dice,
        'dice': dice,  
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['dice']


class DiceLoss(nn.Module):
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        
        num_classes = outputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (outputs * targets_one_hot).sum(dim=(2, 3))
        union = outputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


def compute_metrics(preds, targets):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    preds_vessel = (preds == 1)
    targets_vessel = (targets == 1)
    
    intersection = np.logical_and(preds_vessel, targets_vessel).sum()
    union = np.logical_or(preds_vessel, targets_vessel).sum()
    
    dice = (2. * intersection) / (preds_vessel.sum() + targets_vessel.sum() + 1e-8)
    
    iou = intersection / (union + 1e-8)
    
    accuracy = (preds == targets).sum() / targets.size
    
    return {
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy
    }


def visualize_prediction(image, mask, pred, save_path=None):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
