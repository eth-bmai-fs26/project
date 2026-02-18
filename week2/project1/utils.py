import torch
import numpy as np
from scipy.ndimage import label


def dice_loss(pred, target, eps=1e-6):
    """
    Dice loss for binary segmentation.
    pred: logits (B, 1, H, W)
    target: binary mask (B, 1, H, W)
    """
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice


def poisson_loss(lam, y):
    """
    Poisson loss for count regression.
    lam: predicted rate (B,)
    y: ground-truth counts (B,)
    """
    return (lam - y * torch.log(lam + 1e-8)).mean()


def count_components(mask):
    """
    Counts connected components in a binary segmentation mask.
    mask: Tensor (1, H, W) or (B, 1, H, W)
    """
    mask = mask.squeeze().cpu().numpy()
    _, num = label(mask)
    return num
