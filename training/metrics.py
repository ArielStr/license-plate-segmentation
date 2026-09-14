from __future__ import annotations

import torch


@torch.no_grad()
def binary_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    targets = targets >= 0.5

    intersection = (preds & targets).sum(dim=(1, 2, 3)).float()
    union = (preds | targets).sum(dim=(1, 2, 3)).float()

    iou = (intersection + eps) / (union + eps)

    return iou.mean().item()


@torch.no_grad()
def binary_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    targets = targets >= 0.5

    intersection = (preds & targets).sum(dim=(1, 2, 3)).float()
    total = (
        preds.sum(dim=(1, 2, 3)).float()
        + targets.sum(dim=(1, 2, 3)).float()
    )

    dice = (2.0 * intersection + eps) / (total + eps)

    return dice.mean().item()
