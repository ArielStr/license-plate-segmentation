from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probs = torch.sigmoid(logits)

        p_t = (
            probs * targets
            + (1.0 - probs) * (1.0 - targets)
        )

        focal_weight = (1.0 - p_t) ** self.gamma

        return (focal_weight * bce).mean()

class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        gamma: float = 2.0,
    ):
        super().__init__()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.focal = FocalLoss(gamma=gamma)
        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)

        return (
            self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
        )

def build_loss(
    loss_name: str,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
    focal_gamma: float = 2.0,
) -> nn.Module:

    if loss_name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

    if loss_name == "focal_dice":
        return FocalDiceLoss(
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            gamma=focal_gamma,
        )

    raise ValueError(f"Unsupported loss: {loss_name}")
