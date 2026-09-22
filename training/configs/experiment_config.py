from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str

    seed: int
    batch_size: int

    phase1_epochs: int
    phase1_lr: float

    phase2_epochs: int
    phase2_lr: float

    weight_decay: float

    augmentation_profile: str

    loss: str = "bce_dice"
    bce_weight: float = 1.0
    dice_weight: float = 1.0

    focal_weight: float = 1.0
    focal_gamma: float = 2.0

    phase2_encoder_lr: float | None = None
    phase2_decoder_lr: float | None = None

    scheduler: str | None = None
    scheduler_min_lr: float | None = None

    phase2_batch_schedule: tuple[tuple[int, int], ...] | None = None

    input_width: int = 384
    input_height: int = 128

    architecture: str = "unet"
    encoder_name: str = "resnet34"

    train_size: int | None = None