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


EXPERIMENTS = {
    "baseline_v1": ExperimentConfig(
        name="baseline_v1",
        batch_size=8,
        seed=42,
        phase1_epochs=5,
        phase1_lr=1e-3,
        phase2_epochs=30,
        phase2_lr=1e-4,
        weight_decay=1e-4,
        augmentation_profile="none",
    ),

    "augmentation_v1": ExperimentConfig(
        name="augmentation_v1",
        batch_size=8,
        seed=42,
        phase1_epochs=5,
        phase1_lr=1e-3,
        phase2_epochs=30,
        phase2_lr=1e-4,
        weight_decay=1e-4,
        augmentation_profile="augmentation_v1",
    ),

    "cosine_lr_v1": ExperimentConfig(
        name="cosine_lr_v1",
        batch_size=8,
        seed=42,
        phase1_epochs=5,
        phase1_lr=1e-3,
        phase2_epochs=30,
        phase2_lr=1e-4,
        weight_decay=1e-4,
        augmentation_profile="none",
        scheduler="cosine",
        scheduler_min_lr=1e-6,
    ),
    "batch_size_4_v1": ExperimentConfig(
        name="batch_size_4_v1",
        batch_size=4,
        seed=42,
        phase1_epochs=5,
        phase1_lr=1e-3,
        phase2_epochs=30,
        phase2_lr=1e-4,
        weight_decay=1e-4,
        augmentation_profile="none",
    ),
    "progressive_batch_v1": ExperimentConfig(
        name="progressive_batch_v1",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,

        augmentation_profile="none",

        phase2_batch_schedule=(
            (10, 4),
            (10, 8),
            (10, 16),
        ),
    ),
"loss_dice_emphasis_v1": ExperimentConfig(
    name="loss_dice_emphasis_v1",
    batch_size=4,
    seed=42,
    phase1_epochs=5,
    phase1_lr=1e-3,
    phase2_epochs=30,
    phase2_lr=1e-4,
    weight_decay=1e-4,
    augmentation_profile="none",

    loss="bce_dice",
    bce_weight=0.5,
    dice_weight=1.0,
),
# L2 — Dice only
"loss_dice_only_v1": ExperimentConfig(
    name="loss_dice_only_v1",
    batch_size=4,
    seed=42,
    phase1_epochs=5,
    phase1_lr=1e-3,
    phase2_epochs=30,
    phase2_lr=1e-4,
    weight_decay=1e-4,
    augmentation_profile="none",
    loss="bce_dice",
    bce_weight=0.0,
    dice_weight=1.0,
),

# L3 — BCE only
"loss_bce_only_v1": ExperimentConfig(
    name="loss_bce_only_v1",
    batch_size=4,
    seed=42,
    phase1_epochs=5,
    phase1_lr=1e-3,
    phase2_epochs=30,
    phase2_lr=1e-4,
    weight_decay=1e-4,
    augmentation_profile="none",
    loss="bce_dice",
    bce_weight=1.0,
    dice_weight=0.0,
),
"loss_focal_dice_v1": ExperimentConfig(
    name="loss_focal_dice_v1",
    batch_size=4,
    seed=42,
    phase1_epochs=5,
    phase1_lr=1e-3,
    phase2_epochs=30,
    phase2_lr=1e-4,
    weight_decay=1e-4,
    augmentation_profile="none",
    loss="focal_dice",
    focal_weight=1.0,
    dice_weight=1.0,
    focal_gamma=2.0,
),
"loss_only_focal_v1": ExperimentConfig(
    name="loss_only_focal_v1",
    batch_size=4,
    seed=42,
    phase1_epochs=5,
    phase1_lr=1e-3,
    phase2_epochs=30,
    phase2_lr=1e-4,
    weight_decay=1e-4,
    augmentation_profile="none",
    loss="focal_dice",
    focal_weight=1.0,
    dice_weight=0.0,
    focal_gamma=2.0,
),
"no_freeze_v1": ExperimentConfig(
    name="no_freeze_v1",
    batch_size=4,
    seed=42,

    phase1_epochs=0,
    phase1_lr=1e-3,

    phase2_epochs=30,
    phase2_lr=1e-4,

    weight_decay=1e-4,
    augmentation_profile="none",

    loss="bce_dice",
    bce_weight=1.0,
    dice_weight=1.0,
),
"differential_lr_v1": ExperimentConfig(
    name="differential_lr_v1",
    batch_size=4,
    seed=42,

    phase1_epochs=0,
    phase1_lr=1e-3,

    phase2_epochs=30,
    phase2_lr=1e-4,

    phase2_encoder_lr=1e-4,
    phase2_decoder_lr=1e-3,

    weight_decay=1e-4,
    augmentation_profile="none",

    loss="bce_dice",
    bce_weight=1.0,
    dice_weight=1.0,
),
"long_freeze_v1": ExperimentConfig(
    name="long_freeze_v1",
    batch_size=4,
    seed=42,

    phase1_epochs=10,
    phase1_lr=1e-3,

    phase2_epochs=30,
    phase2_lr=1e-4,

    weight_decay=1e-4,
    augmentation_profile="none",

    loss="bce_dice",
    bce_weight=1.0,
    dice_weight=1.0,
),
"resolution_low_v1": ExperimentConfig(
    name="resolution_low_v1",
    batch_size=4,
    seed=42,

    phase1_epochs=5,
    phase1_lr=1e-3,

    phase2_epochs=30,
    phase2_lr=1e-4,

    weight_decay=1e-4,
    augmentation_profile="none",

    loss="bce_dice",
    bce_weight=1.0,
    dice_weight=1.0,

    input_width=192,
    input_height=64,
),
"resolution_high_v1": ExperimentConfig(
    name="resolution_high_v1",
    batch_size=4,
    seed=42,

    phase1_epochs=5,
    phase1_lr=1e-3,

    phase2_epochs=30,
    phase2_lr=1e-4,

    weight_decay=1e-4,
    augmentation_profile="none",

    loss="bce_dice",
    bce_weight=1.0,
    dice_weight=1.0,

    input_width=768,
    input_height=256,
),
"encoder_resnet18_v1": ExperimentConfig(
    name="encoder_resnet18_v1",
    batch_size=4,
    seed=42,

    phase1_epochs=5,
    phase1_lr=1e-3,

    phase2_epochs=30,
    phase2_lr=1e-4,

    weight_decay=1e-4,
    augmentation_profile="none",

    loss="bce_dice",
    bce_weight=1.0,
    dice_weight=1.0,

    input_width=384,
    input_height=128,

    architecture="unet",
    encoder_name="resnet18",
),

"encoder_resnet50_v1": ExperimentConfig(
    name="encoder_resnet50_v1",
    batch_size=4,
    seed=42,

    phase1_epochs=5,
    phase1_lr=1e-3,

    phase2_epochs=30,
    phase2_lr=1e-4,

    weight_decay=1e-4,
    augmentation_profile="none",

    loss="bce_dice",
    bce_weight=1.0,
    dice_weight=1.0,

    input_width=384,
    input_height=128,

    architecture="unet",
    encoder_name="resnet50",
),

"deeplabv3plus_resnet34_v1": ExperimentConfig(
    name="deeplabv3plus_resnet34_v1",
    batch_size=4,
    seed=42,

    phase1_epochs=5,
    phase1_lr=1e-3,

    phase2_epochs=30,
    phase2_lr=1e-4,

    weight_decay=1e-4,
    augmentation_profile="none",

    loss="bce_dice",
    bce_weight=1.0,
    dice_weight=1.0,

    input_width=384,
    input_height=128,

    architecture="deeplabv3plus",
    encoder_name="resnet34",
),
# ============================================================
    # Dataset V2 experiments — 250 samples
    # ============================================================

    # V2-0 — Controlled baseline
    "baseline_v2": ExperimentConfig(
        name="baseline_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),
    # V2-1 — Batch size 8
    "batch_size_8_v2": ExperimentConfig(
        name="batch_size_8_v2",
        seed=42,
        batch_size=8,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),

    # V2-2 — Batch size 16
    "batch_size_16_v2": ExperimentConfig(
        name="batch_size_16_v2",
        seed=42,
        batch_size=16,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),
# ============================================================
    # V2 — Training strategy
    # ============================================================

    # V2-3 — Data augmentation
    "augmentation_v2": ExperimentConfig(
        name="augmentation_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="augmentation_v1",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),

    # V2-4 — Cosine learning-rate scheduler
    "cosine_lr_v2": ExperimentConfig(
        name="cosine_lr_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        scheduler="cosine",
        scheduler_min_lr=1e-6,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),

    # V2-5 — Progressive batch size
    "progressive_batch_v2": ExperimentConfig(
        name="progressive_batch_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        phase2_batch_schedule=(
            (10, 4),
            (10, 8),
            (10, 16),
        ),

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),
    # ============================================================
    # V2 - Loss experiments
    # ============================================================

    # V2-6 - Dice emphasis: 0.5 * BCE + 1.0 * Dice
    "loss_dice_emphasis_v2": ExperimentConfig(
        name="loss_dice_emphasis_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=0.5,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),

    # V2-7 - Dice only
    "loss_dice_only_v2": ExperimentConfig(
        name="loss_dice_only_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=0.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),

    # V2-8 - BCE only
    "loss_bce_only_v2": ExperimentConfig(
        name="loss_bce_only_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=0.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),

    # V2-9 - Focal + Dice
    "loss_focal_dice_v2": ExperimentConfig(
        name="loss_focal_dice_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="focal_dice",
        focal_weight=1.0,
        dice_weight=1.0,
        focal_gamma=2.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),

    # V2-10 - Focal only
    "loss_only_focal_v2": ExperimentConfig(
        name="loss_only_focal_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="focal_dice",
        focal_weight=1.0,
        dice_weight=0.0,
        focal_gamma=2.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),
    # ============================================================
    # V2 - Fine-tuning strategy
    # ============================================================

    # V2-11 - No frozen-encoder phase
    "no_freeze_v2": ExperimentConfig(
        name="no_freeze_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=0,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),

    # V2-12 - Differential learning rates
    "differential_lr_v2": ExperimentConfig(
        name="differential_lr_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=0,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        phase2_encoder_lr=1e-4,
        phase2_decoder_lr=1e-3,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),

    # V2-13 - Longer frozen-encoder phase
    "long_freeze_v2": ExperimentConfig(
        name="long_freeze_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=10,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),
    # ============================================================
    # V2 - Input resolution
    # ============================================================

    # V2-14 - Low resolution: 64x192
    "resolution_low_v2": ExperimentConfig(
        name="resolution_low_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=192,
        input_height=64,

        architecture="unet",
        encoder_name="resnet34",
    ),

    # V2-15 - High resolution: 256x768
    "resolution_high_v2": ExperimentConfig(
        name="resolution_high_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=768,
        input_height=256,

        architecture="unet",
        encoder_name="resnet34",
    ),
# ============================================================
    # V2 - Architecture / Encoder
    # ============================================================

    # V2-16 - U-Net + ResNet18
    "encoder_resnet18_v2": ExperimentConfig(
        name="encoder_resnet18_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet18",
    ),

    # V2-17 - U-Net + ResNet50
    "encoder_resnet50_v2": ExperimentConfig(
        name="encoder_resnet50_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet50",
    ),

    # V2-18 - DeepLabV3+ + ResNet34
    "deeplabv3plus_resnet34_v2": ExperimentConfig(
        name="deeplabv3plus_resnet34_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="none",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="deeplabv3plus",
        encoder_name="resnet34",
    ),
# ============================================================
    # V2 - Failure-analysis-driven experiment
    # ============================================================

    # High resolution + augmentation
    #
    # Hypothesis:
    # Higher spatial resolution may improve boundary/detail accuracy,
    # while augmentation may improve robustness to appearance
    # variations such as blur, illumination changes and reflections.
    "highres_augmentation_v2": ExperimentConfig(
        name="highres_augmentation_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="augmentation_v1",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=768,
        input_height=256,

        architecture="unet",
        encoder_name="resnet34",
    ),

}


def get_experiment_config(name: str) -> ExperimentConfig:
    if name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(
            f"Unknown experiment: {name}. "
            f"Available experiments: {available}"
        )

    return EXPERIMENTS[name]