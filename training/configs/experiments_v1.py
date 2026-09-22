from training.configs.experiment_config import ExperimentConfig


EXPERIMENTS_V1 = {
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
}