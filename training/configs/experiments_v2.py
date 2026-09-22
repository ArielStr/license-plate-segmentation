from training.configs.experiment_config import ExperimentConfig


EXPERIMENTS_V2 = {
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
    # ============================================================
    # V2 - Failure-analysis-driven blur experiment
    # ============================================================

    # Strong Gaussian blur augmentation
    #
    # Hypothesis:
    # Failure analysis showed that several difficult validation
    # samples are small / visually degraded, while much of the
    # remaining error is concentrated near object boundaries.
    #
    # This experiment isolates stronger blur augmentation while
    # keeping the baseline_v2 training recipe unchanged.
    "strong_blur_v2": ExperimentConfig(
        name="strong_blur_v2",
        seed=42,
        batch_size=4,

        phase1_epochs=5,
        phase1_lr=1e-3,

        phase2_epochs=30,
        phase2_lr=1e-4,

        weight_decay=1e-4,
        augmentation_profile="strong_blur_v2",

        loss="bce_dice",
        bce_weight=1.0,
        dice_weight=1.0,

        input_width=384,
        input_height=128,

        architecture="unet",
        encoder_name="resnet34",
    ),
}