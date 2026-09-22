from .experiment_config import ExperimentConfig

EXPERIMENTS_V3 = {
    # ============================================================
    # V3 — Dataset scaling experiments
    # Dataset: 606 samples
    # Split: 424 train / 91 val / 91 test
    #
    # Goal:
    # Measure the effect of increasing the amount of training data
    # while keeping the validation set and training recipe fixed.
    #
    # Nested subsets:
    # 175 ⊂ 250 ⊂ 350 ⊂ 424
    # ============================================================

    # V3-0 — 175 training samples
    "scaling_175_v3": ExperimentConfig(
        name="scaling_175_v3",
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

        train_size=175,
    ),

    # V3-1 — 250 training samples
    "scaling_250_v3": ExperimentConfig(
        name="scaling_250_v3",
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

        train_size=250,
    ),

    # V3-2 — 350 training samples
    "scaling_350_v3": ExperimentConfig(
        name="scaling_350_v3",
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

        train_size=350,
    ),

    # V3-3 — Full training set: 424 samples
    # This is also the V3 controlled baseline.
    "baseline_v3": ExperimentConfig(
        name="baseline_v3",
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

        train_size=None,
    ),
}