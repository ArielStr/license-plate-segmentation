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

    scheduler: str | None = None
    scheduler_min_lr: float | None = None


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
"seed_smoke_test": ExperimentConfig(
    name="seed_smoke_test",
    seed=42,
    batch_size=8,
    phase1_epochs=1,
    phase1_lr=1e-3,
    phase2_epochs=1,
    phase2_lr=1e-4,
    weight_decay=1e-4,
    augmentation_profile="none",
)
}


def get_experiment_config(name: str) -> ExperimentConfig:
    if name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(
            f"Unknown experiment: {name}. "
            f"Available experiments: {available}"
        )

    return EXPERIMENTS[name]