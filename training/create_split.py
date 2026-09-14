from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = PROJECT_ROOT / "data" / "crops"
MASKS_DIR = PROJECT_ROOT / "data" / "masks"
OUTPUT_PATH = PROJECT_ROOT / "data" / "split.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def find_mask(stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        candidate = MASKS_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No mask found for crop '{stem}' in {MASKS_DIR}"
    )


def collect_samples() -> list[dict]:
    if not CROPS_DIR.exists():
        raise FileNotFoundError(f"Crops directory not found: {CROPS_DIR}")

    if not MASKS_DIR.exists():
        raise FileNotFoundError(f"Masks directory not found: {MASKS_DIR}")

    samples = []

    for crop_path in sorted(CROPS_DIR.iterdir()):
        if (
            not crop_path.is_file()
            or crop_path.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue

        mask_path = find_mask(crop_path.stem)

        samples.append(
            {
                "sample_id": crop_path.stem,
                "crop_path": str(crop_path.relative_to(PROJECT_ROOT)),
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
            }
        )

    if not samples:
        raise RuntimeError("No crop/mask pairs found.")

    return samples


def make_split(samples: list[dict]) -> pd.DataFrame:
    n = len(samples)

    if not np.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    rng = np.random.default_rng(SEED)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))

    # Whatever remains goes to test so the total is always exact.
    n_test = n - n_train - n_val

    train_idx = set(indices[:n_train].tolist())
    val_idx = set(indices[n_train:n_train + n_val].tolist())
    test_idx = set(indices[n_train + n_val:].tolist())

    rows = []

    for idx, sample in enumerate(samples):
        if idx in train_idx:
            split = "train"
        elif idx in val_idx:
            split = "val"
        elif idx in test_idx:
            split = "test"
        else:
            raise RuntimeError(f"Sample index {idx} was not assigned.")

        rows.append({**sample, "split": split})

    df = pd.DataFrame(rows)

    print("=" * 72)
    print("DATASET SPLIT")
    print("=" * 72)
    print(f"Seed:  {SEED}")
    print(f"Total: {n}")
    print(f"Train: {n_train}")
    print(f"Val:   {n_val}")
    print(f"Test:  {n_test}")
    print()

    print(df["split"].value_counts().reindex(["train", "val", "test"]))

    return df


def main() -> None:
    samples = collect_samples()
    df = make_split(samples)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved split file to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
