from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "data" / "split.csv"

SPLIT_ORDER = ["train", "val", "test"]
SIZE_ORDER = ["tiny", "small", "medium", "large"]


def get_size_group(height: int) -> str:
    """Assign a crop to the same size groups used in our failure analysis."""
    if height < 32:
        return "tiny"
    if height < 64:
        return "small"
    if height < 128:
        return "medium"
    return "large"


def load_split_with_sizes() -> pd.DataFrame:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_PATH}\n"
            "Run training/create_split.py first."
        )

    df = pd.read_csv(SPLIT_PATH)

    required_columns = {"sample_id", "crop_path", "split"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"split.csv is missing required columns: {sorted(missing)}"
        )

    rows = []

    for _, row in df.iterrows():
        crop_rel = str(row["crop_path"]).replace("\\", "/")
        crop_path = PROJECT_ROOT / Path(crop_rel)

        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Could not read crop: {crop_path}")

        height, width = image.shape[:2]

        rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "width": width,
                "height": height,
                "size_group": get_size_group(height),
            }
        )

    return pd.DataFrame(rows)


def analyze_split(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("DATASET SPLIT ANALYSIS")
    print("=" * 72)

    print(f"\nTotal samples: {len(df)}")

    for split_name in SPLIT_ORDER:
        split_df = df[df["split"] == split_name]

        print("\n" + "-" * 72)
        print(f"{split_name.upper()} | N={len(split_df)}")
        print("-" * 72)

        counts = (
            split_df["size_group"]
            .value_counts()
            .reindex(SIZE_ORDER, fill_value=0)
        )

        percentages = counts / len(split_df) * 100.0

        print("\nSize groups:")

        for group in SIZE_ORDER:
            print(
                f"  {group.capitalize():6s}: "
                f"{counts[group]:3d} "
                f"({percentages[group]:5.1f}%)"
            )

        print("\nWidth [px]:")
        print(
            f"  mean={split_df['width'].mean():.1f} | "
            f"median={split_df['width'].median():.1f} | "
            f"min={split_df['width'].min()} | "
            f"max={split_df['width'].max()}"
        )

        print("Height [px]:")
        print(
            f"  mean={split_df['height'].mean():.1f} | "
            f"median={split_df['height'].median():.1f} | "
            f"min={split_df['height'].min()} | "
            f"max={split_df['height'].max()}"
        )

    print("\n" + "=" * 72)
    print("SIZE GROUP COMPARISON")
    print("=" * 72)

    comparison = pd.crosstab(
        df["size_group"],
        df["split"],
    )

    comparison = comparison.reindex(
        index=SIZE_ORDER,
        columns=SPLIT_ORDER,
        fill_value=0,
    )

    print("\nCounts:")
    print(comparison)

    percentages = comparison.div(
        comparison.sum(axis=0),
        axis=1,
    ) * 100.0

    print("\nPercentages within each split:")
    print(percentages.round(1))


def main() -> None:
    df = load_split_with_sizes()
    analyze_split(df)


if __name__ == "__main__":
    main()