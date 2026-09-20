from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "evaluation"
    / "failure_patterns"
    / "resolution_high_v2"
    / "all_samples.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "failure_patterns"
    / "resolution_high_v2"
)
OUTPUT_CSV = OUTPUT_DIR / "size_groups.csv"


def assign_size_group(height: int) -> str:
    """
    Group samples according to ORIGINAL crop height.
    """
    if height < 32:
        return "tiny_<32"
    elif height < 64:
        return "small_32_63"
    elif height < 128:
        return "medium_64_127"
    else:
        return "large_128_plus"


def main() -> None:

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "sample_id",
        "original_height",
        "iou",
        "dice",
        "precision",
        "recall",
        "fp_ratio",
        "fn_ratio",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns in CSV: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Assign size group
    # --------------------------------------------------------

    df["size_group"] = (
        df["original_height"]
        .apply(assign_size_group)
    )

    group_order = [
        "tiny_<32",
        "small_32_63",
        "medium_64_127",
        "large_128_plus",
    ]

    df["size_group"] = pd.Categorical(
        df["size_group"],
        categories=group_order,
        ordered=True,
    )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    summary = (
        df.groupby(
            "size_group",
            observed=False,
        )
        .agg(
            n=("sample_id", "count"),

            mean_height=("original_height", "mean"),
            median_height=("original_height", "median"),

            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),

            mean_dice=("dice", "mean"),

            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),

            mean_fp_ratio=("fp_ratio", "mean"),
            mean_fn_ratio=("fn_ratio", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("=" * 90)
    print("SIZE-STRATIFIED FAILURE ANALYSIS")
    print("=" * 90)
    print()

    for _, row in summary.iterrows():

        print(row["size_group"])
        print("-" * 50)

        print(
            f"N:               {int(row['n'])}"
        )

        print(
            f"Height:          "
            f"mean={row['mean_height']:.1f} | "
            f"median={row['median_height']:.1f}"
        )

        print(
            f"IoU:             "
            f"mean={row['mean_iou']:.4f} | "
            f"median={row['median_iou']:.4f}"
        )

        print(
            f"Dice:            "
            f"{row['mean_dice']:.4f}"
        )

        print(
            f"Precision:       "
            f"{row['mean_precision']:.4f}"
        )

        print(
            f"Recall:          "
            f"{row['mean_recall']:.4f}"
        )

        print(
            f"FP ratio:        "
            f"{row['mean_fp_ratio']:.4f}"
        )

        print(
            f"FN ratio:        "
            f"{row['mean_fn_ratio']:.4f}"
        )

        print()

    print("=" * 90)
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 90)


if __name__ == "__main__":
    main()