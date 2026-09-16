from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_CSV = PROJECT_ROOT / "data" / "split.csv"

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "blur_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "blur_scores.csv"
HISTOGRAM_PATH = OUTPUT_DIR / "blur_score_distribution.png"
CONTACT_SHEET_PATH = OUTPUT_DIR / "blur_ranking.png"


# ============================================================
# Visualization configuration
# ============================================================

N_VISUALIZE = 30
N_COLS = 5


# ============================================================
# Sharpness metric
# ============================================================

def compute_laplacian_variance(image_bgr):
    """
    Compute a simple sharpness score using the variance
    of the Laplacian.

    Higher score -> more high-frequency image content.
    Lower score  -> less high-frequency image content.

    Important:
    We initially treat this as a relative score for our
    own training dataset, not as a universal blur threshold.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    return float(laplacian.var())


# ============================================================
# Dataset analysis
# ============================================================

def analyze_training_set():
    df = pd.read_csv(SPLIT_CSV)

    required_columns = {"crop_path", "split"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in split.csv: {missing}"
        )

    train_df = df[df["split"] == "train"].copy()

    print("=" * 72)
    print("BLUR / SHARPNESS ANALYSIS")
    print("=" * 72)
    print(f"Split: train")
    print(f"Training samples: {len(train_df)}")
    print(f"Metric: Variance of Laplacian")
    print("=" * 72)

    results = []

    for _, row in train_df.iterrows():
        image_path = PROJECT_ROOT / row["crop_path"]

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        height, width = image.shape[:2]

        score = compute_laplacian_variance(image)

        results.append(
            {
                "sample": image_path.stem,
                "crop_path": row["crop_path"],
                "width": width,
                "height": height,
                "laplacian_variance": score,
            }
        )

    results_df = pd.DataFrame(results)

    # Low score first -> candidate blurry samples first
    results_df = results_df.sort_values(
        "laplacian_variance",
        ascending=True,
    ).reset_index(drop=True)

    results_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    return results_df


# ============================================================
# Summary
# ============================================================

def print_summary(df):
    scores = df["laplacian_variance"]

    print()
    print("Sharpness score summary")
    print("-" * 72)

    print(f"Min:    {scores.min():.2f}")
    print(f"10%:    {scores.quantile(0.10):.2f}")
    print(f"25%:    {scores.quantile(0.25):.2f}")
    print(f"Median: {scores.median():.2f}")
    print(f"75%:    {scores.quantile(0.75):.2f}")
    print(f"90%:    {scores.quantile(0.90):.2f}")
    print(f"Max:    {scores.max():.2f}")

    print()
    print("10 lowest scores:")
    print("-" * 72)

    for _, row in df.head(10).iterrows():
        print(
            f"{row['sample']:<30} "
            f"{row['laplacian_variance']:>10.2f}"
        )


# ============================================================
# Histogram
# ============================================================

def save_histogram(df):
    scores = df["laplacian_variance"].values

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        scores,
        bins=20,
        edgecolor="black",
    )

    median = np.median(scores)

    ax.axvline(
        median,
        linestyle="--",
        linewidth=2,
        label=f"Median = {median:.1f}",
    )

    ax.set_title(
        "Training-set sharpness distribution\n"
        "Variance of Laplacian"
    )

    ax.set_xlabel(
        "Laplacian variance (higher = more high-frequency content)"
    )

    ax.set_ylabel("Number of samples")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        HISTOGRAM_PATH,
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# Ranked visualization
# ============================================================

def save_ranked_contact_sheet(df):
    """
    Visualize samples across the full score distribution.

    Instead of showing only the lowest-scoring samples,
    we sample evenly from the sorted ranking.

    This lets us visually check whether increasing
    Laplacian variance actually corresponds to increasing
    perceived sharpness.
    """

    n = min(
        N_VISUALIZE,
        len(df),
    )

    indices = np.linspace(
        0,
        len(df) - 1,
        n,
        dtype=int,
    )

    selected = df.iloc[indices]

    n_rows = int(
        np.ceil(n / N_COLS)
    )

    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(18, 3.6 * n_rows),
    )

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(
        axes,
        selected.iterrows(),
    ):
        image_path = (
            PROJECT_ROOT / row["crop_path"]
        )

        image = cv2.imread(
            str(image_path)
        )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        ax.imshow(image_rgb)

        ax.set_title(
            f"{row['sample']}\n"
            f"score={row['laplacian_variance']:.1f}\n"
            f"{row['width']}x{row['height']}",
            fontsize=9,
        )

        ax.axis("off")

    fig.suptitle(
        "Training images ranked by Laplacian variance\n"
        "Low score → High score",
        fontsize=16,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    fig.savefig(
        CONTACT_SHEET_PATH,
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    df = analyze_training_set()

    print_summary(df)

    save_histogram(df)
    save_ranked_contact_sheet(df)

    print()
    print("=" * 72)
    print("Saved:")
    print(RESULTS_CSV)
    print(HISTOGRAM_PATH)
    print(CONTACT_SHEET_PATH)
    print("=" * 72)


if __name__ == "__main__":
    main()