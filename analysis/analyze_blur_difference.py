from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SPLIT_CSV = DATA_DIR / "split.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "blur_difference_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

# Fixed Gaussian blur used ONLY as a probe for measuring
# how much high-frequency information exists in the image.
PROBE_SIGMA = 1.5

# Number of images to display in the ranking panel.
NUM_DISPLAY = 30

# Number of histogram bins.
HIST_BINS = 15


# ============================================================
# Sharpness measurement
# ============================================================

def compute_blur_difference(
    image_bgr: np.ndarray,
    sigma: float = PROBE_SIGMA,
):
    """
    Measure how much an image changes after a fixed Gaussian blur.

    Interpretation:
        larger score -> more high-frequency information -> sharper
        smaller score -> less high-frequency information -> blurrier

    Returns:
        score:
            Mean absolute pixel difference between the original
            image and its Gaussian-blurred version.

        blurred:
            The probe-blurred image.
    """

    blurred = cv2.GaussianBlur(
        image_bgr,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    # Convert to float BEFORE subtraction so that uint8 arithmetic
    # cannot clip/wrap values.
    original_float = image_bgr.astype(np.float32)
    blurred_float = blurred.astype(np.float32)

    absolute_difference = np.abs(
        original_float - blurred_float
    )

    score = float(absolute_difference.mean())

    return score, blurred


# ============================================================
# Dataset analysis
# ============================================================

def analyze_training_set():
    """
    Compute blur-difference score for every training crop.
    """

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find split CSV:\n{SPLIT_CSV}"
        )

    df = pd.read_csv(SPLIT_CSV)

    required_columns = {
        "sample_id",
        "crop_path",
        "split",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in split.csv: {missing}"
        )

    train_df = df[
        df["split"].str.lower() == "train"
    ].copy()

    if len(train_df) == 0:
        raise ValueError(
            "No training samples found in split.csv"
        )

    rows = []

    print("=" * 72)
    print("BLUR-DIFFERENCE SHARPNESS ANALYSIS")
    print("=" * 72)
    print(f"Training samples: {len(train_df)}")
    print(f"Probe sigma: {PROBE_SIGMA}")
    print("=" * 72)

    for _, row in train_df.iterrows():

        sample_id = row["sample_id"]

        crop_path = Path(row["crop_path"])

        # split.csv contains project-relative paths.
        if not crop_path.is_absolute():
            crop_path = PROJECT_ROOT / crop_path

        image = cv2.imread(str(crop_path))

        if image is None:
            print(
                f"WARNING: could not read {crop_path}"
            )
            continue

        score, _ = compute_blur_difference(
            image,
            sigma=PROBE_SIGMA,
        )

        height, width = image.shape[:2]

        rows.append(
            {
                "sample_id": sample_id,
                "crop_path": str(crop_path),
                "width": width,
                "height": height,
                "blur_difference_score": score,
            }
        )

        print(
            f"{sample_id:<24} "
            f"score={score:8.4f} "
            f"size={width}x{height}"
        )

    result_df = pd.DataFrame(rows)

    # Low score = already relatively blurred.
    # High score = relatively sharp.
    result_df = result_df.sort_values(
        "blur_difference_score",
        ascending=True,
    ).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "blur_difference_scores.csv"

    result_df.to_csv(
        csv_path,
        index=False,
    )

    print()
    print(f"Saved scores: {csv_path}")

    return result_df


# ============================================================
# Ranking visualization
# ============================================================

def create_ranking_panel(df: pd.DataFrame):
    """
    Display samples from low score -> high score.

    This is the most important visualization:
    visually inspect whether the ranking corresponds to
    our perception of blur/sharpness.
    """

    if len(df) == 0:
        return

    num_images = min(
        NUM_DISPLAY,
        len(df),
    )

    # Instead of displaying only the first 30 samples,
    # sample evenly across the ENTIRE score distribution.
    indices = np.linspace(
        0,
        len(df) - 1,
        num_images,
        dtype=int,
    )

    display_df = df.iloc[indices].copy()

    cols = 5
    rows = int(
        np.ceil(num_images / cols)
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(18, rows * 3.2),
    )

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(
        axes,
        display_df.iterrows(),
    ):
        image = cv2.imread(
            row["crop_path"]
        )

        if image is None:
            continue

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        ax.imshow(image_rgb)

        ax.set_title(
            f"{row['sample_id']}\n"
            f"score={row['blur_difference_score']:.3f}\n"
            f"{row['width']}x{row['height']}",
            fontsize=8,
        )

        ax.axis("off")

    fig.suptitle(
        "Training images ranked by Original ↔ GaussianBlur difference\n"
        "Low score = less change after blur | High score = more change after blur",
        fontsize=14,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    output_path = (
        OUTPUT_DIR
        / "blur_difference_ranking.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved ranking panel: {output_path}"
    )


# ============================================================
# Distribution visualization
# ============================================================

def create_distribution_plot(
    df: pd.DataFrame,
):
    """
    Plot the distribution of blur-difference scores.
    """

    if len(df) == 0:
        return

    scores = df[
        "blur_difference_score"
    ].to_numpy()

    median_score = float(
        np.median(scores)
    )

    q25 = float(
        np.percentile(scores, 25)
    )

    q75 = float(
        np.percentile(scores, 75)
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.hist(
        scores,
        bins=HIST_BINS,
        edgecolor="black",
    )

    plt.axvline(
        median_score,
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_score:.3f}",
    )

    plt.axvline(
        q25,
        linestyle=":",
        linewidth=2,
        label=f"Q25 = {q25:.3f}",
    )

    plt.axvline(
        q75,
        linestyle=":",
        linewidth=2,
        label=f"Q75 = {q75:.3f}",
    )

    plt.xlabel(
        "Mean |Original - GaussianBlur(Original)|"
    )

    plt.ylabel(
        "Number of samples"
    )

    plt.title(
        "Training-set blur-difference distribution\n"
        f"Gaussian probe sigma = {PROBE_SIGMA}"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "blur_difference_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved distribution: {output_path}"
    )


# ============================================================
# Summary
# ============================================================

def save_summary(
    df: pd.DataFrame,
):
    if len(df) == 0:
        return

    scores = df[
        "blur_difference_score"
    ]

    summary = (
        f"Training samples: {len(df)}\n"
        f"Probe sigma: {PROBE_SIGMA}\n"
        f"\n"
        f"Minimum score: {scores.min():.6f}\n"
        f"Q25: {scores.quantile(0.25):.6f}\n"
        f"Median: {scores.median():.6f}\n"
        f"Q75: {scores.quantile(0.75):.6f}\n"
        f"Maximum score: {scores.max():.6f}\n"
        f"\n"
        f"Interpretation:\n"
        f"Low score  -> image changes little after blur -> likely already blurry.\n"
        f"High score -> image changes strongly after blur -> likely relatively sharp.\n"
    )

    summary_path = (
        OUTPUT_DIR
        / "blur_difference_summary.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)

    print(
        f"Saved summary: {summary_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    df = analyze_training_set()

    create_ranking_panel(df)

    create_distribution_plot(df)

    save_summary(df)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()