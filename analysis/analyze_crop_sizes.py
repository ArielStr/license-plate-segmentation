from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = PROJECT_ROOT / "data" / "crops"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "outputs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Candidate network input sizes: (width, height)
CANDIDATE_CANVASES = [
    (320, 96),
    (384, 128),
    (448, 128),
    (512, 160),
]


@dataclass(frozen=True)
class CropInfo:
    filename: str
    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def area(self) -> int:
        return self.width * self.height


def iter_image_paths(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_crop_info(crops_dir: Path) -> list[CropInfo]:
    if not crops_dir.exists():
        raise FileNotFoundError(f"Crops directory not found: {crops_dir}")

    records: list[CropInfo] = []

    for image_path in iter_image_paths(crops_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"[WARNING] Could not read: {image_path.name}")
            continue

        height, width = image.shape[:2]

        records.append(
            CropInfo(
                filename=image_path.name,
                width=width,
                height=height,
            )
        )

    if not records:
        raise RuntimeError(f"No readable images found in: {crops_dir}")

    return records


def build_dataframe(records: list[CropInfo]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filename": [r.filename for r in records],
            "width": [r.width for r in records],
            "height": [r.height for r in records],
            "aspect_ratio": [r.aspect_ratio for r in records],
            "area": [r.area for r in records],
        }
    )


def letterbox_stats(
    width: int,
    height: int,
    canvas_width: int,
    canvas_height: int,
) -> dict[str, float]:
    """
    Resize while preserving aspect ratio, then pad to the requested canvas.

    Returns:
        scale:
            Uniform resize factor.

        resized_width / resized_height:
            Size after resize and before padding.

        image_fraction:
            Fraction of the final canvas occupied by the resized crop.
            1.0 means no padding at all.

        padding_fraction:
            Fraction of the final canvas that is padding.
    """
    scale = min(canvas_width / width, canvas_height / height)

    resized_width = width * scale
    resized_height = height * scale

    canvas_area = canvas_width * canvas_height
    image_area = resized_width * resized_height

    image_fraction = image_area / canvas_area
    padding_fraction = 1.0 - image_fraction

    return {
        "scale": scale,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "image_fraction": image_fraction,
        "padding_fraction": padding_fraction,
    }


def evaluate_canvases(
    df: pd.DataFrame,
    candidates: list[tuple[int, int]],
) -> pd.DataFrame:
    rows = []

    for canvas_width, canvas_height in candidates:
        padding_fractions = []
        image_fractions = []
        scales = []

        for row in df.itertuples(index=False):
            stats = letterbox_stats(
                width=int(row.width),
                height=int(row.height),
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )

            padding_fractions.append(stats["padding_fraction"])
            image_fractions.append(stats["image_fraction"])
            scales.append(stats["scale"])

        padding = np.asarray(padding_fractions)
        occupancy = np.asarray(image_fractions)
        scale_values = np.asarray(scales)

        rows.append(
            {
                "canvas": f"{canvas_width}x{canvas_height}",
                "canvas_width": canvas_width,
                "canvas_height": canvas_height,
                "canvas_aspect_ratio": canvas_width / canvas_height,
                "mean_padding_pct": padding.mean() * 100,
                "median_padding_pct": np.median(padding) * 100,
                "p90_padding_pct": np.percentile(padding, 90) * 100,
                "mean_image_occupancy_pct": occupancy.mean() * 100,
                "median_scale": np.median(scale_values),
                "min_scale": scale_values.min(),
                "max_scale": scale_values.max(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["mean_padding_pct", "p90_padding_pct"]
    )


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("CROP DATASET SUMMARY")
    print("=" * 72)
    print(f"Number of crops: {len(df)}")

    print("\nWidth [px]")
    print(df["width"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2))

    print("\nHeight [px]")
    print(df["height"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2))

    print("\nAspect ratio (width / height)")
    print(
        df["aspect_ratio"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(3)
    )

    print("\nHeight thresholds")
    for threshold in [24, 30, 32, 40, 50, 64, 96, 128]:
        count = int((df["height"] >= threshold).sum())
        pct = 100.0 * count / len(df)
        print(f"  h >= {threshold:3d}: {count:3d}/{len(df)} ({pct:5.1f}%)")

    print("\nSmallest crops by height")
    print(
        df.nsmallest(10, "height")[
            ["filename", "width", "height", "aspect_ratio"]
        ].to_string(index=False)
    )

    print("\nMost extreme aspect ratios")
    low = df.nsmallest(5, "aspect_ratio")[
        ["filename", "width", "height", "aspect_ratio"]
    ]
    high = df.nlargest(5, "aspect_ratio")[
        ["filename", "width", "height", "aspect_ratio"]
    ]

    print("\nLowest AR:")
    print(low.to_string(index=False))

    print("\nHighest AR:")
    print(high.to_string(index=False))


def save_histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 20,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of crops")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_scatter(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["width"], df["height"])
    plt.title("Crop width vs. height")
    plt.xlabel("Width [px]")
    plt.ylabel("Height [px]")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_crop_info(CROPS_DIR)
    df = build_dataframe(records)

    print_dataset_summary(df)

    crop_csv_path = OUTPUT_DIR / "crop_statistics.csv"
    df.to_csv(crop_csv_path, index=False)

    canvas_df = evaluate_canvases(df, CANDIDATE_CANVASES)
    canvas_csv_path = OUTPUT_DIR / "canvas_comparison.csv"
    canvas_df.to_csv(canvas_csv_path, index=False)

    print("\n" + "=" * 72)
    print("LETTERBOX CANVAS COMPARISON")
    print("=" * 72)
    print(
        canvas_df[
            [
                "canvas",
                "canvas_aspect_ratio",
                "mean_padding_pct",
                "median_padding_pct",
                "p90_padding_pct",
                "median_scale",
            ]
        ].round(3).to_string(index=False)
    )

    best = canvas_df.iloc[0]
    print(
        "\nBest candidate by mean padding: "
        f"{best['canvas']} "
        f"(mean padding {best['mean_padding_pct']:.1f}%)"
    )

    save_histogram(
        df["width"],
        title="Crop Width Distribution",
        xlabel="Width [px]",
        output_path=OUTPUT_DIR / "width_histogram.png",
    )

    save_histogram(
        df["height"],
        title="Crop Height Distribution",
        xlabel="Height [px]",
        output_path=OUTPUT_DIR / "height_histogram.png",
    )

    save_histogram(
        df["aspect_ratio"],
        title="Crop Aspect Ratio Distribution",
        xlabel="Width / Height",
        output_path=OUTPUT_DIR / "aspect_ratio_histogram.png",
    )

    save_scatter(
        df,
        output_path=OUTPUT_DIR / "width_vs_height.png",
    )

    print("\nSaved:")
    print(f"  {crop_csv_path}")
    print(f"  {canvas_csv_path}")
    print(f"  {OUTPUT_DIR / 'width_histogram.png'}")
    print(f"  {OUTPUT_DIR / 'height_histogram.png'}")
    print(f"  {OUTPUT_DIR / 'aspect_ratio_histogram.png'}")
    print(f"  {OUTPUT_DIR / 'width_vs_height.png'}")


if __name__ == "__main__":
    main()
