from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from training.dataset import PlateSegmentationDataset
from training.model import build_model


# ============================================================
# Defaults
# ============================================================

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "resolution_high_v2"
    / "best_model.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "failure_patterns"
    / "resolution_high_v2"
)

DEFAULT_THRESHOLD = 0.5
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 256
DEFAULT_WORST_K = 10

BATCH_SIZE = 1
NUM_WORKERS = 0


# ============================================================
# Segmentation metrics
# ============================================================

def compute_binary_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> dict:

    pred = pred.bool()
    target = target.bool()

    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()

    pred_area = pred.sum().item()
    gt_area = target.sum().item()

    union = tp + fp + fn

    iou = (tp + eps) / (union + eps)

    dice = (
        (2.0 * tp + eps)
        / (pred_area + gt_area + eps)
    )

    precision = (
        (tp + eps)
        / (tp + fp + eps)
    )

    recall = (
        (tp + eps)
        / (tp + fn + eps)
    )

    # Normalize FP/FN by GT area so samples of different
    # sizes are more comparable.
    fp_ratio = (
        fp / gt_area
        if gt_area > 0
        else float("nan")
    )

    fn_ratio = (
        fn / gt_area
        if gt_area > 0
        else float("nan")
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp_pixels": int(tp),
        "fp_pixels": int(fp),
        "fn_pixels": int(fn),
        "gt_area": int(gt_area),
        "pred_area": int(pred_area),
        "fp_ratio": float(fp_ratio),
        "fn_ratio": float(fn_ratio),
    }


# ============================================================
# Original-image statistics
# ============================================================

def compute_original_image_statistics(
    image_bgr: np.ndarray,
    target_width: int,
    target_height: int,
) -> dict:

    height, width = image_bgr.shape[:2]

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    brightness = float(gray.mean())
    contrast = float(gray.std())

    # Higher value usually means a sharper image.
    # This is a useful proxy, not an absolute blur measurement.
    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    area = width * height
    aspect_ratio = width / height

    # Same scale rule used by dataset.letterbox().
    letterbox_scale = min(
        target_width / width,
        target_height / height,
    )

    resized_width = max(
        1,
        int(round(width * letterbox_scale)),
    )

    resized_height = max(
        1,
        int(round(height * letterbox_scale)),
    )

    resized_area = resized_width * resized_height
    canvas_area = target_width * target_height

    canvas_fill_ratio = resized_area / canvas_area

    return {
        "original_width": int(width),
        "original_height": int(height),
        "original_area": int(area),
        "aspect_ratio": float(aspect_ratio),
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "letterbox_scale": float(letterbox_scale),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "canvas_fill_ratio": float(canvas_fill_ratio),
    }


# ============================================================
# Group summary
# ============================================================

FEATURE_COLUMNS = [
    "original_width",
    "original_height",
    "original_area",
    "aspect_ratio",
    "brightness",
    "contrast",
    "sharpness",
    "letterbox_scale",
    "canvas_fill_ratio",
]


def summarize_group(
    df: pd.DataFrame,
    name: str,
) -> list[str]:

    lines = []

    lines.append(name)
    lines.append("-" * len(name))
    lines.append(f"N: {len(df)}")
    lines.append(f"Mean IoU:   {df['iou'].mean():.4f}")
    lines.append(f"Median IoU: {df['iou'].median():.4f}")
    lines.append("")

    for column in FEATURE_COLUMNS:
        lines.append(
            f"{column:20s} "
            f"mean={df[column].mean():.4f} | "
            f"median={df[column].median():.4f}"
        )

    lines.append("")

    return lines


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Quantitatively analyze image characteristics "
            "associated with segmentation failures."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
    )

    parser.add_argument(
        "--worst-k",
        type=int,
        default=DEFAULT_WORST_K,
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("FAILURE PATTERN ANALYSIS")
    print("=" * 72)
    print(f"Device:      {device}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Split:       val")
    print(f"Resolution:  {args.height}x{args.width}")
    print(f"Threshold:   {args.threshold}")
    print(f"Worst K:     {args.worst_k}")
    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = PlateSegmentationDataset(
        split="val",
        target_width=args.width,
        target_height=args.height,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded checkpoint | "
        f"phase={checkpoint.get('phase')} | "
        f"epoch={checkpoint.get('epoch')} | "
        f"val_iou={checkpoint.get('val_iou')}"
    )

    print()
    print("Analyzing validation samples...")
    print()

    rows = []

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        for index, batch in enumerate(loader):

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            sample_id = batch["sample_id"][0]

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= args.threshold
            )

            metrics = compute_binary_metrics(
                predictions[0],
                masks[0] >= 0.5,
            )

            # ------------------------------------------------
            # Load ORIGINAL crop.
            #
            # Important:
            # image characteristics are measured BEFORE
            # resize / letterbox / ImageNet normalization.
            # ------------------------------------------------

            dataset_row = dataset.df.iloc[index]

            crop_rel = str(
                dataset_row["crop_path"]
            ).replace("\\", "/")

            crop_path = (
                PROJECT_ROOT
                / Path(crop_rel)
            )

            original_image = cv2.imread(
                str(crop_path),
                cv2.IMREAD_COLOR,
            )

            if original_image is None:
                raise RuntimeError(
                    f"Could not read original image: {crop_path}"
                )

            image_stats = (
                compute_original_image_statistics(
                    original_image,
                    target_width=args.width,
                    target_height=args.height,
                )
            )

            row = {
                "sample_id": sample_id,
                **metrics,
                **image_stats,
            }

            rows.append(row)

            print(
                f"{sample_id:28s} "
                f"IoU={metrics['iou']:.4f} | "
                f"sharpness={image_stats['sharpness']:.1f} | "
                f"size="
                f"{image_stats['original_width']}x"
                f"{image_stats['original_height']}"
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "iou",
        ascending=True,
    ).reset_index(drop=True)

    df["failure_rank"] = (
        np.arange(len(df)) + 1
    )

    worst_k = min(
        args.worst_k,
        len(df),
    )

    worst_df = df.iloc[:worst_k].copy()
    rest_df = df.iloc[worst_k:].copy()

    # --------------------------------------------------------
    # Save all sample statistics
    # --------------------------------------------------------

    all_samples_path = (
        output_dir / "all_samples.csv"
    )

    df.to_csv(
        all_samples_path,
        index=False,
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    correlation_columns = [
        "iou",
        *FEATURE_COLUMNS,
    ]

    correlations = (
        df[correlation_columns]
        .corr(method="spearman")["iou"]
        .drop("iou")
        .sort_values(ascending=False)
    )

    correlations_df = (
        correlations
        .rename("spearman_correlation_with_iou")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    correlations_path = (
        output_dir / "correlations.csv"
    )

    correlations_df.to_csv(
        correlations_path,
        index=False,
    )

    # --------------------------------------------------------
    # Worst vs rest comparison
    # --------------------------------------------------------

    comparison_rows = []

    for feature in FEATURE_COLUMNS:

        worst_mean = worst_df[feature].mean()
        rest_mean = rest_df[feature].mean()

        difference = worst_mean - rest_mean

        relative_difference = (
            difference / rest_mean
            if rest_mean != 0
            else float("nan")
        )

        comparison_rows.append({
            "feature": feature,
            "worst_10_mean": worst_mean,
            "rest_mean": rest_mean,
            "absolute_difference": difference,
            "relative_difference": relative_difference,
        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        output_dir
        / "worst_10_vs_rest.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # Text summary
    # --------------------------------------------------------

    summary_lines = []

    summary_lines.append(
        "=" * 72
    )
    summary_lines.append(
        "FAILURE PATTERN ANALYSIS"
    )
    summary_lines.append(
        "=" * 72
    )
    summary_lines.append("")

    summary_lines.extend(
        summarize_group(
            df,
            "ALL VALIDATION",
        )
    )

    summary_lines.extend(
        summarize_group(
            worst_df,
            f"WORST {worst_k}",
        )
    )

    summary_lines.extend(
        summarize_group(
            rest_df,
            f"REMAINING {len(rest_df)}",
        )
    )

    summary_lines.append(
        "SPEARMAN CORRELATION WITH IoU"
    )
    summary_lines.append(
        "-" * 29
    )

    for _, row in correlations_df.iterrows():
        summary_lines.append(
            f"{row['feature']:20s} "
            f"{row['spearman_correlation_with_iou']:+.4f}"
        )

    summary_lines.append("")
    summary_lines.append(
        f"WORST {worst_k} SAMPLES"
    )
    summary_lines.append(
        "-" * 20
    )

    for _, row in worst_df.iterrows():
        summary_lines.append(
            f"#{int(row['failure_rank']):02d} "
            f"{row['sample_id']:28s} "
            f"IoU={row['iou']:.4f} | "
            f"sharpness={row['sharpness']:.1f} | "
            f"brightness={row['brightness']:.1f} | "
            f"size={int(row['original_width'])}x"
            f"{int(row['original_height'])}"
        )

    summary_text = "\n".join(
        summary_lines
    )

    summary_path = (
        output_dir / "summary.txt"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print(summary_text)

    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)
    print(f"All samples:       {all_samples_path}")
    print(f"Correlations:      {correlations_path}")
    print(f"Worst 10 vs rest: {comparison_path}")
    print(f"Summary:           {summary_path}")


if __name__ == "__main__":
    main()