from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
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

from training.dataset import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    PlateSegmentationDataset,
)
from training.model import build_model  # noqa: E402


# ============================================================
# Defaults
# ============================================================

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "highres_augmentation_v2"
    / "best_model.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "failure_analysis"
    / "highres_augmentation_v2"
)

DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_K = 10

# Baseline V2 input resolution.
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 256

BATCH_SIZE = 1
NUM_WORKERS = 0


# ============================================================
# Metrics
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

    union = tp + fp + fn

    pred_area = pred.sum().item()
    gt_area = target.sum().item()

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

    area_ratio = (
        pred_area / gt_area
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
        "area_ratio": float(area_ratio),
    }


# ============================================================
# Image utilities
# ============================================================

def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    image = image * IMAGENET_STD + IMAGENET_MEAN

    return np.clip(image, 0.0, 1.0)


def compute_image_statistics(
    image_rgb: np.ndarray,
) -> dict:
    image_uint8 = np.clip(
        image_rgb * 255.0,
        0,
        255,
    ).astype(np.uint8)

    gray = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2GRAY,
    )

    brightness = float(gray.mean())
    contrast = float(gray.std())

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
    }


def create_error_map(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> np.ndarray:
    """
    Error-map colors:

    Green = True Positive
    Red   = False Positive
    Blue  = False Negative
    Black = True Negative
    """

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    error_map = np.zeros(
        (*gt.shape, 3),
        dtype=np.uint8,
    )

    # RGB
    error_map[tp] = [0, 255, 0]
    error_map[fp] = [255, 0, 0]
    error_map[fn] = [0, 0, 255]

    return error_map


# ============================================================
# Visualization
# ============================================================

def save_failure_panel(
    image: np.ndarray,
    gt_mask: np.ndarray,
    probability_map: np.ndarray,
    pred_mask: np.ndarray,
    metrics: dict,
    sample_id: str,
    rank: int,
    output_path: Path,
) -> None:

    error_map = create_error_map(
        pred_mask,
        gt_mask,
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    axes[0].imshow(image)
    axes[0].set_title("Input")

    axes[1].imshow(
        gt_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Ground Truth")

    probability_plot = axes[2].imshow(
        probability_map,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Probability")

    fig.colorbar(
        probability_plot,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    axes[3].imshow(
        pred_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[3].set_title("Prediction")

    axes[4].imshow(error_map)
    axes[4].set_title(
        "Error\n"
        "Green=TP | Red=FP | Blue=FN"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"#{rank:02d} {sample_id} | "
        f"IoU={metrics['iou']:.4f} | "
        f"Dice={metrics['dice']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f}",
        fontsize=12,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the worst validation failures "
            "for a segmentation checkpoint."
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
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
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

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    worst_dir = output_dir / "worst_10"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    worst_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("FAILURE ANALYSIS")
    print("=" * 72)
    print(f"Device:      {device}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Split:       val")
    print(f"Resolution:  {args.height}x{args.width}")
    print(f"Threshold:   {args.threshold}")
    print(f"Worst K:     {args.top_k}")
    print()

    # --------------------------------------------------------
    # Validation dataset
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
    print("Running inference on validation set...")
    print()

    # --------------------------------------------------------
    # First pass:
    # evaluate ALL validation samples.
    #
    # We keep the predictions in memory because the validation
    # set is tiny and we only save visualizations for the
    # worst K afterward.
    # --------------------------------------------------------

    results = []

    with torch.no_grad():

        for batch in loader:

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

            image_np = denormalize_image(
                images[0]
            )

            gt_mask_np = (
                masks[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            probability_np = (
                probabilities[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            pred_mask_np = (
                predictions[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            image_stats = compute_image_statistics(
                image_np
            )

            row = {
                "sample_id": sample_id,
                **metrics,
                **image_stats,
            }

            results.append({
                "row": row,
                "image": image_np,
                "gt_mask": gt_mask_np,
                "probability": probability_np,
                "prediction": pred_mask_np,
            })

            print(
                f"{sample_id:28s} "
                f"IoU={metrics['iou']:.4f}"
            )

    # --------------------------------------------------------
    # Sort by IoU and select worst K
    # --------------------------------------------------------

    results.sort(
        key=lambda item: item["row"]["iou"]
    )

    top_k = min(
        args.top_k,
        len(results),
    )

    worst_results = results[:top_k]

    # --------------------------------------------------------
    # Save panels only for worst K
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(f"WORST {top_k} VALIDATION SAMPLES")
    print("=" * 72)

    worst_rows = []

    for rank, result in enumerate(
        worst_results,
        start=1,
    ):
        row = result["row"]

        worst_rows.append(row)

        filename = (
            f"{rank:02d}_"
            f"{row['sample_id']}.png"
        )

        save_failure_panel(
            image=result["image"],
            gt_mask=result["gt_mask"],
            probability_map=result["probability"],
            pred_mask=result["prediction"],
            metrics=row,
            sample_id=row["sample_id"],
            rank=rank,
            output_path=worst_dir / filename,
        )

        print(
            f"#{rank:02d} "
            f"{row['sample_id']:28s} "
            f"IoU={row['iou']:.4f} "
            f"Precision={row['precision']:.4f} "
            f"Recall={row['recall']:.4f} "
            f"FP={row['fp_pixels']} "
            f"FN={row['fn_pixels']}"
        )

    # --------------------------------------------------------
    # DataFrames
    # --------------------------------------------------------

    all_rows = [
        result["row"]
        for result in results
    ]

    all_df = pd.DataFrame(all_rows)

    worst_df = pd.DataFrame(worst_rows)

    # Detailed CSV only for the failures we inspect.
    worst_metrics_path = (
        output_dir / "worst_10_metrics.csv"
    )

    worst_df.to_csv(
        worst_metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    mean_iou = all_df["iou"].mean()
    median_iou = all_df["iou"].median()

    mean_dice = all_df["dice"].mean()
    median_dice = all_df["dice"].median()

    worst_mean_iou = worst_df["iou"].mean()

    summary_lines = [
        "FAILURE ANALYSIS SUMMARY",
        "=" * 72,
        "",
        f"checkpoint={checkpoint_path}",
        "split=val",
        f"resolution={args.height}x{args.width}",
        f"threshold={args.threshold}",
        "",
        f"N_validation={len(all_df)}",
        f"mean_iou={mean_iou:.6f}",
        f"median_iou={median_iou:.6f}",
        f"mean_dice={mean_dice:.6f}",
        f"median_dice={median_dice:.6f}",
        "",
        f"worst_k={top_k}",
        f"worst_k_mean_iou={worst_mean_iou:.6f}",
        "",
        "Worst samples:",
    ]

    for rank, row in enumerate(
        worst_rows,
        start=1,
    ):
        summary_lines.append(
            f"{rank:02d}. "
            f"{row['sample_id']} | "
            f"IoU={row['iou']:.4f} | "
            f"Precision={row['precision']:.4f} | "
            f"Recall={row['recall']:.4f} | "
            f"FP={row['fp_pixels']} | "
            f"FN={row['fn_pixels']}"
        )

    summary_path = (
        output_dir / "summary.txt"
    )

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    print(f"N:                 {len(all_df)}")
    print(f"Mean IoU:          {mean_iou:.4f}")
    print(f"Median IoU:        {median_iou:.4f}")
    print(f"Mean Dice:         {mean_dice:.4f}")
    print(f"Median Dice:       {median_dice:.4f}")
    print(f"Worst {top_k} Mean IoU: {worst_mean_iou:.4f}")
    print("=" * 72)

    print()
    print(f"Saved metrics: {worst_metrics_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved panels:  {worst_dir}")


if __name__ == "__main__":
    main()