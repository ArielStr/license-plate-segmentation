from __future__ import annotations

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

from training.dataset import PlateSegmentationDataset
from training.model import build_model


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "resolution_high_v3"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "final_test"
)

PANELS_DIR = OUTPUT_DIR / "worst_30_panels"

INPUT_WIDTH = 768
INPUT_HEIGHT = 256

THRESHOLD = 0.5
WORST_K = 30

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
    tn = (~pred & ~target).sum().item()

    pred_area = pred.sum().item()
    gt_area = target.sum().item()

    union = tp + fp + fn

    iou = (
        (tp + eps)
        / (union + eps)
    )

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

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp_pixels": int(tp),
        "fp_pixels": int(fp),
        "fn_pixels": int(fn),
        "tn_pixels": int(tn),
    }


# ============================================================
# Original-size groups
# ============================================================

def get_size_group(
    original_height: int,
) -> str:

    if original_height < 32:
        return "Tiny"

    if original_height < 64:
        return "Small"

    if original_height < 128:
        return "Medium"

    return "Large"


# ============================================================
# Visualization
# ============================================================

def create_error_map(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:

    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    error_map = np.zeros(
        (
            prediction.shape[0],
            prediction.shape[1],
            3,
        ),
        dtype=np.float32,
    )

    # Green = True Positive
    error_map[tp] = [0.0, 1.0, 0.0]

    # Red = False Positive
    error_map[fp] = [1.0, 0.0, 0.0]

    # Blue = False Negative
    error_map[fn] = [0.0, 0.0, 1.0]

    return error_map


def save_failure_panel(
    image: np.ndarray,
    gt_mask: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    sample_id: str,
    rank: int,
    metrics: dict,
    original_width: int,
    original_height: int,
    size_group: str,
    output_path: Path,
) -> None:

    error_map = create_error_map(
        prediction,
        gt_mask,
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    axes[1].imshow(
        gt_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability_plot = axes[2].imshow(
        probability,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title("Probability")
    axes[2].axis("off")

    fig.colorbar(
        probability_plot,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    axes[3].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[3].set_title("Prediction")
    axes[3].axis("off")

    # --------------------------------------------------------
    # Error map
    # --------------------------------------------------------

    axes[4].imshow(error_map)

    axes[4].set_title(
        "Error\n"
        "Green=TP | Red=FP | Blue=FN"
    )

    axes[4].axis("off")

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        f"#{rank:02d} {sample_id} | "
        f"IoU={metrics['iou']:.4f} | "
        f"Dice={metrics['dice']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f}\n"
        f"Original={original_width}x{original_height} | "
        f"Group={size_group}",
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PANELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("FINAL TEST EVALUATION")
    print("=" * 72)

    print(f"Device:      {device}")
    print(f"Checkpoint:  {CHECKPOINT_PATH}")
    print("Model:       U-Net + ResNet34")
    print(
        f"Resolution:  "
        f"{INPUT_HEIGHT}x{INPUT_WIDTH}"
    )
    print(f"Threshold:   {THRESHOLD}")
    print(f"Worst K:     {WORST_K}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = PlateSegmentationDataset(
        split="test",
        target_width=INPUT_WIDTH,
        target_height=INPUT_HEIGHT,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Test samples: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model(
        architecture="unet",
        encoder_name="resnet34",
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print(
        "Loaded checkpoint | "
        f"phase={checkpoint.get('phase')} | "
        f"epoch={checkpoint.get('epoch')} | "
        f"val_iou={checkpoint.get('val_iou')} | "
        f"train_size={checkpoint.get('train_size')}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    with torch.no_grad():

        for index, batch in enumerate(loader):

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            sample_id = batch["sample_id"][0]

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            )

            metrics = compute_binary_metrics(
                predictions[0],
                masks[0] >= 0.5,
            )

            # ------------------------------------------------
            # Original crop metadata
            # ------------------------------------------------

            row = dataset.df.iloc[index]

            crop_rel = str(
                row["crop_path"]
            ).replace("\\", "/")

            crop_path = (
                PROJECT_ROOT
                / Path(crop_rel)
            )

            original_bgr = cv2.imread(
                str(crop_path),
                cv2.IMREAD_COLOR,
            )

            if original_bgr is None:
                raise RuntimeError(
                    f"Could not read crop: "
                    f"{crop_path}"
                )

            original_height, original_width = (
                original_bgr.shape[:2]
            )

            size_group = get_size_group(
                original_height
            )

            # ------------------------------------------------
            # Store visual data
            # ------------------------------------------------

            image_np = (
                images[0]
                .detach()
                .cpu()
                .numpy()
            )

            # Undo ImageNet normalization for display.
            imagenet_mean = np.array(
                [0.485, 0.456, 0.406],
                dtype=np.float32,
            )

            imagenet_std = np.array(
                [0.229, 0.224, 0.225],
                dtype=np.float32,
            )

            image_np = np.transpose(
                image_np,
                (1, 2, 0),
            )

            image_np = (
                image_np * imagenet_std
                + imagenet_mean
            )

            image_np = np.clip(
                image_np,
                0.0,
                1.0,
            )

            gt_np = (
                masks[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            gt_np = (
                gt_np >= 0.5
            ).astype(np.uint8)

            probability_np = (
                probabilities[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            prediction_np = (
                predictions[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            results.append({
                "sample_id": sample_id,

                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],

                "tp_pixels":
                    metrics["tp_pixels"],

                "fp_pixels":
                    metrics["fp_pixels"],

                "fn_pixels":
                    metrics["fn_pixels"],

                "original_width":
                    original_width,

                "original_height":
                    original_height,

                "size_group":
                    size_group,

                "image":
                    image_np,

                "gt":
                    gt_np,

                "probability":
                    probability_np,

                "prediction":
                    prediction_np,
            })

            print(
                f"{sample_id:28s} | "
                f"IoU={metrics['iou']:.4f} | "
                f"Dice={metrics['dice']:.4f} | "
                f"P={metrics['precision']:.4f} | "
                f"R={metrics['recall']:.4f} | "
                f"{size_group}"
            )

    # ========================================================
    # DataFrame
    # ========================================================

    table_rows = []

    for result in results:

        table_rows.append({
            key: value
            for key, value in result.items()
            if key not in {
                "image",
                "gt",
                "probability",
                "prediction",
            }
        })

    df = pd.DataFrame(
        table_rows
    )

    df = df.sort_values(
        "iou",
        ascending=True,
    ).reset_index(drop=True)

    df["rank"] = (
        np.arange(len(df)) + 1
    )

    front_columns = [
        "rank",
        "sample_id",
        "iou",
        "dice",
        "precision",
        "recall",
        "original_width",
        "original_height",
        "size_group",
    ]

    remaining_columns = [
        column
        for column in df.columns
        if column not in front_columns
    ]

    df = df[
        front_columns
        + remaining_columns
    ]

    # --------------------------------------------------------
    # Save complete test results
    # --------------------------------------------------------

    results_path = (
        OUTPUT_DIR
        / "test_results.csv"
    )

    df.to_csv(
        results_path,
        index=False,
    )

    # ========================================================
    # Worst 30
    # ========================================================

    worst_k = min(
        WORST_K,
        len(df),
    )

    worst_df = (
        df.iloc[:worst_k]
        .copy()
    )

    worst_path = (
        OUTPUT_DIR
        / "worst_30.csv"
    )

    worst_df.to_csv(
        worst_path,
        index=False,
    )

    # Fast lookup by sample ID.
    visual_lookup = {
        result["sample_id"]: result
        for result in results
    }

    print()
    print("=" * 72)
    print(
        f"WORST {worst_k} TEST SAMPLES"
    )
    print("=" * 72)

    for rank, (_, row) in enumerate(
        worst_df.iterrows(),
        start=1,
    ):

        sample_id = row["sample_id"]

        result = visual_lookup[
            sample_id
        ]

        metrics = {
            "iou": result["iou"],
            "dice": result["dice"],
            "precision": result["precision"],
            "recall": result["recall"],
        }

        filename = (
            f"{rank:02d}_"
            f"{sample_id}.png"
        )

        save_failure_panel(
            image=result["image"],
            gt_mask=result["gt"],
            probability=result["probability"],
            prediction=result["prediction"],
            sample_id=sample_id,
            rank=rank,
            metrics=metrics,
            original_width=result[
                "original_width"
            ],
            original_height=result[
                "original_height"
            ],
            size_group=result[
                "size_group"
            ],
            output_path=(
                PANELS_DIR
                / filename
            ),
        )

        print(
            f"#{rank:02d} "
            f"{sample_id:28s} | "
            f"IoU={result['iou']:.4f} | "
            f"Dice={result['dice']:.4f} | "
            f"P={result['precision']:.4f} | "
            f"R={result['recall']:.4f} | "
            f"{result['size_group']}"
        )

    # ========================================================
    # Size-group analysis
    # ========================================================

    size_order = [
        "Tiny",
        "Small",
        "Medium",
        "Large",
    ]

    group_rows = []

    for group in size_order:

        group_df = df[
            df["size_group"] == group
        ]

        if len(group_df) == 0:
            continue

        group_rows.append({
            "size_group": group,
            "n": len(group_df),

            "mean_iou":
                group_df["iou"].mean(),

            "median_iou":
                group_df["iou"].median(),

            "mean_dice":
                group_df["dice"].mean(),

            "mean_precision":
                group_df["precision"].mean(),

            "mean_recall":
                group_df["recall"].mean(),
        })

    groups_df = pd.DataFrame(
        group_rows
    )

    groups_path = (
        OUTPUT_DIR
        / "size_group_results.csv"
    )

    groups_df.to_csv(
        groups_path,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary_lines = [
        "FINAL TEST EVALUATION",
        "=" * 72,
        "",
        "Model: U-Net + ResNet34",
        (
            f"Resolution: "
            f"{INPUT_HEIGHT}x{INPUT_WIDTH}"
        ),
        "Augmentation during training: No",
        f"Threshold: {THRESHOLD}",
        f"N_test: {len(df)}",
        "",
        "OVERALL TEST PERFORMANCE",
        "-" * 72,
        (
            f"Mean IoU:       "
            f"{df['iou'].mean():.6f}"
        ),
        (
            f"Median IoU:     "
            f"{df['iou'].median():.6f}"
        ),
        (
            f"Std IoU:        "
            f"{df['iou'].std():.6f}"
        ),
        "",
        (
            f"Mean Dice:      "
            f"{df['dice'].mean():.6f}"
        ),
        (
            f"Median Dice:    "
            f"{df['dice'].median():.6f}"
        ),
        "",
        (
            f"Mean Precision: "
            f"{df['precision'].mean():.6f}"
        ),
        (
            f"Mean Recall:    "
            f"{df['recall'].mean():.6f}"
        ),
        "",
        (
            f"Best IoU:       "
            f"{df['iou'].max():.6f}"
        ),
        (
            f"Worst IoU:      "
            f"{df['iou'].min():.6f}"
        ),
        "",
        "PERFORMANCE BY ORIGINAL PLATE HEIGHT",
        "-" * 72,
    ]

    for _, row in groups_df.iterrows():

        summary_lines.append(
            f"{row['size_group']:6s} | "
            f"N={int(row['n']):3d} | "
            f"Mean IoU={row['mean_iou']:.6f} | "
            f"Median IoU={row['median_iou']:.6f} | "
            f"Dice={row['mean_dice']:.6f} | "
            f"P={row['mean_precision']:.6f} | "
            f"R={row['mean_recall']:.6f}"
        )

    # --------------------------------------------------------
    # IoU thresholds
    # --------------------------------------------------------

    summary_lines.extend([
        "",
        "IOU DISTRIBUTION",
        "-" * 72,
    ])

    for threshold in [
        0.90,
        0.92,
        0.94,
        0.95,
        0.96,
        0.97,
    ]:

        count = int(
            (df["iou"] >= threshold)
            .sum()
        )

        percentage = (
            100.0 * count / len(df)
        )

        summary_lines.append(
            f"IoU >= {threshold:.2f}: "
            f"{count:2d}/{len(df)} "
            f"({percentage:.1f}%)"
        )

    # --------------------------------------------------------
    # Worst 30 summary
    # --------------------------------------------------------

    summary_lines.extend([
        "",
        f"WORST {worst_k}",
        "-" * 72,
        (
            f"Mean IoU:   "
            f"{worst_df['iou'].mean():.6f}"
        ),
        (
            f"Median IoU: "
            f"{worst_df['iou'].median():.6f}"
        ),
        "",
    ])

    for _, row in worst_df.iterrows():

        summary_lines.append(
            f"#{int(row['rank']):02d} "
            f"{row['sample_id']:28s} | "
            f"IoU={row['iou']:.4f} | "
            f"Dice={row['dice']:.4f} | "
            f"{row['size_group']}"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_text = "\n".join(
        summary_lines
    )

    summary_path = (
        OUTPUT_DIR
        / "summary.txt"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print(summary_text)

    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)

    print(
        f"Results:      {results_path}"
    )

    print(
        f"Size groups:  {groups_path}"
    )

    print(
        f"Worst 30:     {worst_path}"
    )

    print(
        f"Summary:      {summary_path}"
    )

    print(
        f"Panels:       {PANELS_DIR}"
    )


if __name__ == "__main__":
    main()