from __future__ import annotations

import sys
from pathlib import Path

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

from training.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PlateSegmentationDataset,
)
from training.model import build_model


# ============================================================
# Configuration
# ============================================================

CHECKPOINTS = {
    "175": (
        PROJECT_ROOT
        / "checkpoints"
        / "scaling_175_v3"
        / "best_model.pt"
    ),
    "250": (
        PROJECT_ROOT
        / "checkpoints"
        / "scaling_250_v3"
        / "best_model.pt"
    ),
    "350": (
        PROJECT_ROOT
        / "checkpoints"
        / "scaling_350_v3"
        / "best_model.pt"
    ),
    "424": (
        PROJECT_ROOT
        / "checkpoints"
        / "baseline_v3"
        / "best_model.pt"
    ),
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "data_scaling_comparison"
)

PANELS_DIR = OUTPUT_DIR / "worst_10_panels"

THRESHOLD = 0.5

INPUT_WIDTH = 384
INPUT_HEIGHT = 128

WORST_K = 10

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

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp_pixels": int(tp),
        "fp_pixels": int(fp),
        "fn_pixels": int(fn),
    }


# ============================================================
# Image utilities
# ============================================================

def denormalize_image(
    image_tensor: torch.Tensor,
) -> np.ndarray:

    image = (
        image_tensor
        .detach()
        .cpu()
        .numpy()
    )

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    image = (
        image * IMAGENET_STD
        + IMAGENET_MEAN
    )

    return np.clip(
        image,
        0.0,
        1.0,
    )


# ============================================================
# Model loading
# ============================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # All data-scaling experiments use:
    # U-Net + ResNet34.
    model = build_model(
        architecture="unet",
        encoder_name="resnet34",
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# Run one model over the complete validation set
# ============================================================

def evaluate_model(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:

    print()
    print("=" * 72)
    print(f"EVALUATING TRAIN SIZE {model_name}")
    print("=" * 72)

    results = {}

    with torch.no_grad():

        for batch in loader:

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

            prediction_np = (
                predictions[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            results[sample_id] = {
                "metrics": metrics,
                "prediction": prediction_np,
            }

            print(
                f"{sample_id:28s} "
                f"IoU={metrics['iou']:.4f}"
            )

    return results


# ============================================================
# Comparison panel
# ============================================================

def save_comparison_panel(
    image: np.ndarray,
    gt_mask: np.ndarray,
    sample_id: str,
    rank: int,
    model_results: dict,
    output_path: Path,
) -> None:

    model_names = list(CHECKPOINTS.keys())

    # Input + GT + one prediction for each model.
    num_columns = 2 + len(model_names)

    fig, axes = plt.subplots(
        1,
        num_columns,
        figsize=(4 * num_columns, 4),
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

    axes[1].set_title(
        "Ground Truth"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    for column_index, model_name in enumerate(
        model_names,
        start=2,
    ):

        result = model_results[model_name][
            sample_id
        ]

        prediction = result["prediction"]
        metrics = result["metrics"]

        axes[column_index].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[column_index].set_title(
            f"Train {model_name}\n"
            f"IoU={metrics['iou']:.4f}"
        )

        axes[column_index].axis("off")

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    iou_175 = (
        model_results["175"][sample_id]
        ["metrics"]["iou"]
    )

    iou_424 = (
        model_results["424"][sample_id]
        ["metrics"]["iou"]
    )

    delta = iou_424 - iou_175

    fig.suptitle(
        f"#{rank:02d} {sample_id} | "
        f"175 → 424 ΔIoU={delta:+.4f}",
        fontsize=13,
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("V3 DATA SCALING COMPARISON")
    print("=" * 72)

    print(f"Device:      {device}")
    print("Split:       val")
    print(
        f"Resolution:  "
        f"{INPUT_HEIGHT}x{INPUT_WIDTH}"
    )
    print(f"Threshold:   {THRESHOLD}")
    print(f"Worst K:     {WORST_K}")

    print()

    # --------------------------------------------------------
    # Check checkpoint paths before doing any work.
    # --------------------------------------------------------

    print("Checkpoints:")

    for model_name, checkpoint_path in CHECKPOINTS.items():

        print(
            f"  {model_name:>3s}: "
            f"{checkpoint_path}"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for "
                f"train size {model_name}: "
                f"{checkpoint_path}"
            )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    dataset = PlateSegmentationDataset(
        split="val",
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

    print()
    print(
        f"Validation samples: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Store image + GT once.
    #
    # All four models use exactly the same validation
    # preprocessing, so these can be shared.
    # --------------------------------------------------------

    sample_visuals = {}

    for batch in loader:

        sample_id = batch["sample_id"][0]

        image_np = denormalize_image(
            batch["image"][0]
        )

        gt_mask_np = (
            batch["mask"][0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        sample_visuals[sample_id] = {
            "image": image_np,
            "gt_mask": gt_mask_np,
        }

    # --------------------------------------------------------
    # Evaluate all models
    # --------------------------------------------------------

    all_model_results = {}

    checkpoint_metadata = {}

    for model_name, checkpoint_path in (
        CHECKPOINTS.items()
    ):

        model, checkpoint = load_model(
            checkpoint_path,
            device,
        )

        print()
        print(
            f"Loaded {model_name} | "
            f"phase={checkpoint.get('phase')} | "
            f"epoch={checkpoint.get('epoch')} | "
            f"val_iou={checkpoint.get('val_iou')} | "
            f"train_size={checkpoint.get('train_size')}"
        )

        results = evaluate_model(
            model_name=model_name,
            model=model,
            loader=loader,
            device=device,
        )

        all_model_results[
            model_name
        ] = results

        checkpoint_metadata[
            model_name
        ] = checkpoint

        # Release model before loading the next one.
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Build comparison DataFrame
    # --------------------------------------------------------

    rows = []

    for sample_id in sample_visuals.keys():

        row = {
            "sample_id": sample_id,
        }

        for model_name in CHECKPOINTS.keys():

            metrics = (
                all_model_results[
                    model_name
                ][sample_id]["metrics"]
            )

            row[
                f"iou_{model_name}"
            ] = metrics["iou"]

            row[
                f"dice_{model_name}"
            ] = metrics["dice"]

            row[
                f"precision_{model_name}"
            ] = metrics["precision"]

            row[
                f"recall_{model_name}"
            ] = metrics["recall"]

        # ----------------------------------------------------
        # IoU changes along the scaling curve
        # ----------------------------------------------------

        row["delta_175_to_250"] = (
            row["iou_250"]
            - row["iou_175"]
        )

        row["delta_250_to_350"] = (
            row["iou_350"]
            - row["iou_250"]
        )

        row["delta_350_to_424"] = (
            row["iou_424"]
            - row["iou_350"]
        )

        row["delta_175_to_424"] = (
            row["iou_424"]
            - row["iou_175"]
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Rank according to the 175-sample model ONLY.
    # --------------------------------------------------------

    df = df.sort_values(
        "iou_175",
        ascending=True,
    ).reset_index(drop=True)

    df["rank_by_175"] = (
        np.arange(len(df)) + 1
    )

    # Put useful columns near the front.
    front_columns = [
        "rank_by_175",
        "sample_id",
        "iou_175",
        "iou_250",
        "iou_350",
        "iou_424",
        "delta_175_to_250",
        "delta_250_to_350",
        "delta_350_to_424",
        "delta_175_to_424",
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
    # Save complete comparison CSV
    # --------------------------------------------------------

    comparison_path = (
        OUTPUT_DIR
        / "all_samples_comparison.csv"
    )

    df.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # Worst K according to 175 model
    # --------------------------------------------------------

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
        / "worst_10_by_175.csv"
    )

    worst_df.to_csv(
        worst_path,
        index=False,
    )

    # --------------------------------------------------------
    # Create visual panels
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        f"WORST {worst_k} SAMPLES "
        f"ACCORDING TO TRAIN-175 MODEL"
    )
    print("=" * 72)

    for rank, (_, row) in enumerate(
        worst_df.iterrows(),
        start=1,
    ):

        sample_id = row["sample_id"]

        visuals = sample_visuals[
            sample_id
        ]

        filename = (
            f"{rank:02d}_"
            f"{sample_id}.png"
        )

        save_comparison_panel(
            image=visuals["image"],
            gt_mask=visuals["gt_mask"],
            sample_id=sample_id,
            rank=rank,
            model_results=all_model_results,
            output_path=(
                PANELS_DIR
                / filename
            ),
        )

        print(
            f"#{rank:02d} "
            f"{sample_id:28s} | "
            f"175={row['iou_175']:.4f} | "
            f"250={row['iou_250']:.4f} | "
            f"350={row['iou_350']:.4f} | "
            f"424={row['iou_424']:.4f} | "
            f"Δ={row['delta_175_to_424']:+.4f}"
        )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    summary_lines = [
        "V3 DATA SCALING COMPARISON",
        "=" * 72,
        "",
        f"N_validation={len(df)}",
        f"threshold={THRESHOLD}",
        (
            f"resolution="
            f"{INPUT_HEIGHT}x{INPUT_WIDTH}"
        ),
        "",
        "ALL VALIDATION SAMPLES",
        "-" * 72,
    ]

    for model_name in CHECKPOINTS.keys():

        mean_iou = (
            df[f"iou_{model_name}"]
            .mean()
        )

        median_iou = (
            df[f"iou_{model_name}"]
            .median()
        )

        summary_lines.append(
            f"Train {model_name:>3s}: "
            f"mean IoU={mean_iou:.6f} | "
            f"median IoU={median_iou:.6f}"
        )

    # --------------------------------------------------------
    # Same worst-10 samples evaluated by every model
    # --------------------------------------------------------

    summary_lines.extend([
        "",
        (
            f"SAME WORST {worst_k} SAMPLES "
            f"FROM TRAIN-175 MODEL"
        ),
        "-" * 72,
    ])

    for model_name in CHECKPOINTS.keys():

        mean_iou = (
            worst_df[f"iou_{model_name}"]
            .mean()
        )

        median_iou = (
            worst_df[f"iou_{model_name}"]
            .median()
        )

        summary_lines.append(
            f"Train {model_name:>3s}: "
            f"mean IoU={mean_iou:.6f} | "
            f"median IoU={median_iou:.6f}"
        )

    # --------------------------------------------------------
    # Improvement statistics
    # --------------------------------------------------------

    delta = df["delta_175_to_424"]

    improved_count = int(
        (delta > 0).sum()
    )

    unchanged_count = int(
        np.isclose(
            delta,
            0.0,
            atol=1e-8,
        ).sum()
    )

    degraded_count = (
        len(df)
        - improved_count
        - unchanged_count
    )

    summary_lines.extend([
        "",
        "175 -> 424 SAMPLE-LEVEL CHANGE",
        "-" * 72,
        (
            f"mean_delta="
            f"{delta.mean():+.6f}"
        ),
        (
            f"median_delta="
            f"{delta.median():+.6f}"
        ),
        (
            f"improved_samples="
            f"{improved_count}/{len(df)}"
        ),
        (
            f"degraded_samples="
            f"{degraded_count}/{len(df)}"
        ),
        (
            f"unchanged_samples="
            f"{unchanged_count}/{len(df)}"
        ),
    ])

    # --------------------------------------------------------
    # Biggest improvements / degradations
    # --------------------------------------------------------

    biggest_improvements = (
        df.sort_values(
            "delta_175_to_424",
            ascending=False,
        )
        .head(10)
    )

    biggest_degradations = (
        df.sort_values(
            "delta_175_to_424",
            ascending=True,
        )
        .head(10)
    )

    summary_lines.extend([
        "",
        "BIGGEST 175 -> 424 IMPROVEMENTS",
        "-" * 72,
    ])

    for _, row in (
        biggest_improvements.iterrows()
    ):

        summary_lines.append(
            f"{row['sample_id']:28s} | "
            f"175={row['iou_175']:.4f} | "
            f"424={row['iou_424']:.4f} | "
            f"delta="
            f"{row['delta_175_to_424']:+.4f}"
        )

    summary_lines.extend([
        "",
        "BIGGEST 175 -> 424 DEGRADATIONS",
        "-" * 72,
    ])

    for _, row in (
        biggest_degradations.iterrows()
    ):

        summary_lines.append(
            f"{row['sample_id']:28s} | "
            f"175={row['iou_175']:.4f} | "
            f"424={row['iou_424']:.4f} | "
            f"delta="
            f"{row['delta_175_to_424']:+.4f}"
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

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print(summary_text)

    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)

    print(
        f"Comparison CSV: {comparison_path}"
    )

    print(
        f"Worst-10 CSV:   {worst_path}"
    )

    print(
        f"Summary:        {summary_path}"
    )

    print(
        f"Panels:         {PANELS_DIR}"
    )


if __name__ == "__main__":
    main()