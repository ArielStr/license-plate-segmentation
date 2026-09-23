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

from training.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PlateSegmentationDataset,
)
from training.model import build_model


# ============================================================
# Configuration
# ============================================================

MODELS = {
    "baseline": {
        "checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "baseline_v3"
            / "best_model.pt"
        ),
        "width": 384,
        "height": 128,
        "label": "Baseline\n128x384",
    },

    "highres": {
        "checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "resolution_high_v3"
            / "best_model.pt"
        ),
        "width": 768,
        "height": 256,
        "label": "High Resolution\n256x768",
    },

    "highres_aug": {
        "checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "highres_augmentation_v3"
            / "best_model.pt"
        ),
        "width": 768,
        "height": 256,
        "label": "HighRes + Aug\n256x768",
    },
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "final_model_comparison"
)

PANELS_DIR = OUTPUT_DIR / "worst_10_panels"

THRESHOLD = 0.5
WORST_K = 10

BATCH_SIZE = 1
NUM_WORKERS = 0


# ============================================================
# Metrics
# ============================================================

def compute_binary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-7,
) -> dict:

    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(
        pred,
        target,
    ).sum()

    fp = np.logical_and(
        pred,
        np.logical_not(target),
    ).sum()

    fn = np.logical_and(
        np.logical_not(pred),
        target,
    ).sum()

    pred_area = pred.sum()
    gt_area = target.sum()

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


def undo_letterbox(
    array: np.ndarray,
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
    interpolation: int,
) -> np.ndarray:
    """
    Convert an array from the letterboxed network canvas back
    to the coordinate system of the original crop.

    Assumes the same centered letterbox rule used by Dataset:
        scale = min(target_width / original_width,
                    target_height / original_height)
    """

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    resized_width = max(
        1,
        int(round(original_width * scale)),
    )

    resized_height = max(
        1,
        int(round(original_height * scale)),
    )

    pad_x = (
        target_width - resized_width
    ) // 2

    pad_y = (
        target_height - resized_height
    ) // 2

    cropped = array[
        pad_y:pad_y + resized_height,
        pad_x:pad_x + resized_width,
    ]

    restored = cv2.resize(
        cropped,
        (
            original_width,
            original_height,
        ),
        interpolation=interpolation,
    )

    return restored


# ============================================================
# Model loading
# ============================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

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
# Evaluate one model
# ============================================================

def evaluate_model(
    model_name: str,
    model_config: dict,
    device: torch.device,
) -> dict:

    width = model_config["width"]
    height = model_config["height"]

    dataset = PlateSegmentationDataset(
        split="val",
        target_width=width,
        target_height=height,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model, checkpoint = load_model(
        model_config["checkpoint"],
        device,
    )

    print()
    print("=" * 72)
    print(f"EVALUATING: {model_name}")
    print("=" * 72)

    print(
        f"Resolution: {height}x{width}"
    )

    print(
        "Checkpoint | "
        f"phase={checkpoint.get('phase')} | "
        f"epoch={checkpoint.get('epoch')} | "
        f"val_iou={checkpoint.get('val_iou')}"
    )

    results = {}

    with torch.no_grad():

        for index, batch in enumerate(loader):

            images = batch["image"].to(device)

            sample_id = batch["sample_id"][0]

            # ------------------------------------------------
            # Get original crop dimensions from split metadata.
            # ------------------------------------------------

            dataset_row = dataset.df.iloc[index]

            crop_rel = str(
                dataset_row["crop_path"]
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
                    f"Could not read original crop: "
                    f"{crop_path}"
                )

            original_height, original_width = (
                original_bgr.shape[:2]
            )

            original_rgb = cv2.cvtColor(
                original_bgr,
                cv2.COLOR_BGR2RGB,
            )

            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            probability_canvas = (
                probabilities[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Undo model-specific letterbox.
            #
            # Important:
            # Restore the probability map first, then apply
            # threshold in original crop coordinates.
            # ------------------------------------------------

            probability_original = undo_letterbox(
                probability_canvas,
                original_width=original_width,
                original_height=original_height,
                target_width=width,
                target_height=height,
                interpolation=cv2.INTER_LINEAR,
            )

            prediction_original = (
                probability_original >= THRESHOLD
            ).astype(np.uint8)

            # ------------------------------------------------
            # Load original GT mask directly from disk.
            #
            # This avoids comparing against a GT mask that has
            # itself gone through a model-specific resize.
            # ------------------------------------------------

            mask_rel = str(
                dataset_row["mask_path"]
            ).replace("\\", "/")

            mask_path = (
                PROJECT_ROOT
                / Path(mask_rel)
            )

            gt_original = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if gt_original is None:
                raise RuntimeError(
                    f"Could not read GT mask: "
                    f"{mask_path}"
                )

            # Safety in case crop/mask dimensions differ.
            if (
                gt_original.shape[1] != original_width
                or
                gt_original.shape[0] != original_height
            ):
                gt_original = cv2.resize(
                    gt_original,
                    (
                        original_width,
                        original_height,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

            gt_original = (
                gt_original >= 128
            ).astype(np.uint8)

            metrics = compute_binary_metrics(
                prediction_original,
                gt_original,
            )

            results[sample_id] = {
                "metrics": metrics,
                "prediction": prediction_original,
                "probability": probability_original,
                "gt": gt_original,
                "image": original_rgb,
            }

            print(
                f"{sample_id:28s} "
                f"IoU={metrics['iou']:.4f}"
            )

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ============================================================
# Visualization
# ============================================================

def save_comparison_panel(
    sample_id: str,
    rank: int,
    all_results: dict,
    output_path: Path,
) -> None:

    baseline_result = (
        all_results["baseline"][sample_id]
    )

    image = baseline_result["image"]
    gt = baseline_result["gt"]

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    # Input
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    # Ground Truth
    axes[1].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    model_order = [
        "baseline",
        "highres",
        "highres_aug",
    ]

    for column, model_name in enumerate(
        model_order,
        start=2,
    ):

        result = (
            all_results[
                model_name
            ][sample_id]
        )

        metrics = result["metrics"]

        axes[column].imshow(
            result["prediction"],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[column].set_title(
            f"{MODELS[model_name]['label']}\n"
            f"IoU={metrics['iou']:.4f}"
        )

        axes[column].axis("off")

    baseline_iou = (
        all_results["baseline"]
        [sample_id]["metrics"]["iou"]
    )

    highres_iou = (
        all_results["highres"]
        [sample_id]["metrics"]["iou"]
    )

    highres_aug_iou = (
        all_results["highres_aug"]
        [sample_id]["metrics"]["iou"]
    )

    fig.suptitle(
        f"#{rank:02d} {sample_id} | "
        f"Baseline={baseline_iou:.4f} | "
        f"HighRes={highres_iou:.4f} | "
        f"HighRes+Aug={highres_aug_iou:.4f}",
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("V3 FINAL MODEL COMPARISON")
    print("=" * 72)

    print(f"Device:    {device}")
    print("Split:     val")
    print(f"Threshold: {THRESHOLD}")

    # --------------------------------------------------------
    # Verify checkpoints
    # --------------------------------------------------------

    print()
    print("Checkpoints:")

    for model_name, config in MODELS.items():

        checkpoint_path = config["checkpoint"]

        print(
            f"  {model_name:12s}: "
            f"{checkpoint_path}"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: "
                f"{checkpoint_path}"
            )

    # --------------------------------------------------------
    # Evaluate all three models
    # --------------------------------------------------------

    all_results = {}

    for model_name, config in MODELS.items():

        all_results[model_name] = (
            evaluate_model(
                model_name=model_name,
                model_config=config,
                device=device,
            )
        )

    # --------------------------------------------------------
    # Make sure all models evaluated exactly the same samples.
    # --------------------------------------------------------

    baseline_ids = set(
        all_results["baseline"].keys()
    )

    for model_name in [
        "highres",
        "highres_aug",
    ]:

        model_ids = set(
            all_results[model_name].keys()
        )

        if model_ids != baseline_ids:
            raise RuntimeError(
                f"Sample mismatch between baseline "
                f"and {model_name}."
            )

    # --------------------------------------------------------
    # Build comparison table
    # --------------------------------------------------------

    rows = []

    for sample_id in sorted(baseline_ids):

        baseline_metrics = (
            all_results["baseline"]
            [sample_id]["metrics"]
        )

        highres_metrics = (
            all_results["highres"]
            [sample_id]["metrics"]
        )

        aug_metrics = (
            all_results["highres_aug"]
            [sample_id]["metrics"]
        )

        row = {
            "sample_id": sample_id,

            "iou_baseline":
                baseline_metrics["iou"],

            "iou_highres":
                highres_metrics["iou"],

            "iou_highres_aug":
                aug_metrics["iou"],

            "dice_baseline":
                baseline_metrics["dice"],

            "dice_highres":
                highres_metrics["dice"],

            "dice_highres_aug":
                aug_metrics["dice"],

            "precision_baseline":
                baseline_metrics["precision"],

            "precision_highres":
                highres_metrics["precision"],

            "precision_highres_aug":
                aug_metrics["precision"],

            "recall_baseline":
                baseline_metrics["recall"],

            "recall_highres":
                highres_metrics["recall"],

            "recall_highres_aug":
                aug_metrics["recall"],
        }

        row["delta_highres_vs_baseline"] = (
            row["iou_highres"]
            - row["iou_baseline"]
        )

        row["delta_aug_vs_highres"] = (
            row["iou_highres_aug"]
            - row["iou_highres"]
        )

        row["delta_aug_vs_baseline"] = (
            row["iou_highres_aug"]
            - row["iou_baseline"]
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Rank by baseline performance.
    # --------------------------------------------------------

    df = df.sort_values(
        "iou_baseline",
        ascending=True,
    ).reset_index(drop=True)

    df["rank_by_baseline"] = (
        np.arange(len(df)) + 1
    )

    front_columns = [
        "rank_by_baseline",
        "sample_id",
        "iou_baseline",
        "iou_highres",
        "iou_highres_aug",
        "delta_highres_vs_baseline",
        "delta_aug_vs_highres",
        "delta_aug_vs_baseline",
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
    # Save all samples
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
    # Worst K baseline samples
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
        / "worst_10_by_baseline.csv"
    )

    worst_df.to_csv(
        worst_path,
        index=False,
    )

    # --------------------------------------------------------
    # Visual panels
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        f"WORST {worst_k} BASELINE SAMPLES"
    )
    print("=" * 72)

    for rank, (_, row) in enumerate(
        worst_df.iterrows(),
        start=1,
    ):

        sample_id = row["sample_id"]

        filename = (
            f"{rank:02d}_"
            f"{sample_id}.png"
        )

        save_comparison_panel(
            sample_id=sample_id,
            rank=rank,
            all_results=all_results,
            output_path=(
                PANELS_DIR
                / filename
            ),
        )

        print(
            f"#{rank:02d} "
            f"{sample_id:28s} | "
            f"B={row['iou_baseline']:.4f} | "
            f"HR={row['iou_highres']:.4f} | "
            f"HR+A={row['iou_highres_aug']:.4f} | "
            f"HR-B="
            f"{row['delta_highres_vs_baseline']:+.4f}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_lines = [
        "V3 FINAL MODEL COMPARISON",
        "=" * 72,
        "",
        f"N_validation={len(df)}",
        f"threshold={THRESHOLD}",
        "",
        (
            "NOTE: Metrics below are recomputed in the "
            "original crop coordinate system."
        ),
        (
            "They are therefore not expected to exactly match "
            "the training-time letterboxed Val IoU values."
        ),
        "",
        "ALL VALIDATION SAMPLES",
        "-" * 72,
    ]

    model_columns = {
        "Baseline 128x384":
            "iou_baseline",

        "HighRes 256x768":
            "iou_highres",

        "HighRes+Aug 256x768":
            "iou_highres_aug",
    }

    for label, column in (
        model_columns.items()
    ):

        summary_lines.append(
            f"{label:24s} | "
            f"mean IoU="
            f"{df[column].mean():.6f} | "
            f"median IoU="
            f"{df[column].median():.6f}"
        )

    # --------------------------------------------------------
    # Same worst baseline samples
    # --------------------------------------------------------

    summary_lines.extend([
        "",
        (
            f"SAME WORST {worst_k} "
            f"BASELINE SAMPLES"
        ),
        "-" * 72,
    ])

    for label, column in (
        model_columns.items()
    ):

        summary_lines.append(
            f"{label:24s} | "
            f"mean IoU="
            f"{worst_df[column].mean():.6f} | "
            f"median IoU="
            f"{worst_df[column].median():.6f}"
        )

    # --------------------------------------------------------
    # Pairwise change counts
    # --------------------------------------------------------

    hr_delta = (
        df["delta_highres_vs_baseline"]
    )

    aug_delta = (
        df["delta_aug_vs_highres"]
    )

    summary_lines.extend([
        "",
        "HIGHRES VS BASELINE",
        "-" * 72,
        (
            f"mean_delta="
            f"{hr_delta.mean():+.6f}"
        ),
        (
            f"median_delta="
            f"{hr_delta.median():+.6f}"
        ),
        (
            f"improved_samples="
            f"{int((hr_delta > 0).sum())}"
            f"/{len(df)}"
        ),
        (
            f"degraded_samples="
            f"{int((hr_delta < 0).sum())}"
            f"/{len(df)}"
        ),
        "",
        "HIGHRES+AUG VS HIGHRES",
        "-" * 72,
        (
            f"mean_delta="
            f"{aug_delta.mean():+.6f}"
        ),
        (
            f"median_delta="
            f"{aug_delta.median():+.6f}"
        ),
        (
            f"improved_samples="
            f"{int((aug_delta > 0).sum())}"
            f"/{len(df)}"
        ),
        (
            f"degraded_samples="
            f"{int((aug_delta < 0).sum())}"
            f"/{len(df)}"
        ),
    ])

    # --------------------------------------------------------
    # Biggest HighRes improvements
    # --------------------------------------------------------

    biggest_hr_improvements = (
        df.sort_values(
            "delta_highres_vs_baseline",
            ascending=False,
        )
        .head(10)
    )

    biggest_hr_degradations = (
        df.sort_values(
            "delta_highres_vs_baseline",
            ascending=True,
        )
        .head(10)
    )

    summary_lines.extend([
        "",
        "BIGGEST HIGHRES IMPROVEMENTS",
        "-" * 72,
    ])

    for _, row in (
        biggest_hr_improvements.iterrows()
    ):

        summary_lines.append(
            f"{row['sample_id']:28s} | "
            f"B={row['iou_baseline']:.4f} | "
            f"HR={row['iou_highres']:.4f} | "
            f"delta="
            f"{row['delta_highres_vs_baseline']:+.4f}"
        )

    summary_lines.extend([
        "",
        "BIGGEST HIGHRES DEGRADATIONS",
        "-" * 72,
    ])

    for _, row in (
        biggest_hr_degradations.iterrows()
    ):

        summary_lines.append(
            f"{row['sample_id']:28s} | "
            f"B={row['iou_baseline']:.4f} | "
            f"HR={row['iou_highres']:.4f} | "
            f"delta="
            f"{row['delta_highres_vs_baseline']:+.4f}"
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
    # Console
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