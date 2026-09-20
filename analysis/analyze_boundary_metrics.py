from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from training.dataset import PlateSegmentationDataset
from training.model import build_model


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "baseline_v2"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "boundary_analysis"
    / "baseline_v2"
)

SPLIT = "val"

INPUT_WIDTH = 384
INPUT_HEIGHT = 128

THRESHOLD = 0.5

TOLERANCES = [0, 1, 2, 3]


# ============================================================
# Standard segmentation metrics
# ============================================================

def compute_iou(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return 1.0

    return intersection / union


# ============================================================
# Boundary extraction
# ============================================================

def extract_boundary(mask):
    """
    Extract a 1-pixel-wide internal boundary from a binary mask.
    """

    mask = mask.astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    eroded = cv2.erode(
        mask,
        kernel,
        iterations=1,
    )

    boundary = mask - eroded

    return boundary.astype(bool)


# ============================================================
# Boundary precision / recall / F1
# ============================================================

def dilate_boundary(boundary, tolerance):
    """
    Dilate a boundary by `tolerance` pixels in every direction.
    """

    if tolerance == 0:
        return boundary.copy()

    kernel_size = 2 * tolerance + 1

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    dilated = cv2.dilate(
        boundary.astype(np.uint8),
        kernel,
        iterations=1,
    )

    return dilated.astype(bool)


def compute_boundary_metrics(
    gt,
    pred,
    tolerance,
):
    """
    Boundary Precision:
        fraction of predicted boundary pixels that lie within
        `tolerance` pixels of the GT boundary.

    Boundary Recall:
        fraction of GT boundary pixels that lie within
        `tolerance` pixels of the predicted boundary.

    Boundary F1:
        harmonic mean of boundary precision and recall.
    """

    gt_boundary = extract_boundary(gt)
    pred_boundary = extract_boundary(pred)

    gt_count = gt_boundary.sum()
    pred_count = pred_boundary.sum()

    if gt_count == 0 and pred_count == 0:
        return 1.0, 1.0, 1.0

    if gt_count == 0 or pred_count == 0:
        return 0.0, 0.0, 0.0

    gt_dilated = dilate_boundary(
        gt_boundary,
        tolerance,
    )

    pred_dilated = dilate_boundary(
        pred_boundary,
        tolerance,
    )

    # Predicted boundary matched by GT
    matched_pred = np.logical_and(
        pred_boundary,
        gt_dilated,
    ).sum()

    # GT boundary matched by prediction
    matched_gt = np.logical_and(
        gt_boundary,
        pred_dilated,
    ).sum()

    precision = matched_pred / pred_count
    recall = matched_gt / gt_count

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )

    return precision, recall, f1


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("BOUNDARY METRIC ANALYSIS")
    print("=" * 80)

    print(f"Device:      {device}")
    print(f"Checkpoint:  {CHECKPOINT_PATH}")
    print(f"Split:       {SPLIT}")
    print(
        f"Resolution:  "
        f"{INPUT_HEIGHT}x{INPUT_WIDTH}"
    )
    print(f"Threshold:   {THRESHOLD}")
    print(f"Tolerances:  {TOLERANCES}")
    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = PlateSegmentationDataset(
        split=SPLIT,
        target_width=INPUT_WIDTH,
        target_height=INPUT_HEIGHT,
        augment=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
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
    print("Analyzing...")
    print()

    results = []

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)
            masks = batch["mask"]
            sample_id = batch["sample_id"][0]

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            )

            gt = (
                masks[0, 0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            pred = (
                predictions[0, 0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            iou = compute_iou(
                gt,
                pred,
            )

            row = {
                "sample_id": sample_id,
                "iou": iou,
            }

            f1_strings = []

            for tolerance in TOLERANCES:

                precision, recall, f1 = (
                    compute_boundary_metrics(
                        gt,
                        pred,
                        tolerance,
                    )
                )

                row[
                    f"boundary_precision_{tolerance}px"
                ] = precision

                row[
                    f"boundary_recall_{tolerance}px"
                ] = recall

                row[
                    f"boundary_f1_{tolerance}px"
                ] = f1

                f1_strings.append(
                    f"{tolerance}px={f1:.4f}"
                )

            results.append(row)

            print(
                f"{sample_id:<30} "
                f"IoU={iou:.4f} | "
                + " | ".join(f1_strings)
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    csv_path = (
        OUTPUT_DIR
        / "boundary_metrics.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"N: {len(df)}")
    print(
        f"Mean IoU:   "
        f"{df['iou'].mean():.4f}"
    )
    print(
        f"Median IoU: "
        f"{df['iou'].median():.4f}"
    )

    print()

    for tolerance in TOLERANCES:

        column = (
            f"boundary_f1_{tolerance}px"
        )

        print(
            f"Boundary F1 @ {tolerance}px | "
            f"mean={df[column].mean():.4f} | "
            f"median={df[column].median():.4f}"
        )

    # --------------------------------------------------------
    # Worst IoU samples
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("WORST 10 BY IoU")
    print("=" * 80)

    worst = df.nsmallest(
        10,
        "iou",
    )

    for _, row in worst.iterrows():

        print(
            f"{row['sample_id']:<30} "
            f"IoU={row['iou']:.4f} | "
            f"BF1@0={row['boundary_f1_0px']:.4f} | "
            f"BF1@1={row['boundary_f1_1px']:.4f} | "
            f"BF1@2={row['boundary_f1_2px']:.4f} | "
            f"BF1@3={row['boundary_f1_3px']:.4f}"
        )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(csv_path)


if __name__ == "__main__":
    main()