from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from training.dataset import (
    PlateSegmentationDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
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
    / "prediction_visualization"
    / "baseline_v2"
)

SPLIT = "val"

INPUT_WIDTH = 384
INPUT_HEIGHT = 128

THRESHOLD = 0.5


# ============================================================
# Helpers
# ============================================================

def denormalize_image(image_tensor):
    """
    Convert normalized CHW tensor back to RGB uint8.
    """

    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    image = image * std + mean
    image = np.clip(image, 0.0, 1.0)

    return (image * 255).astype(np.uint8)


def create_mask_overlay(
    image_rgb,
    mask,
    color,
    alpha=0.45,
):
    """
    Overlay a binary mask on an RGB image.
    """

    result = image_rgb.copy().astype(np.float32)

    mask_bool = mask.astype(bool)

    color_array = np.array(
        color,
        dtype=np.float32,
    )

    result[mask_bool] = (
        (1.0 - alpha) * result[mask_bool]
        + alpha * color_array
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def create_error_overlay(
    image_rgb,
    gt,
    pred,
    alpha=0.65,
):
    """
    TP = green
    FP = red
    FN = blue
    """

    result = image_rgb.copy().astype(np.float32)

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)

    colors = {
        "tp": np.array([0, 255, 0], dtype=np.float32),
        "fp": np.array([255, 0, 0], dtype=np.float32),
        "fn": np.array([0, 0, 255], dtype=np.float32),
    }

    for region, color in [
        (tp, colors["tp"]),
        (fp, colors["fp"]),
        (fn, colors["fn"]),
    ]:
        result[region] = (
            (1.0 - alpha) * result[region]
            + alpha * color
        )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def compute_metrics(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    union = tp + fp + fn

    iou = tp / union if union > 0 else 1.0

    dice_denominator = (
        2 * tp + fp + fn
    )

    dice = (
        2 * tp / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 1.0
    )

    return iou, dice, precision, recall


# ============================================================
# Visualization
# ============================================================

def save_visualization(
    image_rgb,
    gt,
    pred,
    probability,
    sample_id,
    output_path,
):
    iou, dice, precision, recall = compute_metrics(
        gt,
        pred,
    )

    gt_overlay = create_mask_overlay(
        image_rgb,
        gt,
        color=(0, 255, 0),
    )

    pred_overlay = create_mask_overlay(
        image_rgb,
        pred,
        color=(255, 0, 0),
    )

    error_overlay = create_error_overlay(
        image_rgb,
        gt,
        pred,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 8),
    )

    # --------------------------------------------------------
    # Row 1
    # --------------------------------------------------------

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Input")

    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title("Ground Truth overlay")

    axes[0, 2].imshow(pred_overlay)
    axes[0, 2].set_title("Prediction overlay")

    # --------------------------------------------------------
    # Row 2
    # --------------------------------------------------------

    axes[1, 0].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title("Ground Truth mask")

    axes[1, 1].imshow(
        pred,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_title("Prediction mask")

    axes[1, 2].imshow(error_overlay)
    axes[1, 2].set_title(
        "Error overlay\n"
        "Green=TP | Red=FP | Blue=FN"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"{sample_id}\n"
        f"IoU={iou:.4f} | "
        f"Dice={dice:.4f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f}",
        fontsize=14,
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


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

    print("=" * 72)
    print("PREDICTION VISUALIZATION")
    print("=" * 72)

    print(f"Device:     {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Split:      {SPLIT}")
    print(
        f"Resolution: "
        f"{INPUT_HEIGHT}x{INPUT_WIDTH}"
    )
    print(f"Threshold:  {THRESHOLD}")
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
    print("Generating visualizations...")
    print()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():
        for index, batch in enumerate(loader):
            images = batch["image"]
            masks = batch["mask"]
            sample_ids = batch["sample_id"]

            sample_id = sample_ids[0]

            images = images.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            # Remove batch/channel dimensions
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

            probability = (
                probabilities[0, 0]
                .cpu()
                .numpy()
            )

            image_rgb = denormalize_image(
                images[0]
            )

            output_path = (
                OUTPUT_DIR
                / f"{sample_id}.png"
            )

            save_visualization(
                image_rgb=image_rgb,
                gt=gt,
                pred=pred,
                probability=probability,
                sample_id=sample_id,
                output_path=output_path,
            )

            iou, _, _, _ = compute_metrics(
                gt,
                pred,
            )

            print(
                f"{sample_id:<30} "
                f"IoU={iou:.4f}"
            )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()