from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import IMAGENET_MEAN, IMAGENET_STD, PlateSegmentationDataset
from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "evaluation"

BATCH_SIZE = 1
NUM_WORKERS = 0
THRESHOLD = 0.5


def compute_iou_and_dice(pred, target, eps=1e-7):
    pred = pred.bool()
    target = target.bool()

    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()

    iou = (intersection + eps) / (union + eps)

    total = pred.sum().float() + target.sum().float()
    dice = (2.0 * intersection + eps) / (total + eps)

    return iou.item(), dice.item()


def denormalize_image(image_tensor):
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def save_prediction_panel(
    image,
    gt_mask,
    pred_mask,
    sample_id,
    iou,
    dice,
    output_path,
):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image)
    axes[0].set_title("Input")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")

    overlay_uint8 = (image * 255).astype(np.uint8).copy()
    contours, _ = cv2.findContours(
        pred_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        overlay_uint8,
        contours,
        -1,
        (255, 0, 0),
        2,
    )

    axes[3].imshow(overlay_uint8)
    axes[3].set_title("Prediction Overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{sample_id} | IoU={iou:.4f} | Dice={dice:.4f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panels_dir = OUTPUT_DIR / "test_predictions"
    panels_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Threshold: {args.threshold}")

    dataset = PlateSegmentationDataset(split="test")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        "Loaded checkpoint | "
        f"phase={checkpoint.get('phase')} | "
        f"epoch={checkpoint.get('epoch')} | "
        f"val_iou={checkpoint.get('val_iou')}"
    )

    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            sample_id = batch["sample_id"][0]

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = probs >= args.threshold

            iou, dice = compute_iou_and_dice(
                preds[0],
                masks[0] >= 0.5,
            )

            rows.append({
                "sample_id": sample_id,
                "iou": iou,
                "dice": dice,
            })

            image_np = denormalize_image(images[0])
            gt_mask_np = masks[0, 0].detach().cpu().numpy()
            pred_mask_np = (
                preds[0, 0].detach().cpu().numpy().astype(np.uint8)
            )

            save_prediction_panel(
                image_np,
                gt_mask_np,
                pred_mask_np,
                sample_id,
                iou,
                dice,
                panels_dir / f"{sample_id}.png",
            )

            print(
                f"{sample_id:24s} "
                f"IoU={iou:.4f} "
                f"Dice={dice:.4f}"
            )

    df = pd.DataFrame(rows)
    df = df.sort_values("iou").reset_index(drop=True)

    mean_iou = df["iou"].mean()
    mean_dice = df["dice"].mean()
    median_iou = df["iou"].median()
    median_dice = df["dice"].median()

    print()
    print("=" * 72)
    print("TEST RESULTS")
    print("=" * 72)
    print(f"N:           {len(df)}")
    print(f"Mean IoU:    {mean_iou:.4f}")
    print(f"Median IoU:  {median_iou:.4f}")
    print(f"Mean Dice:   {mean_dice:.4f}")
    print(f"Median Dice: {median_dice:.4f}")
    print()
    print("Worst 5 samples:")
    print(df.head(5).to_string(index=False))
    print("=" * 72)

    metrics_path = OUTPUT_DIR / "test_metrics.csv"
    df.to_csv(metrics_path, index=False)

    summary_path = OUTPUT_DIR / "test_summary.txt"
    summary_path.write_text(
        "\n".join([
            f"N={len(df)}",
            f"mean_iou={mean_iou:.6f}",
            f"median_iou={median_iou:.6f}",
            f"mean_dice={mean_dice:.6f}",
            f"median_dice={median_dice:.6f}",
        ]),
        encoding="utf-8",
    )

    print()
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved prediction panels to: {panels_dir}")


if __name__ == "__main__":
    main()
