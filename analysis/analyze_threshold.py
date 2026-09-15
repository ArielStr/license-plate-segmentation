from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from training.dataset import PlateSegmentationDataset
from training.model import build_model


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "threshold_analysis"

BATCH_SIZE = 1
NUM_WORKERS = 0

THRESHOLDS = np.arange(0.10, 0.91, 0.05)


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> float:
    pred = pred.bool()
    target = target.bool()

    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()

    iou = (intersection + eps) / (union + eps)

    return iou.item()


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print()

    # IMPORTANT:
    # Threshold tuning is performed on VALIDATION only.
    dataset = PlateSegmentationDataset(split="val")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
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

    print(f"Validation samples: {len(dataset)}")
    print()

    # ---------------------------------------------------------
    # Forward pass ONCE.
    #
    # Store probability maps and GT masks so that trying another
    # threshold does not require another model inference.
    # ---------------------------------------------------------

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_targets.append(masks.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    print("Inference completed.")
    print(f"Probability tensor: {tuple(all_probs.shape)}")
    print()

    # ---------------------------------------------------------
    # Threshold sweep
    # ---------------------------------------------------------

    rows = []

    for threshold in THRESHOLDS:
        sample_ious = []

        preds = all_probs >= threshold

        for i in range(len(dataset)):
            iou = compute_iou(
                preds[i],
                all_targets[i] >= 0.5,
            )
            sample_ious.append(iou)

        mean_iou = float(np.mean(sample_ious))
        median_iou = float(np.median(sample_ious))

        rows.append(
            {
                "threshold": float(threshold),
                "mean_iou": mean_iou,
                "median_iou": median_iou,
            }
        )

        print(
            f"Threshold={threshold:.2f} | "
            f"Mean IoU={mean_iou:.4f} | "
            f"Median IoU={median_iou:.4f}"
        )

    results = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Find best threshold according to MEAN VALIDATION IoU
    # ---------------------------------------------------------

    best_idx = results["mean_iou"].idxmax()
    best_row = results.loc[best_idx]

    best_threshold = float(best_row["threshold"])
    best_mean_iou = float(best_row["mean_iou"])

    # Find the baseline t=0.50 result for comparison.
    baseline_row = results.loc[
        np.isclose(results["threshold"], 0.50)
    ].iloc[0]

    baseline_iou = float(baseline_row["mean_iou"])
    improvement = best_mean_iou - baseline_iou

    print()
    print("=" * 72)
    print("THRESHOLD ANALYSIS")
    print("=" * 72)
    print(f"Baseline threshold: 0.50")
    print(f"Baseline Mean IoU: {baseline_iou:.4f}")
    print()
    print(f"Best threshold:     {best_threshold:.2f}")
    print(f"Best Mean IoU:      {best_mean_iou:.4f}")
    print(f"Improvement:        {improvement:+.4f}")
    print("=" * 72)

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    csv_path = OUTPUT_DIR / "threshold_results.csv"
    results.to_csv(csv_path, index=False)

    # ---------------------------------------------------------
    # Plot Threshold -> Mean Validation IoU
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        results["threshold"],
        results["mean_iou"],
        marker="o",
    )

    ax.axvline(
        0.50,
        linestyle="--",
        label="Baseline threshold = 0.50",
    )

    ax.scatter(
        [best_threshold],
        [best_mean_iou],
        s=100,
        zorder=3,
        label=f"Best = {best_threshold:.2f}",
    )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean Validation IoU")
    ax.set_title("Threshold Sweep on Validation Set")

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    plot_path = OUTPUT_DIR / "threshold_sweep.png"
    fig.savefig(
        plot_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    summary_path = OUTPUT_DIR / "threshold_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Validation samples: {len(dataset)}\n")
        f.write(f"Baseline threshold: 0.50\n")
        f.write(f"Baseline Mean IoU: {baseline_iou:.6f}\n")
        f.write(f"Best threshold: {best_threshold:.2f}\n")
        f.write(f"Best Mean IoU: {best_mean_iou:.6f}\n")
        f.write(f"Improvement: {improvement:+.6f}\n")

    print()
    print(f"Saved results to: {csv_path}")
    print(f"Saved plot to:    {plot_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()