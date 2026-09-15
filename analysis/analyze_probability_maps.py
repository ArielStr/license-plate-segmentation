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

# Allow importing the existing training modules when this script is run
# directly from the analysis/ directory.
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from training.dataset import IMAGENET_MEAN, IMAGENET_STD, PlateSegmentationDataset
from training.model import build_model


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
METRICS_PATH = PROJECT_ROOT / "evaluation" / "test_metrics.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "probability_analysis"

THRESHOLD = 0.5
NUM_WORST = 5
NUM_BEST = 2
NUM_WORKERS = 0


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def select_samples(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Test metrics file not found: {metrics_path}\n"
            "Run training/evaluate.py first."
        )

    df = pd.read_csv(metrics_path)

    required_columns = {"sample_id", "iou", "dice"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns in {metrics_path}: {sorted(missing_columns)}"
        )

    df = df.sort_values("iou").reset_index(drop=True)

    worst = df.head(NUM_WORST).copy()
    worst["group"] = "worst"

    best = df.tail(NUM_BEST).sort_values("iou", ascending=False).copy()
    best["group"] = "best"

    selected = pd.concat([worst, best], ignore_index=True)

    if selected["sample_id"].duplicated().any():
        raise ValueError(
            "The selected best/worst groups overlap. "
            "Reduce NUM_WORST or NUM_BEST."
        )

    return selected


def save_probability_panel(
    image: np.ndarray,
    gt_mask: np.ndarray,
    probability_map: np.ndarray,
    pred_mask: np.ndarray,
    sample_id: str,
    iou: float,
    dice: float,
    group: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(image)
    axes[0].set_title("Input")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    probability_im = axes[2].imshow(
        probability_map,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    # Draw the exact decision boundary used by thresholding.
    # Suppress the contour only in the degenerate case where the whole
    # probability map lies on one side of the threshold.
    if probability_map.min() < THRESHOLD < probability_map.max():
        axes[2].contour(
            probability_map,
            levels=[THRESHOLD],
            colors="red",
            linewidths=1.2,
        )

    axes[2].set_title(f"Probability Map (red: p={THRESHOLD:.2f})")

    colorbar = fig.colorbar(
        probability_im,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("P(plate)")

    axes[3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Prediction @ {THRESHOLD:.2f}")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{sample_id} | {group.upper()} | "
        f"IoU={iou:.4f} | Dice={dice:.4f}",
        fontsize=14,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = select_samples(METRICS_PATH)
    selected_ids = set(selected["sample_id"])

    print("Selected samples:")
    print(
        selected[
            ["group", "sample_id", "iou", "dice"]
        ].to_string(index=False)
    )
    print()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Threshold: {THRESHOLD}")
    print()

    dataset = PlateSegmentationDataset(split="test")
    loader = DataLoader(
        dataset,
        batch_size=1,
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
    print()

    selected_lookup = selected.set_index("sample_id").to_dict("index")
    processed_ids = set()

    with torch.no_grad():
        for batch in loader:
            sample_id = batch["sample_id"][0]

            if sample_id not in selected_ids:
                continue

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = probs >= THRESHOLD

            image_np = denormalize_image(images[0])
            gt_mask_np = masks[0, 0].detach().cpu().numpy()
            probability_map_np = probs[0, 0].detach().cpu().numpy()
            pred_mask_np = (
                preds[0, 0].detach().cpu().numpy().astype(np.uint8)
            )

            row = selected_lookup[sample_id]
            group = row["group"]
            iou = float(row["iou"])
            dice = float(row["dice"])

            output_path = OUTPUT_DIR / f"{group}_{sample_id}.png"

            save_probability_panel(
                image=image_np,
                gt_mask=gt_mask_np,
                probability_map=probability_map_np,
                pred_mask=pred_mask_np,
                sample_id=sample_id,
                iou=iou,
                dice=dice,
                group=group,
                output_path=output_path,
            )

            processed_ids.add(sample_id)

            print(
                f"Saved {group:5s} | "
                f"{sample_id:24s} | "
                f"IoU={iou:.4f} | "
                f"{output_path.name}"
            )

    missing_ids = selected_ids - processed_ids
    if missing_ids:
        raise RuntimeError(
            "Selected samples were not found in the test dataset: "
            f"{sorted(missing_ids)}"
        )

    selection_path = OUTPUT_DIR / "selected_samples.csv"
    selected.to_csv(selection_path, index=False)

    print()
    print(f"Saved probability panels to: {OUTPUT_DIR}")
    print(f"Saved selected sample list to: {selection_path}")


if __name__ == "__main__":
    main()
