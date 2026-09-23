from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

from training.dataset import IMAGENET_MEAN, IMAGENET_STD, letterbox
from training.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "resolution_high_v3"
    / "best_model.pt"
)

INPUT_WIDTH = 768
INPUT_HEIGHT = 256
THRESHOLD = 0.5


def preprocess_image(
    image_bgr: np.ndarray,
) -> tuple[torch.Tensor, dict]:

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_letterboxed, metadata = letterbox(
        image_rgb,
        target_width=INPUT_WIDTH,
        target_height=INPUT_HEIGHT,
        interpolation=cv2.INTER_LINEAR,
        pad_value=(0, 0, 0),
    )

    image = image_letterboxed.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    image = np.transpose(image, (2, 0, 1))

    tensor = torch.from_numpy(image).float().unsqueeze(0)

    return tensor, metadata


def restore_mask(
    mask: np.ndarray,
    metadata: dict,
) -> np.ndarray:

    top = metadata["top"]
    left = metadata["left"]
    resized_height = metadata["resized_height"]
    resized_width = metadata["resized_width"]

    mask = mask[
        top:top + resized_height,
        left:left + resized_width,
    ]

    mask = cv2.resize(
        mask,
        (
            metadata["original_width"],
            metadata["original_height"],
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:

    model = build_model(
        architecture="unet",
        encoder_name="resnet34",
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

def save_visualization_panel(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # Create segmentation overlay.
    overlay = image_rgb.copy()
    mask_bool = mask.astype(bool)

    highlight = np.zeros_like(overlay)
    highlight[:, :, 0] = 255

    alpha = 0.30

    overlay[mask_bool] = (
        (1.0 - alpha) * overlay[mask_bool]
        + alpha * highlight[mask_bool]
    ).astype(np.uint8)

    # Draw predicted boundary.
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (255, 0, 0),
        1,
    )

    # Create compact README-friendly panel.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Input", fontsize=14)

    axes[1].imshow(
        mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Predicted Mask", fontsize=14)

    axes[2].imshow(overlay)
    axes[2].set_title("Segmentation Overlay", fontsize=14)

    for ax in axes:
        ax.axis("off")

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.90,
        bottom=0.02,
        wspace=0.03,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run license plate segmentation inference."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a cropped license plate image.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="prediction_mask.png",
        help="Output path for the predicted mask.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")

    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    image_tensor, metadata = preprocess_image(image_bgr)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)

    mask = (
        probabilities[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    mask = (mask >= THRESHOLD).astype(np.uint8)

    mask = restore_mask(
        mask,
        metadata,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        mask * 255,
    )

    panel_path = output_path.with_name(
        f"{output_path.stem}_panel.png"
    )

    save_visualization_panel(
        image_bgr=image_bgr,
        mask=mask,
        output_path=panel_path,
    )

    print(f"Saved mask to: {output_path}")
    print(f"Saved visualization to: {panel_path}")


if __name__ == "__main__":
    main()