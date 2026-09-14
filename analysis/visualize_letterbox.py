from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = PROJECT_ROOT / "data" / "crops"
MASKS_DIR = PROJECT_ROOT / "data" / "masks"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "outputs" / "letterbox_examples"

TARGET_WIDTH = 384
TARGET_HEIGHT = 128

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# We intentionally pick representative examples rather than random ones.
NUM_LOW_AR = 2
NUM_MID_AR = 2
NUM_HIGH_AR = 2
NUM_SMALLEST = 2
NUM_LARGEST = 2


def find_mask_for_crop(crop_path: Path) -> Path:
    """
    Find the corresponding mask using the same stem as the crop.

    Supports masks saved with any common image extension.
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = MASKS_DIR / f"{crop_path.stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No mask found for crop {crop_path.name} in {MASKS_DIR}"
    )


def load_image(path: Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    if not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def letterbox(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int,
    pad_value: int | tuple[int, int, int],
) -> tuple[np.ndarray, dict]:
    """
    Resize while preserving aspect ratio and pad to a fixed canvas size.

    Returns:
        padded_image
        metadata with scale, resized size, and padding amounts
    """
    original_height, original_width = image.shape[:2]

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    pad_x = target_width - resized_width
    pad_y = target_height - resized_height

    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )

    metadata = {
        "scale": scale,
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }

    return padded, metadata


def make_overlay(rgb_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create a simple visible overlay without relying on a specific color map.
    """
    overlay = rgb_image.astype(np.float32).copy()

    mask_binary = mask > 127

    # Brighten pixels that belong to the foreground mask.
    overlay[mask_binary] = (
        0.45 * overlay[mask_binary]
        + 0.55 * np.array([255.0, 255.0, 255.0], dtype=np.float32)
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def collect_crop_metadata() -> list[dict]:
    rows = []

    for crop_path in sorted(CROPS_DIR.iterdir()):
        if (
            not crop_path.is_file()
            or crop_path.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue

        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"[WARNING] Could not read: {crop_path.name}")
            continue

        height, width = image.shape[:2]

        rows.append(
            {
                "path": crop_path,
                "filename": crop_path.name,
                "width": width,
                "height": height,
                "area": width * height,
                "aspect_ratio": width / height,
            }
        )

    if not rows:
        raise RuntimeError(f"No readable crops found in {CROPS_DIR}")

    return rows


def select_representative_examples(rows: list[dict]) -> list[dict]:
    """
    Pick representative crops:
      - lowest aspect ratios
      - closest to AR=3
      - highest aspect ratios
      - smallest by area
      - largest by area

    Duplicates are removed while preserving selection order.
    """
    selections = []

    selections.extend(
        sorted(rows, key=lambda x: x["aspect_ratio"])[:NUM_LOW_AR]
    )

    selections.extend(
        sorted(rows, key=lambda x: abs(x["aspect_ratio"] - 3.0))[:NUM_MID_AR]
    )

    selections.extend(
        sorted(rows, key=lambda x: x["aspect_ratio"], reverse=True)[:NUM_HIGH_AR]
    )

    selections.extend(
        sorted(rows, key=lambda x: x["area"])[:NUM_SMALLEST]
    )

    selections.extend(
        sorted(rows, key=lambda x: x["area"], reverse=True)[:NUM_LARGEST]
    )

    unique = []
    seen = set()

    for item in selections:
        stem = item["path"].stem
        if stem in seen:
            continue

        seen.add(stem)
        unique.append(item)

    return unique


def visualize_example(row: dict) -> None:
    crop_path = row["path"]
    mask_path = find_mask_for_crop(crop_path)

    crop = load_image(crop_path, grayscale=False)
    mask = load_image(mask_path, grayscale=True)

    if crop.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Shape mismatch for {crop_path.name}: "
            f"crop={crop.shape[:2]}, mask={mask.shape[:2]}"
        )

    letterboxed_crop, crop_meta = letterbox(
        crop,
        TARGET_WIDTH,
        TARGET_HEIGHT,
        interpolation=cv2.INTER_LINEAR,
        pad_value=(0, 0, 0),
    )

    letterboxed_mask, mask_meta = letterbox(
        mask,
        TARGET_WIDTH,
        TARGET_HEIGHT,
        interpolation=cv2.INTER_NEAREST,
        pad_value=0,
    )

    if (
        crop_meta["resized_width"] != mask_meta["resized_width"]
        or crop_meta["resized_height"] != mask_meta["resized_height"]
        or crop_meta["left"] != mask_meta["left"]
        or crop_meta["right"] != mask_meta["right"]
        or crop_meta["top"] != mask_meta["top"]
        or crop_meta["bottom"] != mask_meta["bottom"]
    ):
        raise RuntimeError(
            f"Crop/mask letterbox mismatch for {crop_path.name}"
        )

    overlay = make_overlay(letterboxed_crop, letterboxed_mask)

    fig = plt.figure(figsize=(15, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(crop)
    ax1.set_title(
        f"Original crop\n"
        f"{row['width']}x{row['height']} | AR={row['aspect_ratio']:.2f}"
    )
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(mask, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("Original mask")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axis("off")
    ax3.text(
        0.02,
        0.95,
        (
            f"Target: {TARGET_WIDTH}x{TARGET_HEIGHT}\n"
            f"Scale: {crop_meta['scale']:.3f}\n"
            f"Resized: "
            f"{crop_meta['resized_width']}x{crop_meta['resized_height']}\n\n"
            f"Padding:\n"
            f"left={crop_meta['left']}\n"
            f"right={crop_meta['right']}\n"
            f"top={crop_meta['top']}\n"
            f"bottom={crop_meta['bottom']}"
        ),
        va="top",
        family="monospace",
        fontsize=11,
    )

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(letterboxed_crop)
    ax4.set_title(f"Letterboxed crop ({TARGET_WIDTH}x{TARGET_HEIGHT})")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(letterboxed_mask, cmap="gray", vmin=0, vmax=255)
    ax5.set_title("Letterboxed mask")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(overlay)
    ax6.set_title("Crop + mask overlay")
    ax6.axis("off")

    fig.suptitle(crop_path.name, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = OUTPUT_DIR / f"{crop_path.stem}_letterbox.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(
        f"{crop_path.name}: "
        f"{row['width']}x{row['height']} -> "
        f"{crop_meta['resized_width']}x{crop_meta['resized_height']} "
        f"+ padding "
        f"(L{crop_meta['left']}, R{crop_meta['right']}, "
        f"T{crop_meta['top']}, B{crop_meta['bottom']})"
    )


def main() -> None:
    if not CROPS_DIR.exists():
        raise FileNotFoundError(f"Crops directory not found: {CROPS_DIR}")

    if not MASKS_DIR.exists():
        raise FileNotFoundError(f"Masks directory not found: {MASKS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = collect_crop_metadata()
    selected = select_representative_examples(rows)

    print("=" * 72)
    print("LETTERBOX VISUAL CHECK")
    print("=" * 72)
    print(f"Dataset crops: {len(rows)}")
    print(f"Selected examples: {len(selected)}")
    print(f"Target canvas: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print()

    for row in selected:
        visualize_example(row)

    print("\nSaved visualizations to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
