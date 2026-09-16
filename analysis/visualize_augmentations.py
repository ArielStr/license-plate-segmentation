import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from training.augmentations import AVAILABLE_PROFILES, build_train_augmentation


SPLIT_CSV = PROJECT_ROOT / "data" / "split.csv"

DEFAULT_NUM_IMAGES = 5
DEFAULT_NUM_AUGMENTATIONS = 7
DEFAULT_SEED = 42


def load_train_samples(
    num_images: int,
    seed: int,
) -> pd.DataFrame:
    """
    Load a reproducible random subset of training samples.
    """

    df = pd.read_csv(SPLIT_CSV)
    train_df = df[df["split"] == "train"].copy()

    if train_df.empty:
        raise RuntimeError("No training samples found in split.csv")

    num_images = min(num_images, len(train_df))

    return train_df.sample(
        n=num_images,
        random_state=seed,
    ).reset_index(drop=True)


def load_image_and_mask(row):
    """
    Load the original RGB crop and its binary mask.
    """

    crop_relative_path = str(row["crop_path"]).replace("\\", "/")
    mask_relative_path = str(row["mask_path"]).replace("\\", "/")

    image_path = PROJECT_ROOT / Path(crop_relative_path)
    mask_path = PROJECT_ROOT / Path(mask_relative_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = (mask > 127).astype(np.uint8)

    sample_name = row["sample_id"]

    return image, mask, sample_name


def apply_augmentation(
    image,
    mask,
    augmentation,
):
    """
    Apply one random realization of the augmentation pipeline.

    Returns:
        augmented_image
        augmented_mask
        replay metadata, when available
    """

    if augmentation is None:
        return image.copy(), mask.copy(), None

    result = augmentation(
        image=image,
        mask=mask,
    )

    return (
        result["image"],
        result["mask"],
        result.get("replay"),
    )


def format_value(value):
    """
    Convert simple replay-metadata values into short readable strings.
    Complex values are skipped to keep titles readable.
    """

    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, str):
        return value

    return None


def extract_replay_metadata(
    replay,
    max_params_per_transform=3,
):
    """
    Extract readable metadata from an Albumentations ReplayCompose result.

    This function is augmentation-agnostic. It does not know about
    shadows, brightness, blur, rotation, etc.
    """

    if replay is None:
        return []

    metadata = []

    for transform in replay.get("transforms", []):
        if not transform.get("applied", False):
            continue

        class_name = transform.get(
            "__class_fullname__",
            "Transform",
        )
        transform_name = class_name.split(".")[-1]

        params = transform.get("params") or {}
        readable_params = []

        for key, value in params.items():
            formatted_value = format_value(value)

            if formatted_value is None:
                continue

            readable_params.append(
                f"{key}={formatted_value}"
            )

            if len(readable_params) >= max_params_per_transform:
                break

        if readable_params:
            metadata.append(transform_name)
            metadata.extend(readable_params)
        else:
            metadata.append(transform_name)

    return metadata


def build_augmentation_title(
    index,
    replay,
):
    """
    Build a generic title for one augmentation realization.
    """

    metadata = extract_replay_metadata(replay)

    if not metadata:
        return f"Aug #{index}\nNo transform"

    return (
        f"Aug #{index}\n"
        + "\n".join(metadata)
    )


def draw_mask_contour(
    ax,
    mask,
):
    """
    Draw the segmentation-mask boundary over an image.
    """

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        contour = contour.squeeze(axis=1)

        if len(contour) < 2:
            continue

        contour = np.vstack([contour, contour[0]])

        ax.plot(
            contour[:, 0],
            contour[:, 1],
            linewidth=1.5,
        )


def save_augmentation_panel(
    image,
    mask,
    sample_name,
    augmentation,
    profile,
    num_augmentations,
    output_dir,
    show_mask,
):
    """
    Create a panel containing the original image and several
    independent augmentation realizations.
    """

    total_images = 1 + num_augmentations

    num_cols = 4
    num_rows = int(np.ceil(total_images / num_cols))

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(16, 4 * num_rows),
    )

    axes = np.array(axes).reshape(-1)

    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original")

    if show_mask:
        draw_mask_contour(
            axes[0],
            mask,
        )

    axes[0].axis("off")

    # Augmented versions
    for i in range(num_augmentations):
        augmented_image, augmented_mask, replay = apply_augmentation(
            image=image,
            mask=mask,
            augmentation=augmentation,
        )

        ax = axes[i + 1]

        ax.imshow(augmented_image)

        ax.set_title(
            build_augmentation_title(
                index=i + 1,
                replay=replay,
            ),
            fontsize=9,
        )

        if show_mask:
            draw_mask_contour(
                ax,
                augmented_mask,
            )

        ax.axis("off")

    # Hide unused cells
    for i in range(total_images, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"{sample_name} | profile={profile}",
        fontsize=16,
    )

    plt.tight_layout()

    output_path = output_dir / f"{sample_name}.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def main(
    profile: str,
    num_images: int,
    num_augmentations: int,
    seed: int,
    show_mask: bool,
):
    augmentation = build_train_augmentation(
        profile=profile,
    )

    samples = load_train_samples(
        num_images=num_images,
        seed=seed,
    )

    output_dir = (
        PROJECT_ROOT
        / "evaluation"
        / "augmentation_visualization"
        / profile
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("AUGMENTATION VISUALIZATION")
    print("=" * 72)
    print(f"Profile: {profile}")
    print(f"Samples: {len(samples)}")
    print(f"Augmentations per sample: {num_augmentations}")
    print(f"Show mask: {show_mask}")
    print(f"Output: {output_dir}")
    print("=" * 72)

    for _, row in samples.iterrows():
        image, mask, sample_name = load_image_and_mask(row)

        save_augmentation_panel(
            image=image,
            mask=mask,
            sample_name=sample_name,
            augmentation=augmentation,
            profile=profile,
            num_augmentations=num_augmentations,
            output_dir=output_dir,
            show_mask=show_mask,
        )

    print("\nDone.")


if __name__ == "__main__":
    PROFILE = "gaussian_blur_v1"
    NUM_IMAGES = 5
    NUM_AUGMENTATIONS = 7
    SEED = 42
    SHOW_MASK = False

    main(
        profile=PROFILE,
        num_images=NUM_IMAGES,
        num_augmentations=NUM_AUGMENTATIONS,
        seed=SEED,
        show_mask=SHOW_MASK,
    )
