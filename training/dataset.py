from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_WIDTH = 384
TARGET_HEIGHT = 128

# ImageNet normalization for the pretrained ResNet34 encoder.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int,
    pad_value,
) -> tuple[np.ndarray, dict]:
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


class PlateSegmentationDataset(Dataset):
    def __init__(
        self,
        split: str,
        split_csv: Path | str | None = None,
        augment=None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got: {split}"
            )

        if split_csv is None:
            split_csv = PROJECT_ROOT / "data" / "split.csv"

        split_csv = Path(split_csv)

        if not split_csv.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_csv}\n"
                "Run training/create_split.py first."
            )

        df = pd.read_csv(split_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if self.df.empty:
            raise RuntimeError(f"No samples found for split='{split}'")

        self.split = split
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        crop_path = PROJECT_ROOT / row["crop_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]

        image_bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {crop_path}")

        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch for {row['sample_id']}: "
                f"image={image.shape[:2]}, mask={mask.shape[:2]}"
            )

        # Augmentations, when added later, must transform image and mask together.
        if self.augment is not None:
            augmented = self.augment(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image, image_meta = letterbox(
            image,
            TARGET_WIDTH,
            TARGET_HEIGHT,
            interpolation=cv2.INTER_LINEAR,
            pad_value=(0, 0, 0),
        )

        mask, mask_meta = letterbox(
            mask,
            TARGET_WIDTH,
            TARGET_HEIGHT,
            interpolation=cv2.INTER_NEAREST,
            pad_value=0,
        )

        if (
            image_meta["resized_width"] != mask_meta["resized_width"]
            or image_meta["resized_height"] != mask_meta["resized_height"]
            or image_meta["left"] != mask_meta["left"]
            or image_meta["right"] != mask_meta["right"]
            or image_meta["top"] != mask_meta["top"]
            or image_meta["bottom"] != mask_meta["bottom"]
        ):
            raise RuntimeError(
                f"Image/mask letterbox mismatch for {row['sample_id']}"
            )

        # RGB uint8 [0,255] -> float32 [0,1]
        image = image.astype(np.float32) / 255.0

        # Normalize for ImageNet-pretrained ResNet34.
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        # Binary mask -> {0, 1}
        mask = (mask > 127).astype(np.float32)

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": row["sample_id"],
        }


if __name__ == "__main__":
    for split_name in ["train", "val", "test"]:
        dataset = PlateSegmentationDataset(split=split_name)
        sample = dataset[0]

        print(
            f"{split_name:5s} | "
            f"N={len(dataset):2d} | "
            f"image={tuple(sample['image'].shape)} | "
            f"mask={tuple(sample['mask'].shape)} | "
            f"id={sample['sample_id']}"
        )
