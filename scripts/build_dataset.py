from pathlib import Path
import json

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

CROPS_DIR = PROJECT_ROOT / "data" / "crops"
MASKS_DIR = PROJECT_ROOT / "data" / "masks"

CROP_PADDING = 0.20


CROPS_DIR.mkdir(parents=True, exist_ok=True)
MASKS_DIR.mkdir(parents=True, exist_ok=True)


def create_full_mask(image_shape, corners):
    height, width = image_shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    polygon = np.array(corners, dtype=np.int32)

    cv2.fillPoly(
        mask,
        [polygon],
        color=255,
    )

    return mask


def get_padded_bbox(corners, image_width, image_height, padding):
    pts = np.array(corners, dtype=np.float32)

    min_x = pts[:, 0].min()
    max_x = pts[:, 0].max()

    min_y = pts[:, 1].min()
    max_y = pts[:, 1].max()

    plate_width = max_x - min_x
    plate_height = max_y - min_y

    pad_x = plate_width * padding
    pad_y = plate_height * padding

    x1 = int(max(0, np.floor(min_x - pad_x)))
    y1 = int(max(0, np.floor(min_y - pad_y)))

    x2 = int(min(image_width, np.ceil(max_x + pad_x)))
    y2 = int(min(image_height, np.ceil(max_y + pad_y)))

    return x1, y1, x2, y2


def process_annotation(annotation_path):
    with annotation_path.open("r", encoding="utf-8") as f:
        annotation = json.load(f)

    image_path = RAW_DIR / annotation["image"]

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return 0

    height, width = image.shape[:2]

    count = 0

    for plate_idx, plate in enumerate(annotation["plates"], start=1):
        corners = plate["corners"]

        full_mask = create_full_mask(
            image.shape,
            corners,
        )

        x1, y1, x2, y2 = get_padded_bbox(
            corners,
            image_width=width,
            image_height=height,
            padding=CROP_PADDING,
        )

        crop = image[y1:y2, x1:x2]
        crop_mask = full_mask[y1:y2, x1:x2]

        sample_name = f"{image_path.stem}_plate_{plate_idx:02d}"

        crop_path = CROPS_DIR / f"{sample_name}.jpg"
        mask_path = MASKS_DIR / f"{sample_name}.png"

        cv2.imwrite(str(crop_path), crop)
        cv2.imwrite(str(mask_path), crop_mask)

        print(
            f"{sample_name}: "
            f"crop={crop.shape[1]}x{crop.shape[0]}"
        )

        count += 1

    return count


def main():
    annotation_paths = sorted(
        ANNOTATIONS_DIR.glob("*.json")
    )

    if not annotation_paths:
        raise RuntimeError(
            f"No annotations found in: {ANNOTATIONS_DIR}"
        )

    total_samples = 0

    for annotation_path in annotation_paths:
        total_samples += process_annotation(annotation_path)

    print()
    print("=" * 60)
    print(f"Created {total_samples} samples")
    print(f"Crops: {CROPS_DIR}")
    print(f"Masks: {MASKS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()