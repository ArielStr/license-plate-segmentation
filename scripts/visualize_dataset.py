from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CROPS_DIR = PROJECT_ROOT / "data" / "crops"
MASKS_DIR = PROJECT_ROOT / "data" / "masks"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def overlay_mask(image, mask):
    overlay = image.copy()

    mask_binary = mask > 127

    highlight = np.zeros_like(image)
    highlight[:, :, 1] = 255

    alpha = 0.35

    overlay[mask_binary] = cv2.addWeighted(
        image[mask_binary],
        1 - alpha,
        highlight[mask_binary],
        alpha,
        0,
    )

    return overlay


def resize_to_height(image, target_height):
    scale = target_height / image.shape[0]

    target_width = int(image.shape[1] * scale)

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def main():
    crop_paths = sorted(
        path
        for path in CROPS_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not crop_paths:
        raise RuntimeError(
            f"No crops found in: {CROPS_DIR}"
        )

    print("Controls: N/SPACE = next, Q = quit")

    for crop_path in crop_paths:
        mask_path = MASKS_DIR / f"{crop_path.stem}.png"

        if not mask_path.exists():
            print(f"Missing mask for: {crop_path.name}")
            continue

        image = cv2.imread(str(crop_path))
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None or mask is None:
            print(f"Could not load: {crop_path.stem}")
            continue

        overlay = overlay_mask(image, mask)

        mask_visual = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR,
        )

        target_height = 180

        image_vis = resize_to_height(
            image,
            target_height,
        )

        mask_vis = resize_to_height(
            mask_visual,
            target_height,
        )

        overlay_vis = resize_to_height(
            overlay,
            target_height,
        )

        panel = np.hstack([
            image_vis,
            mask_vis,
            overlay_vis,
        ])

        cv2.putText(
            panel,
            crop_path.stem,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.imshow(
            "Image | GT Mask | Overlay",
            panel,
        )

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("n"), ord(" "), 13):
                break

            if key == ord("q"):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()