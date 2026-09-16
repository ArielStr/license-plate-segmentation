import random
from typing import Optional

import albumentations as A
import cv2
import numpy as np
class RandomLocalShadow(A.ImageOnlyTransform):
    """
    Simulate realistic local shadows across a license-plate crop.

    Two shadow geometries are supported:
        1. Half-plane shadow:
           one side of a random line is darkened.

        2. Shadow band:
           a relatively narrow strip between two parallel lines
           is darkened.

    Most generated shadows are half-plane shadows.
    Most shadows have moderate intensity, while a small fraction
    are intentionally strong.

    The segmentation mask is not modified.
    """

    def __init__(
        self,
        moderate_strength_range=(0.50, 0.85),
        strong_strength_range=(0.25, 0.50),
        strong_shadow_probability=0.20,
        band_probability=0.20,
        band_width_range=(0.12, 0.35),
        blur_fraction_range=(0.03, 0.10),
        p=0.8,
    ):
        super().__init__(p=p)

        self.moderate_strength_range = moderate_strength_range
        self.strong_strength_range = strong_strength_range
        self.strong_shadow_probability = strong_shadow_probability

        self.band_probability = band_probability
        self.band_width_range = band_width_range

        self.blur_fraction_range = blur_fraction_range

    def apply(
        self,
        img,
        shadow_strength=0.7,
        shadow_type="half_plane",
        line_angle=0.0,
        line_position=0.0,
        shadow_side=1,
        band_width=0.2,
        blur_fraction=0.05,
        **params,
    ):
        height, width = img.shape[:2]

        # Normalized image coordinates.
        #
        # x and y are approximately in [-0.5, 0.5].
        yy, xx = np.mgrid[0:height, 0:width]

        x = xx / max(width - 1, 1) - 0.5
        y = yy / max(height - 1, 1) - 0.5

        theta = np.deg2rad(line_angle)

        # Projection onto the normal direction of the shadow boundary.
        #
        # This gives us a coordinate system in which a shadow can be
        # described using one line (half-plane) or two parallel lines
        # (band).
        projection = (
            x * np.cos(theta)
            + y * np.sin(theta)
            - line_position
        )

        if shadow_type == "half_plane":

            if shadow_side < 0:
                projection = -projection

            shadow_mask = (
                projection > 0
            ).astype(np.float32)

        elif shadow_type == "band":

            half_band_width = band_width / 2.0

            shadow_mask = (
                np.abs(projection) < half_band_width
            ).astype(np.float32)

        else:
            raise ValueError(
                f"Unknown shadow type: {shadow_type}"
            )

        # Blur the boundary so the transition between illuminated and
        # shadowed regions is not always perfectly sharp.
        blur_size = max(
            3,
            int(
                min(height, width)
                * blur_fraction
            ),
        )

        if blur_size % 2 == 0:
            blur_size += 1

        shadow_mask = cv2.GaussianBlur(
            shadow_mask,
            (blur_size, blur_size),
            sigmaX=0,
        )

        # illumination = 1 outside the shadow.
        #
        # Inside the shadow it approaches shadow_strength:
        #
        #   1.00 -> unchanged
        #   0.70 -> moderate shadow
        #   0.30 -> strong shadow
        illumination = (
            1.0
            - shadow_mask
            * (1.0 - shadow_strength)
        )

        augmented = (
            img.astype(np.float32)
            * illumination[..., None]
        )

        return np.clip(
            augmented,
            0,
            255,
        ).astype(img.dtype)

    def get_params(self):

        # ------------------------------------------------------------
        # Shadow geometry
        # ------------------------------------------------------------

        # Real examples suggest that most local shadows resemble a
        # single boundary crossing the plate.
        #
        # Shadow bands exist, but should be less common.
        if np.random.random() < self.band_probability:
            shadow_type = "band"
        else:
            shadow_type = "half_plane"

        # ------------------------------------------------------------
        # Shadow strength
        # ------------------------------------------------------------

        # Most shadows are moderate.
        # Occasionally generate a much stronger shadow to represent
        # the dark real-world examples in the dataset.
        if (
            np.random.random()
            < self.strong_shadow_probability
        ):
            shadow_strength = np.random.uniform(
                *self.strong_strength_range
            )
        else:
            shadow_strength = np.random.uniform(
                *self.moderate_strength_range
            )

        return {
            "shadow_strength": shadow_strength,

            "shadow_type": shadow_type,

            "line_angle": np.random.uniform(
                0.0,
                180.0,
            ),

            "line_position": np.random.uniform(
                -0.20,
                0.20,
            ),

            "shadow_side": np.random.choice(
                [-1, 1]
            ),

            "band_width": np.random.uniform(
                *self.band_width_range
            ),

            "blur_fraction": np.random.uniform(
                *self.blur_fraction_range
            ),
        }

class RandomIlluminationGradient(A.ImageOnlyTransform):
    """
    Apply a smooth linear illumination gradient across the image.

    One side of the image is gradually brightened, while the
    opposite side is gradually darkened.

    Example:

        bright_factor = 1.25
        dark_factor   = 0.60

        1.25 -------- 1.00 -------- 0.60

    The gradient direction is sampled randomly over 360 degrees.

    The segmentation mask is not modified.
    """

    def __init__(
        self,
        dark_factor_range=(0.50, 0.75),
        bright_factor_range=(1.10, 1.30),
        p=0.8,
    ):
        super().__init__(p=p)

        self.dark_factor_range = dark_factor_range
        self.bright_factor_range = bright_factor_range

    def get_params(self):
        angle = random.uniform(
            0.0,
            360.0,
        )

        dark_factor = random.uniform(
            self.dark_factor_range[0],
            self.dark_factor_range[1],
        )

        bright_factor = random.uniform(
            self.bright_factor_range[0],
            self.bright_factor_range[1],
        )

        return {
            "angle": angle,
            "dark_factor": dark_factor,
            "bright_factor": bright_factor,
        }

    def apply(
        self,
        img,
        angle=0.0,
        dark_factor=0.65,
        bright_factor=1.20,
        **params,
    ):
        h, w = img.shape[:2]

        # ------------------------------------------------------------
        # Normalized image coordinates centered around zero
        # ------------------------------------------------------------

        yy, xx = np.mgrid[
            0:h,
            0:w,
        ].astype(np.float32)

        xx = xx / max(w - 1, 1) - 0.5
        yy = yy / max(h - 1, 1) - 0.5

        # ------------------------------------------------------------
        # Gradient direction
        # ------------------------------------------------------------

        theta = np.deg2rad(angle)

        direction_x = np.cos(theta)
        direction_y = np.sin(theta)

        projection = (
            xx * direction_x
            + yy * direction_y
        )

        # ------------------------------------------------------------
        # Normalize projection to [0, 1]
        #
        # 0 -> bright side
        # 1 -> dark side
        # ------------------------------------------------------------

        projection_min = projection.min()
        projection_max = projection.max()

        projection = (
            projection - projection_min
        ) / max(
            projection_max - projection_min,
            1e-6,
        )

        # ------------------------------------------------------------
        # Smooth illumination gradient
        #
        # bright_factor -----------------> dark_factor
        #
        # Example:
        #
        # 1.25 -------- 0.925 -------- 0.60
        #
        # There is no sharp boundary.
        # ------------------------------------------------------------

        illumination = (
            bright_factor
            + projection
            * (dark_factor - bright_factor)
        )

        illumination = illumination[..., None]

        # ------------------------------------------------------------
        # Apply illumination
        # ------------------------------------------------------------

        out = (
            img.astype(np.float32)
            * illumination
        )

        return np.clip(
            out,
            0,
            255,
        ).astype(np.uint8)

    def get_transform_init_args_names(self):
        return (
            "dark_factor_range",
            "bright_factor_range",
        )