import random
from typing import Optional

import albumentations as A
import cv2
import numpy as np

from training.augmentations.blur import RandomGaussianBlur
from training.augmentations.illumination import RandomLocalShadow, RandomIlluminationGradient

AVAILABLE_PROFILES = (
    "none",
    "global_brightness_v1",
    "local_shadow_v1",
    "illumination_gradient_v1",
    "gaussian_blur_v1"
)

def build_gaussian_blur_v1() -> A.ReplayCompose:
    """
    Gaussian-blur augmentation v1.

    Blur is applied to 30% of training samples.
    When applied, sigma is sampled uniformly from 0.5 to 1.5.

    The range is intentionally limited to mild/moderate blur
    to avoid unrealistically degrading samples that are already blurry.
    """

    return A.ReplayCompose([
        RandomGaussianBlur(
            sigma_range=(0.5, 1.5),
            p=0.30,
        ),
    ])
def build_global_brightness_v1() -> A.Compose:
    """
    Apply mild global brightness and contrast variations.

    The transformation affects the entire image uniformly.
    The segmentation mask is unchanged.
    """

    return A.ReplayCompose([
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.15,
            p=0.8,
        ),
    ])

def build_local_shadow_v1() -> A.ReplayCompose:
    return A.ReplayCompose([
        RandomLocalShadow(
            moderate_strength_range=(0.50, 0.85),
            strong_strength_range=(0.25, 0.50),

            strong_shadow_probability=0.20,

            band_probability=0.10,
            band_width_range=(0.12, 0.35),

            blur_fraction_range=(0.03, 0.10),

            p=0.8,
        ),
    ])
def build_illumination_gradient_v1() -> A.ReplayCompose:
    return A.ReplayCompose([
        RandomIlluminationGradient(
            dark_factor_range=(0.40, 0.70),
            bright_factor_range=(1.15, 1.40),
            p=0.8,
        )
    ])
def build_train_augmentation(
    profile: str = "none",
) -> Optional[A.ReplayCompose]:
    """
    Build an augmentation pipeline according to the requested profile.
    """

    profile = profile.lower()

    if profile == "none":
        return None

    if profile == "global_brightness_v1":
        return build_global_brightness_v1()

    if profile == "local_shadow_v1":
        return build_local_shadow_v1()

    if profile == "illumination_gradient_v1":
        return build_illumination_gradient_v1()

    if profile == "gaussian_blur_v1":
        return build_gaussian_blur_v1()

    raise ValueError(
        f"Unknown augmentation profile: '{profile}'. "
        f"Available profiles: {AVAILABLE_PROFILES}"
    )