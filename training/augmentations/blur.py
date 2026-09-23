import random

import albumentations as A
import cv2


class RandomGaussianBlur(A.ImageOnlyTransform):
    """
    Apply a mild Gaussian blur.

    The transformation affects only the image.
    The segmentation mask is unchanged.
    """

    def __init__(
        self,
        sigma_range=(0.3, 1.2),
        p=0.20,
    ):
        super().__init__(p=p)

        self.sigma_range = sigma_range

    def get_params(self):
        return {
            "sigma": random.uniform(
                self.sigma_range[0],
                self.sigma_range[1],
            )
        }

    def apply(
        self,
        img,
        sigma=0.5,
        **params,
    ):
        return cv2.GaussianBlur(
            img,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )

    def get_transform_init_args_names(self):
        return ("sigma_range",)
