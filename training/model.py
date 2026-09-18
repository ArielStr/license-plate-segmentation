from __future__ import annotations

import segmentation_models_pytorch as smp
import torch


def build_model(
    architecture: str = "unet",
    encoder_name: str = "resnet34",
) -> torch.nn.Module:
    """
    Build a binary segmentation model with an ImageNet-pretrained encoder.
    """

    if architecture == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

    elif architecture == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}"
        )

    return model


def freeze_encoder(model: torch.nn.Module) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: torch.nn.Module) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    configs = [
        ("unet", "resnet18"),
        ("unet", "resnet34"),
        ("unet", "resnet50"),
        ("deeplabv3plus", "resnet34"),
    ]

    x = torch.randn(2, 3, 128, 384)

    for architecture, encoder_name in configs:
        model = build_model(
            architecture=architecture,
            encoder_name=encoder_name,
        )

        y = model(x)

        print(
            f"{architecture} + {encoder_name}: "
            f"{tuple(x.shape)} -> {tuple(y.shape)}"
        )
