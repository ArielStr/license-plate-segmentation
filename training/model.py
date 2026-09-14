from __future__ import annotations

import segmentation_models_pytorch as smp
import torch


def build_model() -> torch.nn.Module:
    """
    U-Net with a ResNet34 encoder pretrained on ImageNet.

    Input:
        [B, 3, 128, 384]

    Output:
        [B, 1, 128, 384] logits
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,  # return logits
    )
    return model


def freeze_encoder(model: torch.nn.Module) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: torch.nn.Module) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    model = build_model()

    x = torch.randn(2, 3, 128, 384)
    y = model(x)

    print("Input shape: ", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
