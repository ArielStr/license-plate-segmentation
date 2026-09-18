from __future__ import annotations
import random
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PlateSegmentationDataset
from losses import build_loss
from metrics import binary_dice, binary_iou
from model import build_model, freeze_encoder, unfreeze_encoder
from augmentations import build_train_augmentation
from configs import get_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUM_WORKERS = 0  # safe default for Windows; increase in Colab if desired

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_iou += binary_iou(logits.detach(), masks) * batch_size
        total_dice += binary_dice(logits.detach(), masks) * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "iou": total_iou / total_samples,
        "dice": total_dice / total_samples,
    }


def train_phase(
    name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    best_val_iou,
    checkpoint_dir,
    experiment_name,
    augmentation_profile,
    scheduler=None,
    start_epoch=1,
    total_epochs=None,
):
    if total_epochs is None:
        total_epochs = epochs
    end_epoch = start_epoch + epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        current_batch_size = train_loader.batch_size
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_stats = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        print(
            f"[{name}] "
            f"Epoch {epoch:02d}/{total_epochs:02d} | "
            f"lr={current_lr:.8f} | "
            f"batch_size={current_batch_size} | "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_iou={train_stats['iou']:.4f} "
            f"train_dice={train_stats['dice']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_iou={val_stats['iou']:.4f} "
            f"val_dice={val_stats['dice']:.4f}"
        )

        if val_stats["iou"] > best_val_iou:
            best_val_iou = val_stats["iou"]

            checkpoint_path = checkpoint_dir / "best_model.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_iou": best_val_iou,
                    "phase": name,
                    "epoch": epoch,
                    "batch_size": current_batch_size,
                    "experiment": experiment_name,
                    "augmentation_profile": augmentation_profile,
                },
                checkpoint_path,
            )

            print(
                f"  -> saved new best checkpoint "
                f"(val IoU={best_val_iou:.4f})"
            )
        if scheduler is not None:
            scheduler.step()

    return best_val_iou
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a license plate segmentation experiment."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment configuration name.",
    )

    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_train_loader(
    train_dataset,
    batch_size,
    device,
):
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

def main():
    args = parse_args()
    config = get_experiment_config(args.experiment)
    set_seed(config.seed)

    checkpoint_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / config.name
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Experiment: {config.name}")
    print(f"Seed: {config.seed}")
    print(f"Batch size: {config.batch_size}")
    print(f"Augmentation: {config.augmentation_profile}")
    print(f"Scheduler: {config.scheduler}")
    print(f"Loss: {config.loss}")
    print(f"BCE weight: {config.bce_weight}")
    print(f"Dice weight: {config.dice_weight}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Focal weight: {config.focal_weight}")
    print(f"Focal gamma: {config.focal_gamma}")

    train_augmentation = build_train_augmentation(
        profile=config.augmentation_profile,
    )
    train_dataset = PlateSegmentationDataset(
        split="train",
        augment=train_augmentation,
    )

    val_dataset = PlateSegmentationDataset(
        split="val",
    )

    train_loader = build_train_loader(
        train_dataset=train_dataset,
        batch_size=config.batch_size,
        device=device,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)
    criterion = build_loss(
        loss_name=config.loss,
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        focal_weight=config.focal_weight,
        focal_gamma=config.focal_gamma,
    )

    best_val_iou = -1.0

    # ---------------------------------------------------------
    # Phase 1: freeze pretrained encoder, train decoder only
    # ---------------------------------------------------------
    freeze_encoder(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.phase1_lr,
        weight_decay=config.weight_decay,
    )

    best_val_iou = train_phase(
        name="frozen_encoder",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config.phase1_epochs,
        best_val_iou=best_val_iou,
        checkpoint_dir=checkpoint_dir,
        experiment_name=config.name,
        augmentation_profile=config.augmentation_profile,
    )

    # ---------------------------------------------------------
    # Phase 2: unfreeze encoder and fine-tune full network
    # ---------------------------------------------------------
    unfreeze_encoder(model)

    if (
            config.phase2_encoder_lr is not None
            and config.phase2_decoder_lr is not None
    ):
        encoder_params = list(model.encoder.parameters())

        encoder_param_ids = {
            id(param)
            for param in encoder_params
        }

        decoder_params = [
            param
            for param in model.parameters()
            if id(param) not in encoder_param_ids
        ]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": encoder_params,
                    "lr": config.phase2_encoder_lr,
                },
                {
                    "params": decoder_params,
                    "lr": config.phase2_decoder_lr,
                },
            ],
            weight_decay=config.weight_decay,
        )

        print(
            f"Phase 2 learning rates: "
            f"encoder={config.phase2_encoder_lr:.8f}, "
            f"decoder/rest={config.phase2_decoder_lr:.8f}"
        )

    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.phase2_lr,
            weight_decay=config.weight_decay,
        )

    scheduler = None

    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.phase2_epochs,
            eta_min=config.scheduler_min_lr,
        )
    elif config.scheduler is not None:
        raise ValueError(
            f"Unsupported scheduler: {config.scheduler}"
        )

    if config.phase2_batch_schedule is None:
        best_val_iou = train_phase(
            name="full_finetune",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=config.phase2_epochs,
            best_val_iou=best_val_iou,
            checkpoint_dir=checkpoint_dir,
            experiment_name=config.name,
            augmentation_profile=config.augmentation_profile,
            scheduler=scheduler,
        )

    else:
        scheduled_epochs = sum(
            stage_epochs
            for stage_epochs, _ in config.phase2_batch_schedule
        )

        if scheduled_epochs != config.phase2_epochs:
            raise ValueError(
                f"Phase 2 batch schedule contains {scheduled_epochs} epochs, "
                f"but phase2_epochs={config.phase2_epochs}"
            )

        start_epoch = 1

        for stage_epochs, batch_size in config.phase2_batch_schedule:
            if train_loader.batch_size != batch_size:
                train_loader = build_train_loader(
                    train_dataset=train_dataset,
                    batch_size=batch_size,
                    device=device,
                )

            print()
            print(
                f"Starting Phase 2 batch stage: "
                f"epochs {start_epoch}-{start_epoch + stage_epochs - 1}, "
                f"batch_size={batch_size}"
            )

            best_val_iou = train_phase(
                name="full_finetune",
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epochs=stage_epochs,
                best_val_iou=best_val_iou,
                checkpoint_dir=checkpoint_dir,
                experiment_name=config.name,
                augmentation_profile=config.augmentation_profile,
                scheduler=scheduler,
                start_epoch=start_epoch,
                total_epochs=config.phase2_epochs,
            )

            start_epoch += stage_epochs

    print()
    print(f"Training finished. Best val IoU: {best_val_iou:.4f}")
    print(f"Checkpoint: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
