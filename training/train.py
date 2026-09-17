from __future__ import annotations
import random
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PlateSegmentationDataset
from losses import BCEDiceLoss
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
    total_batches = 0

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

        total_loss += loss.item()
        total_iou += binary_iou(logits.detach(), masks)
        total_dice += binary_dice(logits.detach(), masks)
        total_batches += 1

    return {
        "loss": total_loss / total_batches,
        "iou": total_iou / total_batches,
        "dice": total_dice / total_batches,
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
):
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
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
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"lr={current_lr:.8f} | "
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
    print(f"Checkpoint dir: {checkpoint_dir}")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)
    criterion = BCEDiceLoss()

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

    print()
    print(f"Training finished. Best val IoU: {best_val_iou:.4f}")
    print(f"Checkpoint: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
