from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PlateSegmentationDataset
from losses import BCEDiceLoss
from metrics import binary_dice, binary_iou
from model import build_model, freeze_encoder, unfreeze_encoder
from augmentations import build_train_augmentation
EXPERIMENT_NAME = "cosine_lr_v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / EXPERIMENT_NAME
)
BATCH_SIZE = 8
NUM_WORKERS = 0  # safe default for Windows; increase in Colab if desired

PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3

PHASE2_EPOCHS = 30
PHASE2_LR = 1e-4
PHASE2_MIN_LR = 1e-6

WEIGHT_DECAY = 1e-4
AUGMENTATION_PROFILE = "none"

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

            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_iou": best_val_iou,
                    "phase": name,
                    "epoch": epoch,
                    "experiment": EXPERIMENT_NAME,
                    "augmentation_profile": AUGMENTATION_PROFILE,
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


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Augmentation: {AUGMENTATION_PROFILE}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")

    train_augmentation = build_train_augmentation(
        profile=AUGMENTATION_PROFILE,
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
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
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
        lr=PHASE1_LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_iou = train_phase(
        name="frozen_encoder",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=PHASE1_EPOCHS,
        best_val_iou=best_val_iou,
    )

    # ---------------------------------------------------------
    # Phase 2: unfreeze encoder and fine-tune full network
    # ---------------------------------------------------------
    unfreeze_encoder(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PHASE2_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=PHASE2_EPOCHS,
        eta_min=PHASE2_MIN_LR,
    )

    best_val_iou = train_phase(
        name="full_finetune",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=PHASE2_EPOCHS,
        best_val_iou=best_val_iou,
        scheduler=scheduler,
    )

    print()
    print(f"Training finished. Best val IoU: {best_val_iou:.4f}")
    print(f"Checkpoint: {CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
