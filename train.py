"""Train an EfficientNet on Caltech-UCSD Birds-200-2011.

Default behaviour (good for CPU):
  * Load `efficientnet_b0` with ImageNet-pretrained weights
  * Freeze the backbone, replace the final classifier with a fresh 200-way head
  * Train just the head for 8 epochs

This is "linear probing" + a small twist (we're training the last `nn.Linear`,
not a full mini-MLP). It's the cheapest possible form of transfer learning and
typically gets ~60-70% top-1 on CUB-200 in a few minutes of CPU time.

For higher accuracy (~80%+), pass --no-freeze to unfreeze the backbone and
fine-tune the whole network. Plan for a GPU and several epochs.

Usage:
    python train.py                       # defaults from config.py
    python train.py --epochs 12           # override
    python train.py --no-freeze --lr 1e-4 # full fine-tune
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm import tqdm

import config
import dataset as ds_module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
    """Construct an EfficientNet with a fresh classifier head."""
    factory = getattr(models, model_name, None)
    if factory is None:
        raise ValueError(f"Unknown torchvision model: {model_name}")
    # `weights="DEFAULT"` pulls the best available pretrained weights
    model = factory(weights="DEFAULT")

    # EfficientNets expose classifier as nn.Sequential(Dropout, Linear).
    # Replace the Linear so output dim = num_classes.
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError(f"Don't know how to swap head on {model_name}")

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        # Re-enable grads only on the new head
        for p in model.classifier[-1].parameters():
            p.requires_grad = True
    return model


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, device) -> tuple[float, float]:
    """Return (avg_loss, top1_accuracy) on `loader`."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="val", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx: int):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"epoch {epoch_idx}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
        pbar.set_postfix(
            loss=f"{total_loss / total_n:.3f}",
            acc=f"{total_correct / total_n:.3f}",
        )
    return total_loss / total_n, total_correct / total_n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--model", type=str, default=config.MODEL_NAME)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument(
        "--no-freeze",
        dest="freeze",
        action="store_false",
        default=config.FREEZE_BACKBONE,
        help="Unfreeze backbone and fine-tune the whole network.",
    )
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not config.CUB_ROOT.exists():
        print("CUB-200-2011 not found. Run `python download_data.py` first.")
        return 1

    class_names = ds_module.load_class_names()
    train_loader, val_loader = ds_module.build_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(args.model, config.NUM_CLASSES, args.freeze).to(device)
    trainable = count_trainable(model)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {args.model}  freeze_backbone={args.freeze}\n"
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.CHECKPOINT_DIR / "training_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,elapsed_s\n")

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch:02d}]  "
            f"train_loss={tr_loss:.3f} train_acc={tr_acc:.3f}  "
            f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}  "
            f"lr={lr_now:.2e}  elapsed={elapsed:.0f}s"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{tr_loss:.4f},{tr_acc:.4f},"
                f"{val_loss:.4f},{val_acc:.4f},{lr_now:.6f},{elapsed:.1f}\n"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "num_classes": config.NUM_CLASSES,
                "class_names": class_names,
                "image_size": config.IMAGE_SIZE,
                "val_acc": val_acc,
                "epoch": epoch,
                "freeze_backbone": args.freeze,
            }
            best_path = config.CHECKPOINT_DIR / "best.pt"
            torch.save(ckpt, best_path)
            # Also save class names separately for easy inspection
            with open(config.CHECKPOINT_DIR / "class_names.json", "w") as f:
                json.dump(class_names, f, indent=2)
            print(f"  -> saved new best to {best_path} (val_acc={val_acc:.4f})")

    print(f"\nDone. Best val_acc={best_acc:.4f} in {time.time() - t0:.0f}s.")
    print("Run `python predict.py path/to/bird.jpg` to try the trained model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
