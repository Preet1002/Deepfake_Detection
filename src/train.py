"""Train DFNet.

    python -m src.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import Config, load_config
from .data.dataset import build_dataloaders
from .models.dfnet import build_model
from .utils.common import (
    count_parameters,
    dump_json,
    pick_device,
    save_checkpoint,
    set_seed,
)
from .utils.metrics import compute_metrics, format_metrics


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float,
              min_lr: float) -> float:
    """Linear warmup then cosine decay, computed per optimizer step."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def smooth_targets(targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Label smoothing for BCE: pulls 0/1 towards eps/1-eps.

    Deepfake datasets contain mislabeled and borderline images; smoothing stops
    the network from driving logits to infinity on them.
    """
    if epsilon <= 0:
        return targets
    return targets * (1.0 - epsilon) + 0.5 * epsilon


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader, device: torch.device,
                   criterion: nn.Module) -> tuple[Dict[str, float], float]:
    model.eval()
    losses, scores, labels = [], [], []
    for images, targets in tqdm(loader, desc="  val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        losses.append(criterion(logits, targets).item())
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(targets.cpu().numpy())

    metrics = compute_metrics(np.concatenate(labels), np.concatenate(scores))
    return metrics, float(np.mean(losses))


def train_one_epoch(model, loader, device, criterion, optimizer, config: Config,
                    epoch: int, total_steps: int, warmup_steps: int,
                    global_step: int, scaler) -> tuple[float, int]:
    model.train()
    running_loss, seen = 0.0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.train.epochs}", leave=False)

    for images, targets in progress:
        lr = cosine_lr(global_step, total_steps, warmup_steps,
                       config.train.lr, config.train.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        images = images.to(device, non_blocking=True)
        targets = smooth_targets(targets.to(device, non_blocking=True),
                                 config.train.label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()

        batch = images.size(0)
        running_loss += loss.item() * batch
        seen += batch
        global_step += 1
        progress.set_postfix(loss=f"{running_loss / seen:.4f}", lr=f"{lr:.2e}")

    return running_loss / max(1, seen), global_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DFNet deepfake detector")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint to warm-start from")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.train.seed)
    device = pick_device(config.train.device)

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(config, splits=("train", "val"))
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = build_model(config.model).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"Warm-started from {args.resume}")

    print(f"Device            : {device}", flush=True)
    print(f"Trainable params  : {count_parameters(model):,}", flush=True)
    print(f"Train images      : {len(train_loader.dataset):,} "
          f"{train_loader.dataset.class_counts()}", flush=True)
    print(f"Val images        : {len(val_loader.dataset):,} "
          f"{val_loader.dataset.class_counts()}", flush=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr,
                                  weight_decay=config.train.weight_decay)

    # GradScaler is CUDA-only; MPS runs in fp32 here, which is fast enough.
    use_amp = config.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if config.train.amp and not use_amp:
        print("Note: AMP requested but only supported on CUDA - training in fp32.")

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * config.train.epochs
    warmup_steps = steps_per_epoch * config.train.warmup_epochs

    history = []
    best_auc, best_epoch, global_step = -1.0, -1, 0

    for epoch in range(config.train.epochs):
        start = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, device, criterion, optimizer, config,
            epoch, total_steps, warmup_steps, global_step, scaler,
        )
        val_metrics, val_loss = evaluate_epoch(model, val_loader, device, criterion)
        elapsed = time.time() - start

        print(f"Epoch {epoch + 1:3d}/{config.train.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"{format_metrics(val_metrics)}  ({elapsed:.0f}s)", flush=True)

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": val_loss, **val_metrics})
        dump_json(out_dir / "history.json", history)

        # Select on AUC rather than accuracy: it is threshold-free and far less
        # noisy epoch to epoch on a nearly balanced validation set.
        if val_metrics["auc"] > best_auc:
            best_auc, best_epoch = val_metrics["auc"], epoch
            save_checkpoint(out_dir / "best.pt", model, config.to_dict(),
                            epoch + 1, val_metrics)
            print(f"  -> new best (AUC {best_auc:.4f}), saved {out_dir / 'best.pt'}",
                  flush=True)

        save_checkpoint(out_dir / "last.pt", model, config.to_dict(), epoch + 1, val_metrics)

        if epoch - best_epoch >= config.train.early_stop_patience:
            print(f"Early stopping: no AUC improvement for "
                  f"{config.train.early_stop_patience} epochs.")
            break

    print(f"\nDone. Best val AUC {best_auc:.4f} at epoch {best_epoch + 1}.")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
