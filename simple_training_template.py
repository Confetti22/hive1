#!/usr/bin/env python3
"""
Single-GPU PyTorch training template (local machine)
---------------------------------------------------
• Minimal dependencies (only PyTorch, tqdm, tensorboard, pyyaml).
• YAML-driven hyper-parameters for easy experiment tracking.
• Graceful resume & checkpointing every N epochs.
• Clean function boundaries → easy to swap in your own Dataset/Model.

Usage
~~~~~
$ python train_template.py --cfg config/experiment.yaml  # fresh run
$ python train_template.py --cfg config/experiment.yaml --ckpt runs/exp_001/checkpoints/epoch_010.pth  # resume
"""
from __future__ import annotations
import argparse
import datetime as dt
import os
import shutil
from pathlib import Path
from typing import Tuple, Dict

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from config.load_config import load_cfg
################################################################################
# -----------------------------  Utility Helpers  ---------------------------- #
################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU PyTorch training template")
    parser.add_argument("--cfg", type=str, required=True,default="", help="Path to YAML config file")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda", help="cuda | cpu | cuda:0 …")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: optim.Optimizer | None = None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    return ckpt.get("epoch", 0) + 1  # next epoch number

################################################################################
# ---------------------------  Data & Model Stubs  --------------------------- #
################################################################################



################################################################################
# ---------------------------------  Train  ---------------------------------- #
################################################################################

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device, epoch: int, writer: SummaryWriter):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for step, (inputs, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, targets)

    n_steps = len(loader)
    writer.add_scalar("train/loss", running_loss / n_steps, epoch)
    writer.add_scalar("train/acc", running_acc / n_steps, epoch)


def validate(model: nn.Module, loader: DataLoader, criterion, device, epoch: int, writer: SummaryWriter):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            val_acc += accuracy(outputs, targets)
    n_steps = len(loader)
    writer.add_scalar("val/loss", val_loss / n_steps, epoch)
    writer.add_scalar("val/acc", val_acc / n_steps, epoch)

################################################################################
# --------------------------------  Main  ------------------------------------ #
################################################################################

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    # set_seed(cfg.get("seed", 42))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -------------------------  Experiment folder  ------------------------- #
    exp_name = cfg.get("exp_name", dt.datetime.now().strftime("exp_%Y%m%d_%H%M%S"))
    run_dir = Path("outs") / exp_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    # snapshot cfg for reproducibility
    shutil.copy2(args.cfg, run_dir / "config.yaml")

    # ---------------------------  Data loaders  --------------------------- #
    train_ds = DummyDataset()
    val_ds = DummyDataset()
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 256), shuffle=True, num_workers=cfg.get("num_workers", 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 256), shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=True)

    # ------------------------------  Model  ------------------------------ #
    model = MLP(in_dim=cfg.get("n_features", 128), hidden=tuple(cfg.get("mlp_hidden", [256, 128])), n_classes=cfg.get("n_classes", 2)).to(device)
    
    # -------------------------  loss & optimizer ------------------------------ #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr_start", 1e-4))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.get("lr_gamma", 0.95))

    # ----------------------------  Logging  ------------------------------ #
    writer = SummaryWriter(log_dir=run_dir / "tb")

    # ---------------------------  Resume  ------------------------------- #
    start_epoch = 0
    if args.ckpt:
        start_epoch = load_checkpoint(args.ckpt, model, optimizer)
        print(f"[INFO] Resumed from {args.ckpt} (next epoch = {start_epoch})")

    # ---------------------------  Training  ----------------------------- #
    n_epochs = cfg.num_epochs
    ckpt_every = cfg.save_very_epoch

    for epoch in range(start_epoch, n_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        scheduler.step()
        if (epoch + 1) % cfg.valid_very_epoch == 0 or epoch + 1 == n_epochs:
            validate(model, val_loader, criterion, device, epoch, writer)

        if (epoch + 1) % ckpt_every == 0 or epoch + 1 == n_epochs:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch+1:03d}.pth"
            save_checkpoint({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"[INFO] Saved checkpoint → {ckpt_path.relative_to(Path.cwd())}")

    writer.close()
    print("[Done] Training complete.")


if __name__ == "__main__":
    main()
