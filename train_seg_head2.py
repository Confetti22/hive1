#!/usr/bin/env python3
"""
Concise training script for ConvSegHead with unified evaluation on
validation *and* (optionally) training data.

Key improvements over the original snippet
-----------------------------------------
* **evaluate()** now covers former `seg_valid` / `seg_valid_training_data`.
* Corrected operator precedence bug (`epoch % (4 * valid_every)`).
* Added `with torch.no_grad()` during evaluation.
* Early assertion for out‑of‑range labels to avoid CUDA "device‑side assert".
* Automatic experiment‐folder creation and checkpoint housekeeping.
* Tunable visualisation (`max_viz`).
* Cleaner logging via TensorBoard + HTMLFigureLogger.

Run directly or import `evaluate()` / `main()` elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.load_config import load_cfg
from helper.image_seger import ConvSegHead
from distance_contrast_helper import HTMLFigureLogger
from lib.datasets.dataset4seghead import get_dataset, get_valid_dataset

# -----------------------------------------------------------------------------
# Evaluation helper
# -----------------------------------------------------------------------------

def evaluate(
    loader: DataLoader,
    seg_head: nn.Module,
    loss_fn: nn.Module,
    writer: SummaryWriter,
    img_logger: HTMLFigureLogger,
    epoch: int,
    device: torch.device,
    num_classes: int = 8,
    mode: str = "valid",
    max_viz: int = 4,
):
    """Compute mean loss on *loader* and log a handful of slices."""

    seg_head.eval()
    losses: list[float] = []
    gt_slices: list[np.ndarray] = []
    pred_slices: list[np.ndarray] = []

    with torch.no_grad():
        for inp, lbl in tqdm(loader, desc=f"[{mode}]"):
            inp, lbl = inp.to(device), lbl.to(device)
            mask = lbl >= 0  # ignore unlabeled voxels

            # safety ‑‑ avoid CUDA asserts
            if torch.any(lbl[mask] >= num_classes):
                raise ValueError(
                    f"Found label id >= num_classes in {mode} set (epoch {epoch})"
                )

            logits = seg_head(inp)  # [B, C, D, H, W]
            loss = loss_fn(logits.permute(0, 2, 3, 4, 1)[mask], lbl[mask])
            losses.append(loss.item())

            if len(gt_slices) < max_viz:  # collect slices for visualisation
                probs = F.softmax(logits.squeeze(0), dim=0)
                pred = torch.argmax(probs, dim=0) + 1  # 1‑based for colour map
                z_mid = pred.shape[0] // 2

                gt_slices.append((lbl.squeeze(0)[z_mid] + 1).cpu().numpy())
                pred_slices.append((pred[z_mid] * mask.squeeze(0)[z_mid]).cpu().numpy())

    mean_loss = float(np.mean(losses))
    writer.add_scalar(f"Loss/{mode}", mean_loss, epoch)

    # quick visual
    cmap = plt.get_cmap("nipy_spectral", num_classes)
    ncols = len(gt_slices)
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 6))
    for i in range(ncols):
        axes[0, i].imshow(gt_slices[i], cmap=cmap, vmin=1, vmax=num_classes)
        axes[0, i].axis("off")
        axes[1, i].imshow(pred_slices[i], cmap=cmap, vmin=1, vmax=num_classes)
        axes[1, i].axis("off")
    fig.tight_layout()
    img_logger.add_figure(f"{mode}/pred", fig, global_step=epoch)

    seg_head.train()
    return mean_loss


# thin wrapper to mirror the original call signature -------------------------

def seg_valid_training_data(
    img_logger: HTMLFigureLogger,
    writer: SummaryWriter,
    train_loader: DataLoader,
    seg_head: nn.Module,
    epoch: int,
    loss_fn: nn.Module,
):
    """For backward compatibility with legacy code."""

    dev = next(seg_head.parameters()).device
    return evaluate(
        train_loader,
        seg_head,
        loss_fn,
        writer,
        img_logger,
        epoch,
        device=dev,
        mode="train",
    )


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------

def main(cfg_path: str = "config/seghead.yaml"):
    args = load_cfg(cfg_path)

    # reproducibility -----------------------------------------------------------------
    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # experiment dirs -----------------------------------------------------------------
    exp_root = Path("outs") / "seg_head"
    exp_name = f"test6_level{args.feats_level}_avg_pool{args.feats_avg_kernel}"
    work_dir = exp_root / exp_name
    model_dir = work_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    # data -----------------------------------------------------------------------------
    train_loader = DataLoader(
        get_dataset(args),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=getattr(args, "workers", 4),
    )
    valid_loader = DataLoader(
        get_valid_dataset(args),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=2,
    )

    # logging --------------------------------------------------------------------------
    writer = SummaryWriter(log_dir=str(work_dir))
    val_fig_logger = HTMLFigureLogger(work_dir, html_name="seg_valid.html")
    trn_fig_logger = HTMLFigureLogger(work_dir, html_name="seg_train.html")

    # model / optimisation -------------------------------------------------------------
    seg_head = ConvSegHead(12, args.num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(seg_head.parameters(), lr=args.lr_start)

    # schedule -------------------------------------------------------------------------
    num_epochs = args.num_epochs
    valid_every = max(1, args.valid_very_epoch)
    save_every = max(1, getattr(args, "save_every", 1))
    train_eval_every = 4 * valid_every

    # training -------------------------------------------------------------------------
    for epoch in range(args.start_epoch, num_epochs):
        seg_head.train()
        train_losses = []

        for inp, lbl in train_loader:
            inp, lbl = inp.to(device), lbl.to(device)
            mask = lbl >= 0
            if torch.any(lbl[mask] >= args.num_classes):
                raise ValueError("Label out of range – check dataset vs num_classes")

            optim.zero_grad()
            logits = seg_head(inp)
            loss = loss_fn(logits.permute(0, 2, 3, 4, 1)[mask], lbl[mask])
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        mean_tr_loss = float(np.mean(train_losses))
        writer.add_scalar("Loss/train", mean_tr_loss, epoch)
        print(f"[Epoch {epoch:03d}] train loss: {mean_tr_loss:.4f}")

        # validation ------------------------------------------------------------------
        if epoch % valid_every == 0:
            evaluate(
                valid_loader,
                seg_head,
                loss_fn,
                writer,
                val_fig_logger,
                epoch,
                device,
                num_classes=args.num_classes,
                mode="valid",
            )

        # optional evaluation on training set -----------------------------------------
        if epoch % train_eval_every == 0:
            evaluate(
                train_loader,
                seg_head,
                loss_fn,
                writer,
                trn_fig_logger,
                epoch,
                device,
                num_classes=args.num_classes,
                mode="train",
            )

        # checkpoint ------------------------------------------------------------------
        if (epoch + 1) % save_every == 0:
            ckpt_path = model_dir / f"model_epoch_{epoch + 1:03d}.pth"
            torch.save(seg_head.state_dict(), ckpt_path)
            print(f"Saved: {ckpt_path}")

    val_fig_logger.finalize()
    trn_fig_logger.finalize()
    writer.close()


if __name__ == "__main__":
    main()
