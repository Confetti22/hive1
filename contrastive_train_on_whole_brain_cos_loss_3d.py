#!/usr/bin/env python3
"""
Single-GPU PyTorch **contrastive-learning** template  
Usage
-----
```bash
# fresh run
python train_contrastive_template.py --cfg config/rm009.yaml

# resume from a checkpoint
python train_contrastive_template.py --cfg config/rm009.yaml \
    --ckpt outs/postopk_8_…/checkpoints/epoch_010.pth
```
"""
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import zarr

# ──────────────────────────────────────────────────────────────────────────────
# Project-specific helpers
# ──────────────────────────────────────────────────────────────────────────────
from config.load_config import load_cfg
from helper.contrastive_train_helper import (
    cos_loss_topk,
    get_rm009_eval_data,
    valid_from_roi,
    MLP,
    Contrastive_dataset_3d,
)
from lib.arch.ae import build_final_model, load_compose_encoder_dict

# =============================================================================
# Utility helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive 3-D feature training")
    p.add_argument("--cfg", type=str, default='config/rm009.yaml', help="Path to YAML config")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint to resume")
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu | cuda:0 …")
    return p.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: optim.Optimizer | None = None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    return ckpt.get("epoch", 0) + 1  # resume at *next* epoch


# =============================================================================
# Training helpers
# =============================================================================

def proxy_accuracy(pos_cos: torch.Tensor, neg_cos: torch.Tensor) -> float:
    """Fraction where positive similarity > negative (just a rough metric)."""
    return (pos_cos > neg_cos).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    *,
    n_views: int,
    pos_weight_ratio: float,
    only_pos: bool,
):
    model.train()
    run_loss =  0.0
    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
        batch = torch.cat(batch, dim=0).to(device)  # [B*n_views, C]
        optimizer.zero_grad()
        feats = model(batch).squeeze()
        loss, pos_cos, neg_cos = cos_loss_topk(
            features=feats,
            n_views=n_views,
            pos_weight_ratio=pos_weight_ratio,
            only_pos=only_pos,
        )
        loss.backward()
        optimizer.step()

        run_loss += loss.item()

        it = epoch * len(loader) + step
        writer.add_scalar("train/loss_iter", loss.item(), it)
        writer.add_scalar("train/pos_cos", pos_cos.item(), it)
        writer.add_scalar("train/neg_cos", neg_cos.item(), it)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it)

    n_steps = len(loader)
    writer.add_scalar("train/loss_epoch", run_loss / n_steps, epoch)


@torch.no_grad()
def validate(model: nn.Module, cmpsd_model: nn.Module, eval_data,
             device: torch.device, epoch: int, writer: SummaryWriter, *,
             cnn_ckpt: Path, dims):
    # refresh composite encoder weights
    load_compose_encoder_dict(cmpsd_model, str(cnn_ckpt), mlp_weight_dict=model.state_dict(), dims=dims)
    metrics = valid_from_roi(cmpsd_model, epoch, eval_data, writer)
    for k, v in metrics.items():
        writer.add_scalar(f"val/{k}", v, epoch)


# =============================================================================
# Main
# =============================================================================

def main():
    cli = parse_args()
    cfg = load_cfg(cli.cfg)

    # ---------------- seed & device ---------------- #
    # set_seed(cfg.get("seed", 42))
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    # --------------- experiment folder ------------- #
    avg_pool = cfg.avg_pool_size[0] if "avg_pool_size" in cfg else 8
    exp_name = cfg.get(
        "exp_name",
        f"postopk_{avg_pool}_numparis{cfg.num_pairs}_batch{cfg.batch_size}_nview{cfg.n_views}_d_near{cfg.d_near}_shuffle{cfg.shuffle_very_epoch}",
    )
    run_dir = Path(cfg.get("exp_save_dir", "outs")) / exp_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(cli.cfg, run_dir / "config.yaml")

    # ---------------- data prefixes ---------------- #
    data_prefix = Path("/share/home/shiqiz/data" if cfg.e5 else "/home/confetti/data")
    feats_name = "feats_z16176_z16299C4.zarr"
    feats_map = zarr.open_array(str(data_prefix / "rm009" / feats_name), mode="r")

    # initial dataset/loader
    def make_loader() -> DataLoader:
        ds = Contrastive_dataset_3d(
            feats_map,
            d_near=cfg.d_near,
            num_pairs=cfg.num_pairs,
            n_view=cfg.n_views,
            verbose=False,
        )
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=cfg.get("num_workers", 4), pin_memory=True)

    loader = make_loader()

    # ---------------- models ----------------------- #
    model = MLP(cfg.mlp_filters).to(device)
    cmpsd_model = build_final_model(cfg).to(device)
    cmpsd_model.eval()

    # --------------- optim & sched ----------------- #
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr_start", 5e-4))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.get("lr_gamma", 0.95))

    # --------------- resume logic ------------------ #
    start_epoch = 0
    if cli.ckpt:
        start_epoch = load_checkpoint(cli.ckpt, model, optimizer)
        print(f"[INFO] Resumed from {cli.ckpt} (next epoch = {start_epoch})")
    elif cfg.get("re_use", False):
        ckpt_pth = Path(cfg.reuse_ckpt_path)
        if ckpt_pth.exists():
            start_epoch = load_checkpoint(ckpt_pth, model, optimizer)
            print(f"[INFO] Re-used checkpoint {ckpt_pth} (next epoch = {start_epoch})")
        else:
            raise FileNotFoundError(f"re_use is True but {ckpt_pth} not found")

    # ---------------- logging ---------------------- #
    writer = SummaryWriter(log_dir=run_dir / "tb")

    # ---------------- validation data -------------- #
    eval_data = get_rm009_eval_data(E5=cfg.e5)
    cnn_ckpt = data_prefix / "weights" / "rm009_3d_ae_best.pth"

    # ---------------- training loop ---------------- #
    n_epochs = cfg.num_epochs
    ckpt_every = cfg.save_very_epoch
    shuffle_every = cfg.shuffle_very_epoch
    valid_every = cfg.valid_very_epoch

    for epoch in range(start_epoch, n_epochs):

        train_one_epoch(model, loader, optimizer, device, epoch, writer,
                        n_views=cfg.n_views, pos_weight_ratio=cfg.pos_weight_ratio,
                        only_pos=True)
        scheduler.step()
        # validation
        if (epoch + 1) % valid_every == 0 or epoch + 1 == n_epochs:
            validate(model, cmpsd_model, eval_data, device, epoch, writer,cnn_ckpt=cnn_ckpt, dims=cfg.dims)

        # reshuffle dataset
        if (epoch + 1) % shuffle_every == 0:
            loader = make_loader()

        # checkpoint
        if (epoch + 1) % ckpt_every == 0 or epoch + 1 == n_epochs:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pth"
            save_checkpoint({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"[INFO] Saved checkpoint → {ckpt_path.relative_to(Path.cwd())}")

    # ---------------- finalize --------------------- #
    writer.close()
    print("[Done] Training complete.")


if __name__ == "__main__":
    main()
