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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from pathlib import Path
import zarr

# ──────────────────────────────────────────────────────────────────────────────
# Project-specific helpers
# ──────────────────────────────────────────────────────────────────────────────
from config.load_config import load_cfg
from helper.contrastive_train_helper import (
    cos_loss_topk,
    cos_loss,
    get_t11_eval_data,
    MLP,
    Contrastive_dataset_3d,
    load_checkpoint,
    save_checkpoint,
    log_layer_embeddings,
)
from lib.arch.ae import build_final_model, load_compose_encoder_dict
from lib.core.scheduler import WarmupCosineLR

# =============================================================================
# Utility helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive 3-D feature training")
    p.add_argument("--cfg", type=str, default='config/t11_3d.yaml', help="Path to YAML config")
    p.add_argument("--ckpt", type=str, default=None,)
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu | cuda:0 …")
    return p.parse_args()

# =============================================================================
# Training helpers
# =============================================================================

from collections import defaultdict
from functools import partial
from typing import Sequence, Tuple, Union, Literal, List
Arr   = Union[np.ndarray, torch.Tensor]
Array = Union[Arr, Sequence[Arr]]   # single array or list/tuple of arrays

FEATURE_STORE = defaultdict(list)      # {layer_name: [Tensor, ...]}
HOOK_HANDLES  = []                     # so we can remove them cleanly
LAYER_ORDER   = []

def _hook(layer_name, module, inp, out):
    """
    out : Tensor shape [B, C, D, H, W] or [B, C, H, W]
    We keep it on CPU to avoid GPU memory churn.
    """
    FEATURE_STORE[layer_name].append(out.detach().cpu())

def register_hooks(model, prefix=""):
    """
    Recursively register a forward hook on *leaf* modules that have weights.
    The prefix guarantees unique names.
    """
    for name, m in model.named_children():
        full_name = f"{prefix}{name}"
        # is leaf (= no children) AND has parameters → treat as a layer of interest
        if sum(1 for _ in m.children()) == 0 and sum(p.numel() for p in m.parameters()) > 0:
            LAYER_ORDER.append(full_name)  # <-- capture order
            HOOK_HANDLES.append(m.register_forward_hook(partial(_hook, full_name)))
        else:
            register_hooks(m, f"{full_name}.")



def proxy_accuracy(pos_cos: torch.Tensor, neg_cos: torch.Tensor) -> float:
    """Fraction where positive similarity > negative (just a rough metric)."""
    return (pos_cos > neg_cos).float().mean().item()



def valid_from_roi(model, epoch, eval_data, writer):
    """Evaluate a model on a list of ROIs.

    Works with feature tensors of shape:
        • (C, H, W)                      – old behaviour
        • (C, D, H, W)                  – channel-first 3-D
        • (D, H, W, C)                  – channel-last 3-D
    For 3-D inputs, the middle depth slice (z = D//2) is used for PCA/t-SNE.
    """
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx, data_dic in enumerate(eval_data):
        roi      = data_dic['img']          # numpy (H,W) or (D,H,W)
        label    = data_dic['label']
        if len(label.shape)==2:
            label = label[np.newaxis,:]

        inp = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0).float().to(device)
        _ = model(inp).detach().cpu().numpy().squeeze() # np.ndarray

        #current impl does not support label for different roi
        log_layer_embeddings(
            FEATURE_STORE,
            writer=writer,
            epoch=epoch,
            label_volume=label,  # numpy array
            layer_order=LAYER_ORDER,       # from hook registration
            max_layers=15,
            mode="both",                   # <- t-SNE + UMAP stacked
            tsne_kwargs=dict(perplexity=20),
            umap_kwargs=dict(n_neighbors=30, min_dist=0.05,random_state=42,),
            valid_img_id =idx,
        )
        


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
    loss_fn,
    only_pos: bool = True, # only_pos controls whether to only use topk on positive_pairs in cos_loss_topk, do not infect coss_loss function
):
    model.train()
    run_loss =  0.0
    pos_cos_loss=  0.0
    neg_cos_loss =  0.0
    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
        batch = torch.cat(batch, dim=0).to(device)  # [B*n_views, C]
        optimizer.zero_grad()
        feats = model(batch).squeeze()
        loss, pos_cos, neg_cos = loss_fn(
            features=feats,
            n_views=n_views,
            pos_weight_ratio=pos_weight_ratio,
            only_pos=only_pos,
        )
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        pos_cos_loss +=pos_cos.item()
        neg_cos_loss +=neg_cos.item()

    n_steps = len(loader)
    writer.add_scalar("train/loss", run_loss / n_steps, epoch)
    writer.add_scalar("train/pos_cos", pos_cos_loss/ n_steps, epoch)
    writer.add_scalar("train/neg_cos", neg_cos_loss/ n_steps, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)



@torch.no_grad()
def validate(model: nn.Module, cmpsd_model: nn.Module, eval_data,
             device: torch.device, epoch: int, writer: SummaryWriter, *,
             cnn_ckpt: Path, dims,):
    #discard the last eval layer embeddings
    FEATURE_STORE.clear()
    # refresh composite encoder weights
    load_compose_encoder_dict(cmpsd_model, str(cnn_ckpt), mlp_weight_dict=model.state_dict(), dims=dims)
    valid_from_roi(cmpsd_model, epoch, eval_data, writer)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    cfg.e5 = True 
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sample_name ='t1779'
    # --------------- experiment folder ------------- #
    avg_pool = cfg.avg_pool_size[0] if "avg_pool_size" in cfg else 8
    exp_name = f"test_on_rhems_numparis{cfg.num_pairs}_batch{cfg.batch_size}_nview{cfg.n_views}_d_near{cfg.d_near}_shuffle{cfg.shuffle_very_epoch}_csine_anllr_"
    # run_dir = Path("outs") /'contrastive_run_rm009'/'rm009_v1'/ exp_name
    run_dir = Path("outs") /f'contrastive_run_{sample_name}'/ exp_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.cfg, run_dir / "config.yaml")

    # ---------------- data prefixes ---------------- #
    data_prefix = Path("/share/home/shiqiz/data" if cfg.e5 else "/home/confetti/data")
    feats_name = "ae_feats_nissel_l3_avg8_rhemisphere.zarr"
    feats_map = zarr.open_array(str(data_prefix / sample_name / feats_name), mode="r")
    #load all the feats into memory if it can, will accelate indexing feats
    print(f"{feats_map.shape= }")
    feats_map = feats_map[:]

    ds = Contrastive_dataset_3d(feats_map,d_near=cfg.d_near,num_pairs=cfg.num_pairs,n_view=cfg.n_views,verbose=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, pin_memory=True)

    # ---------------- models ----------------------- #
    level_key = 'l3'
    filters_map={'l1':[32,24,12,12],'l2':[64,32,24,12],'l3':[96,64,32,12]}
    cnn_filters_map ={'l1':[32],'l2':[32,64],'l3':[32,64,96]}
    cnn_kernler_size_map ={'l1':[5],'l2':[5,5],'l3':[5,5,3]}
    cfg.filters = cnn_filters_map[level_key]
    cfg.kernel_size =cnn_kernler_size_map[level_key]
    cfg.mlp_filters = filters_map[level_key]
    cfg.avg_pool_size = [8,8,8] 
    cfg.last_encoder = True 
    cfg.batch_size = 4096
    loss_fn = cos_loss

    model = MLP(cfg.mlp_filters).to(device)

    #cmpsd_model for eval features in cnn and mlp, register hooks to it
    cmpsd_model = build_final_model(cfg).to(device)
    cmpsd_model.eval()
    register_hooks(cmpsd_model.cnn_encoder, "cnn.")
    register_hooks(cmpsd_model.mlp_encoder, "mlp.")
    print("Registered layers:", LAYER_ORDER)

    # --------------- optim & sched ----------------- #
    opt = optim.Adam(model.parameters(), lr=4e-5)
    warmup_epochs= 20

    sched = WarmupCosineLR(opt,warmup_epochs=warmup_epochs,max_epochs=cfg.num_epochs)
    

    # --------------- resume logic ------------------ #
    start_epoch = 0
    if args.ckpt:
        start_epoch = load_checkpoint(args.ckpt, model, opt)
        print(f"[INFO] Resumed from {args.ckpt} (next epoch = {start_epoch})")

    # ---------------- logging ---------------------- #
    writer = SummaryWriter(log_dir=run_dir / "tb")

    # ---------------- validation data -------------- #
    eval_data = get_t11_eval_data(E5=cfg.e5,img_no_list=[1,3])
    cnn_ckpt = data_prefix / "weights" / "t11_3d_ae_best2.pth"

    # ---------------- training loop ---------------- #
    n_epochs = cfg.num_epochs
    ckpt_every = cfg.save_very_epoch
    shuffle_every = cfg.shuffle_very_epoch
    valid_every = cfg.valid_very_epoch

    for epoch in range(start_epoch, n_epochs):

        train_one_epoch(model, loader, opt, device, epoch, writer,
                        n_views=cfg.n_views, pos_weight_ratio=cfg.pos_weight_ratio,loss_fn=loss_fn,
                        only_pos=only_pos)
        sched.step()
        # validation
        if (epoch + 1) % valid_every == 0 or epoch + 1 == n_epochs or epoch ==0:
            validate(model, cmpsd_model, eval_data, device, epoch, writer,cnn_ckpt=cnn_ckpt, dims=cfg.dims,)

        # reshuffle dataset
        if (epoch + 1) % shuffle_every == 0:
            ds = Contrastive_dataset_3d(feats_map,d_near=cfg.d_near,num_pairs=cfg.num_pairs,n_view=cfg.n_views,verbose=False)
            loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, pin_memory=True)

        # checkpoint
        if (epoch + 1) % ckpt_every == 0 or epoch + 1 == n_epochs:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pth"
            save_checkpoint({"model": model.state_dict(), "optim": opt.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"[INFO] Saved checkpoint → {ckpt_path}")

    # ---------------- finalize --------------------- #
    writer.close()
    print("[Done] Training complete.")


if __name__ == "__main__":
    main()
