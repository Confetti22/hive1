#!/usr/bin/env python
"""
Minimal, fast demo of WarmupCosineLR + TensorBoard LR logging.

Run:
    python warmup_cosine_lr_min_demo.py --logdir runs/warmcosine_min
Then (in another shell):
    tensorboard --logdir runs

You'll see a scalar curve named "lr" that shows warm‑up then cosine decay.
This is intentionally tiny so you can confirm everything works quickly.
"""

import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

import argparse
import math
from pathlib import Path

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from lib.core.scheduler import    WarmupCosineLR  


def make_toy_data(n=128, in_dim=10, out_dim=1, device="cpu"):
    x = torch.randn(n, in_dim, device=device)
    true_w = torch.randn(in_dim, out_dim, device=device)
    y = x @ true_w + 0.1 * torch.randn(n, out_dim, device=device)
    return x, y


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny model so training is instant
    model = nn.Linear(args.in_dim, args.out_dim).to(device)
    opt = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)  # max LR
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,     # length of the first cycle (epochs or steps—your call)
        T_mult=2,   # multiply the length each time: 10 → 20 → 40 …
        eta_min=1e-4,  # minimum LR at the valley of each cosine
    )
    # sched = WarmupCosineLR(
    #     opt,
    #     warmup_epochs=args.warmup_epochs,
    #     max_epochs=args.epochs,
    # )

    save_dir = f"{args.logdir}/tooth2"
    os.makedirs(save_dir,exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    print(f"TensorBoard logdir: {writer.log_dir}")

    x, y = make_toy_data(n=args.n_samples, in_dim=args.in_dim, out_dim=args.out_dim, device=device)

    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        # Step scheduler at *start* of epoch so the LR used this epoch reflects schedule.
        sched.step()
        lr = sched.get_last_lr()[0]
        writer.add_scalar("lr", lr, epoch)

        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()

        writer.add_scalar("loss", float(loss.detach().cpu()), epoch)
        print(f"epoch {epoch:02d}  lr={lr:.6f}  loss={loss.item():.4f}")

    writer.close()



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Minimal WarmupCosineLR demo")
    p.add_argument("--logdir", type=str, default="runs/warmcosine_min")
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--warmup-epochs", dest="warmup_epochs", type=int, default=30)
    p.add_argument("--cosine-epochs", dest="cosine_epochs", type=int, default=100,
                   help="length of cosine phase; <epochs-warmup_epochs> shrinks period")
    p.add_argument("--base-lr", dest="base_lr", type=float, default=0.1)
    p.add_argument("--n-samples", dest="n_samples", type=int, default=128)
    p.add_argument("--in-dim", dest="in_dim", type=int, default=10)
    p.add_argument("--out-dim", dest="out_dim", type=int, default=1)
    args = p.parse_args()
    train(args)
