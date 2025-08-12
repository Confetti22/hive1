# train_seghead_main.py
import argparse
from pathlib import Path
import shutil

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from config.load_config import load_cfg
from lib.arch.seg import ConvSegHead
from train_seghead_helper import accuracy, seg_valid
from lib.loss.ce_dice_combo import ComboLoss
from lib.utils.loss_utils import compute_class_weights_from_dataset
from lib.datasets.dataset4seghead import get_dataset, get_valid_dataset
from distance_contrast_helper import HTMLFigureLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='config/seghead.yaml', help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def save_checkpoint(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    return ckpt.get("epoch", 0) + 1


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    model.train()
    train_loss = []
    total_top1 = []
    total_top3= []

    for input, label in tqdm(loader, desc=f"Train Epoch {epoch}", leave=False):
        input, label = input.to(device), label.to(device)
        mask = label >= 0

        optimizer.zero_grad()
        logits = model(input)
        logits_flat = logits.permute(0, 2, 3, 4, 1)[mask]
        labels_flat = label[mask]

        loss = criterion(logits_flat, labels_flat)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        top1, top3 = accuracy(logits_flat, labels_flat, topk=(1, 3))
        total_top1.append(top1.item())
        total_top3.append(top3.item())
        

    avg_loss = sum(train_loss) / len(train_loss)
    avg_top1 = sum(total_top1)/len(total_top1)
    avg_top3 = sum(total_top3)/len(total_top3)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("top1_acc/train", avg_top1, epoch)
    writer.add_scalar("top3_acc/train", avg_top3, epoch)
    return avg_loss


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    exp_name = f"continuefeats_level{cfg.feats_level}_avg_pool{cfg.feats_avg_kernel}__focal_combo_loss" 
    run_dir = Path("outs") / "seg_head" / exp_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.cfg, run_dir / "config.yaml")

    writer = SummaryWriter(log_dir=run_dir / "tb")
    img_logger = HTMLFigureLogger(log_dir=run_dir, html_name="seg_valid_result.html")
    train_img_logger = HTMLFigureLogger(log_dir=run_dir, html_name="train_seg_valid_result.html")

    dataset = get_dataset(cfg)
    valid_dataset = get_valid_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    class_weights = compute_class_weights_from_dataset(dataset, cfg.num_classes)
    loss_fn = ComboLoss(class_weights=class_weights, focal=cfg.get("use_focal", True))

    model = ConvSegHead(cfg.in_channels, cfg.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_start)

    start_epoch = 0
    if args.ckpt:
        start_epoch = load_checkpoint(args.ckpt, model, optimizer)
        print(f"Resumed from {args.ckpt} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.num_epochs):
        avg_loss = train_one_epoch(model, loader, loss_fn, optimizer, device, epoch, writer)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        if epoch % cfg.valid_very_epoch == 0:
            val_loss ,avg_top1, avg_top3 = seg_valid(img_logger, valid_loader, model, epoch, device=device, loss_fn=loss_fn)
            writer.add_scalar("Loss/valid", val_loss, epoch)
            writer.add_scalar("top1_acc/valid", avg_top1, epoch)
            writer.add_scalar("top3_acc/valid", avg_top3, epoch)

        if epoch % (4 * cfg.valid_very_epoch) == 0:
            seg_valid(train_img_logger, loader, model, epoch, device=device, loss_fn=loss_fn)

        if (epoch + 1) % cfg.save_very_epoch == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pth"
            save_checkpoint({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    img_logger.finalize()
    train_img_logger.finalize()
    writer.close()


if __name__ == "__main__":
    main()