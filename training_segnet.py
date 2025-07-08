import itertools                      # NEW
import time, zarr, math, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.load_config import load_cfg
from helper.image_seger import ConvSegHead
from lib.arch.ae import build_final_model        # already in your script
from distance_contrast_helper import HTMLFigureLogger
from lib.datasets.dataset4seghead import get_dataset, get_valid_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg_path = 'config/seghead.yaml'
args = load_cfg(cfg_path)

args.filters = [6,12,24]
args.mlp_filters =[24,16,12]
# %% ---------- models ---------------------------------------------------------
C            = 12          # channels returned by cmpsd_model
num_classes  = 8
cmpsd_model  = build_final_model(args).to(device)     # NEW
seg_head     = ConvSegHead(C, num_classes).to(device)

cmpsd_model.train()
seg_head.train()

optimizer = torch.optim.Adam(
    itertools.chain(cmpsd_model.parameters(), seg_head.parameters()),
    lr=args.lr_start
)
loss_fn = nn.CrossEntropyLoss()

# %% ---------- validation helper ---------------------------------------------
def seg_valid(img_logger, valid_loader, cmpsd_model, seg_head, epoch):
    cmpsd_model.eval()
    seg_head.eval()

    valid_loss, gt_maskes, pred_maskes = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            feats  = cmpsd_model(inputs)          # NEW
            logits = seg_head(feats)              # CHANGED

            mask  = labels >= 0
            loss  = loss_fn(
                logits.permute(0, 2, 3, 4, 1)[mask],   # [N_vox,K]
                labels[mask]
            )
            valid_loss.append(loss.item())

            probs = F.softmax(logits, dim=1)
            pred  = torch.argmax(probs, dim=1) + 1
            pred  = pred * mask

            pred_np  = pred.cpu().numpy()
            label_np = (labels + 1).cpu().numpy()
            z_mid    = pred_np.shape[1] // 2

            gt_maskes.extend(label_np[:, z_mid, :, :])
            pred_maskes.extend(pred_np[:, z_mid, :, :])

    num_classes = 8                       # your label count
    cmap        = plt.get_cmap('nipy_spectral', num_classes)
    max_cols    = 4                       # ≤ 4 columns in the gallery

    # --- 1) merge GT + prediction for visualisation ----------------------------
    combined_imgs = []
    for gt, pred in zip(gt_maskes, pred_maskes):
        # Put GT on the left, prediction on the right
        combined = np.hstack((gt, pred))         # shape: (H, 2*W)
        combined_imgs.append(combined)

    # --- 2) build a grid -------------------------------------------------------
    n_imgs  = len(combined_imgs)
    n_cols  = min(max_cols, n_imgs)
    n_rows  = math.ceil(n_imgs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,figsize=(4 * n_cols, 4 * n_rows),squeeze=False)

    for idx, img in enumerate(combined_imgs):
        r, c = divmod(idx, n_cols)
        axes[r, c].imshow(img, cmap=cmap, vmin=1, vmax=num_classes)
        axes[r, c].set_title(f"Sample {idx}", fontsize=10)
        axes[r, c].axis("off")

    # Hide any empty cells (when images % max_cols ≠ 0)
    for idx in range(n_imgs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.tight_layout()

    img_logger.add_figure('gt/pred',fig,global_step = epoch)
    cmpsd_model.train()
    seg_head.train()

    return sum(valid_loss) / len(valid_loss)

# %% ---------- data loaders & loggers (unchanged) -----------------------------

dataset        = get_dataset(args)
loader         = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)
valid_dataset  = get_valid_dataset(args)
valid_loader   = DataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=False)

exp_save_dir   = 'outs/seg_head'
# exp_name       = f"test8_level{args.feats_level}_avg_pool{args.feats_avg_kernel}"
exp_name       = f"training_from_scratch_level{args.feats_level}_avg_pool{args.feats_avg_kernel}"
model_save_dir = f"{exp_save_dir}/{exp_name}"
os.makedirs(model_save_dir, exist_ok=True)

writer          = SummaryWriter(f'{exp_save_dir}/{exp_name}')
img_logger      = HTMLFigureLogger(exp_save_dir + '/' + exp_name, html_name="seg_valid_result.html")
train_img_logger= HTMLFigureLogger(exp_save_dir + '/' + exp_name, html_name="train_seg_valid_result.html")

# %% ---------- training loop --------------------------------------------------
for epoch in tqdm(range(args.num_epochs)):
    train_loss = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        mask = labels >= 0

        optimizer.zero_grad()
        feats  = cmpsd_model(inputs)          # NEW
        logits = seg_head(feats)              # CHANGED

        loss = loss_fn(logits.permute(0,2,3,4,1)[mask], labels[mask])
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    avg_loss = sum(train_loss) / len(train_loss)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"[Epoch {epoch}] train loss: {avg_loss:.4f}")

    if epoch % args.valid_very_epoch == 0:
        v_loss = seg_valid(img_logger, valid_loader, cmpsd_model, seg_head, epoch)
        writer.add_scalar('Loss/valid', v_loss, epoch)

    if (epoch % (4 * args.valid_very_epoch)) == 0:
        seg_valid(train_img_logger, loader, cmpsd_model, seg_head, epoch)

    if (epoch + 1) % 50 == 0:
        save_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'cmpsd_model': cmpsd_model.state_dict(),
            'seg_head'   : seg_head.state_dict()
        }, save_path)
        print(f"Saved models to {save_path}")

img_logger.finalize()
train_img_logger.finalize()