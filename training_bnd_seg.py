#%%
import itertools                      
from typing import Sequence, Union
from pathlib import Path
import time, zarr, math, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tifffile as tif
from config.load_config import load_cfg
from helper.contrastive_train_helper import log_layer_embeddings 
from lib.arch.ae import build_final_model        # already in your script
from distance_contrast_helper import HTMLFigureLogger
from lib.datasets.dataset4seghead import get_dataset, get_valid_dataset
from lib.core.metric import accuracy

from lib.arch.ae import load_ae2encoder,load_encoder2encoder,load_mlp_ckpt_to_convmlp
from lib.arch.seg import ConvSegHead
from collections import defaultdict
from functools import partial
from torchsummary import summary
import math

from collections import defaultdict
from typing import Sequence, Tuple, Union, Literal, List
Arr   = Union[np.ndarray, torch.Tensor]
Array = Union[Arr, Sequence[Arr]]   # single array or list/tuple of arrays

# %% ---------- validation helper ---------------------------------------------
# ───────────────────────── feature hooks ──────────────────────────
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



def bnd_seg_valid(img_logger, valid_loader, cmpsd_model,epoch,valid_first_batch=False,seg_head=None):
    cmpsd_model.eval()
    if seg_head:
        seg_head.eval()

    valid_loss, gt_maskes, pred_maskes = [], [], []
    total_top1 = []

    with torch.no_grad():
        for inputs, masks ,bnd_labels in tqdm(valid_loader):
            inputs, masks,bnd_labels = inputs.to(device), masks.to(device),bnd_labels.to(device)
            valid_mask = masks >= 0
            bnd_labels =bnd_labels[:, None, :, :, None]            # (4, 1, 249, 249, 1) 
            valid_mask =valid_mask[:, None, :, :, None]            # (4, 1, 249, 249, 1) 

            optimizer.zero_grad()
            logits = cmpsd_model(inputs)          # NEW
            if seg_head:
                logits = seg_head(logits)          # NEW
            logits_flat = logits.permute(0, 2, 3, 4, 1)[valid_mask]
            bnd_labels_flat = bnd_labels[valid_mask]

            loss = loss_fn(logits_flat,bnd_labels_flat)

            valid_loss.append(loss.item())

            top1 = accuracy(logits_flat, bnd_labels_flat)
            total_top1.append(top1)

            probs  = torch.sigmoid(logits)
            pred  = (probs >= 0.5).long()
            
            pred = pred.permute(0, 2, 3, 4, 1) 
            pred  = pred * valid_mask

            pred_np  = pred.cpu().squeeze().numpy()
            bnd_label_np = (bnd_labels).cpu().squeeze().numpy()

            gt_maskes.append(bnd_label_np)
            pred_maskes.append(pred_np)

            if valid_first_batch:break

    gt_maskes = np.concatenate([ arr for arr in gt_maskes], axis=0)
    pred_maskes= np.concatenate([ arr for arr in pred_maskes], axis=0)

    num_classes = 3                       # your label count
    # cmap        = plt.get_cmap('nipy_spectral', num_classes)
    cmap = ListedColormap([
    (0, 0, 0, 0),   # label 0 – fully transparent (or pick another color)
    (1, 0, 0, 0),   # label 1 – red
    (0, 0, 1, 0)    # label 2 – blue
    ])
    max_cols    = 4                       # ≤ 4 columns in the gallery

    # --- 1) merge GT + prediction for visualisation ----------------------------
    combined_imgs = []
    for idx ,(gt, pred) in enumerate(zip(gt_maskes, pred_maskes)):
        # Put GT on the left, prediction on the right
        merged = gt.copy()
        merged[pred>0] = 2 
        valid_save_dir = img_logger.images_dir
        tif.imwrite(f"{valid_save_dir}/{idx}_{epoch}.tif",merged)
        combined = np.hstack((gt,merged))
        combined_imgs.append(combined)
    # --- 2) build a grid -------------------------------------------------------
    n_imgs  = len(combined_imgs)
    n_cols  = min(max_cols, n_imgs)
    n_rows  = math.ceil(n_imgs / n_cols)
    h, w = combined_imgs[0].shape        # (H, W) of a single mask

    dpi      = 57                       # logical pixels per inch
    fig_w_in = (w * n_cols) / dpi        # make sure W *cols  ≥ fig_w * dpi
    fig_h_in = (h * n_rows) / dpi        #               same for H

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w_in, fig_h_in),
        dpi=dpi,                          # keep >= original resolution
        squeeze=False
    )

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
    if seg_head:
        seg_head.train()

    avg_valid_loss = sum(valid_loss)/len(valid_loss)
    avg_top1 = sum(total_top1)/len(total_top1)

    return avg_valid_loss, avg_top1

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg_path = 'config/seghead.yaml'
args = load_cfg(cfg_path)
args.e5 = True
data_prefix   = Path("/share/home/shiqiz/data" if args.e5 else "/home/confetti/data")

exp_save_dir   = 'outs/seg_bnd'
exp_name       = f"bnd_seg_scratch_3moduler_level{args.feats_level}_avg_pool{args.feats_avg_kernel}"
# exp_name       = f"bnd_seg_on_stractch"
model_save_dir = f"{exp_save_dir}/{exp_name}"
os.makedirs(model_save_dir, exist_ok=True)

writer          = SummaryWriter(f'{exp_save_dir}/{exp_name}')
img_logger      = HTMLFigureLogger(exp_save_dir + '/' + exp_name, html_name="seg_valid_result.html")
train_img_logger= HTMLFigureLogger(exp_save_dir + '/' + exp_name, html_name="train_seg_valid_result.html")


args.filters = [32,64]
args.last_encoder= len(args.filters)==3

args.mlp_filters =[64,32,24,12]
args.data_path_dir = f"{data_prefix}/rm009/boundary_seg/rois"
args.mask_path_dir = f"{data_prefix}/rm009/boundary_seg/masks"
args.bnd_path_dir = f"{data_prefix}/rm009/boundary_seg/bnd_masks"
args.valid_data_path_dir = f"{data_prefix}/rm009/boundary_seg/valid_rois"
args.valid_mask_path_dir = f"{data_prefix}/rm009/boundary_seg/valid_masks"
args.valid_bnd_path_dir = f"{data_prefix}/rm009/boundary_seg/valid_bnd_masks"
args.e5_data_path_dir = f"{data_prefix}/rm009/boundary_seg/rois"
args.e5_mask_path_dir = f"{data_prefix}/rm009/boundary_seg/masks"
args.e5_bnd_path_dir = f"{data_prefix}/rm009/boundary_seg/bnd_masks"
args.e5_valid_data_path_dir = f"{data_prefix}/rm009/boundary_seg/valid_rois"
args.e5_valid_mask_path_dir = f"{data_prefix}/rm009/boundary_seg/valid_masks"
args.e5_valid_bnd_path_dir = f"{data_prefix}/rm009/boundary_seg/valid_bnd_masks"
args.feats_level = len(args.filters)
args.feats_avg_kernel = 8

cmpsd_model = build_final_model(args).to(device)

cnn_ckpt_path = data_prefix / "weights" / "rm009_3d_ae_best.pth"
mlp_ckpt_pth = "outs/contrastive_run_rm009/rm009_v1/l2_avg8_roi_postopk_numparis16384_batch4096_nview4_d_near6_shuffle20_csine_anllr/checkpoints/epoch_8300.pth" 

# load_encoder2encoder(cmpsd_model.cnn_encoder, cnn_ckpt_path)
# cmpsd_model.cnn_encoder.requires_grad_(False)   # no grads
# cmpsd_model.cnn_encoder.eval()                  # BN/Dropout → inference

# mlp_weights_dict = torch.load(mlp_ckpt_pth)['model']
# load_mlp_ckpt_to_convmlp(cmpsd_model.mlp_encoder,mlp_weight_dict=mlp_weights_dict,dims=3)
# cmpsd_model.mlp_encoder.requires_grad_(False)   # no grads
# cmpsd_model.mlp_encoder.eval()                  # BN/Dropout → inference
# verify it's really frozen
n_frozen = sum(p.numel() for p in cmpsd_model.cnn_encoder.parameters())
print(f"[info] frozen cnn_encoder params: {n_frozen:,}")

C=12
seg_head     = ConvSegHead(C, 1).to(device)

register_hooks(cmpsd_model.mlp_encoder, "mlp.")  # you can add cnn if needed
print("Registered layers:", LAYER_ORDER)

summary(cmpsd_model, (1, *args.input_size))

# Flip the *whole* model back to training mode (the encoder stays in eval() because we set it explicitly)
cmpsd_model.train()
seg_head.train()

args.lr_start = 1e-4
args.lr_end = 1e-5
warmup_epochs = 10
max_epochs = args.num_epochs

trainable_params = filter(lambda p: p.requires_grad, cmpsd_model.parameters())
optimizer = torch.optim.Adam( itertools.chain(cmpsd_model.parameters(), seg_head.parameters()),lr=args.lr_start)
# optimizer = torch.optim.Adam(trainable_params, lr=args.lr_start)


scheduler= torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=20,     # length of the first cycle (epochs or steps—your call)
    T_mult=2,   # multiply the length each time: 10 → 20 → 40 …
)

# %% ---------- data loaders & loggers (unchanged) -----------------------------

dataset        = get_dataset(args,bnd=True)
loader         = DataLoader(dataset, batch_size=6, shuffle=True, drop_last=False)
valid_dataset  = get_valid_dataset(args,bnd=True)
valid_loader   = DataLoader(valid_dataset, batch_size=6, shuffle=False, drop_last=False)

from train_seghead_helper import ComboLoss,compute_class_weights_from_dataset
from lib.loss.l1 import WeightedL1Loss

class_weights = compute_class_weights_from_dataset(dataset, num_classes=2)
# class_weights[0]=0.1
# class_weights[1]=0.9
loss_fn = WeightedL1Loss(class_weights.to(device),reduction='mean', logits=True) 
# loss_fn= ComboLoss(weight_ce=1.0,
#                       weight_dice=1.0,
#                       smooth=1e-6,
#                       class_weights= 20,   # pos_weight (or α) if desired
#                       focal=True,
#                       binary=True)


# %% ---------- training loop --------------------------------------------------
for epoch in tqdm(range(args.num_epochs)):
    train_loss = []
    total_top1 = []
    for inputs, masks,bnd_labels in loader:
        inputs, masks,bnd_labels = inputs.to(device), masks.to(device),bnd_labels.to(device)
        valid_mask = masks >= 0
        bnd_labels =bnd_labels[:, None, :, :, None]            # (4, 1, 249, 249, 1) 
        valid_mask =valid_mask[:, None, :, :, None]            # (4, 1, 249, 249, 1) 

        optimizer.zero_grad()
        logits = cmpsd_model(inputs)          # NEW
        logits = seg_head(logits)              # CHANGED
        
        logits_flat = logits.permute(0, 2, 3, 4, 1)[valid_mask]
        bnd_labels_flat = bnd_labels[valid_mask]

        loss = loss_fn(logits_flat,bnd_labels_flat)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())

        top1= accuracy(logits_flat, bnd_labels_flat)
        total_top1.append(top1)

        FEATURE_STORE.clear()   # free memory

    avg_loss = sum(train_loss) / len(train_loss)
    avg_top1 = sum(total_top1)/len(total_top1)
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar('lr',scheduler.get_last_lr()[0] , epoch)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar("top1_acc/train", avg_top1, epoch)
    print(f"Epoch {epoch:02d} | loss={loss:.4f} | lr={current_lr:.6f}")
    
    if epoch % args.valid_very_epoch == 0:
        v_loss,avg_top1= bnd_seg_valid(img_logger, valid_loader, cmpsd_model, epoch=epoch ,valid_first_batch=True,seg_head=seg_head)
        writer.add_scalar('Loss/valid', v_loss, epoch)
        writer.add_scalar("top1_acc/valid", avg_top1, epoch)
    

    if (epoch % (4 * args.valid_very_epoch)) == 0:
        bnd_seg_valid(train_img_logger, loader, cmpsd_model,  epoch,valid_first_batch=True,seg_head=seg_head)

    if (epoch + 1) % 50 == 0:
        save_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'cmpsd_model': cmpsd_model.state_dict(),
            'seg_head'   : seg_head.state_dict()
        }, save_path)
        print(f"Saved models to {save_path}")

img_logger.finalize()
train_img_logger.finalize()