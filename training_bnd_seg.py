#%%
import itertools                      
from typing import Sequence, Union
from pathlib import Path
from matplotlib import cm
from PIL import Image
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
def print_load_result(load_result):
    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys

    if not missing and not unexpected:
        print("✅ All weights loaded successfully.")
    else:
        print("⚠️ Some weights were not loaded exactly:")
        if missing:
            print(f"   • Missing keys ({len(missing)}):\n     {missing}")
        if unexpected:
            print(f"   • Unexpected keys ({len(unexpected)}):\n     {unexpected}")




def bnd_seg_valid(img_logger, valid_loader, cmpsd_model,epoch,valid_first_batch=False,seg_head=None):
    cmpsd_model.eval()
    if seg_head:
        seg_head.eval()

    valid_loss, gt_maskes, pred_maskes = [], [], []

    with torch.no_grad():
        for inputs, masks ,targets in tqdm(valid_loader):
            inputs, masks,targets = inputs.to(device), masks.to(device),targets.to(device)
            masks = masks.unsqueeze(1).unsqueeze(1)
            targets= targets.unsqueeze(1).unsqueeze(1)

            optimizer.zero_grad()
            logits = cmpsd_model(inputs)          # NEW
            if seg_head:
                logits = seg_head(logits)          # NEW

            loss = loss_fn(logits,targets,masks)
            valid_loss.append(loss.item())


            probs  = torch.sigmoid(logits)
            pred  = (probs >= 0.5).long()
            
            valid_mask = masks.permute(0, 2, 3, 4, 1)            # (4, 1, 249, 249, 1) 
            pred = pred.permute(0, 2, 3, 4, 1) 
            pred  = pred * valid_mask

            pred_np  = pred.cpu().squeeze().numpy()
            bnd_label_np = (targets).cpu().squeeze().numpy()

            gt_maskes.append(bnd_label_np)
            pred_maskes.append(pred_np)

            if valid_first_batch:break

    gt_maskes = np.concatenate([ arr for arr in gt_maskes], axis=0)
    pred_maskes= np.concatenate([ arr for arr in pred_maskes], axis=0)

    # cmap        = plt.get_cmap('nipy_spectral', num_classes)
    cmap = [
        [0,0,0],
        [0,1,1],
        [1,0,1],
        [1,1,1]
    ]
    max_cols    = 4                       # ≤ 4 columns in the gallery

    # --- 1) merge GT + prediction for visualisation ----------------------------
    combined_imgs = []
    for idx ,(gt, pred) in enumerate(zip(gt_maskes, pred_maskes)):
        # Put GT on the left, prediction on the right
        merged = np.zeros((gt.shape[0],gt.shape[1],3),dtype=np.float32)
        code = (gt.astype(np.uint8) << 1) + pred.astype(np.uint8)
        merged[code ==0]=cmap[0]
        merged[code ==1]=cmap[1]
        merged[code ==2]=cmap[2]
        merged[code ==3]=cmap[3]

        valid_save_dir = img_logger.images_dir
        rgb_uint8 = (merged*255).astype(np.uint8)
        img = Image.fromarray(rgb_uint8)
        img.save(f"{valid_save_dir}/{idx}_{epoch}.tiff",format='TIFF')
        
        rgb_gt = np.stack([gt] * 3, axis=-1)
        combined = np.hstack((rgb_gt,merged))
        combined_imgs.append(combined)
    # --- 2) build a grid -------------------------------------------------------
    n_imgs  = len(combined_imgs)
    n_cols  = min(max_cols, n_imgs)
    n_rows  = math.ceil(n_imgs / n_cols)
    h, w,c = combined_imgs[0].shape        # (H, W) of a single mask

    dpi      = 100                       # logical pixels per inch
    fig_w_in = (2*w * n_cols) / dpi        # make sure W *cols  ≥ fig_w * dpi
    fig_h_in = (2*h * n_rows) / dpi        #               same for H

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w_in, fig_h_in),
        dpi=dpi,                          # keep >= original resolution
        squeeze=False
    )

    for idx, img in enumerate(combined_imgs):
        r, c = divmod(idx, n_cols)
        axes[r, c].imshow(img )
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

    return avg_valid_loss


#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg_path = 'config/seghead.yaml'
args = load_cfg(cfg_path)
args.e5 =False 
data_prefix   = Path("/share/home/shiqiz/data" if args.e5 else "/home/confetti/data")

exp_save_dir   = 'outs/seg_bnd'
exp_name       = f"bnd_seg_finetune_scratch_3moduler_level{args.feats_level}_avg_pool{args.feats_avg_kernel}_cldice_smallerlr"
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
seg_scratch_pth = "/home/confetti/e5_workspace/hive1/outs/seg_bnd/bnd_seg_finetune_scratch_3moduler_level2_avg_pool8_cldice_smallerlr/model_epoch_150.pth"
ckpt = torch.load(seg_scratch_pth)
load_result = cmpsd_model.load_state_dict(ckpt['cmpsd_model'])
print_load_result(load_result)

C=12
seg_head     = ConvSegHead(C, 1).to(device)
load_result = seg_head.load_state_dict(ckpt['seg_head'])
print_load_result(load_result)
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
n_learnable = sum(p.numel() for p in cmpsd_model.cnn_encoder.parameters())
print(f"[info] learnable cnn_encoder params: {n_learnable:,}")
n_learnable = sum(p.numel() for p in cmpsd_model.mlp_encoder.parameters())
print(f"[info] learnable mlp params: {n_learnable:,}")


register_hooks(cmpsd_model.mlp_encoder, "mlp.")  # you can add cnn if needed
print("Registered layers:", LAYER_ORDER)

summary(cmpsd_model, (1, *args.input_size))

# Flip the *whole* model back to training mode (the encoder stays in eval() because we set it explicitly)
cmpsd_model.train()
seg_head.train()

args.lr_start = 1e-5
# args.lr_end = 1e-5
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

dataset        = get_dataset(args,bnd=True,bool_mask=True)
loader         = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)
valid_dataset  = get_valid_dataset(args,bnd=True,bool_mask=True)
valid_loader   = DataLoader(valid_dataset, batch_size=4, shuffle=False, drop_last=False)

from lib.loss.cldice import get_loss

loss_fn = get_loss()
start_epoch = 150
# %% ---------- training loop --------------------------------------------------
for epoch in tqdm(range(start_epoch,args.num_epochs)):
    train_loss = []
    total_top1 = []
    for inputs, masks,targets in loader:
        inputs, masks,targets = inputs.to(device), masks.to(device),targets.to(device)
        optimizer.zero_grad()
        logits = cmpsd_model(inputs)          # NEW
        logits = seg_head(logits)  

        masks = masks.unsqueeze(1).unsqueeze(1)
        targets= targets.unsqueeze(1).unsqueeze(1)

        loss = loss_fn(logits,targets,masks)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())

        FEATURE_STORE.clear()   # free memory

    avg_loss = sum(train_loss) / len(train_loss)
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar('lr',scheduler.get_last_lr()[0] , epoch)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"Epoch {epoch:02d} | loss={loss:.4f} | lr={current_lr:.6f}")
    
    if epoch % args.valid_very_epoch == 0:
        v_loss= bnd_seg_valid(img_logger, valid_loader, cmpsd_model, epoch=epoch ,valid_first_batch=True,seg_head=seg_head)
        writer.add_scalar('Loss/valid', v_loss, epoch)

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