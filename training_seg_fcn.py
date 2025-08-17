#%%
import itertools                      
from pathlib import Path
import math, os
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config.load_config import load_cfg
from lib.arch.ae import build_semantic_seg_model
from distance_contrast_helper import HTMLFigureLogger
from lib.datasets.dataset4seghead import get_dataset, get_valid_dataset
from helper.contrastive_train_helper import log_layer_embeddings

from functools import partial
from lib.core.metric import accuracy

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
    Hook to store output features. Only records when model is in eval mode.
    out : Tensor shape [B, C, D, H, W] or [B, C, H, W]
    """
    if module.training:
        return  # Skip during training

    if len(FEATURE_STORE[layer_name]) < 3:
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

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        # Input shape: (B, C, H, W) or (B, C, D, H, W)
        # avoid D==1
        batch_size = x.size(0)
        spatial_dims = x.dim() - 2  # Number of spatial dims (2 for 2D, 3 for 3D)
        tv = 0.0

        if spatial_dims == 2:
            # (B, C, H, W)
            h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
            w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
            count_h = self._tensor_size(x[:, :, 1:, :])
            count_w = self._tensor_size(x[:, :, :, 1:])
            tv = (h_tv / count_h + w_tv / count_w)
        elif spatial_dims == 3:
            # (B, C, D, H, W)
            d_tv = torch.pow(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], 2).sum()
            h_tv = torch.pow(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], 2).sum()
            w_tv = torch.pow(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], 2).sum()
            count_d = self._tensor_size(x[:, :, 1:, :, :])
            count_h = self._tensor_size(x[:, :, :, 1:, :])
            count_w = self._tensor_size(x[:, :, :, :, 1:])
            tv = (d_tv / count_d + h_tv / count_h + w_tv / count_w)
        else:
            raise ValueError("Unsupported input dimensions. Expected 4D or 5D input.")

        return self.TVLoss_weight * 2 * tv / batch_size

    def _tensor_size(self, t):
        return t.numel()


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


def bnd_seg_valid(img_logger, valid_loader, seg_model,epoch,visualize_first_batch=False):
    seg_model.eval()

    valid_loss  = []
    gt_maskes   = []   # list of 2-D numpy arrays
    pred_maskes = []
    total_top1 = []
    total_top3= []

    with torch.no_grad():
        for idx ,(inputs, targets) in enumerate(tqdm(valid_loader)):
            inputs, targets = inputs.to(device),targets.to(device)

            bottle_neck,cnn_out,logits = seg_model(inputs)          # NEW
            mask = targets >=0
            #targets is the middle slice of the 3d output, so take the middle slice of the output
            logits_channel_last =  logits.permute(0, 2, 3, 4, 1) #B,D,H,W,C
            logits_middle_slice = logits_channel_last[:,int(logits_channel_last.shape[1]//2),:,:,:]

            logits_flat = logits_middle_slice[mask]
            targets_flat = targets[mask]

            loss = supervised_loss_fn(logits_flat, targets_flat)
            
            valid_loss.append(loss.item())

            top1, top3 = accuracy(logits_flat, targets_flat, topk=(1, 3))
            total_top1.append(top1)
            total_top3.append(top3)

            # ---------- prediction ----------
            if (visualize_first_batch and idx ==2) or not visualize_first_batch:
                probs = F.softmax(logits_middle_slice, dim=-1)               # softmax over channel K
                pred  = torch.argmax(probs, dim=-1) + 1         # [B,D,H,W], +1 keeps 0 for ignore
                pred  = pred * mask
                            # ---------- move to CPU once ----------
                pred_np  = pred.cpu().numpy()                # [B,D,H,W]
                label_np = (targets + 1).cpu().numpy()

                # one 2-D slice per sample
                gt_maskes.extend(label_np)
                pred_maskes.extend(pred_np)

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

    avg_valid_loss = sum(valid_loss)/len(valid_loss)
    avg_top1 = sum(total_top1)/len(total_top1)
    avg_top3 = sum(total_top3)/len(total_top3)

    seg_model.train()
    return avg_valid_loss, avg_top1,avg_top3

 
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg_path = 'config/semisupervised.yaml'
args = load_cfg(cfg_path)

model_save_dir = f"{args.exp_save_dir}/{args.exp_name}"
os.makedirs(model_save_dir, exist_ok=True)

writer          = SummaryWriter(f'{args.exp_save_dir}/{args.exp_name}')
img_logger      = HTMLFigureLogger(args.exp_save_dir + '/' + args.exp_name, html_name="seg_valid_result.html")
train_img_logger= HTMLFigureLogger(args.exp_save_dir + '/' + args.exp_name, html_name="train_seg_valid_result.html")

seg_model= build_semantic_seg_model(args).to(device)
seg_model = seg_model

cpkt = torch.load('outs/cortex_semantic_seg/with_tvloss_lambda1e-5/model_epoch_50.pth')
load_result = seg_model.load_state_dict(cpkt['seg_model'])
print("\n\n")
print(f"{load_result=}")

print(seg_model)
summary(seg_model,(1,20,512,512))
exit(0)

start_epoch = 0 

if args.tsne_emb:
    register_hooks(seg_model.cnn_module.decoder.up_layers[2],'seghead.last_cnn')
    register_hooks(seg_model.mlp_module, "seghead.mlp")  # you can add cnn if needed
    print("Registered layers:", LAYER_ORDER)

seg_model.train()

optimizer = torch.optim.Adam( itertools.chain(seg_model.parameters()),lr=args.lr_start)

from lib.core.scheduler import WarmupCosineLR
scheduler = WarmupCosineLR(optimizer,args.lr_warmup,args.epochs)


# %% ---------- data loaders & loggers (unchanged) -----------------------------

train_batch_size =1
VALID_BATCH_SIZE = 1 #set batch_size of valid_loader ==1 to make sure each item in feature_store is from one image 
dataset        = get_dataset(args,bnd=False,bool_mask=False,crop_roi=True)
train_loader   = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=False)
valid_dataset  = get_valid_dataset(args,bnd=False,bool_mask=False,crop_roi =True)
valid_loader   = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, drop_last=False)

from lib.datasets.simple_dataset import get_dataset as simple_get_dataset
from lib.datasets.simple_dataset import get_valid_dataset as simple_get_valid_dataset
temp_conf=args.copy()
temp_conf.data_path_dir = '/home/confetti/data/rm009/boundary_seg/new_boundary_seg_data/low_resol_rois'
temp_conf.e5_data_path_dir = '/share/home/shiqiz/rm009/boundary_seg/new_boundary_seg_data/low_resol_rois'
temp_conf.valid_data_path_dir = '/home/confetti/data/rm009/boundary_seg/new_boundary_seg_data/low_resol_rois_valid'
temp_conf.e5_valid_data_path_dir = '/share/home/shiqiz/rm009/boundary_seg/new_boundary_seg_data/low_resol_rois'
low_recon_targets = simple_get_dataset(temp_conf)
valid_low_recon_targets = simple_get_valid_dataset(temp_conf)
recon_target_loader = DataLoader(low_recon_targets, batch_size=train_batch_size, shuffle=True, drop_last=False)



#~~~~~~~ weighted l1 loss ~~~~~~~~#
from lib.loss.ce_dice_combo import ComboLoss
from lib.utils.loss_utils import compute_class_weights_from_dataset
class_weights = compute_class_weights_from_dataset(dataset, args.mlp_filters[-1])
supervised_loss_fn = ComboLoss(class_weights=class_weights, focal=args.get("use_focal", True))
tv_loss_fn = TVLoss(TVLoss_weight=1e-2)

# %% ---------- training loop --------------------------------------------------
for epoch in tqdm(range(start_epoch,args.epochs)):
    train_loss = []
    total_top1 = []
    for inputs, targets,recon_target in train_loader:
        inputs, targets, recon_target= inputs.to(device), targets.to(device), recon_target.to(device)
        optimizer.zero_grad()
        bottle_necks,cnn_outs,logits = seg_model(inputs)          

        mask = targets >=0
        #targets is the middle slice of the 3d output, so take the middle slice of the output
        logits_channel_last =  logits.permute(0, 2, 3, 4, 1) #B,D,H,W,C
        logits_middle_slice = logits_channel_last[:,int(logits_channel_last.shape[1]//2),:,:,:]

        logits_flat = logits_middle_slice[mask]
        targets_flat = targets[mask]

        loss = supervised_loss_fn(logits_flat, targets_flat)
        
        #unsepervised tv_loss
        if args.tv_loss:
            # logits (B,C,D,H,W)
            probs = F.softmax(logits,dim=1)
            tv_loss  = tv_loss_fn(probs) 
            loss    += tv_loss
        if args.recon_loss:
            reconed    =  decoder(bottle_necks)
            recon_loss =  recon_loss_fn(reconed,recon_target)
            loss      +=  recon_loss
            

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        FEATURE_STORE.clear()   # free memory


    scheduler.step()
    avg_loss = sum(train_loss) / len(train_loss)
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar('lr',scheduler.get_last_lr()[0] , epoch)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"Epoch {epoch:02d} | loss={loss:.4f} | lr={current_lr:.6f}")
    
    if epoch % args.valid_very_epoch == 0:
        val_loss ,avg_top1, avg_top3  = bnd_seg_valid(img_logger, valid_loader, seg_model, epoch=epoch ,visualize_first_batch=True)

        if args.tsne_emb:
            # random_indexs = random.sample(range(0, len(valid_dataset)), 3)

            random_indexs =(0,1,2)
            print(f"begin valid, {len(valid_dataset)= },len of feature_dict={len(FEATURE_STORE[LAYER_ORDER[0]])}")

            for index in random_indexs:
                _,valid_volume = valid_dataset[index]
                valid_volume = valid_volume.numpy()
                log_layer_embeddings(
                    FEATURE_STORE,
                    writer=writer,
                    epoch=epoch,
                    label_volume=valid_volume,  # numpy array
                    layer_order=LAYER_ORDER,       # from hook registration
                    max_layers=12,
                    mode="both",                   # <- t-SNE + UMAP stacked
                    tsne_kwargs=dict(perplexity=20),
                    umap_kwargs=dict(n_neighbors=30, min_dist=0.05,random_state=42,),
                    valid_img_idx=index,
                )
        FEATURE_STORE.clear()


        writer.add_scalar("Loss/valid", val_loss, epoch)
        writer.add_scalar("top1_acc/valid", avg_top1, epoch)
        writer.add_scalar("top3_acc/valid", avg_top3, epoch)



    if (epoch % ( args.valid_very_epoch)) == 0:
        bnd_seg_valid(train_img_logger, train_loader, seg_model,  epoch,visualize_first_batch=True)
        FEATURE_STORE.clear()

    if (epoch + 1) % 50 == 0:
        save_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'seg_model': seg_model.state_dict(),
        }, save_path)
        print(f"Saved models to {save_path}")

img_logger.finalize()
train_img_logger.finalize()