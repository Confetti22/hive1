#%%
import itertools                      
from pathlib import Path
import math, os
import numpy as np
import random
import torch
import torch.nn.functional as F
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
from lib.loss.tv import TVLoss
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

            logits = seg_model(inputs)          # NEW
            mask = targets >=0
            #targets is the middle slice of the 3d output, so take the middle slice of the output
            logits_channel_last =  logits.permute(0, 2, 3, 4, 1) #B,D,H,W,C
            logits_middle_slice = logits_channel_last[:,int(logits_channel_last.shape[1]//2),:,:,:]

            logits_flat = logits_middle_slice[mask]
            targets_flat = targets[mask]

            loss = loss_fn(logits_flat, targets_flat)
            
            valid_loss.append(loss.item())

            top1, top3 = accuracy(logits_flat, targets_flat, topk=(1, 3))
            total_top1.append(top1)
            total_top3.append(top3)

            # ---------- prediction ----------
            if (visualize_first_batch and idx ==0) or not visualize_first_batch:
                probs = F.softmax(logits_middle_slice, dim=-1)               # softmax over channel K
                pred  = torch.argmax(probs, dim=-1) + 1         # [B,D,H,W], +1 keeps 0 for ignore
                pred  = pred * mask
                            # ---------- move to CPU once ----------
                pred_np  = pred.cpu().numpy()                  # [B,D,H,W]
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
cpkt = torch.load('outs/seg_bnd/semantic_seg/model_epoch_100.pth')
load_result = seg_model.load_state_dict(cpkt['seg_model'])
print(f"{load_result=}")
# print(seg_model)
# summary(seg_model,(1,20,1024,1024))

register_hooks(seg_model.cnn_module.decoder.up_layers[2],'seghead.last_cnn')
register_hooks(seg_model.mlp_module, "seghead.mlp")  # you can add cnn if needed
print("Registered layers:", LAYER_ORDER)

seg_model.train()

optimizer = torch.optim.Adam( itertools.chain(seg_model.parameters()),lr=args.lr_start)

from lib.core.scheduler import WarmupCosineLR
scheduler = WarmupCosineLR(optimizer,args.lr_warmup,args.epochs)


# %% ---------- data loaders & loggers (unchanged) -----------------------------

dataset        = get_dataset(args,bnd=False,bool_mask=False,crop_roi=True)
loader         = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)
valid_dataset  = get_valid_dataset(args,bnd=False,bool_mask=False,crop_roi =True)

#set batch_size of valid_loader ==1 to make sure each item in feature_store is from one image 
valid_loader   = DataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=False)

#~~~~~~~ weighted l1 loss ~~~~~~~~#
from lib.loss.ce_dice_combo import ComboLoss
from lib.utils.loss_utils import compute_class_weights_from_dataset
class_weights = compute_class_weights_from_dataset(dataset, args.mlp_filters[-1])
loss_fn = ComboLoss(class_weights=class_weights, focal=args.get("use_focal", True))

start_epoch = 80 
# %% ---------- training loop --------------------------------------------------
for epoch in tqdm(range(start_epoch,args.epochs)):
    train_loss = []
    total_top1 = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        cnn_out,logits = seg_model(inputs)          

        mask = targets >=0
        #targets is the middle slice of the 3d output, so take the middle slice of the output
        logits_channel_last =  logits.permute(0, 2, 3, 4, 1) #B,D,H,W,C
        logits_middle_slice = logits_channel_last[:,int(logits_channel_last.shape[1]//2),:,:,:]

        logits_flat = logits_middle_slice[mask]
        targets_flat = targets[mask]

        loss = loss_fn(logits_flat, targets_flat)
        
        #unsepervised tv_loss
        if args.tv_loss:
            tv_loss  = TVloss(cnn_out) 
            loss    += tv_loss

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
        random_indexs = random.sample(range(0, len(valid_dataset)), 3)
        print(f"begin valid, {len(valid_dataset)= },len of feature_dict={FEATURE_STORE[LAYER_ORDER[0]]}")
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



    if (epoch % (4 * args.valid_very_epoch)) == 0:
        bnd_seg_valid(train_img_logger, loader, seg_model,  epoch,visualize_first_batch=True)

    if (epoch + 1) % 50 == 0:
        save_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'seg_model': seg_model.state_dict(),
        }, save_path)
        print(f"Saved models to {save_path}")

img_logger.finalize()
train_img_logger.finalize()