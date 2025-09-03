#%%
import itertools                      
from pathlib import Path
import math, os
import numpy as np
import shutil
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
from lib.arch.ae import ConvMLP
from distance_contrast_helper import HTMLFigureLogger
from lib.datasets.dataset4seghead import get_dataset, get_valid_dataset
from helper.contrastive_train_helper import log_layer_embeddings
import tifffile as tif

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
VALID_M = 4
import numpy as np

def merge_arrays_to_grid(array1, array2, K):
    B, H, W = array1.shape
    assert array2.shape == (B, H, W), "array2 must have same shape as array1"
    K = min(K,B)

    # Get first K images from each array
    row1 = array1[:K]   # (K, H, W)
    row2 = array2[:K]   # (K, H, W)

    # Concatenate images in each row horizontally → shape: (H, K*W)
    row1_concat = np.concatenate(row1, axis=1)
    row2_concat = np.concatenate(row2, axis=1)

    # Stack the two rows vertically → shape: (2*H, K*W)
    merged_image = np.concatenate([row1_concat, row2_concat], axis=0)

    return merged_image


def _hook(layer_name, module, inp, out):
    """
    Hook to store output features. Only records when model is in eval mode.
    out : Tensor shape [B, C, D, H, W] or [B, C, H, W]
    """
    if module.training:
        return  # Skip during training
    if len(FEATURE_STORE[layer_name]) < VALID_M:
        FEATURE_STORE[layer_name].append(out[:1].detach().cpu())

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


def test_function(img_logger, dataset,test_idxes, seg_model,epoch):
    """
    recored prediction on given img idxes
    """
    seg_model.eval()

    gt_maskes   = []   # list of 2-D numpy arrays
    pred_maskes = []
    label_volumes = []

    with torch.no_grad():
        for idx in test_idxes: 
            batch = dataset[idx]
            if args.recon_loss:
                (inputs, targets,recon_targets) = batch
            else:
                (inputs, targets) = batch
            inputs = inputs.unsqueeze(0)
            targets = targets.unsqueeze(0)
            inputs, targets = inputs.to(device),targets.to(device)
            bottle_neck,logits = seg_model(inputs)          # NEW
            mask = targets >=0
            #targets is the middle slice of the 3d output, so take the middle slice of the output
            logits_channel_last =  logits.permute(0, 2, 3, 4, 1) #B,D,H,W,C
            logits_middle_slice = logits_channel_last[:,int(logits_channel_last.shape[1]//2),:,:,:]

            # ---------- prediction ----------
            probs = F.softmax(logits_middle_slice, dim=-1)               # softmax over channel K
            pred  = torch.argmax(probs, dim=-1) + 1         # [B,D,H,W], +1 keeps 0 for ignore
            pred  = pred * mask
                        # ---------- move to CPU once ----------
            pred_np  = pred.cpu().numpy()                # [B,D,H,W]
            label_np = (targets + 1).cpu().numpy()
            label_volumes.append(label_np[0])
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

    seg_model.train()
    return label_volumes


def recon_valid(writer, valid_loader, encoder,decoder,epoch):
    encoder.eval()
    decoder.eval()

    recon_targets   = []   # list of 2-D numpy arrays
    reconeds = []
    recon_losses = []

    with torch.no_grad():
        for idx ,batch in enumerate(tqdm(valid_loader)):
            (inputs, targets,recon_target) = batch

            inputs ,recon_target = inputs.to(device), recon_target.to(device)
            bottle_necks = encoder(inputs)          
            reconed     =  decoder(bottle_necks).squeeze()
            recon_target = recon_target.squeeze()
            recon_loss  =  recon_loss_fn(reconed,recon_target) 
            recon_losses.append(recon_loss.item())

            # ---------- prediction ----------
            if idx < VALID_M:
                reconed = reconed.detach().cpu().squeeze().numpy() #B*H*W
                recon_target = recon_target.detach().cpu().squeeze().numpy() #B*H*W
                recon_targets.append(recon_target)
                reconeds.append(reconed)

    stacked_recon_targets  = np.stack(recon_targets, axis= 0)
    stacked_reconeds= np.stack(reconeds, axis= 0)
    merged = merge_arrays_to_grid(stacked_recon_targets,stacked_reconeds,4)
    merged = (merged - merged.min()) / (merged.max() - merged.min())
    writer.add_image('x and re_x ',merged,epoch,dataformats='HW')
    # tif.imwrite( f'temp_/reconed_{epoch}.tif', stacked_reconeds)
    # tif.imwrite(f'temp_/target{epoch}.tif', stacked_recon_targets)

    avg_valid_recon_loss = sum(recon_losses)/len(recon_losses) if recon_losses else 0
    encoder.train()
    decoder.train()
    return avg_valid_recon_loss

def bnd_seg_valid(img_logger, valid_loader, seg_model,epoch):
    seg_model.eval()

    valid_loss  = []
    ce_losses = []
    dice_losses = []
    gt_maskes   = []   # list of 2-D numpy arrays
    pred_maskes = []
    total_top1 = []
    total_top3= []
    label_volumes = []

    with torch.no_grad():
        for idx ,batch in enumerate(tqdm(valid_loader)):
            if args.recon_loss:
                (inputs, targets,recon_targets) = batch
            else:
                (inputs, targets) = batch

            inputs, targets = inputs.to(device),targets.to(device)
            bottle_neck,logits = seg_model(inputs)          # NEW
            mask = targets >=0
            #targets is the middle slice of the 3d output, so take the middle slice of the output
            logits_channel_last =  logits.permute(0, 2, 3, 4, 1) #B,D,H,W,C
            logits_middle_slice = logits_channel_last[:,int(logits_channel_last.shape[1]//2),:,:,:]

            logits_flat = logits_middle_slice[mask]
            targets_flat = targets[mask]

            loss = torch.tensor(0).to(device) 
            if args.supervised_loss:
                loss, ce_loss, dice_loss = supervised_loss_fn(logits_flat, targets_flat)
                valid_loss.append(loss.item())
                ce_losses.append(ce_loss.item())
                dice_losses.append(dice_loss.item())

            top1, top3 = accuracy(logits_flat, targets_flat, topk=(1, 3))
            total_top1.append(top1)
            total_top3.append(top3)

            # ---------- prediction ----------
            if idx < VALID_M:
                probs = F.softmax(logits_middle_slice, dim=-1)               # softmax over channel K
                pred  = torch.argmax(probs, dim=-1) + 1         # [B,D,H,W], +1 keeps 0 for ignore
                pred  = pred * mask
                            # ---------- move to CPU once ----------
                pred_np  = pred.cpu().numpy()                # [B,D,H,W]
                label_np = (targets + 1).cpu().numpy()
                if len(label_volumes) < VALID_M:
                    label_volumes.append(label_np[0])
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

    avg_valid_loss = sum(valid_loss)/len(valid_loss) if valid_loss else 0
    avg_top1 = sum(total_top1)/len(total_top1) 
    avg_top3 = sum(total_top3)/len(total_top3)
    avg_ce_loss = sum(ce_losses)/len(ce_losses) if ce_losses else 0
    avg_dice_loss = sum(dice_losses)/len(dice_losses) if dice_losses else 0

    seg_model.train()
    return avg_valid_loss, avg_top1, avg_top3, avg_ce_loss, avg_dice_loss, label_volumes

 
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg_path = 'config/semisupervised.yaml'
args = load_cfg(cfg_path)

if args.test_mode:
    use_ratio = 0.1
    args.exp_name = f"_{args.exp_name}"
else:
    use_ratio = 1


model_save_dir = f"{args.exp_save_dir}/{args.exp_name}"
os.makedirs(model_save_dir, exist_ok=True)
shutil.copy(cfg_path, f"{model_save_dir}/config.yaml")


writer          = SummaryWriter(f'{args.exp_save_dir}/{args.exp_name}')
img_logger      = HTMLFigureLogger(args.exp_save_dir + '/' + args.exp_name, html_name="seg_valid_result.html")
test_img_logger = HTMLFigureLogger(args.exp_save_dir + '/' + args.exp_name, html_name="seg_valid_result_test.html")
train_img_logger= HTMLFigureLogger(args.exp_save_dir + '/' + args.exp_name, html_name="train_seg_valid_result.html")

seg_model= build_semantic_seg_model(args).to(device)
seg_model.train()

if args.recon_loss:
    seg_model.cnn_module.decoder.requires_grad_(False)
    seg_model.cnn_module.decoder.eval()
    seg_model.mlp_module.requires_grad_(False)
    seg_model.mlp_module.eval()

print("\n","frozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if not p.requires_grad])
print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")
    
# print(seg_model)
# summary(seg_model,(1,20,512,512))

if args.tv_loss:
    tv_loss_fn = TVLoss(TVLoss_weight=args.tv_loss_weight)

if args.recon_loss:
    from lib.loss.l1 import L1Loss
    decoder = ConvMLP(filters=(args.filters[-1], 1),dims=3,l2_norm=False,last_act=True).to(device)
    decoder.train()
    recon_loss_fn = L1Loss(args) 
    print(f"using recon_loss, the {decoder= }")

    
#optimizer and scheduler
if args.recon_loss:    
    optimizer = torch.optim.Adam( itertools.chain(seg_model.parameters(),decoder.parameters()),lr=args.lr_start)
else:
    optimizer = torch.optim.Adam( itertools.chain(seg_model.parameters()),lr=args.lr_start)

from lib.core.scheduler import WarmupCosineLR
scheduler = WarmupCosineLR(optimizer,args.lr_warmup,args.epochs)


# %% ---------- data loaders & loggers (unchanged) -----------------------------

if args.e5:
    train_batch_size = 16 
else:
    train_batch_size = 4 


VALID_BATCH_SIZE = 1 #set batch_size of valid_loader ==1 to make sure each item in feature_store is from one image 

dataset        = get_dataset(args,bnd=False,bool_mask=False,use_ratio=use_ratio,
                             crop_roi=True,recon_target_flag = args.recon_loss)
train_loader   = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=False)
valid_dataset  = get_valid_dataset(args,bnd=False,bool_mask=False,use_ratio= 1,
                                   crop_roi =True,recon_target_flag = args.recon_loss)
valid_loader   = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True, drop_last=False)
fix_valid_loader   = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, drop_last=False)

test_idxes = [6,56,88,140]


if args.tsne_emb:
    register_hooks(seg_model.cnn_module.decoder.up_layers[2],'seghead.last_cnn')
    register_hooks(seg_model.mlp_module, "seghead.mlp")  # you can add cnn if needed
    print("Registered layers:", LAYER_ORDER)

#~~~~~~~ weighted l1 loss ~~~~~~~~#
from lib.loss.ce_dice_combo import ComboLoss
from lib.utils.loss_utils import compute_class_weights_from_dataset

if args.supervised_loss:
    class_weights = compute_class_weights_from_dataset(dataset, args.mlp_filters[-1],recon_target_flag=args.recon_loss)
    supervised_loss_fn = ComboLoss(class_weights=class_weights, focal=args.get("use_focal", True))


# %% ---------- training loop --------------------------------------------------
from pprint import pprint
from lib.arch.ae import modify_key,delete_key
ckpt = torch.load("outs/cortex_semantic_seg4/supervised1/model_epoch_100.pth")
pprint(ckpt['seg_model'].keys())
# encoder_dict = modify_key(ckpt['seg_model'],source='cnn_module.encoder.', target='')
# encoder_dict = delete_key(encoder_dict,pattern_lst=('cnn_module.decoder','mlp_module'))
# load_result = seg_model.cnn_module.encoder.load_state_dict(encoder_dict)

load_result = seg_model.load_state_dict(ckpt['seg_model'])

print(f"load seg_model {load_result}")

start_epoch = 0 

for epoch in tqdm(range(start_epoch,args.epochs)):
    train_loss = []
    ce_losses = []
    dice_losses = []
    tv_losses = []
    recon_losses = []
    total_top1 = []
    for  batch_idx,batch in enumerate(train_loader):
        if args.recon_loss:
            inputs, targets,recon_target = batch
            inputs, targets, recon_target= inputs.to(device), targets.to(device), recon_target.to(device)
        else:
            inputs, targets = batch
            inputs, targets= inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        bottle_necks,logits = seg_model(inputs)          

        mask = targets >=0 #B,H,W
        #targets is the middle slice of the 3d output, so take the middle slice of the output
        logits_channel_last =  logits.permute(0, 2, 3, 4, 1) #B,D,H,W,C
        logits_middle_slice = logits_channel_last[:,int(logits_channel_last.shape[1]//2),:,:,:]

        logits_flat = logits_middle_slice[mask]
        targets_flat = targets[mask]

        loss = torch.tensor(0.0).to(device) 
        if args.supervised_loss:
            loss ,ce_loss, dice_loss = supervised_loss_fn(logits_flat, targets_flat)
            train_loss.append(loss.item())
            ce_losses.append(ce_loss.item())
            dice_losses.append(dice_loss.item())
        
        #unsepervised tv_loss
        if args.tv_loss:
            # logits (B,C,D,H,W)
            probs = F.softmax(logits,dim=1)
            tv_loss  = tv_loss_fn(probs) 
            loss    += tv_loss
            tv_losses.append(tv_loss.item())
        if args.recon_loss:
            reconed    =  decoder(bottle_necks).squeeze()
            recon_target = recon_target.squeeze()
            recon_loss =  recon_loss_fn(reconed,recon_target) * args.recon_loss_weight
            loss      +=  recon_loss
            recon_losses.append(recon_loss.item())
        
        loss.backward()
        optimizer.step()
        FEATURE_STORE.clear()   # free memory
        
    scheduler.step()
    avg_loss = sum(train_loss) / len(train_loss) if train_loss else 0
    avg_ce_loss = sum(ce_losses) / len(ce_losses) if ce_losses else 0
    avg_dice_loss = sum(dice_losses) / len(dice_losses) if dice_losses else 0
    avg_tv_loss = sum(tv_losses) / len(tv_losses) if len(tv_losses)!= 0 else 0
    avg_recon_loss = sum(recon_losses)/len(recon_losses) if len(recon_losses) !=0  else 0

    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar('lr',scheduler.get_last_lr()[0] , epoch)
    if avg_loss: 
        writer.add_scalar('Loss/train', avg_loss, epoch)
    if avg_ce_loss:
        writer.add_scalar('ce_Loss/train', avg_ce_loss, epoch)
    if avg_dice_loss:
        writer.add_scalar('dice_Loss/train', avg_dice_loss, epoch)
    if args.tv_loss:
        writer.add_scalar('tv_Loss/train', avg_tv_loss, epoch)
    if args.recon_loss:
        writer.add_scalar('recon_Loss/train', avg_recon_loss, epoch)

    print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} |tv_loss={avg_tv_loss:.4f}| recon_loss={avg_recon_loss:.4f} lr={current_lr:.6f}")
    
    
    #validation for unsupervised_recon
    if epoch % args.valid_very_epoch == 0 and args.recon_loss:
        valid_recon_loss = recon_valid(writer,valid_loader,seg_model.cnn_module.encoder,decoder,epoch)
        writer.add_scalar("recon_Loss/valid", valid_recon_loss, epoch)
        print(f"Epoch {epoch:02d} | valid_recon_loss={valid_recon_loss:.4f}" ,"\n")

    #validation for supervised segmentation task loss
    if epoch % args.valid_very_epoch == 0 and  args.supervised_loss:

        val_loss ,avg_top1, avg_top3, val_ce_loss, val_dice_loss,label_volumes = bnd_seg_valid(img_logger, valid_loader, seg_model, epoch=epoch )
        writer.add_scalar("Loss/valid", val_loss, epoch)
        writer.add_scalar("acc/top_1_valid", avg_top1, epoch)
        writer.add_scalar("acc/top_3_valid", avg_top3, epoch)
        writer.add_scalar("ce_Loss/valid", val_ce_loss, epoch)
        writer.add_scalar("dice_Loss/valid", val_dice_loss, epoch)
        #this will recod first batch of valid_loader and thus will be random
        # if args.tsne_emb:
        #     for idx,label_volume in enumerate(label_volumes): 
        #         log_layer_embeddings(
        #             FEATURE_STORE,
        #             writer=writer,
        #             epoch=epoch,
        #             label_volume=label_volume,  # numpy array
        #             layer_order=LAYER_ORDER,       # from hook registration
        #             max_layers=12,
        #             mode="both",                   # <- t-SNE + UMAP stacked
        #             tsne_kwargs=dict(perplexity=20),
        #             umap_kwargs=dict(n_neighbors=30, min_dist=0.05,random_state=42,),
        #             valid_img_idx=idx,
        #         )
        FEATURE_STORE.clear()

        # recod the embeddings from the specified idx in valid_dataset
        label_volumes = test_function(test_img_logger, list(valid_dataset),test_idxes ,seg_model, epoch=epoch )

        if args.tsne_emb:
            for idx,label_volume in enumerate(label_volumes): 
                log_layer_embeddings(FEATURE_STORE,writer=writer,epoch=epoch,label_volume=label_volume,layer_order=LAYER_ORDER, 
                    mode="both", tsne_kwargs=dict(perplexity=20),umap_kwargs=dict(n_neighbors=30, min_dist=0.05,random_state=42,),
                    valid_img_idx=idx,comment='fix_valid'
                    )
        FEATURE_STORE.clear()




    if (epoch % ( args.valid_very_epoch)) == 0:
        val_loss ,avg_top1, avg_top3, val_ce_loss, val_dice_loss,label_volumes = bnd_seg_valid(train_img_logger, train_loader, seg_model,  epoch)
        FEATURE_STORE.clear()
        writer.add_scalar("top1_acc/train", avg_top1, epoch)
        writer.add_scalar("top3_acc/train", avg_top3, epoch)

    if (epoch + 1) % 50 == 0:
        save_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
        if args.recon_loss:
            torch.save({
                'seg_model': seg_model.state_dict(),
                'decoder_model':decoder.state_dict(),
            }, save_path)
        else:
            torch.save({
                'seg_model': seg_model.state_dict(),
            }, save_path)

        print(f"Saved models to {save_path}")

img_logger.finalize()
train_img_logger.finalize()