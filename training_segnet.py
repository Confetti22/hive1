#%%
import itertools                      
from typing import Sequence, Union
from pathlib import Path
import time, zarr, math, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import zoom
from confettii.plot_helper import three_pca_as_rgb_image
import tifffile as tif
from config.load_config import load_cfg
from helper.image_seger import ConvSegHead
from helper.contrastive_train_helper import tsne_grid_plot,umap_tsne_grid_plot, plot_pca_maps,class_balance
from lib.arch.ae import build_final_model        # already in your script
from distance_contrast_helper import HTMLFigureLogger
from lib.datasets.dataset4seghead import get_dataset, get_valid_dataset
from lib.core.metric import accuracy
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



def seg_valid(img_logger, valid_loader, cmpsd_model, seg_head, epoch,last_val_label=False):
    cmpsd_model.eval()
    seg_head.eval()

    valid_loss, gt_maskes, pred_maskes = [], [], []
    total_top1 = []
    total_top3= []

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            feats  = cmpsd_model(inputs)          # NEW
            logits = seg_head(feats)              # CHANGED

            mask  = labels >= 0

            logits_flat  = logits.permute(0, 2, 3, 4, 1)[mask]   # [N_vox, K]
            labels_flat  = labels[mask]
            loss         = loss_fn(logits_flat, labels_flat)
            valid_loss.append(loss.item())

            top1, top3 = accuracy(logits_flat, labels_flat, topk=(1, 3))
            total_top1.append(top1)
            total_top3.append(top3)

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

    avg_valid_loss = sum(valid_loss)/len(valid_loss)
    avg_top1 = sum(total_top1)/len(total_top1)
    avg_top3 = sum(total_top3)/len(total_top3)

    return avg_valid_loss, avg_top1,avg_top3

def log_layer_embeddings(
    writer,
    epoch,
    label_volume,
    layer_order,
    max_layers=12,
    mode="tsne",
    tsne_kwargs=None,
    umap_kwargs=None,
):
    """
    Collect features from FEATURE_STORE (global dict name->tensor),
    slice middle Z, color by label_volume, class-balance, and plot
    t-SNE/UMAP grids.

    Parameters
    ----------
    writer : SummaryWriter
    epoch : int
    label_volume : np.ndarray [D,H,W]
    layer_order : sequence[str]
        Ordered list of layer names (e.g., LAYER_ORDER from registration).
    max_layers : int
        Cap number plotted.
    mode : 'tsne' | 'umap' | 'both'
    tsne_kwargs, umap_kwargs : dict
        Passed into reducers.
    """
    import numpy as np
    from scipy.ndimage import zoom

    pca_img_lst            = []
    tsne_encoded_feats_lst = []
    tsne_label_lst         = []
    plot_tag_lst           = []

    # iterate in the provided order
    for k in list(layer_order)[:max_layers]:
        if k not in FEATURE_STORE:
            continue

        # you stored single tensor per key; if you still store list use [-1]
        out_t = FEATURE_STORE[k][-1] if isinstance(FEATURE_STORE[k], (list, tuple)) else FEATURE_STORE[k]
        feat = out_t.detach().cpu().squeeze().numpy()  # assume [C,D,H,W] or [D,H,W,C]? adjust below

        # Reorder to [D,H,W,C] assuming channel-first
        if feat.ndim == 4 and feat.shape[0] not in (label_volume.shape[0],):  # crude heuristic
            # assume feat is [C,D,H,W] -> moveaxis 0->-1
            feat = np.moveaxis(feat, 0, -1)
        elif feat.ndim == 4 and feat.shape[-1] not in (label_volume.shape[0],):
            # already channel-last; leave
            pass
        else:
            raise ValueError(f"Unexpected feature shape for {k}: {feat.shape}")


        feat2d = feat[int(feat.shape[0]//2),:]            # [H,W,C]
        lbl2d  = label_volume[int(label_volume.shape[0]//2),:]    # [H,W]

        H, W, C = feat2d.shape

        # PCA→RGB preview
        rgb_img = three_pca_as_rgb_image(feat2d.reshape(-1, C), (H, W))
        pca_img_lst.append(rgb_img)

        # Align labels to feature resolution
        zoom_factor = [H / lbl2d.shape[0], W / lbl2d.shape[1]]
        label_zoomed = zoom(lbl2d, zoom_factor, order=0)

        mask = label_zoomed >= 0
        fg_feats_flat = feat2d[mask]
        fg_label_flat = label_zoomed[mask]

        blced_feats, blced_labels = class_balance(fg_feats_flat, fg_label_flat)
        tsne_encoded_feats_lst.append(blced_feats)
        tsne_label_lst.append(blced_labels)
        plot_tag_lst.append(k)

    # Embedding grid(s)
    umap_tsne_grid_plot(
        tsne_encoded_feats_lst,
        tsne_label_lst,
        writer,
        tag=f"embed_grid/epoch{epoch}",
        step=epoch,
        tag_list=plot_tag_lst,
        mode=mode,
        tsne_kwargs=tsne_kwargs,
        umap_kwargs=umap_kwargs,
        filter_label=0,
    )

    # PCA image grid
    plot_pca_maps(
        pca_img_lst,
        writer=writer,
        tag=f"pca/epoch{epoch}",
        step=epoch,
        ncols=len(pca_img_lst),
    )

    FEATURE_STORE.clear()  # free memory

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg_path = 'config/seghead.yaml'
args = load_cfg(cfg_path)

args.filters = [6,12,24]
args.mlp_filters =[24,16,12]
args.data_path_dir = "/home/confetti/data/rm009/v1_roi1_seg/rois"
args.valid_data_path_dir = "/home/confetti/data/rm009/v1_roi1_seg_valid/rois"
args.feats_level = 3
args.feats_avg_kernel = 8
args.last_encoder=True


C            = 12          # channels returned by cmpsd_model
num_classes  = 8
cmpsd_model  = build_final_model(args).to(device)     # NEW
seg_head     = ConvSegHead(C, num_classes).to(device)

cmpsd_model.train()
seg_head.train()


data_prefix = Path("/share/home/shiqiz/data" if args.e5 else "/home/confetti/data")

mlp_feat_names = ['rm009_smallestv1roi_oridfar256_l2_pool8_10000.pth','rm009_smallestv1roi_postopk_nview4_l2_avg8.pth','rm009_smallestv1roi_postopk_nview6_l2_avg8.pth']
cnn_ckpt_pth = data_prefix / "weights" / "rm009_3d_ae_best.pth"
mlp_ckpt_pth = data_prefix/ "weights"/ mlp_feat_names[0]


summary(cmpsd_model,(1,*args.input_size))
summary(seg_head,(12,64,64,64))
print(f"{cmpsd_model= }")
print(f"{seg_head= }")


#register for later feature_extraction
register_hooks(cmpsd_model.cnn_encoder, "cnn.")
register_hooks(cmpsd_model.mlp_encoder, "mlp.")
register_hooks(seg_head,              "head.")
print("Registered layers:", LAYER_ORDER)

args.lr_start = 1e-3
args.lr_end = 1e-5
warmup_epochs = 10
max_epochs = args.num_epochs

optimizer = torch.optim.Adam(
    itertools.chain(cmpsd_model.parameters(), seg_head.parameters()),
    lr=args.lr_start
)
from lib.core.scheduler import WarmupCosineLR

scheduler = WarmupCosineLR(optimizer,
                           warmup_epochs=warmup_epochs,
                           max_epochs=max_epochs)


loss_fn = nn.CrossEntropyLoss()


# %% ---------- data loaders & loggers (unchanged) -----------------------------

dataset        = get_dataset(args)
loader         = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)
valid_dataset  = get_valid_dataset(args)
valid_loader   = DataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=False)

valid_label_volume = tif.imread("/home/confetti/data/rm009/v1_roi1_seg_valid/masks/0010_mask.tiff")
valid_label_volume = valid_label_volume.astype(np.int8)
valid_label_volume = valid_label_volume -1 # class number rearrange from 0 to N-1

exp_save_dir   = 'outs/seg_head'
# exp_name       = f"test8_level{args.feats_level}_avg_pool{args.feats_avg_kernel}"
exp_name       = f"finetune_level{args.feats_level}_avg_pool{args.feats_avg_kernel}_3"
model_save_dir = f"{exp_save_dir}/{exp_name}"
os.makedirs(model_save_dir, exist_ok=True)

writer          = SummaryWriter(f'{exp_save_dir}/{exp_name}')
img_logger      = HTMLFigureLogger(exp_save_dir + '/' + exp_name, html_name="seg_valid_result.html")
train_img_logger= HTMLFigureLogger(exp_save_dir + '/' + exp_name, html_name="train_seg_valid_result.html")

# %% ---------- training loop --------------------------------------------------
for epoch in tqdm(range(args.num_epochs)):
    train_loss = []
    total_top1 = []
    total_top3= []
    for inputs, label in loader:
        inputs, label = inputs.to(device), label.to(device)
        mask = label >= 0

        optimizer.zero_grad()
        feats  = cmpsd_model(inputs)          # NEW
        logits = seg_head(feats)              # CHANGED
        logits_flat = logits.permute(0, 2, 3, 4, 1)[mask]
        labels_flat = label[mask]

        loss = loss_fn(logits_flat,labels_flat)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())

        top1, top3 = accuracy(logits_flat, labels_flat, topk=(1, 3))
        total_top1.append(top1)
        total_top3.append(top3)

        FEATURE_STORE.clear()   # free memory

    avg_loss = sum(train_loss) / len(train_loss)
    avg_top1 = sum(total_top1)/len(total_top1)
    avg_top3 = sum(total_top3)/len(total_top3)
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar('lr',scheduler.get_last_lr()[0] , epoch)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar("top1_acc/train", avg_top1, epoch)
    writer.add_scalar("top3_acc/train", avg_top3, epoch)
    print(f"Epoch {epoch:02d} | loss={loss:.4f} | lr={current_lr:.6f}")
    
    if epoch % args.valid_very_epoch == 0:
        v_loss,avg_top1,avg_top3= seg_valid(img_logger, valid_loader, cmpsd_model, seg_head, epoch)
        writer.add_scalar('Loss/valid', v_loss, epoch)
        writer.add_scalar("top1_acc/valid", avg_top1, epoch)
        writer.add_scalar("top3_acc/valid", avg_top3, epoch)
        # log_tsne(writer, epoch, valid_label_volume) #log_tsne will use the last feature collected by FEATURE_STORE dict
        log_layer_embeddings(
            writer=writer,
            epoch=epoch,
            label_volume=valid_label_volume,  # numpy array
            layer_order=LAYER_ORDER,       # from hook registration
            max_layers=12,
            mode="both",                   # <- t-SNE + UMAP stacked
            tsne_kwargs=dict(perplexity=20),
            umap_kwargs=dict(n_neighbors=30, min_dist=0.05,random_state=42,),
        )

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