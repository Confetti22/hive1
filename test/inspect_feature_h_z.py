#%%
import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
#%%

from pathlib import Path
from typing import Dict, Union
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from lib.arch.ae import ConvMLP
import matplotlib.pyplot as plt

from confettii.plot_helper import three_pca_as_rgb_image
from sklearn.manifold import TSNE
from scipy.ndimage import zoom

from config.load_config import load_cfg
from helper.contrastive_train_helper import (
    get_t11_eval_data,
    valid_from_roi,
    MLP,
    load_checkpoint,
    tsne_grid_plot,
    plot_pca_maps,
)
from lib.arch.ae import build_final_model, load_compose_encoder_dict

def valid_from_roi(model, it, eval_data, writer):
    """Evaluate a model on a list of ROIs.

    Works with feature tensors of shape:
        • (C, H, W)                      – old behaviour
        • (C, D, H, W)                  – channel-first 3-D
        • (D, H, W, C)                  – channel-last 3-D
    For 3-D inputs, the middle depth slice (z = D//2) is used for PCA/t-SNE.
    """
    model.eval()

    pca_img_lst               = []
    tsne_encoded_feats_lst    = []
    tsne_label_lst            = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx, data_dic in enumerate(eval_data):
        roi      = data_dic['img']          # numpy (H,W) or (D,H,W)
        label    = data_dic['label']

        inp = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0).float().to(device)
        outs = model(inp).detach().cpu().numpy().squeeze()        # np.ndarray

        if outs.ndim == 3:                                  # (C, H, W)
            feats_map = np.moveaxis(outs, 0, -1)            # → (H, W, C)
        elif outs.ndim == 4:
            C, D, H, W = outs.shape
            z_mid = D // 2
            feats_map = np.moveaxis(outs[:, z_mid], 0, -1)   # (H, W, C)
        H, W, C = feats_map.shape
        feats_flat = feats_map.reshape(-1, C)
        # ---------------------------------------------------------------- visualisations
        rgb_img = three_pca_as_rgb_image(feats_flat, (H, W))
        pca_img_lst.append(rgb_img)
        tsne_encoded_feats_lst.append(feats_flat)

        # ---------------------------------------------------------------- labels (resampled to H×W)
        zoom_factor = [y / x for y, x in zip((H, W), label.shape)]
        label_rs    = zoom(label, zoom_factor, order=0)
        tsne_label_lst.append(label_rs.ravel())

    # ---------- logging ----------
    tsne_grid_plot(tsne_encoded_feats_lst,tsne_label_lst,writer,tag=f"tsne/{tag_suffix}",step=1)
    # plt.tight_layout()
    # plt.show()
    plot_pca_maps(pca_img_lst,writer=writer,tag=f"pca/{tag_suffix}",step=it,ncols=len(pca_img_lst))

# ─────────────────────────────────────────────────────────────
# 1.  ─ helper: log eigenvalue distribution ───────────────────
# ─────────────────────────────────────────────────────────────
def log_layer_eigen_hist(mlp: nn.Module,
                         layer_idx: int,
                         writer: SummaryWriter,
                         tag_prefix: str,
                         step: int = 0):
    """
    Compute eigenvalues of  W·Wᵀ  for the *hidden* Linear layer #layer_idx
    (this is the square of the singular values of W) and push a histogram
    to TensorBoard.

    Parameters
    ----------
    mlp        : the patched `MLPWithIntermediates` holding current weights
    layer_idx  : 0-based index within `mlp.layers`
    writer     : global SummaryWriter
    tag_prefix : something like 'eig/<ckpt_stem>'
    step       : global_step in TensorBoard (use layer_idx or any counter)
    """
    # grab weight — shape [out_features, in_features]
    W = mlp.layers[layer_idx].weight.detach().cpu().numpy()
    # squared singular values  =  eigenvalues( W·Wᵀ ); guaranteed real, ≥ 0
    eigvals = np.linalg.eigvals(W @ W.T).real      #  shape (out_features,)

    writer.add_histogram(
        tag  = f"{tag_prefix}/layer{layer_idx}",
        values = eigvals,
        global_step = step
    )


def log_conv1x1_eigen_spectrum(
    mlp: ConvMLP,
    layer_idx: int,
    writer: SummaryWriter,
    tag_prefix: str,
    step: int = 0,
):
    """
    Log the squared eigen-value spectrum of a 1×1 Conv layer’s weight matrix.

    Plots (rank, λ²) where λ are the eigen-values of  W · Wᵀ  (so λ ≥ 0).
    The curve is sent to TensorBoard with writer.add_figure.
    """
    # --------------------- pull weight and form 2-D matrix --------------------
    W = mlp.layers[layer_idx].weight              # (out_c, in_c, 1[, 1])
    out_c, in_c = W.shape[:2]
    Wmat = W.view(out_c, in_c)                    # (out_c, in_c)

    # --------------------- eigen-values, square, sort -------------------------
    eigvals = np.linalg.eigvals(
        (Wmat @ Wmat.T).detach().cpu().numpy()
    ).real                                        # λ ≥ 0
    eigvals_sq = np.square(eigvals)               # λ²
    eigvals_sq_sorted = np.sort(eigvals_sq)[::-1] # descending

    # --------------------- build figure ---------------------------------------
    ranks = np.arange(1, eigvals_sq_sorted.size + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, eigvals_sq_sorted, marker="o", lw=1)
    ax.set_xlabel("Rank (1 = largest)")
    ax.set_ylabel(r"$\lambda^2$")
    ax.set_title(f"Squared eigen-value spectrum – layer {layer_idx}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # --------------------- push to TensorBoard --------------------------------
    writer.add_figure(f"{tag_prefix}/layer{layer_idx}_spectrum", fig, global_step=step)
    plt.close(fig)
# -----------------------------------------------------------
# 1.  -- utilities ------------------------------------------
# -----------------------------------------------------------
class MLPWithIntermediates(MLP):
    """
    Extension of your original MLP that can return the raw (L2-normalised)
    feature map after *any* intermediate layer, or all layers in one go.
    """
    def forward(self, x,
                until: int | None = None,      # 0-based layer index, None = full net
                return_all: bool = False):     # True -> list[Tensor] for every layer
        feats = []
        for li, layer in enumerate(self.layers):
            x = layer(x)
            if li != len(self.layers) - 1:       # no ReLU on the head
                x = self.relu(x)
            # store *before* the final L2 normalisation
            feats.append(x.clone())

        # L2-normalise every requested feature map
        feats = [f / f.norm(p=2, dim=-1, keepdim=True) for f in feats]

        if return_all:
            return feats
        if until is not None:
            return feats[until]
        return feats[-1]                       # default behaviour (as before)

class ConvMLPWithIntermediates(ConvMLP):
    """
    Same API as the dense MLP variant:
      • until = k   → return features after layer k
      • return_all  → return list for every layer
    """
    def forward(self, x, until: int | None = None, return_all: bool = False):
        feats = []
        for li, layer in enumerate(self.layers):
            x = layer(x)
            if li != len(self.layers) - 1:         # no ReLU on the head
                x = self.relu(x)
            # L2-norm across channel dim
            feats.append(x / x.norm(p=2, dim=1, keepdim=True))

        if return_all:
            return feats
        if until is not None:
            return feats[until]
        return feats[-1]

class EncoderWrapper(nn.Module):
    """Return features **after layer `li`** of the Conv-MLP."""
    def __init__(self, backbone, layer_idx):
        super().__init__()
        self.backbone  = backbone
        self.layer_idx = layer_idx

    def forward(self, x):
        x = self.backbone.cnn_encoder(x)               # B,C,(D,)H,W
        x = self.backbone.mlp_encoder(x, until=self.layer_idx)  # same shape
        return x.squeeze(0)                            # drop batch → C,(D,)H,W



#%%

cfg = load_cfg("config/rm009.yaml")
writer = SummaryWriter(log_dir="outs/eval_mlp_layers")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_data = get_t11_eval_data(E5=cfg.e5,img_no_list=[1])
data_prefix = Path("/share/home/shiqiz/data" if cfg.e5 else "/home/confetti/data")


mlp_feat_names = ['rm009_smallestv1roi_oridfar256_l2_pool8_10000.pth','rm009_smallestv1roi_postopk_nview4_l2_avg8.pth','rm009_smallestv1roi_postopk_nview6_l2_avg8.pth']
cnn_ckpt = data_prefix / "weights" / "rm009_3d_ae_best.pth"
mlp_ckpt_path_lst = [data_prefix/ "weights"/ckpt_name for ckpt_name in mlp_feat_names]

cfg.filters = [6,12,24]
cfg.mlp_filters =[24,16,12]
cfg.kernel_size =[5,5,3]
cfg.last_encoder= True

cmpsd_model = build_final_model(cfg).to(device)
cmpsd_model.eval()
cmpsd_model_ckpt_path = '/home/confetti/e5_workspace/hive1/outs/seg_head/training_from_scratch_level3_avg_pool7/model_epoch_1900.pth'
cmpsd_model_ckpt = torch.load(cmpsd_model_ckpt_path)
cmpsd_model.load_state_dict(cmpsd_model_ckpt['cmpsd_model'])
#%%

cmpsd_model.mlp_encoder = ConvMLPWithIntermediates(cfg.mlp_filters,
                                                   dims=cfg.dims).to(device)

ckpt_tag = 'training_from_scratch_level3_avg_pool7_1900'
n_layers = len(cmpsd_model.mlp_encoder.layers)

for li in range(n_layers):
    print(f"begin####{ckpt_tag}/layer{li}#####")
    wrapped = EncoderWrapper(cmpsd_model, li).to(device)
    tag_suffix = f"{ckpt_tag}/layer{li}"

    # --- PCA & t-SNE ---
    valid_from_roi(wrapped, li, eval_data, writer)   # inside: tag+=tag_suffix
    # --- eigen-histogram ---
    log_conv1x1_eigen_spectrum(cmpsd_model.mlp_encoder,li, writer, f"eig/{ckpt_tag}", li)

writer.close()
print("✅  Finished validating every checkpoint & every MLP layer.")

# -----------------------------------------------------------
# 3.  -- evaluation loop ------------------------------------
# for ckpt_path in mlp_ckpt_path_lst:
#     load_checkpoint(ckpt_path, cmpsd_model.mlp_encoder)
#     ckpt_tag = Path(ckpt_path).stem
#     n_layers = len(cmpsd_model.mlp_encoder.layers)

#     for li in range(n_layers):
#         print(f"begin####{ckpt_tag}/layer{li}#####")
#         wrapped = EncoderWrapper(cmpsd_model, li).to(device)
#         tag_suffix = f"{ckpt_tag}/layer{li}"

#         # --- PCA & t-SNE ---
#         valid_from_roi(wrapped, li, eval_data, writer)   # inside: tag+=tag_suffix

#         # --- eigen-histogram ---
#         log_conv1x1_eigen_spectrum(cmpsd_model.mlp_encoder,li, writer, f"eig/{ckpt_tag}", li)

# writer.close()
# print("✅  Finished validating every checkpoint & every MLP layer.")
#%%

print(cmpsd_model)

# %%
