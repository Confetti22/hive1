#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from torchvision import transforms
import numpy as np
import tifffile as tif
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from confettii.plot_helper import grid_plot_list_imgs
from lib.utils.gaussian_blur import GaussianBlur
import torch
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple

TensorLike = Union[np.ndarray, torch.Tensor]

def _to_float_tensor(x: TensorLike) -> torch.Tensor:
    """`np.ndarray` → float32 `torch.Tensor` (keeps channel-last layout)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.float()

def _normalize(x: torch.Tensor, p: int = 2, eps: float = 1e-6) -> torch.Tensor:
    """Normalise along the **last** (channel) axis."""
    return F.normalize(x, p=p, dim=-1, eps=eps)

def _gauss_kernel_1d(ksize: int, sigma: float = None, dtype=None, device=None):
    if sigma is None:                       # mimic your random σ ∈ [0.1, 2]
        sigma = float(torch.empty(1).uniform_(0.1, 2.0))
    half = (ksize - 1) // 2
    x = torch.arange(-half, half + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def _gauss_blur_2d(slice_hw_c: torch.Tensor, ksize: int = 11) -> torch.Tensor:
    """Depth-wise separable Gaussian blur for arbitrary channels."""
    H, W, C = slice_hw_c.shape
    hwc = slice_hw_c.permute(2,0,1).unsqueeze(0)      # (1,C,H,W)

    k1d = _gauss_kernel_1d(ksize, dtype=hwc.dtype, device=hwc.device)
    kx  = k1d.view(1,1,1,ksize).repeat(C,1,1,1)       # horizontal
    ky  = k1d.view(1,1,ksize,1).repeat(C,1,1,1)       # vertical

    pad = (ksize - 1) // 2
    hwc = F.pad(hwc, (pad,pad,pad,pad), mode="reflect")
    hwc = F.conv2d(hwc, kx, groups=C)
    hwc = F.conv2d(hwc, ky, groups=C)

    return hwc.squeeze(0).permute(1,2,0)              # (H,W,C)

def _laplacian_2d(slice_hw_c: torch.Tensor) -> torch.Tensor:
    """
    Apply a 3 × 3 Laplacian kernel channel-wise to one slice (H, W, C).
    """
    c_h_w = slice_hw_c.permute(2, 0, 1)           # → (C, H, W)
    C     = c_h_w.shape[0]
    lap   = torch.tensor([[0, 1, 0],
                          [1,-4, 1],
                          [0, 1, 0]], dtype=c_h_w.dtype, device=c_h_w.device)
    lap   = lap.view(1, 1, 3, 3).repeat(C, 1, 1, 1)   # (C,1,3,3)
    c_h_w = F.conv2d(c_h_w.unsqueeze(0), lap, groups=C).squeeze(0)
    return c_h_w.permute(1, 2, 0)                 # → (H, W, C)


def _process_volume(
    vol: torch.Tensor,
    fn_slice,
    norm_p: int
) -> torch.Tensor:
    """
    Apply a 2-D operation `fn_slice` to every z-slice of a 3-D volume.

    `vol` is (D, H, W, C); result keeps the same layout.
    """
    D, H, W, C = vol.shape
    out = []
    for z in range(D):
        res = fn_slice(vol[z])
        out.append(_normalize(res, p=norm_p))
    return torch.stack(out, dim=0)


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────
def get_edges_blur(
    feats: TensorLike,
    ksize: int = 11,
    norm_p: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    input: h,w,c or d,h,w,c
    Edge map by Gaussian blur subtraction.

    Returns `(orig_norm, blur_norm, edge_norm)`,
    all in the same layout as the input.
    """
    x = _to_float_tensor(feats)

    if x.ndim == 3:                                       # (H, W, C)
        orig  = _normalize(x, p=norm_p)
        blur  = _normalize(_gauss_blur_2d(x, ksize), p=norm_p)
        edge  = _normalize(orig - blur, p=norm_p)
        return orig, blur, edge

    elif x.ndim == 4:                                     # (D, H, W, C)
        orig = _process_volume(x, lambda s: s,             norm_p)
        blur = _process_volume(x, lambda s: _gauss_blur_2d(s, ksize), norm_p)
        edge = _normalize(orig - blur, p=norm_p)
        return orig, blur, edge

    else:
        raise ValueError("Input must be (H,W,C) or (D,H,W,C).")


def get_edges_laplacian(
    feats: TensorLike,
    norm_p: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    input: h,w,c or d,h,w,c
    Edge map by Laplacian filter.
    Returns `(orig_norm, lap_norm)` in the same layout as the input.
    """
    x = _to_float_tensor(feats)

    if x.ndim == 3:                                       # (H, W, C)
        orig = _normalize(x, p=norm_p)
        lap  = _normalize(_laplacian_2d(x), p=norm_p)
        return orig, lap

    elif x.ndim == 4:                                     # (D, H, W, C)
        orig = _process_volume(x, lambda s: s, norm_p)
        lap  = _process_volume(x, _laplacian_2d, norm_p)
        return orig, lap

    else:
        raise ValueError("Input must be (H,W,C) or (D,H,W,C).")

#%%
#first extract feats
#then rescale the feats_map into the same spatial shap as orignal img space
#then using edge_detector to draw boarder at the interface of different feature region
from collections import defaultdict
from functools import partial
from typing import Sequence, Tuple, Union, Literal, List

from config.load_config import load_cfg
from helper.contrastive_train_helper import (
    load_checkpoint,
    log_layer_embeddings,
)

from lib.arch.ae import build_final_model, load_compose_encoder_dict

Arr   = Union[np.ndarray, torch.Tensor]
Array = Union[Arr, Sequence[Arr]]   # single array or list/tuple of arrays
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
#define model 
cfg = load_cfg('../config/rm009.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
cfg.filters = [32,64,96]
cfg.kernel_size =[5,5,3]
cfg.mlp_filters =[96,48,24,12]
cfg.avg_pool_size = [8,8,8]
cfg.last_encoder= True
cfg.e5 = False

cmpsd_model = build_final_model(cfg).to(device)
cmpsd_model.eval()
register_hooks(cmpsd_model.cnn_encoder, "cnn.")
register_hooks(cmpsd_model.mlp_encoder, "mlp.")
print("Registered layers:", LAYER_ORDER)
#%%
#load ckpt
data_prefix = Path("/share/home/shiqiz/data" if cfg.e5 else "/home/confetti/data")
mlp_feat_names = ['rm009_postopk_1000.pth','rm009_smallestv1roi_oridfar256_l2_pool8_10000.pth','rm009_smallestv1roi_postopk_nview4_l2_avg8.pth','rm009_smallestv1roi_postopk_nview6_l2_avg8.pth']

cnn_ckpt = data_prefix / "weights" / "rm009_3d_ae_best.pth"
mlp_ckpt_path = data_prefix/ "weights"/mlp_feat_names[0]
load_compose_encoder_dict(cmpsd_model, str(cnn_ckpt),mlp_ckpt_path , dims=cfg.dims)

#input data
input_path="/home/confetti/data/rm009/rm009_roi/z16200_z16275_d64_hw1536.tif"
input=tif.imread(input_path)
inp = torch.from_numpy(input).unsqueeze(0).unsqueeze(0).float().to(device)
print(f"input.shape={input.shape}")

#extract feature and rescale into orignal image space
outs = cmpsd_model(inp).detach().cpu().numpy().squeeze() # np.ndarray
#%%
#selet feature from any specific activation layer
k = LAYER_ORDER[-1]
out_t = FEATURE_STORE[k][-1] if isinstance(FEATURE_STORE[k], (list, tuple)) else FEATURE_STORE[k]
feat = out_t.detach().cpu().squeeze().numpy()  # assume [C,D,H,W] or [D,H,W,C]? adjust below
print(f"{feat.shape= }")
#%%
# Reorder to [D,H,W,C] assuming channel-first
feat = np.moveaxis(feat, 0, -1)
print(f"{feat.shape= }")
#%%
# feat2d = feat[int(feat.shape[0]//2),:]            # [H,W,C]

wo_feats,wo_blurred,wo_subtracted_img=get_edges_blur(feat,ksize=13)

print(f"{wo_feats.shape=}")
print(f"{wo_blurred.shape=}")
print(f"{wo_subtracted_img.shape=}")
print("without downsample")

mean_last = lambda arr: np.mean(arr, axis=-1)
mean_list = [img.mean(-1) for img in (wo_feats, wo_blurred, wo_subtracted_img)]
grid_plot_list_imgs(images=mean_list)
#%%
lap_returns=get_edges_laplacian(feat)
lap_returns[0].shape
#%%
[print(f"{img.shape= }"for img in lap_returns)]
#%%
grid_plot_list_imgs(lap_returns)

# %%
# %%

wo_feats,wo_laplacian=get_edges_laplacian(features_list)
print("using laplacian,with downsample")
print("using laplacian,without downsample")
my_tools.plot([input,wo_feats,wo_laplacian])






#%%
import numpy as np

patch_size=32
offset=(515,512)

# %%
# i_pos and radius determines the location and size of the template patch
import numpy as np
i_pos_list=[np.array([380,380]),np.array([850,200]),np.array([300,160])]
i_pos=i_pos_list[0]
radius=20

#%%
# %%



#%%
import pickle
import add_one_confetti.add_one as my_tools
pth_ncc="results/simclr400_stride1_wo_pca_ncc.pkl"
with open(pth_ncc,'rb')as file:
    ncc_map=pickle.load(file)
print(ncc_map.shape)
my_tools.plot_ncc(ncc_map,i_pos,radius,save_path=None)

# ncc_map=transforms.ToTensor()(ncc_map.astype(np.float32))
# ncc_map=ncc_map.permute(0,2,1)
# ncc_map=nomalize(ncc_map)
# my_tools.plot(ncc_map)


#%%
import numpy as np
import matplotlib.pyplot as plt

# Assuming ncc_map is defined
# Example ncc_map for demonstration purposes

x = np.arange(0, ncc_map.shape[0])
y = np.arange(0, ncc_map.shape[1])
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()
levels = [1, 0.9, 0.7, 0.5, 0.4, 0, -0.2,-1]
#reverse the levels
levels=levels[::-1]
ncc_map=ncc_map[::-1]
CS = ax.contourf(X, Y, ncc_map, levels=levels)
CS_lines = ax.contour(X, Y, ncc_map, levels=levels, colors='black')
ax.clabel(CS_lines, inline=True, fontsize=2)
ax.set_title('Contour Plot with Specific Levels')
plt.colorbar(CS)  # Optional: Add a colorbar to the plot
plt.show()
# %%
#!/usr/bin/env python3

# %%
