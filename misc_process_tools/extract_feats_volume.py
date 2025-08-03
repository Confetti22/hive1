#%%
#!/usr/bin/env python3
"""Refactored 3-D feature-extraction pipeline

This script generalises the original IMS-specific workflow so it can also work
with large multi-page TIFF volumes.  Key improvements:

* Modular design – clearly separated I/O, tiling logic, feature extraction and
  Zarr writing.
* Dimension-safe indexing – no more hard-coded axis lengths, preventing the
  off-by-one bug in the original `index` slice.
* Works with arbitrary encoder back-bones; the only requirement is that the
  model returns a 1-D feature vector for every patch.
* A helper `image_to_feature_coord()` converts a coordinate in raw image space
  to the corresponding index in the feature volume.
* Extensive type hints and docstrings for better readability and IDE support.

Author: ChatGPT (o3) – 2025-07-08
"""


from pathlib import Path
from typing import Iterable, Tuple
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import zarr
from tifffile import TiffFile, memmap
from confettii.feat_extract import SlidingWindowND
from typing import Tuple, Union

# -----------------------------------------------------------------------------
#                       I / O   a n d   v o l u m e   r e a d e r
# -----------------------------------------------------------------------------
class VolumeReader:
    """Handle large 3-D volumes stored as `.ims` or multi-page `.tiff`.

    The class exposes a unified API:

    ```python
    with VolumeReader(path) as vol:
        patch = vol.read_block(offset=(z,y,x), size=(d,h,w))
    ```
    """

    def __init__(self, path: str | Path, channel: int = 0):
        self.path = Path(path)
        self.channel = channel
        self._handle = None  # Lazily opened

    # ------------------------------------------------------------------ context
    def __enter__(self):
        suffix = self.path.suffix.lower()
        if suffix == ".ims":
            from helper.image_reader import Ims_Image  # local import to avoid heavy deps
            self._handle = Ims_Image(str(self.path), channel=self.channel)
        elif suffix in {".tif", ".tiff"}:
            # Memory-map to spare RAM – works well for large volumes
            self._handle = memmap(self.path)
        else:
            raise ValueError(f"Unsupported volume format: {self.path.suffix}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None and hasattr(self._handle, "close"):
            self._handle.close()

    # --------------------------------------------------------------------- meta
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return full volume shape (D, H, W)."""
        if hasattr(self._handle, "rois"): #if ims image
            return tuple(int(x) for x in self._handle.rois[0][3:])
        return self._handle.data.shape  # tifffile memmap

    # ------------------------------------------------------------- random block
    def read_block(self, *, offset: Tuple[int, int, int], size: Tuple[int, int, int]) -> np.ndarray:
        """Read a 3-D sub-volume starting at *offset* with *size* (all z-first)."""
        z, y, x = offset
        d, h, w = size
        if hasattr(self._handle, "from_roi"):
            coords = np.array([z, y, x, d, h, w])  # IMS path
            return self._handle.from_roi(coords=coords, level=0)  
        return self._handle[z : z + d, y : y + h, x : x + w]  # tifffile path 

# -----------------------------------------------------------------------------
#                  C o o r d i n a t e   m a p p i n g   h e l p e r
# -----------------------------------------------------------------------------

def image_to_feature_coord(img_coord: Tuple[int, int, int], *,
                           img_offset: Tuple[int, int, int],
                           roi_stride: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Map *img_coord* (z, y, x) to the corresponding feature volume index.

    Parameters
    ----------
    img_coord : tuple[int,int,int]
        Raw image coordinate.
    img_offset : tuple[int,int,int]
        Offset of the *first* processed voxel (often the beginning of the region).
    roi_stride : tuple[int,int,int]
        Stride of patch centres along each axis.
    """
    return tuple(((c - o) // s) for c, o, s in zip(img_coord, img_offset, roi_stride))

# -----------------------------------------------------------------------------
#                           F e a t u r e   e x t r a c t o r
# -----------------------------------------------------------------------------


# ────────────────────────────────────────────────────────── helpers
def _lookup(module: nn.Module, attr_path: str) -> nn.Module:
    tgt = module
    for attr in attr_path.split("."):
        tgt = getattr(tgt, attr) if attr else tgt
    return tgt


def _register_hook(layer: nn.Module, buffer: dict[str, torch.Tensor]):
    def hook(_, __, out):
        buffer["feat"] = out.detach()

    return layer.register_forward_hook(hook)


# ───────────────────────────────────────────────── extract
def extract_features_to_zarr(
    *,
    vol_path: Union[str, Path],
    channel:int = 0, #channel for ims image
    model: nn.Module,
    zarr_path: Union[str, Path],
    global_offset: Tuple[int,int,int]= (0,0,0),
    whole_volume_size =None,
    region_size: Tuple[int, int, int],
    roi_size: Tuple[int, int, int],
    roi_stride: Tuple[int, int, int],
    batch_size: int = 256,
    device: str = "cuda",
    # NEW ↓
    layer_path: str = "",           # path to layer inside the model (“” = model output)
    pool_size: int | None = None,   # applied after model, so if model contains pool_size, do not apply twice
) -> None:
    """Extract a *single* feature map (optionally pooled) and store it in Zarr."""
    model.eval().to(device)

    # Hook the requested layer
    layer = model if layer_path == "" else _lookup(model, layer_path)
    activ: dict[str, torch.Tensor] = {}
    handle = _register_hook(layer, activ)

    pool = None
    if pool_size and pool_size > 1:
        pool = nn.AvgPool3d(kernel_size=[pool_size] * 3, stride=1, padding=0).to(device)

    # ---------------------------------------------------------------- size probe
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *roi_size, device=device)
        _ = model(dummy)
        feat_sample = activ["feat"]
        if pool:
            feat_sample = pool(feat_sample)
        feat_dim = feat_sample.numel()

    # ───────────────────────────── Zarr grid bookkeeping (unchanged logic)
    step = [int(2 * (1 / 2) * r_size / r_stride - 1) for r_size, r_stride in zip(roi_size, roi_stride)]
    margin = [int(s * s_size) for s, s_size in zip(step, roi_stride)]
    region_stride = [int(r_size - m) for r_size, m in zip(region_size, margin)]

    with VolumeReader(vol_path,channel=channel) as volume:
        if whole_volume_size:
            d, h, w = whole_volume_size
        else:
            d, h, w = volume.shape
        num_blocks = [
            math.ceil((d - region_size[0]) / region_stride[0]) + 1,
            math.ceil((h - region_size[1]) / region_stride[1]) + 1,
            math.ceil((w - region_size[2]) / region_stride[2]) + 1,
        ]

        chunk_shape = [
            math.floor((region_size[0] - roi_size[0]) / roi_stride[0]) + 1,
            math.floor((region_size[1] - roi_size[1]) / roi_stride[1]) + 1,
            math.floor((region_size[2] - roi_size[2]) / roi_stride[2]) + 1,
        ]

        zarr_shape = tuple(nb * cs for nb, cs in zip(num_blocks, chunk_shape)) + (feat_dim,)
        zarr_chunk = tuple(chunk_shape) + (feat_dim,)
        store = zarr.open(str(zarr_path), mode="w", shape=zarr_shape, dtype="float32", chunks=zarr_chunk)

        pbar = tqdm(total=math.prod(num_blocks), unit="block", desc="Feature extraction")

        # --------------------------------------------------- iterate volume grid
        for bz in range(num_blocks[0]):
            for by in range(num_blocks[1]):
                for bx in range(num_blocks[2]):
                    offset = (
                        bz * region_stride[0] + global_offset[0],
                        by * region_stride[1] + global_offset[1],
                        bx * region_stride[2] + global_offset[2],
                    )
                    block = volume.read_block(offset=offset, size=region_size)

                    pad = [max(0, region_size[i] - block.shape[i]) for i in range(3)]
                    if any(pad):
                        block = np.pad(block, [(0, pad[0]), (0, pad[1]), (0, pad[2])], mode="constant")

                    dataset = SlidingWindowND(block, window=roi_size, stride=roi_stride)
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

                    feats_all = []
                    with torch.no_grad():
                        for patches in loader:
                            patches = patches.to(device)
                            _ = model(patches)            # fills activ["feat"]
                            feat = activ["feat"]
                            if pool:
                                feat = pool(feat)
                            feats_all.append(feat.flatten(1).cpu())  # [B, feat_dim]

                    feats_block = torch.cat(feats_all, dim=0).numpy().reshape(*chunk_shape, feat_dim)

                    store[
                        slice(bz * chunk_shape[0], (bz + 1) * chunk_shape[0]),
                        slice(by * chunk_shape[1], (by + 1) * chunk_shape[1]),
                        slice(bx * chunk_shape[2], (bx + 1) * chunk_shape[2]),
                        :
                    ] = feats_block.astype("float32")

                    pbar.update(1)
        pbar.close()

    handle.remove()
    print(f"Finished – features saved to {zarr_path}")

# -----------------------------------------------------------------------------
#                               e n t r y   p o i n t
# -----------------------------------------------------------------------------
    
#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
from config.load_config import load_cfg
from lib.arch.ae import build_encoder_model, load_encoder2encoder 
#%%

cfg = load_cfg("config/t11_3d.yaml")
avg_pool = 8 
cfg.avg_pool_size = [avg_pool] * 3
cfg.last_encoder = True 
# cfg.filters = [32,64]
# cfg.kernel_size = [5,5]
# %%

#%%
E5 =False
if E5:
    data_prefix = "/share/home/shiqiz/data"
    workspace_prefix = "/share/home/shiqiz/workspace/hive1"
else:
    data_prefix = "/home/confetti/data"
    workspace_prefix = '/home/confetti/e5_workspace/hive1'


model = build_encoder_model(cfg, dims=3)
load_encoder2encoder(model, f"{data_prefix}/weights/t11_3d_ae_best2.pth")
vol_path = "/home/confetti/e5_data/t1779/t1779.ims" 
# vol_path = '/share/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/Analysis/ROIReconstruction/ROIImage/z13750_c1.ims'
save_zarr_path = f"{data_prefix}/t1779/test_ae_feats_nissel_l3_avg8_rhemisphere.zarr"

#%%
extract_features_to_zarr(
    vol_path= vol_path,
    channel=2,
    model=model,
    zarr_path=save_zarr_path,
    # global_offset=(3392,2000,7008),
    global_offset=(6400,2000,7008),
    # whole_volume_size=(6784,5024,4200),
    whole_volume_size=(64,5024,4200),
    region_size=(64, 1536, 1536),
    roi_size=(64,64,64),
    roi_stride=(16,16,16),
    batch_size= 1024,
    device="cuda",
    # layer_path="down_layers.0",  # pick *one* internal layer
    # pool_size=8,                 # or None / 1 for “no pooling”
)


#%%
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
#                       q u i c k   v a l i d a t i o n   u t i l s
# -----------------------------------------------------------------------------

def summarise_zarr(path: str | Path):
    """Print basic metadata (shape, chunks, dtype)."""
    arr = zarr.open(str(path), mode="r")
    print(f"Shape  : {arr.shape}\nChunks : {arr.chunks}\nDType  : {arr.dtype}\n")


def plot_zarr_slices(path: str | Path, n: int = 6, *, pca_rgb: bool = False, channel_axis: int = -1):
    """Maximum-projection slices along Z/Y/X for a quick sanity check.

    Parameters
    ----------
    path : str | Path
        Zarr array path.
    n : int, default 6
        How many slices per axis to show.
    pca_rgb : bool, default False
        If *True*, also show an RGB PCA view of the middle Z-slice.
    channel_axis : int, default -1
        Channel axis location. Use -1 for channel-last (D,H,W,C),
        or 0 for channel-first (C,D,H,W).
    """
    if channel_axis not in (-1, 0):
        raise ValueError("channel_axis must be -1 (C-last) or 0 (C-first).")

    arr = zarr.open(str(path), mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D array, got shape {arr.shape}.")

    if channel_axis == -1:
        D, H, W, C = arr.shape
        z_img = lambda z: arr[z, :, :, :].max(-1)
        y_img = lambda y: arr[:, y, :, :].max(-1)
        x_img = lambda x: arr[:, :, x, :].max(-1).T
        get_mid_slice_for_pca = lambda mid_z: arr[mid_z, :, :, :]  # (H, W, C)
    else:  # channel_axis == 0  -> (C, D, H, W)
        C, D, H, W = arr.shape
        z_img = lambda z: arr[:, z, :, :].max(0)
        y_img = lambda y: arr[:, :, y, :].max(0)
        x_img = lambda x: arr[:, :, :, x].max(0).T
        # Convert (C, H, W) -> (H, W, C) for PCA
        get_mid_slice_for_pca = lambda mid_z: np.moveaxis(arr[:, mid_z, :, :], 0, -1)

    z_lin = np.linspace(0, D - 1, n, dtype=int)
    y_lin = np.linspace(0, H - 1, n, dtype=int)
    x_lin = np.linspace(0, W - 1, n, dtype=int)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    for i, z in enumerate(z_lin):
        axes[0, i].imshow(z_img(z), cmap="gray")
        axes[0, i].set_title(f"Z {z}")
        axes[0, i].axis("off")
    for i, y in enumerate(y_lin):
        axes[1, i].imshow(y_img(y), cmap="gray")
        axes[1, i].set_title(f"Y {y}")
        axes[1, i].axis("off")
    for i, x in enumerate(x_lin):
        axes[2, i].imshow(x_img(x), cmap="gray")
        axes[2, i].set_title(f"X {x}")
        axes[2, i].axis("off")

    plt.tight_layout()

    if pca_rgb:
        try:
            from sklearn.decomposition import PCA
            mid_z = D // 2
            mid = get_mid_slice_for_pca(mid_z)   # (H, W, C)
            Hm, Wm, Cm = mid.shape
            if Cm < 3:
                print(f"PCA-RGB needs >=3 channels, but got C={Cm}. Skipping.")
            else:
                flat = mid.reshape(-1, Cm).astype(np.float32)
                pca = PCA(n_components=3).fit_transform(flat)
                rgb = (pca - pca.min(0)) / (pca.ptp(0) + 1e-7)
                rgb = rgb.reshape(Hm, Wm, 3)
                plt.figure(figsize=(6, 6))
                plt.title("Mid-Z PCA-RGB")
                plt.imshow(rgb)
                # plt.axis("off")
        except ImportError:
            print("Install scikit-learn for PCA-RGB view → `pip install scikit-learn`.\n")
    plt.show()

#%%
save_zarr_path = f"/home/confetti/data/t1779/test_feat3_l3_avg8_rhemisphere.zarr"
summarise_zarr(save_zarr_path)
plot_zarr_slices(save_zarr_path, n=8, pca_rgb=True,channel_axis=-1)
#%%
img_coord = (120,3499,5250)  # arbitrary voxel in raw image space
feat_idx = image_to_feature_coord(img_coord, img_offset=(0, 0, 0), roi_stride=(8,8,8))
print("Image coord", img_coord, "--> feature idx", feat_idx)
