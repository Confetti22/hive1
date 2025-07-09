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
        if hasattr(self._handle, "shape"):
            return tuple(int(x) for x in self._handle.shape)
        return self._handle.data.shape  # tifffile memmap

    # ------------------------------------------------------------- random block
    def read_block(self, *, offset: Tuple[int, int, int], size: Tuple[int, int, int]) -> np.ndarray:
        """Read a 3-D sub-volume starting at *offset* with *size* (all z-first)."""
        z, y, x = offset
        d, h, w = size
        if hasattr(self._handle, "from_roi"):
            coords = np.array([z, y, x, d, h, w])
            return self._handle.from_roi(coords=coords, level=0)  # IMS path
        # tifffile path – array-like interface supports basic slicing
        return self._handle[z : z + d, y : y + h, x : x + w]



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

def extract_features_to_zarr(
    *,
    vol_path: str | Path,
    model: nn.Module,
    zarr_path: str | Path,
    region_size: Tuple[int, int, int],
    roi_size: Tuple[int, int, int],
    roi_stride: Tuple[int, int, int],
    batch_size: int = 256,
    device: str = "cuda",
) -> None:
    """Slide through *vol_path*, extract CNN features and store them in a Zarr array.

    The function supports volumes that **do not** divide evenly by the tiling
    parameters – the last tiles are padded on the high side.
    """

    model.eval().to(device)

    step = [int(2*(1/2)*r_size/r_stride -1) for r_size, r_stride in zip(roi_size,roi_stride) ]
    step_size = roi_stride
    margin = [int(s * s_size) for s,s_size in zip(step,step_size)]
    region_stride  = [int(r_size - m) for r_size, m in zip(region_size,margin)]

    with VolumeReader(vol_path) as volume:
        # ----------------------------------------------------------------- grid
        d, h, w = volume.shape
        num_blocks = [
            math.ceil((d - region_size[0]) / region_stride[0]) + 1,
            math.ceil((h - region_size[1]) / region_stride[1]) + 1,
            math.ceil((w - region_size[2]) / region_stride[2]) + 1,
        ]

        # Number of feature vectors *per* block
        chunk_shape = [
            math.floor((region_size[0] - roi_size[0]) / roi_stride[0]) + 1,
            math.floor((region_size[1] - roi_size[1]) / roi_stride[1]) + 1,
            math.floor((region_size[2] - roi_size[2]) / roi_stride[2]) + 1,
        ]

        # Extract feature dimension from a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *roi_size, device=device)
            feat_dim = model(dummy).numel()

        zarr_shape = tuple(nb * cs for nb, cs in zip(num_blocks, chunk_shape)) + (feat_dim,)
        zarr_chunk = tuple(chunk_shape) + (feat_dim,)

        store = zarr.open(str(zarr_path), mode="w", shape=zarr_shape, dtype="float32", chunks=zarr_chunk)

        total_blocks = math.prod(num_blocks)
        pbar = tqdm(total=total_blocks, desc="Feature extraction", unit="block")

        # ----------------------------------------------------------- iterate grid
        for bz in range(num_blocks[0]):
            for by in range(num_blocks[1]):
                for bx in range(num_blocks[2]):
                    # ------------------------------------------------- offsets
                    z_off = bz * region_stride[0]
                    y_off = by * region_stride[1]
                    x_off = bx * region_stride[2]
                    offset = (z_off, y_off, x_off)

                    block = volume.read_block(offset=offset, size=region_size)

                    # Pad high side if we ran beyond volume edge
                    pad_z = max(0, region_size[0] - block.shape[0])
                    pad_y = max(0, region_size[1] - block.shape[1])
                    pad_x = max(0, region_size[2] - block.shape[2])
                    if pad_z or pad_y or pad_x:
                        block = np.pad(block, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")

                    dataset = SlidingWindowND(block, window=roi_size, stride=roi_stride)
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

                    feats = []
                    with torch.no_grad():
                        for patches in loader:
                            patches = patches.to(device)
                            out = model(patches)  # shape: [B, feat_dim]
                            feats.append(out.cpu().numpy())
                    feats_block = np.concatenate(feats, axis=0).reshape(*chunk_shape, feat_dim)

                    # -------------------------------------------- write to Zarr
                    z_slice = slice(bz * chunk_shape[0], (bz + 1) * chunk_shape[0])
                    y_slice = slice(by * chunk_shape[1], (by + 1) * chunk_shape[1])
                    x_slice = slice(bx * chunk_shape[2], (bx + 1) * chunk_shape[2])
                    store[z_slice, y_slice, x_slice, :] = feats_block.astype("float32")

                    pbar.update(1)
        pbar.close()

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

cfg = load_cfg("../config/rm009.yaml")
avg_pool = 3
cfg.avg_pool_size = [avg_pool] * 3

model = build_encoder_model(cfg, dims=3)
load_encoder2encoder(model, "/home/confetti/data/weights/rm009_3d_ae_best.pth")
vol_path= "/home/confetti/data/rm009/rm009_roi/z16176_z16299C4.tif"
save_zarr_path = "/home/confetti/data/rm009/rm009_roi/feats_z16176_z16299C4.zarr"
#%%

extract_features_to_zarr(
    vol_path= vol_path,
    model=model,
    zarr_path=save_zarr_path,
    region_size=(64, 1536, 1536),
    roi_size=(24,24,24),
    roi_stride=(4,4,4),
    batch_size=2048,
    device="cuda",
)

# --------------------------------------------------------------------- demo
img_coord = (10000, 5000, 3000)  # arbitrary voxel in raw image space
feat_idx = image_to_feature_coord(img_coord, img_offset=(0, 0, 0), roi_stride=(16, 16, 16))
print("Image coord", img_coord, "--> feature idx", feat_idx)



#%%
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
#                       q u i c k   v a l i d a t i o n   u t i l s
# -----------------------------------------------------------------------------

def summarise_zarr(path: str | Path):
    """Print basic metadata (shape, chunks, dtype)."""
    arr = zarr.open(str(path), mode="r")
    print(f"Shape  : {arr.shape}\nChunks : {arr.chunks}\nDType  : {arr.dtype}\n")


def plot_zarr_slices(path: str | Path, n: int = 6, *, pca_rgb: bool = False):
    """Maximum‑projection slices along Z/Y/X for a quick sanity check.

    Parameters
    ----------
    path : str | Path
        Zarr array path.
    n : int, default 6
        How many slices per axis to show.
    pca_rgb : bool, default False
        If *True*, also show an RGB PCA view of the middle Z‑slice.
    """
    arr = zarr.open(str(path), mode="r")
    D, H, W, C = arr.shape
    z_lin = np.linspace(0, D - 1, n, dtype=int)
    y_lin = np.linspace(0, H - 1, n, dtype=int)
    x_lin = np.linspace(0, W - 1, n, dtype=int)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    for i, z in enumerate(z_lin):
        axes[0, i].imshow(arr[z, :, :, :].max(-1), cmap="gray")
        axes[0, i].set_title(f"Z {z}")
        axes[0, i].axis("off")
    for i, y in enumerate(y_lin):
        axes[1, i].imshow(arr[:, y, :, :].max(-1), cmap="gray")
        axes[1, i].set_title(f"Y {y}")
        axes[1, i].axis("off")
    for i, x in enumerate(x_lin):
        axes[2, i].imshow(arr[:, :, x, :].max(-1).T, cmap="gray")
        axes[2, i].set_title(f"X {x}")
        axes[2, i].axis("off")

    plt.tight_layout()

    if pca_rgb:
        try:
            from sklearn.decomposition import PCA
            mid_z = D // 3
            flat = arr[mid_z].reshape(-1, C).astype(np.float32)
            pca = PCA(n_components=3).fit_transform(flat)
            rgb = (pca - pca.min(0)) / (pca.ptp(0) + 1e-7)
            rgb = rgb.reshape(H, W, 3)
            plt.figure(figsize=(6, 6))
            plt.title("Mid‑Z PCA‑RGB")
            plt.imshow(rgb)
            # plt.axis("off")
        except ImportError:
            print("Install scikit‑learn for PCA‑RGB view → `pip install scikit‑learn`.\n")
    plt.show()


#%%
summarise_zarr(save_zarr_path)
plot_zarr_slices(save_zarr_path, n=8, pca_rgb=True)
#%%
img_coord = (120,3499,5250)  # arbitrary voxel in raw image space
feat_idx = image_to_feature_coord(img_coord, img_offset=(0, 0, 0), roi_stride=(4,4,4))
print("Image coord", img_coord, "--> feature idx", feat_idx)
