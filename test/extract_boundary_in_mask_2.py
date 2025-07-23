"""
Cortical Layer Boundary Extraction (3‑D)
======================================

This module repairs noisy 3‑D cortical laminar masks and extracts a *single*
binary boundary volume marking interfaces **between labeled cortical layers**
(ignoring background). It is designed for large histology volumes like your
V1 ROI with shape `(Z, Y, X) = (64, 3500, 5250)` and isotropic 4 µm voxels.

**Input label coding (your data):**

```
0  -> background (incl. white matter)
1  -> L1
8  -> L23
2  -> L4A
3  -> L4B
6  -> L4Ca
4  -> L4Cb
5  -> L5
7  -> L6
```

We remap these to *ordered* indices 1..8 internally:

```
ORDERED_LAYERS = [1, 8, 2, 3, 6, 4, 5, 7]
# becomes → 1..8
# mapping_raw2ord[raw_code] = ordered_index
# mapping_ord2raw[ordered_index] = raw_code
```

### Supported Local Pattern Types
Because some columns / sub‑regions of your ROI may contain only a subset of
layers (common in V1 specializations), we support:

1. **Full lamination** – all 8 layers present.
2. **Only L1** – e.g., pia/dura region or annotation artifact.
3. **Only L4B** – e.g., blob‑style annotations for key lamina.

The pipeline never *hallucinates* missing layers; it repairs holes and closes
gaps *within* the labels that are actually present. Inter‑label boundaries are
emitted **only between voxels of two different nonzero labels actually present
in the cleaned volume**.

---

## Gap Statistics & Default Structuring Sizes
- Typical layer thicknesses (vox):
  - L1 ~45
  - L23 ~60
  - L4A ~70
  - L4B ~50
  - L4Ca ~40
  - L4Cb ~55
  - L5 ~70
  - L6 ~40
- Max observed inter‑layer gap usually <15 vox; rare outliers up to ~30 vox.
- Slow variation along Z: each boundary surface shifts <~100 vox over the 64‑slice stack.

These priors drive *adaptive* erosion radii, closing radii, and slice‑wise
smoothing.

---

## Pipeline Overview

**High‑level steps** (all 3‑D, but memory‑aware / slice‑wise options provided):

1. **Remap labels → ordered indices 1..8** (fast int LUT) for convenience.
2. **Per‑slice cleaning (Y,X plane)** – fill small holes, drop speckles, mild
   closing to bridge ≤15‑voxel gaps; optional more aggressive 30‑voxel bridging.
3. **3‑D domain mask** = union of all cleaned slices; fill enclosed voids.
4. **Seed generation** – safe erosion (≤4 voxels) of each label to create
   “confident cores”. Fallback to centroid seed if layer vanishes after erosion.
5. **3‑D seeded watershed** (skimage) constrained to cortex domain to close
   inter‑label gaps (handles up to ~30‑voxel voids). No OpenCV used.
6. **Z‑axis slow‑change smoothing** – majority / median filter of label index
   across small Z windows; optional morphological voting to remove one‑slice
   spikes inconsistent with <100‑voxel drift prior.
7. **Boundary extraction** – identify voxel faces where neighboring voxels carry
   *different* nonzero labels; background ignored. Combines all layer‑adjacency
   surfaces into one binary volume.
8. **Surface thinning** – reduce multi‑voxel thick boundaries to ~1‑voxel
   (or keep 2‑voxel safety band if training needs class weight balancing).
9. **Optional re‑map back to raw codes** if you need a cleaned multi‑label
   volume in the original code space.

---

## Memory & Performance Notes
Your volume is large (~64×3500×5250 ≈ 1.18e9 vox).

- Use `np.uint8` or `np.uint16` where possible.
- Consider **memory‑mapped arrays** (e.g., `np.memmap` to `.npy` file) for
  intermediates.
- Many operations run **slice‑wise** (loop over Z) to reduce peak RAM.
- Watershed step can be done either full‑volume (requires RAM) or in Z blocks;
  we provide both; default = full‑volume if memory allows.

---

## Quick Start

```python
import numpy as np
import tifffile  # or your preferred IO
from cortical_layer_boundary_3d import compute_cortical_boundaries

L_raw = tifffile.imread("v1_roi_labels.tif")  # shape (64, 3500, 5250)

boundary_mask, L_clean_ord, L_clean_raw = compute_cortical_boundaries(
    L_raw,
    do_full_watershed=True,      # False will do slice‑wise gap closing only
    thin=True,
)

# Save
tifffile.imwrite("v1_roi_boundaries_uint8.tif", boundary_mask.astype(np.uint8))
```

---

## Module Code
"""

from __future__ import annotations
import numpy as np
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional
import os 
from scipy import ndimage as ndi
import tifffile
from pathlib import Path
from skimage import morphology, segmentation, measure, util

# -----------------------------------------------------------------------------
# Label mapping
# -----------------------------------------------------------------------------



# Raw codes (your data) in pia→white order
DEFAULT_LABEL_ORDER_RAW = (1, 8, 2, 3, 6, 4, 5, 7)
N_LAYERS = len(DEFAULT_LABEL_ORDER_RAW)

# build LUTs
_raw2ord = {raw: i+1 for i, raw in enumerate(DEFAULT_LABEL_ORDER_RAW)}  # 1..8
_ord2raw = {i+1: raw for i, raw in enumerate(DEFAULT_LABEL_ORDER_RAW)}

import numpy as np

def block_mode_labels(L: np.ndarray, factors=(8,8,8), prefer_nonzero=True) -> np.ndarray:
    """
    Downsample a uint8 label volume by blockwise mode pooling.
    factors: (fz, fy, fx) ints.
    If prefer_nonzero=True, ties that include nonzero labels favor the most frequent nonzero.
    """
    fz, fy, fx = factors
    Z, Y, X = L.shape

    # Trim to exact multiples (pad if you prefer; here we crop)
    Z2, Y2, X2 = Z // fz, Y // fy, X // fx
    Lc = L[:Z2*fz, :Y2*fy, :X2*fx]

    # Reshape into blocks: (Z2, fz, Y2, fy, X2, fx)
    B = Lc.reshape(Z2, fz, Y2, fy, X2, fx)
    # Move block dims last, flatten block
    B = B.transpose(0,2,4,1,3,5).reshape(Z2, Y2, X2, fz*fy*fx)

    # bincount per block
    max_label = int(L.max())
    out = np.zeros((Z2, Y2, X2), dtype=np.uint8)
    for idx in np.ndindex(Z2, Y2, X2):
        block = B[idx]
        counts = np.bincount(block, minlength=max_label+1)

        if prefer_nonzero:
            # If any nonzero present, zero loses unless it dominates alone
            if counts[1:].sum() > 0:
                counts0 = counts.copy()
                counts0[0] = 0
                lab = np.argmax(counts0)
            else:
                lab = 0
        else:
            lab = np.argmax(counts)
        out[idx] = lab
    return out


def remap_raw_to_ordered(L_raw: np.ndarray, background: int = 0) -> np.ndarray:
    """Map raw laminar label codes to ordered indices 1..N.

    Unknown / background codes become 0.
    Works in‑place safe if you copy first.
    """
    L_ord = np.zeros_like(L_raw, dtype=np.uint8)
    for r, o in _raw2ord.items():
        L_ord[L_raw == r] = o
    # anything else stays 0
    return L_ord


def remap_ordered_to_raw(L_ord: np.ndarray) -> np.ndarray:
    L_raw = np.zeros_like(L_ord, dtype=np.uint8)
    for o, r in _ord2raw.items():
        L_raw[L_ord == o] = r
    return L_raw

# -----------------------------------------------------------------------------
# Cleaning utilities (2‑D slice level)
# -----------------------------------------------------------------------------

@dataclass
class CleanParams:
    hole_area: int = 400   # 2‑D area threshold per slice
    min_area: int = 400    # remove speckles smaller than this
    close_r_small: int = 3 # mild closing (bridges ≤3 vox radii ~ 6 vox gap)
    close_r_big: int = 8   # optional extra pass (bridges ≤16 vox gap)
    do_big_close: bool = True


def _clean_slice(mask2d: np.ndarray, p: CleanParams) -> np.ndarray:
    """Clean a single 2‑D binary slice."""
    if mask2d.dtype != bool:
        mask2d = mask2d.astype(bool)
    out = morphology.remove_small_holes(mask2d, area_threshold=p.hole_area)
    out = morphology.remove_small_objects(out, min_size=p.min_area)
    if p.close_r_small > 0:
        out = morphology.binary_closing(out, morphology.disk(p.close_r_small))
    if p.do_big_close and p.close_r_big > p.close_r_small:
        out = morphology.binary_closing(out, morphology.disk(p.close_r_big))
    return out


def clean_volume_slicewise(L_ord: np.ndarray, p: CleanParams) -> np.ndarray:
    """Apply slice‑wise cleaning to each label independently.

    Returns a cleaned ordered‑label volume (same shape) with holes patched and
    thin gaps bridged in‑plane.
    """
    Z, Y, X = L_ord.shape
    out = np.zeros_like(L_ord, dtype=np.uint8)
    for o in range(1, N_LAYERS+1):
        # process each slice to control RAM / anisotropy differences
        Mk = L_ord == o
        Mk_clean = np.zeros_like(Mk)
        for z in range(Z):
            Mk_clean[z] = _clean_slice(Mk[z], p)
        out[Mk_clean] = o
    return out

# -----------------------------------------------------------------------------
# 3‑D domain & seeds
# -----------------------------------------------------------------------------

@dataclass
class SeedParams:
    erode_r: int = 2  # safe (< half min layer thickness ~40/2=20)
    min_seed_vox: int = 50  # fallback: if erosion nukes layer, shrink r then centroid


def make_domain_mask(L_ord_clean: np.ndarray, bridge_r: int = 4) -> np.ndarray:
    """Domain = union of all nonzero labels, lightly dilated to span small inter‑label gaps.
    `bridge_r=4` (~8 vox gap) can be increased to 8 (~16 gap) if you see holes.
    """
    dom = L_ord_clean > 0
    if bridge_r > 0:
        dom = morphology.binary_dilation(dom, morphology.ball(bridge_r))
    # fill 3‑D holes completely enclosed by domain
    dom = ndi.binary_fill_holes(dom)
    return dom


def make_layer_seeds(L_ord_clean: np.ndarray, seed_p: SeedParams) -> np.ndarray:
    """Create eroded 'core' seeds for each label (ordered coding)."""
    seeds = np.zeros_like(L_ord_clean, dtype=np.int32)
    se = morphology.ball(seed_p.erode_r) if seed_p.erode_r > 0 else None
    for o in range(1, N_LAYERS+1):
        Mk = L_ord_clean == o
        if not Mk.any():
            continue
        if se is not None:
            core = morphology.binary_erosion(Mk, se)
        else:
            core = Mk.copy()
        # fallback: if erosion removes everything, relax
        if core.sum() < seed_p.min_seed_vox:
            core = Mk  # just use full mask
            if core.sum() == 0:
                continue
        seeds[core] = o
    return seeds

# -----------------------------------------------------------------------------
# Watershed gap closing
# -----------------------------------------------------------------------------

@dataclass
class WatershedParams:
    compactness: float = 0.0   # 0 → pure flooding
    prefer_brighter: bool = False  # we will invert distance if needed


def run_watershed_gap_closing(
    L_ord_clean: np.ndarray,
    domain_mask: np.ndarray,
    seeds: np.ndarray,
    ws_p: WatershedParams,
) -> np.ndarray:
    """Run multi‑label watershed inside the cortex domain.

    Cost image: we use distance *to background* (so interior of cortex low cost,
    edges high). Markers = eroded seeds. Watershed fills unlabeled domain.
    """
    # distance to domain boundary (invert so growth stops at edges?)
    dist_out = ndi.distance_transform_edt(domain_mask)  # inside distances
    # scikit-image watershed *labels basins of minima*.
    # We want seeds to grow everywhere; using negative distance makes edges high cost.
    elev = -dist_out.astype(np.float32)
    L_ws = segmentation.watershed(
        elev,
        markers=seeds,
        mask=domain_mask,
        compactness=ws_p.compactness,
    )
    # preserve original labels where present? (optional)
    # For stability, re‑impose original label where conflict minimal
    keep = L_ord_clean > 0
    L_ws[keep] = L_ord_clean[keep]
    return L_ws.astype(np.uint8)

# -----------------------------------------------------------------------------
# Z‑axis slow‑change smoothing
# -----------------------------------------------------------------------------

@dataclass
class ZSmoothParams:
    enable: bool = True
    win: int = 3            # median window size in slices (must be odd)
    max_jump_ignore: int = 100  # px; currently advisory (for future surf fitting)
    drop_singletons: bool = True  # remove 1‑slice blips


def z_median_filter_labels(L: np.ndarray, win: int) -> np.ndarray:
    """Apply per‑voxel median filter along Z.

    NOTE: median over categorical labels is a little hacky but works when layers
    don't swap wildly slice‑to‑slice (your <100px shift prior). Background counts.
    """
    if win <= 1:
        return L
    # pad reflect to keep shape
    pad = win // 2
    Lpad = np.pad(L, ((pad,pad),(0,0),(0,0)), mode='edge')
    out = np.empty_like(L)
    for z in range(L.shape[0]):
        slab = Lpad[z:z+win]
        # median along axis 0
        out[z] = np.median(slab, axis=0).astype(L.dtype)
    return out


def z_remove_singletons(L: np.ndarray) -> np.ndarray:
    """If a label appears only in a single slice column between two identical labels,
    replace it by the majority of neighbors. Cheap noise killer."""
    Z = L.shape[0]
    if Z < 3:
        return L
    out = L.copy()
    mid = L[1:-1]
    up = L[0:-2]
    dn = L[2:]
    mask = (mid != up) & (mid != dn) & (up == dn)
    out[1:-1][mask] = up[mask]
    return out


def apply_z_smoothing(L: np.ndarray, zp: ZSmoothParams) -> np.ndarray:
    if not zp.enable:
        return L
    out = z_median_filter_labels(L, zp.win)
    if zp.drop_singletons:
        out = z_remove_singletons(out)
    return out

from skimage import morphology

def thicken(mask, r: int):
    if r <= 0:
        return mask
    se = morphology.ball(r)  # or morphology.disk(r) if you prefer per-slice
    return morphology.binary_dilation(mask, se)
# -----------------------------------------------------------------------------
# Boundary extraction
# -----------------------------------------------------------------------------

from typing import Tuple

@dataclass
class BoundaryParams:
    thin: bool = True                      # legacy 1-voxel skeleton
    thick_r: int = 0                       # dilation radius if thin=False
    include_bg_layers: Tuple[int,...] = (1, 8)  # ordered labels to include vs bg

def extract_internal_boundaries(
    L_ord: np.ndarray,
    thin: bool = True,
    include_bg_layers: Tuple[int,...] = (1, 8),
    thick_r: int = 0,
) -> np.ndarray:
    """
    Return binary volume marking faces where adjacent voxels belong to two
    *different* labels of interest.

    Includes all inter-laminar (nonzero↔nonzero) boundaries.
    Additionally, if `include_bg_layers` is nonempty, includes boundaries where
    background (0) touches any layer listed (e.g., L1 and L6).

    If `thin` is True (default), returns a ~1-voxel skeletonized band
    (legacy behavior). If `thin=False`, the raw 1-voxel adjacency mask is
    optionally *dilated* by `thick_r` voxels (ball structuring element) to
    achieve ~4-6 voxel thickness (use thick_r=2 for ~5 vox; 3 for ~7).
    """
    L = L_ord
    b = np.zeros_like(L, dtype=bool)

    def _bmask_pair(a, b_):
        diff = (a != b_)
        if not include_bg_layers:
            return diff & (a > 0) & (b_ > 0)
        # include selected label↔bg contacts
        return diff & (
            ((a > 0) & (b_ > 0)) |
            ((a == 0) & np.isin(b_, include_bg_layers)) |
            ((b_ == 0) & np.isin(a, include_bg_layers))
        )

    # Z neighbors
    if L.shape[0] > 1:
        mask = _bmask_pair(L[:-1], L[1:])
        dz = np.zeros_like(L, dtype=bool)
        dz[:-1] = mask
        dz[1:] |= mask
        b |= dz

    # Y neighbors
    mask = _bmask_pair(L[:, :-1], L[:, 1:])
    dy = np.zeros_like(L, dtype=bool)
    dy[:, :-1] = mask
    dy[:, 1:] |= mask
    b |= dy

    # X neighbors
    mask = _bmask_pair(L[:, :, :-1], L[:, :, 1:])
    dx = np.zeros_like(L, dtype=bool)
    dx[:, :, :-1] = mask
    dx[:, :, 1:] |= mask
    b |= dx

    # post-processing
    if thin:
        try:
            b = morphology.skeletonize(b.astype(np.uint8)) > 0
        except Exception as e:  # pragma: no cover
            warnings.warn(f"skeletonize_3d failed ({e}); returning raw boundary.")
    else:
        if thick_r > 0:
            b = morphology.binary_dilation(b, morphology.ball(thick_r))

    return b.astype(np.uint8)
# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field

@dataclass
class PipelineParams:
    clean: CleanParams = field(default_factory=CleanParams)
    zsmooth: ZSmoothParams = field(default_factory=ZSmoothParams)
    seeds: SeedParams = field(default_factory=SeedParams)              
    ws: WatershedParams = field(default_factory=WatershedParams)   
    boundary: BoundaryParams  = field(default_factory=BoundaryParams)

    bridge_r: int = 4   # domain dilation radius
    do_full_watershed: bool = True
    thin: bool = True
    xy_win: int = 0            # NEW: median window across Y & X (odd; 0/1 disables)


def apply_xy_smoothing(L: np.ndarray, win: int) -> np.ndarray:
    """Mild median smoothing across Y & X (each slice independently).

    win: odd kernel width; 0/1 disables.
    """
    if win is None or win <= 1:
        return L
    out = ndi.median_filter(L, size=(1, win, win), mode='nearest')
    return out.astype(L.dtype)


ORDERED_NAMES = {
    1: "l1",
    2: "l23",
    3: "l4a",
    4: "l4b",
    5: "l4ca",
    6: "l4cb",
    7: "l5",
    8: "l6",
}

def compute_pairwise_boundaries(
    L_ord: np.ndarray,
    include_bg_layers: tuple[int, ...] = (),
    thin: bool = True,
) -> dict[tuple[int,int], np.ndarray]:
    """
    Returns {(lo,hi): uint8 mask} for each adjacent ordered label pair (lo<hi).
    Optionally includes bg (0) ↔ layer if layer in include_bg_layers.
    """
    from skimage import morphology

    Z, Y, X = L_ord.shape
    pair_masks: dict[tuple[int,int], np.ndarray] = {}

    def _add_contacts(A, B, axis):
        """A and B are shape-aligned neighbor slices (face-adjacent)."""
        diff = A != B
        if not diff.any():
            return

        a_vals = A[diff]
        b_vals = B[diff]

        # Background inclusion logic – build a keep mask
        if include_bg_layers:
            # For each differing pair decide if to keep
            keep = (
                ((a_vals > 0) & (b_vals > 0)) |                              # layer↔layer
                ((a_vals == 0) & np.isin(b_vals, include_bg_layers)) |       # bg↔layer
                ((b_vals == 0) & np.isin(a_vals, include_bg_layers))
            )
        else:
            keep = (a_vals > 0) & (b_vals > 0)

        if not keep.any():
            return

        # Indices (multi-d) where diff is True AND kept
        diff_idx = np.nonzero(diff)
        keep_idx = tuple(d[keep] for d in diff_idx)

        # For each kept contact, mark *both* voxels (A-side and B-side)
        # We do this by creating helper arrays of same shape as L_ord initialized False,
        # then OR them into pair_masks.
        if axis == 0:
            # A at z, B at z+1
            zA, yA, xA = keep_idx
            zB, yB, xB = zA + 1, yA, xA
        elif axis == 1:
            zA, yA, xA = keep_idx
            zB, yB, xB = zA, yA + 1, xA
        else:  # axis == 2
            zA, yA, xA = keep_idx
            zB, yB, xB = zA, yA, xA + 1

        a_keep = a_vals[keep]
        b_keep = b_vals[keep]

        # Normalize pair ordering
        lo = np.minimum(a_keep, b_keep)
        hi = np.maximum(a_keep, b_keep)

        # Now group by (lo,hi)
        pair_labels = np.stack([lo, hi], axis=1)  # (N,2)
        # Use a structured array or view to get unique pairs
        pair_view = pair_labels.view([('lo','u1'),('hi','u1')])
        unique_pairs, inverse = np.unique(pair_view, return_inverse=True)

        for pidx, rec in enumerate(unique_pairs):
            plo = int(rec['lo']); phi = int(rec['hi'])
            key = (plo, phi)
            if key not in pair_masks:
                pair_masks[key] = np.zeros_like(L_ord, dtype=bool)
            mask = (inverse == pidx)
            # mark both sides
            pair_masks[key][zA[mask], yA[mask], xA[mask]] = True
            pair_masks[key][zB[mask], yB[mask], xB[mask]] = True

    # Scan 6-neighbor axes
    if Z > 1:
        _add_contacts(L_ord[:-1],     L_ord[1:],     axis=0)
    if Y > 1:
        _add_contacts(L_ord[:, :-1],  L_ord[:, 1:],  axis=1)
    if X > 1:
        _add_contacts(L_ord[:, :, :-1], L_ord[:, :, 1:], axis=2)

    # Thin (skeletonize) per pair if requested
    out = {}
    for key, m in pair_masks.items():
        if thin:
            try:
                ms = morphology.skeletonize(m.astype(np.uint8)) > 0
            except Exception:
                ms = m
            out[key] = ms.astype(np.uint8)
        else:
            out[key] = m.astype(np.uint8)
    return out


def save_pairwise_boundaries(
    pair_masks: Dict[tuple[int,int], np.ndarray],
    base_name: str,
    save_dir: str | os.PathLike,
    ord2name: Dict[int,str],
    make_flat_copy: bool = False,
    compress: bool = False,
    upscale_factors: tuple[int,int] = (2,2),   # (fy, fx)
    skip_empty: bool = True,
    subdir_pattern: str = "pair_{a}_{b}",      # or "{name_a}_{name_b}"
):
    """
    Save pairwise boundary masks, each in its own subdirectory, after upscaling.

    Output structure:
        save_dir/
            pair_a_b/ (or custom pattern)
                <base>_a_nameA_b_nameB_up2x.tif

    Parameters
    ----------
    pair_masks : {(a,b): uint8 mask}
        Each mask can be 2-D or 3-D. a<b; a or b may be 0 (background).
    base_name : str
        Prefix from source volume (e.g., 'Z55200').
    save_dir : path-like
        Root directory.
    ord2name : dict[int,str]
        Ordered label → short name. Unmapped / background → 'bg'.
    make_flat_copy : bool
        Also write upscaled TIFFs directly in save_dir (legacy style).
    compress : bool
        Enable TIFF compression (deflate) if True.
    upscale_factors : (fy, fx)
        Y and X integer upscaling factors (Z unchanged).
    skip_empty : bool
        Skip masks with all zeros.
    subdir_pattern : str
        Format string for subdirectory naming. Available fields: a, b, name_a, name_b.
    """
    import tifffile
    from pathlib import Path

    fy, fx = upscale_factors
    if fy < 1 or fx < 1:
        raise ValueError("Upscale factors must be ≥1")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tif_kwargs = {}
    if compress:
        tif_kwargs["compression"] = "deflate"

    out_paths = {}
    for (a, b), arr in sorted(pair_masks.items()):
        if skip_empty and not arr.any():
            continue

        name_a = ord2name.get(a, "bg")
        name_b = ord2name.get(b, "bg")

        # Upscale (nearest neighbor)
        if (fy, fx) != (1, 1):
            arr_up = upscale_mask_xy(arr, fy=fy, fx=fx)
        else:
            arr_up = arr

        sub_dir_name = subdir_pattern.format(a=a, b=b, name_a=name_a, name_b=name_b)
        sub_dir = save_dir / sub_dir_name
        sub_dir.mkdir(exist_ok=True)

        fname = f"{base_name}_{a}_{name_a}_{b}_{name_b}_up{fy}x{fx}.tif"
        fpath = sub_dir / fname
        tifffile.imwrite(fpath, arr_up.astype(np.uint8), **tif_kwargs)

        if make_flat_copy:
            flat_path = save_dir / fname
            if flat_path != fpath:
                tifffile.imwrite(flat_path, arr_up.astype(np.uint8), **tif_kwargs)

        out_paths[(a, b)] = fpath

    return out_paths


from typing import Optional, Tuple
import numpy as np

def upscale_mask_xy(arr: np.ndarray, fy: int = 2, fx: int = 2) -> np.ndarray:
    """
    Upscale a 2-D or 3-D mask in (Y,X) or (Z,Y,X) by factors fy, fx using nearest neighbor.
    For 3-D, Z is unchanged.
    """
    if arr.ndim == 2:
        return arr.repeat(fy, axis=0).repeat(fx, axis=1)
    elif arr.ndim == 3:
        return arr.repeat(fy, axis=1).repeat(fx, axis=2)
    else:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {arr.shape}")


def _ensure_3d(L: np.ndarray) -> tuple[np.ndarray, bool]:
    if L.ndim == 2:
        return L[None, ...], True
    if L.ndim == 3:
        return L, False
    raise ValueError(f"Expected 2-D or 3-D label array, got shape {L.shape}.")

def compute_cortical_boundaries(
    L_in: np.ndarray,
    params: "PipelineParams" | None = None,
    do_full_watershed: Optional[bool] = None,
    thin: Optional[bool] = None,
    coding_type: str = "raw",
    smooth_mask = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if params is None:
        params = PipelineParams()
    if do_full_watershed is not None:
        params.do_full_watershed = do_full_watershed
    if thin is not None:
        params.thin = thin

    was_2d = False
    if L_in.ndim == 2:
        L_in = L_in[None, ...]
        was_2d = True

    coding_type = coding_type.lower()
    if coding_type not in ("raw", "ordered"):
        raise ValueError(f"coding_type must be 'raw' or 'ordered', got {coding_type!r}")

    if coding_type == "raw":
        L_ord = remap_raw_to_ordered(L_in)
    else:
        uniq = np.unique(L_in)
        if not np.all(np.isin(uniq, np.arange(0, 9))):
            raise ValueError(f"Unexpected labels in ordered input: {uniq}.")
        L_ord = L_in.astype(np.uint8, copy=False)

    if smooth_mask:
        L_clean = clean_volume_slicewise(L_ord, params.clean)
        domain  = make_domain_mask(L_clean, params.bridge_r)
        seeds   = make_layer_seeds(L_clean, params.seeds)

        if params.do_full_watershed:
            L_ws = run_watershed_gap_closing(L_clean, domain, seeds, params.ws)
        else:
            L_ws = L_clean

        L_ws = apply_z_smoothing(L_ws, params.zsmooth)
        if params.xy_win and params.xy_win > 1:
            L_ws = apply_xy_smoothing(L_ws, params.xy_win)
    else:
        L_ws = L_ord

    boundary = extract_internal_boundaries(
        L_ws,
        thin=params.boundary.thin,
        include_bg_layers=params.boundary.include_bg_layers,
        thick_r=params.boundary.thick_r,
    )

    if coding_type == "raw":
        L_ws_raw = remap_ordered_to_raw(L_ws)
    else:
        L_ws_raw = L_ws

    if was_2d:
        boundary = boundary[0]
        L_ws     = L_ws[0]
        L_ws_raw = L_ws_raw[0]

    return boundary.astype(np.uint8), L_ws.astype(np.uint8), L_ws_raw.astype(np.uint8)

import resource, atexit
@atexit.register                     # runs automatically on normal exit
def report_max_mem():
    """
    Print the maximum resident-set size (RSS) the process ever reached.
    On Linux ru_maxrss is in *kilobytes*; on macOS it is in *bytes*.
    """
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # normalize to MB for readability
    print(f"\n[MEM] Peak RSS: {max_rss_kb/1024:.1f} MB")

if __name__ == "__main__":  # simple smoke test

    save_dir = "/home/confetti/data/rm009/boundary_seg/valid_bnd_masks"
    os.makedirs(save_dir,exist_ok=True)

    from pathlib import Path
    dir_path = Path("/home/confetti/data/rm009/boundary_seg/valid_masks")
    tif_paths = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in {".tif", ".tiff"})

    SAVE_PAIR_WISE_BND  = False 

    for mask_path in tif_paths:
        base_name = Path(mask_path).stem

        L_raw = tifffile.imread(mask_path)

        # Promote to 3-D if needed
        L_raw3d, was_2d = _ensure_3d(L_raw)

        params = PipelineParams(
            boundary=BoundaryParams(
                thin= False,        # we want thick band
                thick_r=1,         # ~5 vox; raise to 3 if you want ~7
                include_bg_layers=(1,8)
            ),
            xy_win=8,
        )

        bnd, Ls_ord, Ls_raw = compute_cortical_boundaries(
            L_raw3d,
            params=params,
            coding_type='ordered',
            smooth_mask=False,
        )

        if SAVE_PAIR_WISE_BND:
            # Pairwise boundaries (works on promoted 3-D)
            pair_masks = compute_pairwise_boundaries(
                Ls_ord,
                include_bg_layers=params.boundary.include_bg_layers,
                thin=True,
            )
            for key, m in pair_masks.items():
                pair_masks[key] = thicken(m, r=2).astype(np.uint8)

            pair_paths = save_pairwise_boundaries(
                pair_masks,
                base_name=base_name,           # e.g., "Z55200"
                save_dir=save_dir,
                ord2name=ORDERED_NAMES,
                make_flat_copy=False,
                compress=True,
                upscale_factors=(1,1),         # <- upscale H,Y and W,X by 2
                skip_empty=True,
                subdir_pattern="pair_{a}_{b}", # or "{name_a}_{name_b}"
            )

         # If original was 2-D, squeeze leading Z for 2-D outputs
        if was_2d:
            bnd_out    = bnd
            Ls_ord_out = Ls_ord
            Ls_raw_out = Ls_raw
            # Also squeeze each pairwise boundary
            pair_masks = {k: v[0] for k, v in pair_masks.items()}
        else:
            bnd_out    = bnd
            Ls_ord_out = Ls_ord
            Ls_raw_out = Ls_raw       

        tifffile.imwrite(f"{save_dir}/{base_name}_bnd_thick.tif", bnd_out.astype('uint8'))
        # tifffile.imwrite(f"{save_dir}/{base_name}_ord.tif",       Ls_ord_out.astype('uint8'))
        # tifffile.imwrite(f"{save_dir}/{base_name}_raw.tif",       Ls_raw_out.astype('uint8'))
        print(f"[{base_name}] boundary mask saved (was_2d={was_2d})")

