import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from sklearn.manifold import TSNE
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union

import random
import pickle
import numpy as np
import tifffile as tif
from helper.image_reader import Ims_Image
from confettii.plot_helper import three_pca_as_rgb_image



def save_checkpoint(state: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

from pathlib import Path
from typing import Union
import torch
from torch import nn, optim


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> int:
    """
    Load weights (and optionally optimizer state) from a checkpoint.
    Handles both raw state_dict and wrapped dict with epoch/optimizer.

    Automatically reshapes 2D Linear weights to 5D 1x1 Conv weights if needed.

    Returns
    -------
    int
        The epoch to resume from (0 if no epoch information is stored).
    """
    ckpt = torch.load(path, map_location="cpu")

    # ───────────────────────────────────── case 1: raw state_dict ──
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) or hasattr(v, "dtype")
                                      for v in ckpt.values()):
        ckpt = {"model": ckpt}

    # ──────────────────────────────── case 2: wrapped dict (Lightning, custom)
    state_dict = ckpt.get("model") or ckpt.get("state_dict")
    if state_dict is None:
        raise RuntimeError(f"Checkpoint at {path} doesn’t contain a model state-dict.")

    # Adapt shape mismatches (e.g., [out, in] → [out, in, 1, 1, 1])
    adapted_dict = {}
    model_state = model.state_dict()
    for k, v in state_dict.items():
        if k in model_state:
            expected_shape = model_state[k].shape
            if v.shape != expected_shape:
                if v.ndim == 2 and len(expected_shape) == 5 and expected_shape[2:] == (1, 1, 1):
                    v = v.view(*expected_shape)  # expand to Conv3D weight shape
                elif v.ndim == 1 and len(expected_shape) == 1:
                    pass  # biases usually match
                else:
                    print(f"[warn] Skipping {k}: shape {v.shape} → {expected_shape} mismatch")
                    continue
        adapted_dict[k] = v

    model.load_state_dict(adapted_dict, strict=False)

    if optimizer and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])

    return ckpt.get("epoch", 0) + 1

class Contrastive_dataset_3d_one_stage(Dataset):
    """
    sample roi in ims file; return positive roi pairs and negative roi pairs for contrastive_learning
    """
    def __init__ (self,ims_path,d_near,num_pairs,n_view = 2,verbose = False):

        self.ims_vol =Ims_Image(ims_path,channel=0) 
        level = 0
        D,H,W= self.ims_vol.rois[level][3:]
        d_near = int(d_near) 

        margin = 10 
        # Generate random (x, y, z) locations within the given range
        lz, hz = d_near + margin +int(D//4),  int(D*3/4) - d_near - margin
        ly, hy = d_near + margin +int(H//4),  int(H*3/4) - d_near - margin
        lx, hx = d_near + margin +int(W//4),  int(W//2) - d_near - margin

        self.loc_lst = np.stack([
            np.random.randint(lz, hz, size=num_pairs),
            np.random.randint(ly, hy, size=num_pairs),
            np.random.randint(lx, hx, size=num_pairs)
        ], axis=1)

        self.sample_num = num_pairs
        self.all_near_shifts = generate_sphereshell__shifts(R= d_near,r= 24 ,dims=3)
        self.n_view =n_view


    def __len__(self):
        return self.sample_num
    
    def __getitem__(self,idx):

        z, y, x = self.loc_lst[idx].T    # Unpack coordinates
        roi = self.ims_vol.from_roi(coords=(z,y,x,64,64,64))
        roi = roi.astype(np.float32)
        roi=torch.from_numpy(roi)
        roi=torch.unsqueeze(roi,0)
        # for each call, the positive pair is resampled within the near_shifts range, 
        # maybe fix the positive pair will be better for stable training
        pair_locs = [self.positve_pair_loc_generate([z,y,x]) for _ in range(self.n_view -1)]
        pair_roi = [self.get_roi_given_loc(pair_loc) for pair_loc in pair_locs]
        res = [roi]
        for pair in pair_roi:
            res.append(pair)

        #res: x1,neigb1(x1),neigb2(x1),..,neigbN(x1)
        return res

    def get_roi_given_loc(self,loc):
        z, y, x = loc   # Unpack coordinates
        roi = self.ims_vol.from_roi(coords=(z,y,x,64,64,64))
        roi = roi.astype(np.float32)
        roi=torch.from_numpy(roi)
        roi=torch.unsqueeze(roi,0)
        return roi 

    def positve_pair_loc_generate(self,loc):
        shift = random.choice(self.all_near_shifts)
        return loc+shift


class Contrastive_dataset_3d_fix_paris(Dataset):
    """
    fix the positive pair at the init

    """
    def __init__(self, feats_map, d_near, num_pairs, n_view=2, verbose=False):
        D, H, W, C = feats_map.shape
        self.feats_map = feats_map
        d_near = int(d_near)
        margin = 10

        # Generate random (z, y, x) locations within the safe volume
        lz, hz = d_near + margin, D - d_near - margin
        ly, hy = d_near + margin, H - d_near - margin
        lx, hx = d_near + margin, W - d_near - margin

        self.loc_lst = np.stack([
            np.random.randint(lz, hz, size=num_pairs),
            np.random.randint(ly, hy, size=num_pairs),
            np.random.randint(lx, hx, size=num_pairs)
        ], axis=1)

        self.sample_num = num_pairs
        self.n_view = n_view
        self.all_near_shifts = generate_sphereshell__shifts(R=d_near, r=0, dims=3)

        # Fix positive pairs at initialization
        self.fixed_positive_pairs = [
            [self.positve_pair_loc_generate(self.loc_lst[i]) for _ in range(n_view - 1)]
            for i in range(self.sample_num)
        ]

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        z, y, x = self.loc_lst[idx]
        feat = self.feats_map[z, y, x, :]  # Shape: (C)
        feat = torch.from_numpy(feat)

        # Use fixed positive pairs
        pair_locs = self.fixed_positive_pairs[idx]
        pair_feats = [torch.from_numpy(self.get_feats_given_loc(loc, self.feats_map)) for loc in pair_locs]

        # res: x1, neigb1(x1), neigb2(x1), ..., neigbN(x1)
        return [feat] + pair_feats

    def get_feats_given_loc(self, loc, feat_map):
        z, y, x = loc
        return feat_map[z, y, x, :]

    def positve_pair_loc_generate(self, loc):
        shift = random.choice(self.all_near_shifts)
        loc = np.array(loc)
        new_loc = loc + shift
        return np.clip(new_loc, [0, 0, 0], np.array(self.feats_map.shape[:3]) - 1)  # Ensure valid bounds

import numpy as np
import random
import torch
from torch.utils.data import Dataset

class Contrastive_dataset_3d(Dataset):
    """
    Supports both 4-D (D, H, W, C) and 3-D (H, W, C) feature maps.

    Optional region-of-interest limits:
        lz, ly, lx – inclusive lower bounds
        hz, hy, hx – exclusive  upper bounds

    If any bound is None it is computed from d_near + margin.
    """
    def __init__(
        self,
        feats_map,
        d_near: int,
        num_pairs: int,
        n_view: int = 2,
        *,
        verbose: bool = False,
        margin: int = 10,
        lz: int | None = None,
        ly: int | None = None,
        lx: int | None = None,
        hz: int | None = None,
        hy: int | None = None,
        hx: int | None = None,
    ):
        self.feats_map = feats_map
        self.dims = feats_map.ndim - 1  # 3-D (volumetric) or 2-D (single slice)
        self.verbose = verbose
        self.n_view = n_view
        d_near = int(d_near)

        if self.dims == 3:
            D, H, W, C = feats_map.shape
            # --------- derive bounds (fallback to auto-computed) ----------
            lz = lz if lz is not None else d_near 
            ly = ly if ly is not None else d_near + margin
            lx = lx if lx is not None else d_near + margin
            hz = hz if hz is not None else D - d_near 
            hy = hy if hy is not None else H - d_near - margin
            hx = hx if hx is not None else W - d_near - margin

            if not (0 <= lz < hz <= D and 0 <= ly < hy <= H and 0 <= lx < hx <= W):
                raise ValueError("Sampling bounds are out of volume range.")

            self.loc_lst = np.stack([
                np.random.randint(lz, hz, size=num_pairs),
                np.random.randint(ly, hy, size=num_pairs),
                np.random.randint(lx, hx, size=num_pairs)
            ], axis=1)

        elif self.dims == 2:
            H, W, C = feats_map.shape
            ly = ly if ly is not None else d_near + margin
            lx = lx if lx is not None else d_near + margin
            hy = hy if hy is not None else H - d_near - margin
            hx = hx if hx is not None else W - d_near - margin

            if not (0 <= ly < hy <= H and 0 <= lx < hx <= W):
                raise ValueError("Sampling bounds are out of image range.")

            self.loc_lst = np.stack([
                np.random.randint(ly, hy, size=num_pairs),
                np.random.randint(lx, hx, size=num_pairs)
            ], axis=1)

        else:
            raise ValueError("Feature map must be 4-D (D, H, W, C) or 3-D (H, W, C).")

        self.sample_num = num_pairs
        self.all_near_shifts = generate_sphereshell__shifts(R=d_near, r=0, dims=self.dims)

    # ------------------------------------------------------------------ required
    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        loc = self.loc_lst[idx]
        feat = torch.from_numpy(self._get_feats_given_loc(loc)).float()
        pair_locs = [self._positive_pair_loc_generate(loc) for _ in range(self.n_view - 1)]
        pair_feats = [torch.from_numpy(self._get_feats_given_loc(pl)).float() for pl in pair_locs]
        return [feat] + pair_feats

    # ------------------------------------------------------------------ helpers
    def _get_feats_given_loc(self, loc):
        if self.dims == 3:
            z, y, x = loc
            return self.feats_map[z, y, x, :]
        else:  # dims == 2
            y, x = loc
            return self.feats_map[y, x, :]

    def _positive_pair_loc_generate(self, loc):
        shift = random.choice(self.all_near_shifts)
        return loc + shift

class Contrastive_dataset_multiple_2d(Dataset):
    def __init__ (self,feats_map,d_near,n_view = 2,verbose = False):

        N,C,H,W = feats_map.shape
        self.feats_maps = feats_map
        self.feats_map_shape = (C,H,W)
        self.data_length = N
        self.margin = 10

        d_near = int(d_near//(2**3*0.5))

        self.all_near_shifts = generate_sphereshell__shifts(R= d_near,r= 0, dims=2)
        self.n_view =n_view


    def __len__(self):
        return self.data_length
    
    def __getitem__(self,idx):

        C,H,W = self.feats_map_shape
        margin = self.margin
        ith_feat_map = self.feats_maps[idx]  # Shape: (C)
        y = random.randint(margin, H - margin - 1)
        x = random.randint(margin, W - margin - 1)
        sampled_feat = ith_feat_map[:,y,x]

        sampled_feat = torch.from_numpy(sampled_feat)
        # for each call, the positive pair is resampled within the near_shifts range, 
        # maybe fix the positive pair will be better for stable training
        pair_locs = [self.positve_pair_loc_generate([y,x]) for _ in range(self.n_view -1)]
        pair_feats = [ith_feat_map[:,pair_loc[0],pair_loc[1]] for pair_loc in pair_locs]
        res = [sampled_feat]
        for pair in pair_feats:
            res.append(torch.from_numpy(pair).float())

        #res: x1,neigb1(x1),neigb2(x1),..,neigbN(x1)
        return res

    def positve_pair_loc_generate(self,loc):
        shift = random.choice(self.all_near_shifts)
        return loc+shift



# You also need to modify your generate_sphereshell__shifts function to accept a `dims` argument.
def generate_sphereshell__shifts(R, r=0, dims=3):
    """Generate integer shifts within a sphere shell (radius R, inner radius r)."""
    shifts = []
    ranges = [range(-R, R+1)] * dims
    for shift in np.array(np.meshgrid(*ranges)).T.reshape(-1, dims):
        norm = np.linalg.norm(shift)
        if r < norm <= R:
            shifts.append(shift)
    return np.array(shifts)


import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, filters=[24, 18, 12, 8]):
        super(MLP, self).__init__()
        
        layers = []
        for in_features, out_features in zip(filters[:-1], filters[1:]):
            layers.append(nn.Linear(in_features, out_features))
        
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Last layer, no activation
        return x / x.norm(p=2, dim=-1, keepdim=True)

def cos_loss(features,n_views,pos_weight_ratio=5,enhanced =False):

    #labels for positive pairs
    N = features.shape[0]
    labels = torch.cat([torch.arange(int(N//n_views)) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    cos_similarity_matrix = torch.matmul(features, features.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    # print(f"{labels.shape=}")
    # print(f"{features.shape=}")
    # print(f"{mask.shape=}")
    cos_similarity_matrix = cos_similarity_matrix[~mask].view(cos_similarity_matrix.shape[0], -1)

    # select and combine multiple positives and the negatives
    pos_coses = cos_similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    neg_coses = cos_similarity_matrix[~labels.bool()].view(cos_similarity_matrix.shape[0], -1)

    if enhanced:
        mean_pos_cose = pos_coses.abs().mean()
        mean_neg_cose = neg_coses.abs().mean()

        # Filter pos_coses: keep only those with abs > mean_pos_cose
        filtered_pos = pos_coses[pos_coses.abs() > mean_pos_cose]
        if filtered_pos.numel() > 0:
            pos_loss = ((filtered_pos - 1) ** 2).mean()
        else:
            pos_loss = torch.tensor(0.0, device=pos_coses.device)

        # Filter neg_coses: keep only those with abs < mean_neg_cose
        filtered_neg = neg_coses[neg_coses.abs() < mean_neg_cose]
        if filtered_neg.numel() > 0:
            neg_loss = (filtered_neg ** 2).mean()
        else:
            neg_loss = torch.tensor(0.0, device=neg_coses.device)
    
    else:
        pos_loss = ((pos_coses -1 )**2).mean()
        neg_loss = (neg_coses**2).mean()

    pos_weight = (pos_weight_ratio)/(pos_weight_ratio+1)
    neg_weight = (1)/(pos_weight_ratio+1)

    
    # pos_loss =(torch.exp( torch.abs((pos_coses-1)/T) ) -1 ).mean()
    # neg_loss = (torch.exp( torch.abs((neg_coses)/T) ) -1 ).mean() 
    return pos_loss*pos_weight +neg_loss*neg_weight, pos_coses.mean(),neg_coses.mean()


def cos_loss_topk(features,n_views,pos_weight_ratio=5,only_pos=False):
    #labels for positive pairs
    N = features.shape[0]
    labels = torch.cat([torch.arange(int(N//n_views)) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    cos_similarity_matrix = torch.matmul(features, features.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)

    cos_similarity_matrix = cos_similarity_matrix[~mask].view(cos_similarity_matrix.shape[0], -1)

    # select and combine multiple positives and the negatives
    pos_coses = cos_similarity_matrix[labels.bool()].view(labels.shape[0], -1) #shape (N,n_view-1)
    neg_coses = cos_similarity_matrix[~labels.bool()].view(cos_similarity_matrix.shape[0], -1) # shape (N, N - n_view)

    k_pos = min(int(n_views/4),1)
    k_neg = int(N/10)
    filtered_pos, topk_pos_indices = torch.topk(pos_coses.abs(),k_pos,dim=-1) #shape :N,k_pos
    if only_pos:
        filtered_neg = neg_coses
    else:
        filtered_neg, topk_neg_indices = torch.topk(neg_coses.abs(),k_neg,dim=-1,largest=False) #shape: N,k_neg
    filtered_pos_loss = ((filtered_pos - 1) ** 2).mean()
    filtered_neg_loss = (filtered_neg ** 2).mean()
 
    pos_weight = (pos_weight_ratio)/(pos_weight_ratio+1)
    neg_weight = (1)/(pos_weight_ratio+1)
    
    return pos_weight*filtered_pos_loss + neg_weight+filtered_neg_loss, pos_coses.mean(),neg_coses.mean()





def compute_ncc_map(loc_idx, encoded, shape):
    ncc_list =[]
    for  idx in loc_idx :
        att = encoded @ encoded[idx]
        ncc_list.append(att.reshape(shape))
    return ncc_list





def plot_ncc_maps(img_lst,loc_idx_lst,locations,writer, tag="ncc_plot", step=0,ncols=4,fig_size=4):
    num_plots = len(img_lst)
    ncols = ncols
    nrows = int(np.ceil(num_plots/ ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_size*ncols,  fig_size*nrows))

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            ax.imshow(img_lst[i], cmap='viridis')
            ax.plot(locations[loc_idx_lst[i], 1], locations[loc_idx_lst[i], 0], '*r')
        ax.axis('off')
    plt.tight_layout()
    # Save plot to TensorBoard

    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)

# plot_pca_maps then also works when len(img_lst)==1
def plot_pca_maps(
    img_lst,
    writer,
    tag: str = "pca_plot",
    step: int = 0,
    ncols: int = 4,
    fig_size: int = 4,
):
    num_plots = len(img_lst)
    nrows = int(np.ceil(num_plots / ncols))

    # Create the grid of sub-plots
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_size * ncols, fig_size * nrows))

    # Make sure `axes` is always a 1-D iterable
    axes = np.atleast_1d(axes).ravel()

    # Fill the axes with images (or leave them blank)
    for i, ax in enumerate(axes):
        if i < num_plots:
            ax.imshow(img_lst[i])
        ax.axis("off")

    plt.tight_layout()

    # Log to TensorBoard
    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)


from collections import defaultdict
from typing import Sequence, Tuple, Union, Literal, List
Arr   = Union[np.ndarray, torch.Tensor]
Array = Union[Arr, Sequence[Arr]]   # single array or list/tuple of arrays

def class_balance(
    feats_lst : Array,
    label_lst : Array,
    n_per_class: Union[int, Literal["min", "max"]] = "min",
    shuffle   : bool  = True,
    random_state: int | None = None
) -> Tuple[List[Arr], List[Arr]]:
    """
    Return class-balanced versions of feats_lst & label_lst.

    Parameters
    ----------
    feats_lst, label_lst
        • Either 1-D sequences (length = N) **or**
        • 2-D/ND arrays with first-axis length = N.
    n_per_class : int | "min" | "max"
        • "min" → use size of the smallest class  (down-sample, default)  
        • "max" → use size of the largest class   (up-sample w/ replacement)  
        • int   → explicit quota (down- or up-sampling as needed).
    shuffle : bool
        Shuffle the final order.
    random_state : int | None
        Seed for reproducibility.

    Returns
    -------
    feats_balanced, labels_balanced : **lists** with equal counts per class.
    """
    rng = np.random.default_rng(random_state)

    # ------- convert to python lists of arrays for uniform handling ----------
    feats_is_array  = isinstance(feats_lst, (np.ndarray, torch.Tensor))
    labels_is_array = isinstance(label_lst, (np.ndarray, torch.Tensor))

    feats_seq  = [feats_lst]  if feats_is_array  else list(feats_lst)
    labels_seq = [label_lst]  if labels_is_array else list(label_lst)

    N = len(labels_seq[0])                  # sample count
    for arr in labels_seq + feats_seq:
        assert len(arr) == N, "length mismatch among inputs"

    # -------------------  gather indices per class ---------------------------
    class_to_idx = defaultdict(list)
    labels_np    = labels_seq[0] if isinstance(labels_seq[0], np.ndarray) \
                   else np.array(labels_seq[0])
    for idx, y in enumerate(labels_np):
        class_to_idx[int(y)].append(idx)

    counts = {c: len(idxs) for c, idxs in class_to_idx.items()}
    if isinstance(n_per_class, int):
        quota = n_per_class
    elif n_per_class == "max":
        quota = max(counts.values())
    else:  # "min"
        quota = min(counts.values())

    # -------------------- sample (with replacement if needed) ----------------
    chosen_idx = []
    for c, idxs in class_to_idx.items():
        if len(idxs) >= quota:
            chosen = rng.choice(idxs, quota, replace=False)
        else:   # minority class, need to up-sample
            chosen = rng.choice(idxs, quota, replace=True)
        chosen_idx.extend(chosen)

    if shuffle:
        rng.shuffle(chosen_idx)

    # -------------------- slice / gather back --------------------------------
    def _gather(seq):
        if len(seq) == 1:    # originally a single array → return array
            arr = seq[0]
            if isinstance(arr, torch.Tensor):
                return arr[chosen_idx]
            return arr[chosen_idx, ...]
        else:                # list/tuple input → return list/tuple
            return [a[chosen_idx] for a in seq]

    balanced_feats  = _gather(feats_seq)
    balanced_labels = _gather(labels_seq)

    return balanced_feats, balanced_labels



def tsne_plot(encoded, labels, ax, title='tsne'):
    # Filter out label 0
    labels = np.array(labels)
    mask = labels != 0
    filtered_encoded = encoded[mask]
    filtered_labels = labels[mask]

    tsne_model = TSNE(n_components=2, perplexity=20, random_state=42)
    reduced_data = tsne_model.fit_transform(filtered_encoded)
    
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=1.2, c=filtered_labels, cmap='tab10')
    ax.legend(*scatter.legend_elements(), title="Digits")
    ax.set_title(title)

def tsne_grid_plot(
    encoded_list,
    labels_list,
    writer,
    tag: str = "tsne_grid",
    step: int = 0,
    tag_list=None,
):
    """
    Plot one or more t-SNE scatter plots side-by-side and log the composite figure
    to TensorBoard.

    Parameters
    ----------
    encoded_list : list[array-like]
        List of feature arrays to embed/plot; length = N plots.
    labels_list : list[array-like]
        Parallel list of integer / categorical labels for coloring points.
    writer : SummaryWriter
        TensorBoard writer handle.
    tag : str, default="tsne_grid"
        TensorBoard *grid* tag used for the combined figure logged via add_figure().
    step : int, default=0
        Global step for TensorBoard.
    tag_list : list[str] or None
        Optional per-plot titles. Length must be == N *or* 1.
        If length==1 and N>1, an index suffix "_i" is appended automatically.
        If None, defaults to [f"{tag}_{i}" for i in range(N)].
    """
    import matplotlib.pyplot as plt  # local import to avoid polluting global namespace

    num_plots = len(encoded_list)
    if num_plots != len(labels_list):
        raise ValueError(f"encoded_list ({num_plots}) and labels_list ({len(labels_list)}) length mismatch.")

    # Normalize tag_list
    if tag_list is None:
        per_plot_tags = [f"{tag}_{i}" for i in range(num_plots)]
    else:
        if len(tag_list) == 1 and num_plots > 1:
            base = tag_list[0]
            per_plot_tags = [f"{base}_{i}" for i in range(num_plots)]
        elif len(tag_list) == num_plots:
            per_plot_tags = list(tag_list)
        else:
            raise ValueError(
                f"tag_list length ({len(tag_list)}) must be 1 or match num_plots ({num_plots})."
            )

    # Create figure/axes
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]  # make iterable

    # Draw each subplot
    for ax, encoded, labels, title_str in zip(axes, encoded_list, labels_list, per_plot_tags):
        tsne_plot(encoded, labels, ax, title=title_str)

    plt.tight_layout()

    # Log the *grid* figure under the supplied 'tag'
    writer.add_figure(tag, fig, global_step=step)

    # Close to free memory (TensorBoard now owns the rendered image)
    plt.close(fig)

def get_vsi_eval_data(img_no_list =[1,3,4,5]):
    eval_feats_path = '/home/confetti/data/wide_filed/test_hp_raw_feats.pkl'
    eval_label_path = '/home/confetti/data/wide_filed/test_hp_label.tif'
    eval_img = tif.imread("/home/confetti/data/wide_filed/test_hp_img.tif")


    with open(eval_feats_path,'rb') as f:
        eval_feats = pickle.load(f)
    
    eval_labels = tif.imread(eval_label_path)
    eval_labels = zoom(eval_labels,(1/16),order=0)
    eval_labels = eval_labels.ravel()
    print(f"eval_labels:{eval_labels.shape}")

    H,W = [int(x//16)for x in eval_img.shape]
    coords = [[i, j] for i in range(H) for j in range(W)]
    coords = np.array(coords)
    eval_datas ={}
    eval_datas['img']=eval_img
    eval_datas['label']=eval_labels
    eval_datas['feats']=eval_feats
    eval_datas['locs']=coords
    # sample point idx for ncc plot
    eval_datas['loc_idx']=[940,2162,4895,8768,10020]
    eval_datas = [eval_datas]

    return eval_datas 

def get_t11_eval_data(E5,img_no_list =[1,3,4,5],ncc_seed_point = True):
    eval_datas = []
    # === eval feats and ncc_points and labels ===#
    # Set the path prefix based on E5 flag
    if E5:
        prefix = "/share/home/shiqiz/data"
    else:
        prefix = "/home/confetti/data"

    # === eval feats and ncc_points and labels ===#
    for img_no in img_no_list:
        eval_label_path = f"{prefix}/t1779/test_data_part_brain/human_mask_{img_no:04d}.tif"
        vol = tif.imread(f"{prefix}/t1779/test_data_part_brain/{img_no:04d}.tif")
        z_slice = vol[int(vol.shape[0]//2),:,:]

        eval_dic = {}
        eval_dic['img'] = vol
        eval_dic['label'] = tif.imread(eval_label_path)
        eval_dic['z_slice'] = z_slice
        eval_datas.append(eval_dic)

    # sample point idx for ncc plot
    if ncc_seed_point:
        eval_datas[0]['loc_idx']=[940,2162,978,4024,3607]
        eval_datas[1]['loc_idx']=[1015,1883,968,3459,4152]
        # eval_datas[2]['loc_idx']=[6907, 1982, 2367, 2193,2296]
        eval_datas[2]['loc_idx']=[6907, 1982, 376, 4692, 2459]
        eval_datas[3]['loc_idx']=[1779, 3453, 3653, 3223,978]
    return eval_datas


def get_rm009_eval_data(E5,img_no_list=[1,2]):
    eval_datas = []
    # === eval feats and ncc_points and labels ===#
    # Set the path prefix based on E5 flag
    if E5:
        prefix = "/share/home/shiqiz/data"
    else:
        prefix = "/home/confetti/data"

    # === eval feats and ncc_points and labels ===#
    for img_no in img_no_list:
        vol = tif.imread(f"{prefix}/rm009/seg_valid/{img_no:04d}.tif")
        mask = tif.imread(f"{prefix}/rm009/seg_valid/{img_no:04d}_human_mask.tif")
        
        z_slice = vol[int(vol.shape[0]//2),:,:]

        eval_dic = {}
        eval_dic['img'] = vol
        eval_dic['label'] = mask 
        eval_dic['z_slice'] = z_slice
        eval_datas.append(eval_dic)


    return eval_datas


def valid_from_roi(model, it, eval_data, writer):
    """Evaluate a model on a list of ROIs.

    Works with feature tensors of shape:
        • (C, H, W)                      – old behaviour
        • (C, D, H, W)                  – channel-first 3-D
        • (D, H, W, C)                  – channel-last 3-D
    For 3-D inputs, the middle depth slice (z = D//2) is used for PCA/t-SNE.
    """
    model.eval()

    ncc_valid_imgs            = []
    pca_img_lst               = []
    ncc_seedpoints_idx_lsts   = []
    tsne_encoded_feats_lst    = []
    tsne_label_lst            = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx, data_dic in enumerate(eval_data):
        roi      = data_dic['img']          # numpy (H,W) or (D,H,W)
        label    = data_dic['label']
        idxes    = data_dic.get('loc_idx', [])

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
        # ---------------------------------------------------------------- NCC plots
        if len(idxes) > 0:
            ncc_imgs = compute_ncc_map(idxes, feats_flat, shape=(H, W))
            ncc_valid_imgs.extend(ncc_imgs)
            ncc_seedpoints_idx_lsts.extend(idxes)
            rows, cols = np.indices((H, W))
            locations = np.stack([rows.ravel(), cols.ravel()], axis=1)
        # ---------------------------------------------------------------- labels (resampled to H×W)
        zoom_factor = [y / x for y, x in zip((H, W), label.shape)]
        label_rs    = zoom(label, zoom_factor, order=0)
        tsne_label_lst.append(label_rs.ravel())

    # ---------- logging ----------
    tsne_grid_plot(tsne_encoded_feats_lst,tsne_label_lst,writer,tag='tsne',step=1)
    plot_pca_maps(pca_img_lst,writer=writer,tag='pca',step=it,ncols=len(pca_img_lst))

    if len(idxes) > 0: 
        plot_ncc_maps(ncc_valid_imgs, ncc_seedpoints_idx_lsts,locations,writer=writer, tag=f"ncc",step=it,ncols=len(idxes))




def valid_from_feats(model,it,eval_data,writer):
    "eval from ae_feats, older version"
    model.eval()

    ncc_valid_imgs=[]
    ncc_seedpoints_idx_lsts =[]
    tsne_encoded_feats_lst4tsne=[]
    tsne_label_lst4tsne=[]
    for idx,data_dic in enumerate(eval_data):
        feats=data_dic['feats']
        locations =data_dic['locs'] 
        labels= data_dic['label']
        idxes = data_dic['loc_idx']

        feats = torch.from_numpy(feats).float().to('cuda')
        encoded = model(feats)
        encoded = encoded.detach().cpu().numpy()
        n = int(np.sqrt(len(encoded.shape[0])))
        ncc_lst = compute_ncc_map(idxes,encoded,(n,n))
        ncc_valid_imgs.extend(ncc_lst)
        ncc_seedpoints_idx_lsts.extend(idxes)
        tsne_encoded_feats_lst4tsne.append(encoded)
        tsne_label_lst4tsne.append(labels)

    tsne_grid_plot(tsne_encoded_feats_lst4tsne,tsne_label_lst4tsne,writer,tag=f'tsne',step =it)
    plot_ncc_maps(ncc_valid_imgs, ncc_seedpoints_idx_lsts,locations,writer=writer, tag=f"ncc",step=it,ncols=len(idxes))




