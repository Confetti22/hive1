import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union

import pickle
import numpy as np
import tifffile as tif
from confettii.plot_helper import three_pca_as_rgb_image

from collections import defaultdict
from typing import Sequence, Tuple, Union, Literal, List
Arr   = Union[np.ndarray, torch.Tensor]
Array = Union[Arr, Sequence[Arr]]   # single array or list/tuple of arrays



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

            # --- save to disk (optional) ---
    save_parent_dir = Path(writer.log_dir).resolve()
    save_dir = save_parent_dir/'valid_imgs'

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{tag}_step{step}.png"
        fig.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
        # optional: print or log path
        print(f"Saved figure to {save_dir / fname}")

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

def _scatter_embedding(ax, emb, labels, title, filter_label=0, cmap='tab10'):
    """
    Generic scatter: emb [N,2], labels [N].
    Filters out points whose label == filter_label (if not None).
    """
    import numpy as np
    labels = np.asarray(labels)
    if filter_label is not None:
        mask = labels != filter_label
        emb = emb[mask]
        labels = labels[mask]

    sc = ax.scatter(emb[:, 0], emb[:, 1], s=1.2, c=labels, cmap=cmap)
    # legend_elements() can be heavy for large class counts; optional:
    try:
        ax.legend(*sc.legend_elements(), title="Label", loc="best", markerscale=4)
    except Exception:
        pass
    ax.set_title(title)
    from sklearn.manifold import TSNE

def _run_tsne(X, tsne_kwargs=None):
    if tsne_kwargs is None:
        tsne_kwargs = {}
    # defaults
    tsne_defaults = dict(n_components=2, perplexity=20, random_state=42, init='pca')
    tsne_defaults.update(tsne_kwargs)
    return TSNE(**tsne_defaults).fit_transform(X)


def _run_umap(X, umap_kwargs=None):
    try:
        import umap
    except ImportError as e:
        raise RuntimeError(
            "UMAP is not installed. `pip install umap-learn` to enable mode='umap' or 'both'."
        ) from e

    if umap_kwargs is None:
        umap_kwargs = {}
    umap_defaults = dict(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
    umap_defaults.update(umap_kwargs)
    reducer = umap.UMAP(**umap_defaults)
    return reducer.fit_transform(X)


import warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

def umap_tsne_grid_plot(
        encoded_list,labels_list,writer,tag: str = "embed_grid",step: int = 0,
        tag_list=None,mode: str = "tsne",tsne_kwargs=None,umap_kwargs=None,filter_label=0,
         dpi=300, ext='png'
):
    """
    Plot one or more embeddings (t-SNE and/or UMAP) side-by-side (and stacked if both)
    and log to TensorBoard.

    Parameters
    ----------
    encoded_list : list[array-like, shape (Ni, Di)]
    labels_list  : list[array-like, shape (Ni,)]
    writer       : SummaryWriter
    tag          : str   TensorBoard root tag for the *composite* figure
    step         : int   Global step
    tag_list     : list[str] or None
        Same semantics as before (len==1 auto-expanded; len==N must match N).
    mode         : {'tsne','umap','both','tsne+umap','umap+tsne'}
        Controls what to plot. 'both' → 2-row figure (row0 t-SNE, row1 UMAP).
    tsne_kwargs  : dict passed to sklearn.manifold.TSNE
    umap_kwargs  : dict passed to umap.UMAP
    filter_label : value to exclude (e.g., background 0); set to None to keep all.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    num_plots = len(encoded_list)
    if num_plots != len(labels_list):
        raise ValueError(
            f"encoded_list ({num_plots}) and labels_list ({len(labels_list)}) length mismatch."
        )

    # Normalize titles
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

    mode_norm = mode.lower()
    if mode_norm in ("tsne+umap", "umap+tsne"):
        mode_norm = "both"
    if mode_norm not in ("tsne", "umap", "both"):
        raise ValueError(f"Unknown mode='{mode}'. Expected 'tsne','umap','both'.")

    # Run reducers -------------------------------------------------------------
    tsne_embeds = None
    umap_embeds = None

    if mode_norm in ("tsne", "both"):
        tsne_embeds = []
        for X in encoded_list:
            tsne_embeds.append(_run_tsne(X, tsne_kwargs))

    if mode_norm in ("umap", "both"):
        umap_embeds = []
        for X in encoded_list:
            umap_embeds.append(_run_umap(X, umap_kwargs))

    # Figure layout ------------------------------------------------------------
    if mode_norm == "both":
        nrows = 2
        ncols = num_plots
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        # Ensure we can iterate uniformly
        axes_tsne = axes[0, :] if ncols > 1 else [axes[0, 0]]
        axes_umap = axes[1, :] if ncols > 1 else [axes[1, 0]]

        # Row 0: t-SNE
        for ax, emb, labels, base_title in zip(axes_tsne, tsne_embeds, labels_list, per_plot_tags):
            _scatter_embedding(ax, emb, labels, f"{base_title} (t-SNE)", filter_label)

        # Row 1: UMAP
        for ax, emb, labels, base_title in zip(axes_umap, umap_embeds, labels_list, per_plot_tags):
            _scatter_embedding(ax, emb, labels, f"{base_title} (UMAP)", filter_label)

    else:
        # single-row figure
        nrows = 1
        ncols = num_plots
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4))
        if num_plots == 1:
            axes = [axes]  # flatten
        # Choose which embeddings we have
        embeds = tsne_embeds if mode_norm == "tsne" else umap_embeds
        suffix = "(t-SNE)" if mode_norm == "tsne" else "(UMAP)"
        for ax, emb, labels, base_title in zip(axes, embeds, labels_list, per_plot_tags):
            _scatter_embedding(ax, emb, labels, f"{base_title} {suffix}", filter_label)

    plt.tight_layout()

        # --- save to disk (optional) ---
    save_parent_dir = Path(writer.log_dir).resolve()
    save_dir = save_parent_dir/'valid_imgs'

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{tag}_step{step}.{ext}"
        fig.savefig(save_dir / fname, dpi=dpi, bbox_inches='tight')
        # optional: print or log path
        print(f"Saved figure to {save_dir / fname}")

    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)


def log_layer_embeddings(
    FEATURE_STORE,
    writer,
    epoch,
    label_volume,
    layer_order,
    max_layers=12,
    mode="tsne",
    tsne_kwargs=None,
    umap_kwargs=None,
    dpi=300, ext='png',
    valid_img_idx = -1,
    pca_flat = True,
    comment = "",

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
        out_t = FEATURE_STORE[k][valid_img_idx] if isinstance(FEATURE_STORE[k], (list, tuple)) else FEATURE_STORE[k]

        feat = out_t.detach().cpu().squeeze().numpy()  # could be [D,H,W,C], [H,W,C], [C,D,H,W], or [C,H,W]

        # --- Normalize to channel-last; if 4D, also standardize to [D,H,W,C] ---
        if feat.ndim == 4:
            # [C,D,H,W] -> [D,H,W,C]
            feat = np.moveaxis(feat, 0, -1)
            feat2d = feat[feat.shape[0] // 2]  # mid-slice -> [H,W,C]
        elif feat.ndim == 3:
            # [C,H,W] -> [H,W,C]
            feat = np.moveaxis(feat, 0, -1)
            feat2d = feat
        else:
            raise ValueError(f"Unexpected feature shape for {k}: {feat.shape}")

        # pick label mid-slice (features are 2D now)
        label_volume = np.squeeze(label_volume)
        if len(label_volume.shape) ==3:
            lbl2d = label_volume[label_volume.shape[0] // 2]  # [H,W]
        else:
            lbl2d = label_volume 
        lbl2d = lbl2d.astype(np.int8)

        # if min==0, then the background is 0
        if lbl2d.min() == 0:
            lbl2d = lbl2d -1

        H, W, C = feat2d.shape

        # PCA→RGB preview
        if pca_flat:
            rgb_img = three_pca_as_rgb_image(feat2d.reshape(-1, C), (H, W))
            pca_img_lst.append(rgb_img)

        # Align labels to feature resolution
        zoom_factor = [H / lbl2d.shape[0], W / lbl2d.shape[1]]
        label_zoomed = zoom(lbl2d, zoom_factor, order=0)

        mask = label_zoomed >= 0
        fg_feats_flat = feat2d[mask]
        fg_label_flat = label_zoomed[mask]

        blced_feats, blced_labels = class_balance(fg_feats_flat, fg_label_flat,n_per_class=300)
        tsne_encoded_feats_lst.append(blced_feats)
        tsne_label_lst.append(blced_labels)
        plot_tag_lst.append(k)

    # Embedding grid(s)
    print(f"{valid_img_idx= }")
    umap_tsne_grid_plot(
        tsne_encoded_feats_lst,
        tsne_label_lst,
        writer,
        tag=f"embed_grid{valid_img_idx}_{comment}",
        step=epoch,
        tag_list=plot_tag_lst,
        mode=mode,
        tsne_kwargs=tsne_kwargs,
        umap_kwargs=umap_kwargs,
        filter_label=None,
        dpi=dpi, ext=ext,
    )

    # PCA image grid
    if pca_flat:
        plot_pca_maps(
            pca_img_lst,
            writer=writer,
            tag=f"pca{valid_img_idx}_{comment}",
            step=epoch,
            ncols=len(pca_img_lst),
        )

    # FEATURE_STORE.clear()  # free memory














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

def get_t11_eval_data(E5,img_no_list =[1,3,4,5],ncc_seed_point = False):
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




