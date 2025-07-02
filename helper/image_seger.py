import time
import zarr
import numpy as np
import napari
from magicgui import widgets
from scipy.spatial.distance import pdist, squareform
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F





def get_target_feats_map(target_shape,roi_offset,lb,stride ):

    vol_start_idx = [ stride - (offset - lb)%stride for  offset,lb in zip(roi_offset,lb )]
    feats_offset = [ int( (start + offset - lb)//stride) for start, offset, lb in zip(vol_start_idx,roi_offset,lb )]

    feats_map = zarr.open_array('/home/confetti/data/t1779/mlp_feats.zarr',mode='a')
    print(f"feats_map.shape{feats_map.shape}")
    C,D,H,W = feats_map.shape
    # Desired region
    (lz, ly, lx) = feats_offset
    (hz, hy, hx) = [l + s for l, s in zip(feats_offset, target_shape)]

    # Clip to valid bounds
    max_z, max_y, max_x = D,H,W # assuming feats_map has shape (C, Z, Y, X)
    clipped_lz = max(0, lz)
    clipped_ly = max(0, ly)
    clipped_lx = max(0, lx)
    clipped_hz = min(max_z, hz)
    clipped_hy = min(max_y, hy)
    clipped_hx = min(max_x, hx)

    # Compute slices for existing data
    index = (
        slice(None),
        slice(clipped_lz, clipped_hz),
        slice(clipped_ly, clipped_hy),
        slice(clipped_lx, clipped_hx)
    )

    existing_data = feats_map[index]
    target_feats = np.zeros((C, *target_shape), dtype=feats_map.dtype)
    z_start = clipped_lz - lz
    y_start = clipped_ly - ly
    x_start = clipped_lx - lx
    z_end = z_start + (clipped_hz - clipped_lz)
    y_end = y_start + (clipped_hy - clipped_ly)
    x_end = x_start + (clipped_hx - clipped_lx)
    target_feats[:, z_start:z_end, y_start:y_end, x_start:x_end] = existing_data

    target_feats_map = np.moveaxis(target_feats,0,-1) # D,H,W,C
    print(f"target_feats.shape {target_feats_map.shape}")
    return target_feats_map





def map2sample_space(mapped_seg_out,sample_shape,vol_start_idx,stride):
    # feats_lst = np.moveaxis(feats_slice,0,-1).reshape(-1,C) 
    mapped_seg_out = np.squeeze(mapped_seg_out)
    zoomed_seg_out = np.kron(mapped_seg_out,np.ones((stride,stride,stride),dtype=int))

    lzp = vol_start_idx[0] 
    ret =( sample_shape[0] -  vol_start_idx[0] ) % stride 
    hzp = ret if  ret else stride

    lyp = vol_start_idx[1] 
    ret =( sample_shape[1] -  vol_start_idx[1] ) % stride 
    hyp = ret if  ret else stride

    lxp = vol_start_idx[2] 
    ret =( sample_shape[2] -  vol_start_idx[2] ) % stride 
    hxp = ret if  ret else stride

    seg_out =np.pad(zoomed_seg_out,pad_width=((lzp,hzp),(lyp,hyp),(lxp,hxp)),constant_values=0).astype(int)

    return seg_out 


def _compute_seg1(label_mask: np.ndarray, feature_map: np.ndarray, dist_matrix=None, spatail_decay=True,) -> np.ndarray:
    """
    using similarity and distance matrix to compute seg 

    Parameters
    ----------
    label_mask : np.ndarray
        A 3D array of shape (D, H, W) containing integer class labels for each voxel.
    feature_map : np.ndarray
        A 4D array with dimensions ordered as (D, H, W, C), where C is the number of feature channels.
    dist_matrix : optional
        Precomputed distance matrix for spatial weighting (default: None).
    spatial_decay : bool, optional
        Whether to apply spatial decay weighting (default: True).

    Returns
    -------
    np.ndarray
        A 3D array of shape (D, H, W) with predicted class labels for each voxel.
    """

    print(f"label_mask.shape {label_mask.shape}")

    unique_labels = np.unique(label_mask)
    unique_labels = unique_labels[unique_labels != 0]  # ignore background (if 0)

    if len(unique_labels) < 2:
        return np.zeros(label_mask.shape, dtype=np.uint8)

    D,H, W, C = feature_map.shape
    flat_feats = feature_map.reshape(-1, C)
    num_pixels = flat_feats.shape[0]
    class_similarities = np.full((num_pixels, len(unique_labels)), -np.inf)

    for class_idx, class_label in enumerate(unique_labels):
        class_mask = label_mask == class_label
        if not np.any(class_mask):
            continue

        class_feats = feature_map[class_mask]
        class_indices = np.where(class_mask.reshape(-1))[0]

        if spatail_decay and dist_matrix is not None:
            sim = (flat_feats @ class_feats.T) * dist_matrix[:, class_indices]
        else:
            sim = flat_feats @ class_feats.T

        max_sim = sim.max(axis=1)
        class_similarities[:, class_idx] = max_sim

    # Choose class with the highest similarity
    predicted_classes = np.argmax(class_similarities, axis=1)
    mapped_seg_label = np.array([unique_labels[i] for i in predicted_classes])
    mapped_seg_label = mapped_seg_label.reshape(D,H,W)
    return mapped_seg_label 


def _compute_seg2(label_mask: np.ndarray, feature_map: np.ndarray,  spatial_decay=True,d_sigma=16) -> np.ndarray:
    """
    using similarity to compute seg , compute distance when needed, will be faster

    Parameters
    ----------
    label_mask : np.ndarray
        A 3D array of shape (D, H, W) containing integer class labels for each voxel.
    feature_map : np.ndarray
        A 4D array with dimensions ordered as (D, H, W, C), where C is the number of feature channels.
    spatial_decay : bool, optional
        Whether to apply spatial decay weighting (default: True).
    Returns
    -------
    np.ndarray
        A 3D array of shape (D, H, W) with predicted class labels for each voxel.
    """
    
    print(f"label_mask.shape {label_mask.shape}")

    unique_labels = np.unique(label_mask)
    unique_labels = unique_labels[unique_labels != 0]  # ignore background (if 0)

    if len(unique_labels) < 2:
        return np.zeros(label_mask.shape, dtype=np.uint8)

    D,H, W, C = feature_map.shape
    flat_feats = feature_map.reshape(-1, C)
    num_pixels = flat_feats.shape[0]
    class_similarities = np.full((num_pixels, len(unique_labels)), -np.inf)

    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing='ij'
    )
    all_coords = np.stack([z_coords, y_coords, x_coords], axis=-1).reshape(-1, 3)

    for class_idx, class_label in enumerate(unique_labels):
        class_mask = label_mask == class_label
        if not np.any(class_mask):
            continue

        class_feats = feature_map[class_mask]
        class_indices = np.where(class_mask.reshape(-1))[0]
        class_coords = all_coords[class_indices]

        # Compute cosine similarity
        sim = flat_feats @ class_feats.T

        if spatial_decay:
            voxel_coords = all_coords[:, np.newaxis, :]  # shape: (num_pixels, 1, 3)
            class_coords_exp = class_coords[np.newaxis, :, :]  # shape: (1, num_class_points, 3)
            dists = np.linalg.norm(voxel_coords - class_coords_exp, axis=-1)  # shape: (num_pixels, num_class_points)
            decay_weights = np.exp(-dists**2 / (2*d_sigma**2))
            sim *= decay_weights  # element-wise weighting of similarity

        max_sim = sim.max(axis=1)
        class_similarities[:, class_idx] = max_sim

    # Choose class with the highest similarity
    predicted_classes = np.argmax(class_similarities, axis=1)
    mapped_seg_label = np.array([unique_labels[i] for i in predicted_classes])
    mapped_seg_label = mapped_seg_label.reshape(D,H,W)
    return mapped_seg_label

def replicate_nonzero_slices(arr, n):
    """
    Replicates each non-zero z-slice of arr to n slices before and after.
    
    Parameters:
    - arr: np.ndarray of shape (D, H, W), dtype=int
    - n: int, number of slices to replicate before and after
    
    Returns:
    - arr_copy: np.ndarray with the replicated slices
    """
    D, H, W = arr.shape
    arr_copy = arr.copy()
    
    # Find indices of non-zero slices along the z-axis
    nonzero_z_indices = [i for i in range(D) if np.any(arr[i])]
    
    for idx in nonzero_z_indices:
        start = max(0, idx - n)
        end = min(D, idx + n + 1)
        for i in range(start, end):
            arr_copy[i] = arr[idx]
    
    return arr_copy

class SegmentationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 12),
            nn.ReLU(),
            nn.Linear(12, num_classes)  # Multiclass logits
        )

    def forward(self, x):
        return self.classifier(x)
    

class ConvSegHead(nn.Module):
    """
    A small 3D convolutional head for voxel-wise classification.
    adding padding to insure the output shape is the same as input
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv3d(in_channels // 2, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # x shape: [B, C, D, H, W]
        return self.net(x)  # logits per class

def _seg_via_mlp_head(user_mask, feature_map, num_epochs=100, lr=1e-3, return_prob=False):
    """
    Args:
        user_mask: numpy of shape (D, H, W) or (H, W), where labeled voxels have integer class labels >= 0.
        feature_map: numpy of shape (D, H, W, C) or (H, W, C)
    
    Returns:
        predicted_mask: numpy of shape (D, H, W) or (H, W), with predicted class labels (int64)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    feature_map = torch.from_numpy(feature_map)
    user_mask = torch.from_numpy(user_mask)

    if feature_map.dim() == 4:
        D, H, W, C = feature_map.shape
        coords = torch.nonzero(user_mask > 0, as_tuple=False)  # [N, 3]
        z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
        prompt_features = feature_map[z, y, x]  # [N, C]
    elif feature_map.dim() == 3:
        H, W, C = feature_map.shape
        coords = torch.nonzero(user_mask > 0, as_tuple=False)  # [N, 2]
        y, x = coords[:, 0], coords[:, 1]
        prompt_features = feature_map[y, x]  # [N, C]
    else:
        raise ValueError("feature_map must be 3D or 4D")

    labels = user_mask[tuple(coords.T)] - 1  # Convert to 0-based index
    num_classes = labels.max().item() + 1

    if num_classes < 2:
        raise ValueError("Need at least 2 labeled classes in the mask.")

    head = SegmentationHead(C, num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    prompt_features = prompt_features.to(device)
    prompt_labels = labels.to(device)

    for epoch in tqdm(range(num_epochs)):
        head.train()
        optimizer.zero_grad()
        logits = head(prompt_features)
        loss = loss_fn(logits, prompt_labels)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item():.4f}")

    # Predict over full volume/image
    flat_features = feature_map.reshape(-1, C).to(device)
    with torch.no_grad():
        head.eval()
        logits = head(flat_features)
        probs = F.softmax(logits, dim=1)

    if return_prob:
        prob_vol = probs.reshape((-1, num_classes)).reshape(*feature_map.shape[:-1], num_classes)
        if feature_map.dim() == 4:
            prob_vol = prob_vol.permute(3, 0, 1, 2)  # [K, D, H, W]
        else:
            prob_vol = prob_vol.permute(2, 0, 1)      # [K, H, W]
        return prob_vol.detach().cpu().numpy()
    else:
        pred_mask = torch.argmax(probs, dim=1).reshape(feature_map.shape[:-1]) + 1
        return pred_mask.detach().cpu().numpy()



def _seg_via_conv_head(user_input_label, feature_map, num_epochs=100, lr=1e-3, return_prob=False):
    """
    Args:
        user_input_label: numpy of shape (D, H, W) or (H, W), labels >= 1
        feature_map: numpy of shape (D, H, W, C) or (H, W, C)
    Returns:
        predicted_mask or probabilities of shape (D, H, W) or (H, W)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    feature_map = torch.from_numpy(feature_map).float()
    user_input_label = torch.from_numpy(user_input_label).long()

    is_3d = feature_map.dim() == 3  # (H, W, C)
    is_4d = feature_map.dim() == 4  # (D, H, W, C)

    if is_4d:
        D, H, W, C = feature_map.shape
        feat = feature_map.permute(3, 0, 1, 2).to(device)  # [C, D, H, W]
        labels = user_input_label.to(device) - 1  # 0-based
        mask = (labels >= 0)
        spatial_shape = (D, H, W)
    elif is_3d:
        H, W, C = feature_map.shape
        feat = feature_map.permute(2, 0, 1).unsqueeze(1).to(device)  # [C, 1, H, W]
        labels = user_input_label.to(device) - 1  # 0-based
        mask = (labels >= 0)
        spatial_shape = (H, W)
    else:
        raise ValueError("feature_map must be 3D or 4D")

    num_classes = labels.max().item() + 1
    if num_classes < 2:
        raise ValueError("Need at least 2 labeled classes in the mask.")

    head = ConvSegHead(C, num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epochs)):
        head.train()
        optimizer.zero_grad()
        logits = head(feat)  # [K, D, H, W] or [K, 1, H, W]

        logits = logits if is_4d else logits.squeeze(1)  # remove dummy D=1 for 2D

        # Permute logits to [D, H, W, K] or [H, W, K]
        logits_flat = logits.permute(*range(1, logits.ndim), 0)[mask]  # [N, K]
        labels_flat = labels[mask]

        loss = loss_fn(logits_flat, labels_flat)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"Conv head epoch {epoch}, loss: {loss.item():.4f}")

    # Inference
    head.eval()
    with torch.no_grad():
        logits = head(feat)
        logits = logits if is_4d else logits.squeeze(1)
        probs = F.softmax(logits, dim=0)  # [K, D, H, W] or [K, H, W]

    if return_prob:
        return probs.detach().cpu().numpy()
    else:
        pred = torch.argmax(probs, dim=0) + 1  # back to 1-based label
        return pred.detach().cpu().numpy()


def seg_via_mlp_head(roi_offset, roi_size, label: np.ndarray, lb, stride=16):
    vol_start_idx = [(stride - (offset - base) % stride) for offset, base in zip(roi_offset, lb)]

    processed_label = replicate_nonzero_slices(label, n=18)
    mapped_label = processed_label[
        vol_start_idx[0]::stride,
        vol_start_idx[1]::stride,
        vol_start_idx[2]::stride
    ][:-1, :-1, :-1]

    target_feats_map = get_target_feats_map(mapped_label.shape, roi_offset=roi_offset, lb=lb, stride=stride)

    start_time = time.time()
    mapped_seg = _seg_via_mlp_head(user_mask=mapped_label, feature_map=target_feats_map, num_epochs=2000)
    print(f"mlp compute seg time: {time.time() - start_time:.2f}s")

    return map2sample_space(mapped_seg, roi_size, vol_start_idx, stride)

def seg_via_conv_head(roi_offset, roi_size, label: np.ndarray, lb, stride=16):
    """
    Wrapper that parallels seg_via_mlp_head but uses ConvSegHead.
    """
    # compute starting indices to align with stride
    vol_start_idx = [(stride - (offset - lb0) % stride) for offset, lb0 in zip(roi_offset, lb)]
    # replicate and map label slices
    processed_label = replicate_nonzero_slices(label, n=18)
    mapped_label = processed_label[
        vol_start_idx[0]::stride,
        vol_start_idx[1]::stride,
        vol_start_idx[2]::stride
    ][:-1, :-1, :-1]

    target_feats_map = get_target_feats_map(mapped_label.shape,
                                           roi_offset=roi_offset,
                                           lb=lb,
                                           stride=stride)

    start_time = time.time()
    mapped_seg = _seg_via_conv_head(
        user_input_label = mapped_label,
        feature_map = target_feats_map,
        num_epochs=2000
    )
    print(f"compute conv seg time: {time.time() - start_time:.2f}s")

    return map2sample_space(mapped_seg, roi_size, vol_start_idx, stride)




# --- Segmentation utilities ---

def seg_by_computing_sim(roi_offset, roi_size, label: np.ndarray, lb, stride=16):
    vol_start_idx = [(stride - (offset - base) % stride) for offset, base in zip(roi_offset, lb)]

    # Map label from sample to feature space
    processed_label = replicate_nonzero_slices(label, n=18)
    mapped_label = processed_label[
        vol_start_idx[0]::stride,
        vol_start_idx[1]::stride,
        vol_start_idx[2]::stride
    ][:-1, :-1, :-1]

    target_feats_map = get_target_feats_map(mapped_label.shape, roi_offset=roi_offset, lb=lb, stride=stride)


    start_time = time.time()
    mapped_seg = _compute_seg2(label_mask=mapped_label, feature_map=target_feats_map, spatial_decay=False)
    print(f"somputing_sim compute seg time: {time.time() - start_time:.2f}s")

    return map2sample_space(mapped_seg, roi_size, vol_start_idx, stride)



# --- UI Controller Class ---

class SimpleSeger2(widgets.Container):
    def __init__(self, viewer1: napari.Viewer, viewer2: napari.Viewer, simple_viewer):
        super().__init__()
        self.viewer1 = viewer1
        self.simple_viewer = simple_viewer

        self.stride = 16
        self.lb = [int(x + 1.5 * self.stride) for x in [3392, 2512, 3504]]
        self.roi_size = [64, 64, 64]
        self.init_label_data()

        self._setup_layers()
        self._setup_buttons()
        self._register_callbacks()

    def init_label_data(self):
        shape = tuple(self.roi_size)
        self.last_seg_data = np.zeros(shape, dtype=np.uint8)
        self.last_label_data = np.zeros(shape, dtype=np.uint8)
        self.current_label_data = np.zeros(shape, dtype=np.uint8)

    def _setup_layers(self):
        zero_data = np.zeros(self.roi_size, dtype=np.uint8)
        self.label_layer = self.viewer1.add_labels(zero_data, name='Label')
        self.segout_layer = self.viewer1.add_labels(zero_data, name='Segout')
        self.label_layer.brush_size = 30
        self.label_layer.mode = 'PAINT'
        self.viewer1.layers.selection = [self.label_layer]

    def _setup_buttons(self):
        self.method_selector = widgets.ComboBox(
                choices=["computing_sim", "mlp_head","conv_head"],
                value="computing_sim",
                label="Segmentation Method"
            )
        self.seg_button = widgets.PushButton(text="Seg")
        self.clear_button = widgets.PushButton(text="Clear")
        self.undo_button = widgets.PushButton(text="Undo")

        self.seg_button.clicked.connect(self.run_seg)
        self.clear_button.clicked.connect(self.clear_labels)
        self.undo_button.clicked.connect(self.undo_labels)

        self.extend([self.method_selector,self.seg_button, self.clear_button, self.undo_button])

    def _register_callbacks(self):
        self.simple_viewer.roi_layer.events.data.connect(self.prepare_seg)

    # --- Button Actions ---

    def prepare_seg(self):
        roi_size = self.read_roi_size()
        self.label_layer.data = np.zeros(roi_size, dtype=np.uint8)
        self.segout_layer.data = np.zeros(roi_size, dtype=np.uint8)
        self.current_label_data = np.zeros(roi_size, dtype=np.uint8)

    def run_seg(self):
        self._backup_current_state()
        roi_offset = self.read_roi_offset()
        roi_size = self.read_roi_size()
        label_data = self.label_layer.data.copy()

        method = self.method_selector.value
        if method == "computing_sim":
            seg_out = seg_by_computing_sim(roi_offset, roi_size, label_data, self.lb, stride=self.stride)
        elif method == "mlp_head":
            seg_out = seg_via_mlp_head(roi_offset, roi_size, label_data, self.lb, stride=self.stride)
        elif method =='conv_head':
            seg_out = seg_via_conv_head(roi_offset, roi_size, label_data, self.lb, stride=self.stride)
        else:
            print(f"Unknown method: {method}")
            return

        self.segout_layer.data = seg_out
        self.viewer1.layers.selection = [self.label_layer]

    def clear_labels(self):
        shape = self.label_layer.data.shape
        self.label_layer.data = np.zeros(shape, dtype=np.uint8)
        self.segout_layer.data = np.zeros(shape, dtype=np.uint8)
        self.viewer1.layers.selection = [self.label_layer]

    def undo_labels(self):
        self.label_layer.data = self.last_label_data
        self.segout_layer.data = self.last_seg_data
        self.viewer1.layers.selection = [self.label_layer]

    # --- Utilities ---

    def _backup_current_state(self):
        self.last_label_data = self.current_label_data.copy()
        self.last_seg_data = self.segout_layer.data.copy()
        self.current_label_data = self.label_layer.data.copy()

    def read_roi_offset(self):
        return self.simple_viewer.get_roi_offset

    def read_roi_size(self):
        return self.simple_viewer.get_roi_size