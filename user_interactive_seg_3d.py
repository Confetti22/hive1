#%%
import torch
import numpy as np
import napari
from magicgui import magicgui,widgets
from magicgui.widgets import Container
from train_helper import get_eval_data,MLP
from scipy.ndimage import zoom
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize
from helper.simple_viewer import SimpleViewer
import zarr
import time
import matplotlib.pyplot as plt
from helper.image_reader import Ims_Image, wrap_image
import tifffile as tif

#%%
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

def compute_seg(label_mask: np.ndarray, feature_map: np.ndarray, dist_matrix=None, spatail_decay=True,) -> np.ndarray:
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


def seg(roi_offset,roi_size,label:np.ndarray,lb,stride = 16):

    vol_start_idx = [ stride - (offset - lb)%stride for  offset,lb in zip(roi_offset,lb )]

    # skipped the first stride cube (both imcompele and complete cube , cube is the roi when extrating feats)
    #map label from sample space to feature space
    processed_label = replicate_nonzero_slices(label,n=18)
    mapped_label = processed_label[ vol_start_idx[0]::stride, vol_start_idx[1]::stride,vol_start_idx[2]::stride] 
    mapped_label = mapped_label[:-1,:-1,:-1]
    target_feats_map = get_target_feats_map(mapped_label.shape,roi_offset=roi_offset,lb=lb,stride=stride)

    current = time.time()
    loc_lst = list(np.ndindex(target_feats_map.shape[:-1]))
    d_sigma = 16
    dist = pdist(loc_lst, metric='euclidean')
    dist_matrix = squareform(dist)
    print(f"distacne_matrix: {dist_matrix.shape}")
    dist_matrix = np.exp(- dist_matrix**2/(2*d_sigma**2))
    print(f"compute_dis_matrix: {time.time() - current}")
    current = time.time()

    mapped_seg_out = compute_seg(label_mask = mapped_label,feature_map=target_feats_map, dist_matrix= dist_matrix, spatail_decay=True) 
    print(f"compute seg time :{time.time() - current}")
    seg_out = map2sample_space(mapped_seg_out,roi_size,vol_start_idx,stride)
    return seg_out

    
#%%


# level = 0
# stride = 16
# ims_vol = Ims_Image(ims_path="/home/confetti/e5_data/t1779/t1779.ims", channel=2)
# raw_volume_size =ims_vol.rois[level][3:] #the data shape at r3 for test
# print(f"raw_volume_size{raw_volume_size}")
# whole_volume_size = [int(element//2) for element in raw_volume_size]
# whole_volume_offset = [int(element//4) for element in raw_volume_size]
# valid_offset = [ int(x + int((3/2) *stride)) for x in whole_volume_offset]
# valid_size = [ int(x - int((3/2) *stride)) for x in whole_volume_size]
# lb = valid_offset
# hb = [ x+ y for x, y in zip(valid_offset,valid_size)] 

# roi_offset = [7436,5600,4850]
# roi_size =[128,1024,1024]
# raw_img = ims_vol.from_roi(coords=[*roi_offset,*roi_size],level=0)
# img_shape = raw_img.shape
#%%
tif_path = '/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/LuZhonghua/CrabeatingMacaque_Brain_CM001/Reconstruction_/SliceImage/10.0/PBM_PFC_CMD47-050_090_640nm_10X.tif'
tif_img = tif.imread(tif_path)
print(f"{tif_img.shape=}")
#%%

current  = time.time()
print(f"reading from ims, raw_img shape {raw_img.shape}, time :{time.time() - current}")


label_data = np.zeros((img_shape),dtype=np.uint8)

viewer = napari.Viewer(ndisplay=3)
image_layer = viewer.add_image(raw_img,name = 'raw_roi')
label_layer = viewer.add_labels(label_data,name ='Label')
segout_layer = viewer.add_labels(label_data,name = 'Segout')

label_layer.brush_size = 30
label_layer.mode = 'PAINT'

viewer.layers.selection = [label_layer]  # Keep selected



# --- Define separate buttons ---
seg_button = widgets.PushButton(text="Seg")
clear_button = widgets.PushButton(text="Clear")
undo_button = widgets.PushButton(text="Undo")

last_seg_data = np.zeros((img_shape),dtype=np.uint8) 
last_label_data = np.zeros((img_shape),dtype=np.uint8)
current_label_data = np.zeros((img_shape),dtype=np.uint8)


# --- Seg button action ---
@seg_button.clicked.connect
def run_seg():
    label_data = label_layer.data.copy()
    
    global last_label_data, current_label_data,last_seg_data  # <-- declare them as global

    last_label_data = current_label_data
    current_label_data  = label_data
    last_seg_data = segout_layer.data.copy() 

    # need current roi_size and a lb related to pre_computed_feats_map
    seg_out = seg(roi_offset,roi_size,label_data,lb,stride = 16)
    segout_layer.data = seg_out 

    viewer.layers.selection = [label_layer]  # Keep selected
# --- Clear button action ---

@clear_button.clicked.connect
def clear_labels():
    label_layer.data = np.zeros_like(label_layer.data)
    segout_layer.data = np.zeros_like(label_layer.data)
    viewer.layers.selection = [label_layer]  # Keep selected

@undo_button.clicked.connect
def undo_labels():
    global last_label_data, current_label_data,last_seg_data  # <-- declare them as global
    label_layer.data = last_label_data 
    segout_layer.data = last_seg_data 
    viewer.layers.selection = [label_layer]  # Keep selected




# --- Combine buttons into a container widget ---
control_panel = Container(widgets=[seg_button, clear_button,undo_button])

# Add widget to napari
viewer.window.add_dock_widget(control_panel, area='right')

# viewer_widget = SimpleViewer(viewer=viewer)
# viewer.window.add_dock_widget(viewer_widget,area='right')
napari.run()

