import numpy as np
import napari
from magicgui import widgets
from scipy.spatial.distance import pdist, squareform
import zarr
import time




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

    

class SimpleSeger(widgets.Container):
    def __init__(self, viewer1: napari.Viewer, viewer2: napari.Viewer,simple_viewer):
        super().__init__()

        self.simple_viewer = simple_viewer

        stride = 16
        whole_volume_offset =  [3392, 2512, 3504]
        valid_offset = [ int(x + int((3/2) *stride)) for x in whole_volume_offset]
        self.lb = valid_offset

        current  = time.time()
        roi_offset = [7500,5600,4850]
        roi_size =[64,64,64]


        label_data = np.zeros((roi_size),dtype=np.uint8)
        self.viewer1 = viewer1

        self.label_layer = self.viewer1.add_labels(label_data,name ='Label')
        self.segout_layer = self.viewer1.add_labels(label_data,name = 'Segout')

        self.last_seg_data = np.zeros((roi_size),dtype=np.uint8) 
        self.last_label_data = np.zeros((roi_size),dtype=np.uint8)
        self.current_label_data = np.zeros((roi_size),dtype=np.uint8)

        self.label_layer.brush_size = 30
        self.label_layer.mode = 'PAINT'
        self.viewer1.layers.selection = [self.label_layer]  # Keep selected
        
        self.regiser_callbacks()

        


    def regiser_callbacks(self):
        # --- Define separate buttons ---
        self.simple_viewer.roi_layer.events.data.connect(self.prepare_seg)
        self.seg_button = widgets.PushButton(text="Seg")
        self.seg_button.clicked.connect(self.run_seg)
        self.clear_button = widgets.PushButton(text="Clear")
        self.clear_button.clicked.connect(self.clear_labels)
        self.undo_button = widgets.PushButton(text="Undo")
        self.undo_button.clicked.connect(self.undo_labels)

        self.extend([
            self.seg_button, 
            self.clear_button,
            self.undo_button,
            ])

    # --- Seg button action ---
    def prepare_seg(self):
        roi_size = self.read_roi_size_from_simpleviewer()
        self.label_layer.data = np.zeros(roi_size,dtype=np.uint8)
        self.segout_layer.data = np.zeros(roi_size,dtype=np.uint8)
        self.current_label_data = np.zeros((roi_size),dtype=np.uint8)



    def run_seg(self,):
        label_data = self.label_layer.data.copy()
        

        self.last_label_data = self.current_label_data
        self.current_label_data  = label_data
        self.last_seg_data = self.segout_layer.data.copy() 

        roi_offset =self.read_offset_from_simpleviewer()
        roi_size = self.read_roi_size_from_simpleviewer()

        # need current roi_size and a lb related to pre_computed_feats_map
        seg_out = seg(roi_offset,roi_size,label_data,self.lb,stride = 16)
        self.segout_layer.data = seg_out 

        self.viewer1.layers.selection = [self.label_layer]  # Keep selected

    def clear_labels(self,):
        label_layer = self.label_layer
        label_layer.data = np.zeros_like(label_layer.data)
        self.segout_layer.data = np.zeros_like(label_layer.data)
        self.viewer1.layers.selection = [label_layer]  # Keep selected

    def undo_labels(self,):
        label_layer = self.label_layer
        label_layer.data = self.last_label_data 
        self.segout_layer.data = self.last_seg_data 
        self.viewer1.layers.selection = [label_layer]  # Keep selected
    
    def read_offset_from_simpleviewer(self):
        return self.simple_viewer.get_roi_offset

    def read_roi_size_from_simpleviewer(self):
        return self.simple_viewer.get_roi_size





