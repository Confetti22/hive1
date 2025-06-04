#%%
import sys
import os

# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
#%%

from lib.arch.ae import build_final_model,load_compose_encoder_dict
from config.load_config import load_cfg
import time
from torchsummary import summary

device ='cuda'
print(f'{os.getcwd()}=')
args = load_cfg('config/vsi_ae_2d.yaml')
cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)

cnn_ckpt_pth = '/home/confetti/data/weights/vsi_2d_ae_best.pth'
mlp_ckpt_pth = '/home/confetti/e5_workspace/hive/contrastive_run_vsi/avg5_batch2048_nview2_pos_weight_2/model_final.pth'
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth)
print(cmpsd_model)
summary(cmpsd_model,(1,*args.input_size))
exit(0)

#%%
import tifffile as tif
import torch
from lib.utils.preprocess_img import pad_to_multiple_of_unit
from confettii.plot_helper import kmeans_grid_results 
import numpy as np
import matplotlib.pyplot as plt
img = tif.imread('/home/confetti/e5_data/wide_filed/nuclei_channel/074.tif')
plt.imshow(img)
#%%


img = img[2080:2080+1536,3180:3180+1536]



zoom_factor= 8

img = pad_to_multiple_of_unit(img,unit=zoom_factor) 


input = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device) #add batch and channel dim B*C*H*W
out = cmpsd_model(input).cpu().detach().squeeze().numpy()
C,H,W = out.shape

# arr: shape (C, H, W)
lengths = np.linalg.norm(out, axis=0)  # shape (H, W)
print(f"{out.shape=}")
print(f"{lengths.shape=}")
print(f"{lengths=}")
#%%
# feat_list = np.moveaxis(out,0,-1).reshape(-1,C)
# kmeans_grid_results(feat_list,K_values=[4,6,8],img_shape=(H,W))


#%%
# %%
from scipy.spatial.distance import pdist,squareform
import numpy as np
import napari
from magicgui import magicgui,widgets
from magicgui.widgets import Container
from scipy.ndimage import zoom
viewer = napari.Viewer()
viewer.add_image(img, name='img',contrast_limits=[0,4000])

img_shape = img.shape
label_data = tif.imread('/home/confetti/e5_data/wide_filed/test_seg_Label.tif')
# label_data = np.zeros((img_shape),dtype=np.uint8)
label_layer = viewer.add_labels(label_data,name ='Label')
label_layer.brush_size = 30
label_layer.mode = 'PAINT'
print(f"img_shape type :{type(img_shape),img_shape}")

#current only suit for 2d
scaled_img_shape = tuple( int(x// zoom_factor)  for x in img_shape)
print(f"scaled_img_shaep :{type(scaled_img_shape), {scaled_img_shape}}")

loc_lst = list(np.ndindex(scaled_img_shape))
d_sigma = 16
dist = pdist(loc_lst, metric='euclidean')
dist_matrix = squareform(dist)
print(f"distacne_matrix: {dist_matrix.shape}")
dist_matrix = np.exp(- dist_matrix**2/(2*d_sigma**2))

# %%
segout_layer = viewer.add_labels(label_data,name = 'Segout')

viewer.layers.selection = [label_layer]  # Keep selected

# --- Define separate buttons ---
seg_button = widgets.PushButton(text="Seg")
clear_button = widgets.PushButton(text="Clear")
undo_button = widgets.PushButton(text="Undo")

last_seg_data = np.zeros((img_shape),dtype=np.uint8) 
last_label_data = np.zeros((img_shape),dtype=np.uint8)
current_label_data = np.zeros((img_shape),dtype=np.uint8)


def seg_func(ori_label_mask: np.ndarray, feature_map: np.ndarray, dist_matrix= dist_matrix, spatail_decay= False, scale_factor=1,class_assign_thres = 0.5) -> np.ndarray:
    print(f"begin to seg")
    current = time.time()
    label_mask = zoom(ori_label_mask, zoom=(1/scale_factor),order=0)
    print(f"label_mask.shape {label_mask.shape}")

    unique_labels = np.unique(label_mask)
    unique_labels = unique_labels[unique_labels != 0]  # ignore background (if 0)

    if len(unique_labels) < 1:
        return np.zeros(ori_label_mask.shape, dtype=np.uint8)

    C,H, W = feature_map.shape
    print(f"in seg_func:{feature_map.shape=}")
    feature_map = np.moveaxis(feature_map,0,-1)
    print(f"in seg_func after moveaxis:{feature_map.shape=}")
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


    max_similarities = np.max(class_similarities, axis=1)
    predicted_classes = np.argmax(class_similarities, axis=1)

    # Choose class with the highest similarity
    # Apply threshold: if max similarity < thres → assign class 0
    seg_label = np.array([
        unique_labels[i] if max_similarities[idx] >= class_assign_thres else 0
        for idx, i in enumerate(predicted_classes)
    ])
    # seg_label = np.array([unique_labels[i] for i in predicted_classes])

    seg_label = seg_label.reshape(H, W)

    zoomed_seg_label = zoom(seg_label, zoom=scale_factor,order=0)
    print(f"finished time used:{time.time()-current}")

    return zoomed_seg_label.astype(np.uint8)

# --- Seg button action ---
@seg_button.clicked.connect
def run_seg():
    label_data = label_layer.data.copy()
    
    global last_label_data, current_label_data,last_seg_data  # <-- declare them as global

    last_label_data = current_label_data
    current_label_data  = label_data
    last_seg_data = segout_layer.data.copy() 

    seg_result = seg_func(label_data, feature_map= out,dist_matrix= dist_matrix,spatail_decay= True,scale_factor=zoom_factor,class_assign_thres=0)
    segout_layer.data = seg_result

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

napari.run()
# %%