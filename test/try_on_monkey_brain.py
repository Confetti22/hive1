#%%
import torch.nn as nn
import torch

from ..helper.extrac_feats_helper import Encoder ,TraverseDataset3d,get_feature_list, load_cfg
import torch.nn as nn
from sklearn.cluster import KMeans

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchsummary import summary
import torch
import tifffile as tif
import torch.nn.functional as F
import torch.nn as nn
import time

def modify_key(weight_dict,source,target):
    new_weight_dict = {}
    for key, value in weight_dict.items():
        new_key = key.replace(source,target)
        new_weight_dict[new_key] = value
    return new_weight_dict


def delete_key(weight_dict,pattern_lst):
    new_weight_dict = {k: v for k, v in weight_dict.items() if not k.startswith(pattern_lst)}
    return new_weight_dict 

def load_encoder_dict(model):
    ckpt_pth = '/home/confetti/data/visor_ae_weights/1024data_k553_input128_998.pth'
    ckpt = torch.load(ckpt_pth)
    removed_module_dict = modify_key(ckpt['model'],source='module.',target='')
    deleted_unwanted_dict = delete_key(removed_module_dict,('fc1', 'fc2','contrastive_projt','up_layers','conv_out'))

    model.cnn_encoder.load_state_dict(deleted_unwanted_dict,strict=False)



class BaiscEncoder(nn.Module):
    def __init__(self,
                 in_channel: int = 1,
                 encoder_filters =[32,64,96],
                 encoder_block_type: str = 'single',
                 pad_mode: str = 'reflect',
                 act_mode: str = 'elu',
                 norm_mode: str = 'none',
                 encoder_kernel_size =[5,3,3],
                 init_mode: str = 'none',
                 **kwargs
                 ):
        super().__init__()
        self.cnn_encoder= Encoder(in_channel,encoder_filters,
                    pad_mode,act_mode,norm_mode,kernel_size=encoder_kernel_size,init_mode=init_mode,
                    block_type=encoder_block_type,
                    **kwargs)
        self.sum_layer = nn.AdaptiveAvgPool3d(output_size=1)
    
    def forward(self,x):
        x = self.cnn_encoder(x)
        x = self.sum_layer(x)
        return x

def build_model(cfg):
    kwargs = {
        'in_channel': cfg.MODEL.IN_PLANES,
        'input_size': cfg.DATASET.input_size,
        'encoder_filters': cfg.MODEL.FILTERS,
        'clf_filters': cfg.MODEL.clf_filters,
        'encoder_kernel_size':cfg.MODEL.kernel_size,
        'encoder_block_type': cfg.MODEL.BLOCK_TYPE,
        'pad_mode': cfg.MODEL.PAD_MODE,
        'act_mode': cfg.MODEL.ACT_MODE,
        'norm_mode': cfg.MODEL.NORM_MODE,
        'cluster_feats_dim': cfg.cluster_feats_dim,
        'K':cfg.K,
    }

    model = BaiscEncoder(**kwargs)
    return model



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(96, 48)  
        self.fc2 = nn.Linear(48, 24)  
        self.fc3 = nn.Linear(24, 12)  
        self.relu = nn.ReLU()  

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on the output layer
        return x / x.norm(p=2, dim=-1, keepdim=True)

#%%

device ='cuda'
E5 = False 

cnn_cfg = load_cfg('/home/confetti/e5_workspace/deepcluster4brain/deepcluster_out/logs/test6_3cnntraining_fixed_dataorder_samller_data/cfg.yaml')
cnn_win_size =cnn_cfg.DATASET.input_size[0]
stride = 16 
batch_size = 512 

#define model
cnn_model = build_model(cnn_cfg)
cnn_model.eval()

print(cnn_model)
cnn_model.to(device)
summary(cnn_model,(1,*cnn_cfg.DATASET.input_size))
load_encoder_dict(cnn_model)


mlp = MLP().to(device)
mlp.eval()
mlp_ckpt_pth = "/home/confetti/e5_workspace/deepcluster4brain/runs/test14_8192_batch4096_nview2_pos_weight_2_shuffle_every50/model_epoch_41999.pth"
mlp_ckpt= torch.load(mlp_ckpt_pth)
mlp.load_state_dict(mlp_ckpt)


def extract_feats(img_vol,win_size,cnn,mlp):
    """
    img_vol: need to be precropped
    """

    draw_border_dataset = TraverseDataset3d(img_vol,stride=stride,win_size=win_size)  
    border_draw_loader = DataLoader(draw_border_dataset,batch_size,shuffle=False,drop_last=False)
    print(f"len of dataset is {len(draw_border_dataset)}")

    current = time.time()
    feats_lst = get_feature_list('cuda',cnn,mlp,border_draw_loader,save_path=None)
    out_shape = draw_border_dataset.sample_shape

    print(f"extracting feature from image consume {time.time()-current} seconds")
    return feats_lst,out_shape

#%%
import torch
import numpy as np
import napari
import matplotlib.pyplot as plt
import tifffile as tif
from image_reader import Ims_Image

tif_pth = '/home/confetti/data/CM001-1slice/PBM_PFC_CMD47-050_105_640nm_10X.tif'

ims_vol = Ims_Image('/home/confetti/data/z03120_c3.ims')

img_vol = ims_vol.from_roi(coords=[0,1024,1024,30,1536,1536],level=0)
img_vol = np.pad(img_vol,pad_width=((17,17),(0,0),(0,0)))
img_shape = img_vol.shape


feats_lst,outs_shape = extract_feats(img_vol,win_size=64,cnn=cnn_model,mlp=mlp)
print(f"{outs_shape=}")
# import pickle
# with open('/home/confetti/data/c001_feats.pkl', 'wb') as f:
#     pickle.dump(feats_lst, f)

# %%
from magicgui import magicgui,widgets
from magicgui.widgets import Container
from scipy.ndimage import zoom
from scipy.spatial.distance import pdist, squareform

viewer = napari.Viewer(ndisplay=2)
z_slice = img_vol[32,24:-24,24:-24]
viewer.add_image(z_slice,name ='img')

feature_map = feats_lst.reshape(outs_shape[1],outs_shape[2],12)
print(f"{feature_map.shape=}")
#%%

img_shape = z_slice.shape
label_data = np.zeros((img_shape),dtype=np.uint8)
label_layer = viewer.add_labels(label_data,name ='Label')
label_layer.brush_size = 30
label_layer.mode = 'PAINT'
print(f"img_shape type :{type(img_shape),img_shape}")

#current only suit for 2d
scaled_img_shape = tuple( int(x//stride)  for x in img_shape)
print(f"scaled_img_shaep :{type(scaled_img_shape), {scaled_img_shape}}")
loc_lst = list(np.ndindex(scaled_img_shape))
d_sigma = 16
dist = pdist(loc_lst, metric='euclidean')
dist_matrix = squareform(dist)
print(f"distacne_matrix: {dist_matrix.shape}")
dist_matrix = np.exp(- dist_matrix**2/(2*d_sigma**2))


segout_layer = viewer.add_labels(label_data,name = 'Segout')

viewer.layers.selection = [label_layer]  # Keep selected

# --- Define separate buttons ---
seg_button = widgets.PushButton(text="Seg")
clear_button = widgets.PushButton(text="Clear")
undo_button = widgets.PushButton(text="Undo")

last_seg_data = np.zeros((img_shape),dtype=np.uint8) 
last_label_data = np.zeros((img_shape),dtype=np.uint8)
current_label_data = np.zeros((img_shape),dtype=np.uint8)


def seg_func(label_mask: np.ndarray, feature_map: np.ndarray, dist_matrix=None, spatail_decay=True, stride=1) -> np.ndarray:
    label_mask = zoom(label_mask, zoom=(1/stride),order=0)
    print(f"label_mask.shape {label_mask.shape}")

    unique_labels = np.unique(label_mask)
    unique_labels = unique_labels[unique_labels != 0]  # ignore background (if 0)

    if len(unique_labels) < 2:
        return np.zeros(label_mask.shape, dtype=np.uint8)

    H, W, C = feature_map.shape
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
    seg_label = np.array([unique_labels[i] for i in predicted_classes])
    seg_label = seg_label.reshape(H, W)

    zoomed_seg_label = zoom(seg_label, zoom=stride,order=0)

    return zoomed_seg_label.astype(np.uint8)

# --- Seg button action ---
@seg_button.clicked.connect
def run_seg():
    label_data = label_layer.data.copy()
    
    global last_label_data, current_label_data,last_seg_data  # <-- declare them as global

    last_label_data = current_label_data
    current_label_data  = label_data
    last_seg_data = segout_layer.data.copy() 


    seg_result = seg_func(label_data, feature_map,dist_matrix=dist_matrix,spatail_decay=True,stride=stride)
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




# from magicgui.widgets import FunctionGui

# def test_func():
#     global last_label_data
#     print(f"in test_func,, last_label_data.shape {last_label_data.shape}")

# class MyGui(FunctionGui):
#     def __init__(self,func,button_name):
#         super().__init__(
#           func,
#           call_button=button_name,
#           layout='vertical',
#         )
# test_widg = MyGui(test_func,button_name='test1')

#using magicgui decorator will wrap the function in a FunctionGui object and turn it into a widget 
@magicgui(call_button='test2',layout='vertical')
def test_widg(viewer):
    global last_label_data
    print("called with viewer",viewer)
    print(f"in test_func,, last_label_data.shape {last_label_data.shape}")

print(type(test_widg))
#the parameter of test_widg is now turned into a sub-widget
test_widg.viewer.value = viewer
#alternative way
print(f"the viewer attribute type {type(test_widg['viewer'])}")
# test_widg['viewer'].value = viewer


# --- Combine buttons into a container widget ---
control_panel = Container(widgets=[seg_button, clear_button,undo_button])

# Add widget to napari
viewer.window.add_dock_widget(control_panel, area='right')
viewer.window.add_dock_widget(test_widg, area='right')

# viewer_widget = SimpleViewer(viewer=viewer)
# viewer.window.add_dock_widget(viewer_widget,area='right')
napari.run()
# %%
