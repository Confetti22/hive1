#%%
from helper.graph_cut_helper import GraphCutFastBFS
import torch
import numpy as np
import napari
from magicgui import widgets
from magicgui.widgets import Container
from scipy.ndimage import zoom
import time

from lib.arch.ae import build_final_model,load_compose_encoder_dict,build_encoder_model,load_encoder2encoder
from config.load_config import load_cfg
from helper.image_seger import _compute_seg2,_seg_via_conv_head, _seg_via_mlp_head
import tifffile as tif
import torch
from lib.utils.preprocess_img import pad_to_multiple_of_unit
from helper.image_reader import Ims_Image
import numpy as np

#define model 
device ='cuda'
args = load_cfg('config/t11_3d.yaml')
args.avg_pool_size = (8,8,8) 

cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)

#load trained weights
cnn_ckpt_pth = '/home/confetti/data/weights/t11_3d_ae_best2.pth'
mlp_ckpt_pth ='/home/confetti/data/weights/t11_3d_mlp_best_new_format.pth'
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth,dims=args.dims)

#optinal encoder model
encoder_model = build_encoder_model(args,dims=3) 
encoder_model.eval().to(device)
load_encoder2encoder(encoder_model,cnn_ckpt_pth)



#prepare data
ims_vol =Ims_Image('/home/confetti/e5_data/t1779/t1779.ims',channel=2)
roi_offset =[6980,3425,4040]
roi_size =[64,1536,1536]
vol = ims_vol.from_roi(coords=[*roi_offset,*roi_size],level=0)

vol = tif.imread('/home/confetti/data/t1779/test_data_part_brain/0001.tif')
zoom_factor= 8
print(f"{vol.shape= }")
vol = pad_to_multiple_of_unit(vol,unit=zoom_factor) 
print(f"{vol.shape= }")

input = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(device) #add batch and channel dim B*C*H*W
mlp_out = cmpsd_model(input).cpu().detach().squeeze().numpy()
print(f"{mlp_out.shape= }")
C,H,W = mlp_out.shape


feats_map = mlp_out[:,:,:]
feats_map = np.moveaxis(feats_map,0,-1) #  feats_map shape h,w,C
feats_map_shape = feats_map.shape[:2]

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
rgb_vis = pca.fit_transform(feats_map.reshape(-1,C)).reshape(H,W, 3)

z_slice = vol[32]
input_label_shape = z_slice.shape


#%%

def simple_seg_func(label_mask: np.ndarray, feature_map: np.ndarray,  spatial_decay=True) -> np.ndarray:
    "did not use distance matrix"
    feats_map_shape = feats_map.shape[:-1]  # (D, H, W, C) or (H, W, C)
    input_label_shape = label_mask.shape        # (D, H, W) or (H, W)

    #downscale label_mask into the spatial_shape of feature_map
    zoom_factors = [ x/y for x, y in zip(feats_map_shape,input_label_shape)]
    label_mask = zoom(label_mask, zoom= zoom_factors,order=0)
    print(f"label_mask.shape {label_mask.shape}")
    
    feature_map=np.expand_dims(feature_map,axis=0)
    label_mask = np.expand_dims(label_mask,axis=0)
    result = _compute_seg2(label_mask,feature_map,spatial_decay)
    result = np.squeeze(result)

    #upscale the result into the label/roi spatial_shape
    zoom_factors = [  y/x for x, y in zip(feats_map_shape,input_label_shape)]
    zoomed_seg_label = zoom(result, zoom=zoom_factors,order=0)

    return zoomed_seg_label.astype(np.uint8)

def graph_cutseg_func(label_mask: np.ndarray, feats_map: np.ndarray) -> np.ndarray:
    feats_map_shape = feats_map.shape[:-1]  # (D, H, W, C) or (H, W, C)
    input_label_shape = label_mask.shape        # (D, H, W) or (H, W)

    zoom_factors = [ x/y for x, y in zip(feats_map_shape,input_label_shape)]
    label_mask = zoom(label_mask, zoom= zoom_factors,order=0)
    print(f"label_mask.shape {label_mask.shape}")
    
    unique_labels = np.unique(label_mask)
    unique_labels = unique_labels[unique_labels != 0]  # ignore background (if 0)

    if len(unique_labels) < 2:
        return np.zeros(label_mask.shape, dtype=np.uint8)

    cut_sigma = cut_sigma_slider.value 
    cut_lambda = cut_lambda_slider.value 
    graph_cut = GraphCutFastBFS(feats_map, label_mask, cut_sigma, cut_lambda)
    graph_cut.start_cut()
    result = graph_cut.TREE  # or graph_cut.output_array()

    zoom_factors = [  y/x for x, y in zip(feats_map_shape,input_label_shape)]
    zoomed_seg_label = zoom(result, zoom=zoom_factors,order=0)

    return zoomed_seg_label.astype(np.uint8)


def seg_head_seg_func(label_mask: np.ndarray, feats_map: np.ndarray,lr=0.001,num_epochs= 2000, return_prob = False,mode='mlp') -> np.ndarray:
    feats_map_shape = feats_map.shape[:-1]  # (D, H, W, C) or (H, W, C)
    input_label_shape = label_mask.shape        # (D, H, W) or (H, W)

    zoom_factors = [ x/y for x, y in zip(feats_map_shape,input_label_shape)]
    label_mask = zoom(label_mask, zoom= zoom_factors,order=0)

    if mode == 'mlp':
        probs_map = _seg_via_mlp_head(label_mask,feats_map,num_epochs=num_epochs,return_prob=True)
    else:
        probs_map = _seg_via_conv_head(label_mask,feats_map,num_epochs=num_epochs,return_prob=True)

    zoom_factors = [  y/x for x, y in zip(feats_map_shape,input_label_shape)]

    zoomed_seg_prob = zoom(probs_map, zoom=zoom_factors,order=0)
    if return_prob:
        return zoomed_seg_prob
    else:
        pred_mask = np.argmax(zoomed_seg_prob, dim=0) + 1  # Convert back to 1-based labels
        return pred_mask.astype(np.uint8)
        

viewer = napari.Viewer(ndisplay=2)
viewer.add_image(z_slice,name ='img')


label_data = np.zeros((input_label_shape),dtype=np.uint8)
label_data = tif.imread('/home/confetti/data/t1779/test_data_part_brain/0001_user_input.tif')
label_data = label_data.astype(int)
label_layer = viewer.add_labels(label_data,name ='Label')
label_layer.brush_size = 30
label_layer.mode = 'PAINT'

segout_layer = viewer.add_labels(label_data,name = 'Segout')
viewer.layers.selection = [label_layer]  # Keep selected



# --- Define separate buttons ---
seg_button = widgets.PushButton(text="Seg")
method_button = widgets.ComboBox(value='sim2',choices=['graphcut','mlp_seg','conv_seg','similarity'])
pcafeats_button = widgets.ComboBox(value='non-pca',choices=['pca','non-pca'])
cut_sigma_slider =widgets.ComboBox(label='sigma',value=0.01, choices=[ 0.001, 0.005,0.01,0.05])
cut_lambda_slider =widgets.ComboBox(label='labmda',value=0.01, choices=[ 0.01, 0.1, 0.5,1, 1.5,2,10,]) 
clear_button = widgets.PushButton(text="Clear")
undo_button = widgets.PushButton(text="Undo")



last_seg_data = np.zeros((input_label_shape),dtype=np.uint8) 
last_label_data = np.zeros((input_label_shape),dtype=np.uint8)
current_label_data = np.zeros((input_label_shape),dtype=np.uint8)


# --- Seg button action ---
@seg_button.clicked.connect
def run_seg():
    label_data = label_layer.data.copy()
    
    global last_label_data, current_label_data,last_seg_data  # <-- declare them as global

    last_label_data = current_label_data
    current_label_data  = label_data
    last_seg_data = segout_layer.data.copy() 
    mode = method_button.value
    feats_map_mode = pcafeats_button.value
    if feats_map_mode =='pca':
        feats = rgb_vis
    else:
        feats = feats_map
    current = time.time()
    print(f"begin cutting")
    if mode =="graphcut":
        seg_result = graph_cutseg_func(label_data, feats)
    elif mode =='mlp_seg':
        seg_result = seg_head_seg_func(label_data, feats,mode='mlp')
    elif mode =='conv_seg':
        seg_result = seg_head_seg_func(label_data, feats,mode='conv')
    elif mode =='similariy':
        seg_result = simple_seg_func(label_data, feats,True)
    else:
        print('wrong seg method mode')

    print(f"finished cutting: {time.time()-current:.3f}")
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



control_panel = Container(widgets=[method_button,pcafeats_button,cut_lambda_slider,cut_sigma_slider,seg_button, clear_button,undo_button])

viewer.window.add_dock_widget(control_panel, area='right')
napari.run()

