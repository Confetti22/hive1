#%%
from graph_cut_helper import GraphCutFastBFS
import torch
import torch.nn.functional as F
import numpy as np
import napari
from magicgui import magicgui,widgets
from magicgui.widgets import Container
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import zoom

from lib.arch.ae import build_final_model,load_compose_encoder_dict,build_encoder_model,load_encoder2encoder
from config.load_config import load_cfg

device ='cuda'
args = load_cfg('config/t11_3d.yaml')
args.avg_pool_size = (8,8,8) 

cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)
cnn_ckpt_pth = '/home/confetti/data/weights/t11_3d_ae_best2.pth'
mlp_ckpt_pth ='/home/confetti/data/weights/t11_3d_mlp_best_new_format.pth'
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth,dims=args.dims)

encoder_model = build_encoder_model(args,dims=3) 
encoder_model.eval().to(device)
load_encoder2encoder(encoder_model,cnn_ckpt_pth)

import tifffile as tif
import torch
from lib.utils.preprocess_img import pad_to_multiple_of_unit
from helper.image_reader import Ims_Image
import numpy as np
import matplotlib.pyplot as plt

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
feats_map = np.moveaxis(feats_map,0,-1) # h,w,C
feats_map_shape = feats_map.shape[:2]

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
rgb_vis = pca.fit_transform(feats_map.reshape(-1,C)).reshape(H,W, 3)

z_slice = vol[32]
z_slcie_shape = z_slice.shape


#%%

def simple_seg_func(label_mask: np.ndarray, feature_map: np.ndarray, dist_matrix=None, spatail_decay=True,) -> np.ndarray:
    zoom_factors = [ x/y for x, y in zip(feats_map_shape,z_slcie_shape)]
    label_mask = zoom(label_mask, zoom= zoom_factors,order=0)
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
    result = np.array([unique_labels[i] for i in predicted_classes])
    result = result.reshape(H, W)

    zoom_factors = [  y/x for x, y in zip(feats_map_shape,z_slcie_shape)]
    zoomed_seg_label = zoom(result, zoom=zoom_factors,order=0)

    return zoomed_seg_label.astype(np.uint8)


def graph_cutseg_func(label_mask: np.ndarray, feats_map: np.ndarray) -> np.ndarray:
    zoom_factors = [ x/y for x, y in zip(feats_map_shape,z_slcie_shape)]
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


    zoom_factors = [  y/x for x, y in zip(feats_map_shape,z_slcie_shape)]
    zoomed_seg_label = zoom(result, zoom=zoom_factors,order=0)

    return zoomed_seg_label.astype(np.uint8)
import torch
import torch.nn as nn
from tqdm.auto import tqdm

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


def seg_head_seg_func(label_mask: np.ndarray, feats_map: np.ndarray,lr=0.001,num_epochs= 2000, return_prob = False) -> np.ndarray:
    zoom_factors = [ x/y for x, y in zip(feats_map_shape,z_slcie_shape)]
    label_mask = zoom(label_mask, zoom= zoom_factors,order=0)
    print(f"label_mask.shape {label_mask.shape}")
    

    unique_labels = np.unique(label_mask)
    unique_labels = unique_labels[unique_labels != 0]  # ignore background (if 0)

    if len(unique_labels) < 2:
        return np.zeros(label_mask.shape, dtype=np.uint8)
    device = 'cuda'
    H, W, C = feats_map.shape
    feats_map = torch.from_numpy(feats_map)
    label_mask = torch.from_numpy(label_mask)
    # Get labeled coordinates and labels (ignore 0)
    coords = torch.nonzero(label_mask > 0, as_tuple=False)  # [N, 2]
    labels = label_mask[coords[:, 0], coords[:, 1]] - 1  # Convert to 0-based class index

    num_classes = labels.max().item() + 1
    if num_classes < 2:
        raise ValueError("Need at least 2 labeled classes in the mask.")

    y, x =  coords[:, 0], coords[:, 1]
    prompt_features = feats_map[y, x]  # [N, C]

    head = SegmentationHead(C, num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    prompt_features =prompt_features.to(device)
    prompt_labels = labels.to(device)

    for epoch in tqdm(range(num_epochs)):
        head.train()

        optimizer.zero_grad()
        logits = head(prompt_features)
        loss = loss_fn(logits, prompt_labels)
        loss.backward()
        optimizer.step()
        print(f"loss:{loss.item()}")

    # Predict over full volume
    flat_features = feats_map.reshape(-1, C).to(device)
    with torch.no_grad():
        head.eval()
        logits = head(flat_features)  # [D*H*W, K]
        probs = F.softmax(logits, dim=1)  # [D*H*W, K]

    zoom_factors = [  y/x for x, y in zip(feats_map_shape,z_slcie_shape)]

    if return_prob:
        prob_map = probs.reshape(H, W, num_classes).permute(2, 0, 1)  # [K, D, H, W]
        prob_map = prob_map.detach().cpu().numpy()
        zoomed_seg_prob = zoom(prob_map, zoom=zoom_factors,order=0)
        return zoomed_seg_prob
    else:
        pred_mask = torch.argmax(probs, dim=1).reshape( H, W) + 1  # Convert back to 1-based labels
        pred_mask = pred_mask.detach().cpu().numpy()
        zoomed_seg_label = zoom(pred_mask, zoom=zoom_factors,order=0)
        return zoomed_seg_label.astype(np.uint8)
        








viewer = napari.Viewer(ndisplay=2)
viewer.add_image(z_slice,name ='img')


label_data = np.zeros((z_slcie_shape),dtype=np.uint8)
label_data = tif.imread('/home/confetti/data/t1779/test_data_part_brain/0001_user_input.tif')
label_data = label_data.astype(int)
label_layer = viewer.add_labels(label_data,name ='Label')
label_layer.brush_size = 30
label_layer.mode = 'PAINT'

segout_layer = viewer.add_labels(label_data,name = 'Segout')
viewer.layers.selection = [label_layer]  # Keep selected

# --- Define separate buttons ---
seg_button = widgets.PushButton(text="Seg")
method_button = widgets.ComboBox(value='seg_head',choices=['graphcut','seg_head','naiveclf'])
pcafeats_button = widgets.ComboBox(value='pca',choices=['pca','non-pca'])
cut_sigma_slider =widgets.ComboBox(label='sigma',value=0.01, choices=[ 0.001, 0.005,0.01,0.05])
cut_lambda_slider =widgets.ComboBox(label='labmda',value=0.01, choices=[ 0.01, 0.1, 0.5,1, 1.5,2,10,]) 
clear_button = widgets.PushButton(text="Clear")
undo_button = widgets.PushButton(text="Undo")

last_seg_data = np.zeros((z_slcie_shape),dtype=np.uint8) 
last_label_data = np.zeros((z_slcie_shape),dtype=np.uint8)
current_label_data = np.zeros((z_slcie_shape),dtype=np.uint8)

d_sigma = 8 
loc_lst = list(np.ndindex(feats_map_shape))
dist = pdist(loc_lst, metric='euclidean')
dist_matrix = squareform(dist)
print(f"distacne_matrix: {dist_matrix.shape}")
dist_matrix = np.exp(- dist_matrix**2/(2*d_sigma**2))

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
    print(f"begin cutting")
    if mode =="graphcut":
        seg_result = graph_cutseg_func(label_data, feats)
    elif mode =='seg_head':
        seg_result = seg_head_seg_func(label_data, feats)
    else:
        seg_result = simple_seg_func(label_data, feats,dist_matrix,True)
    print(f"finished cutting")
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
control_panel = Container(widgets=[method_button,pcafeats_button,cut_lambda_slider,cut_sigma_slider,seg_button, clear_button,undo_button])

# Add widget to napari
viewer.window.add_dock_widget(control_panel, area='right')
viewer.window.add_dock_widget(test_widg, area='right')

# viewer_widget = SimpleViewer(viewer=viewer)
# viewer.window.add_dock_widget(viewer_widget,area='right')
napari.run()

