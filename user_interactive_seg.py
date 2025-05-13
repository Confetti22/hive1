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
import matplotlib.pyplot as plt
from helper.image_reader import wrap_image


#%%


def load_mlp():
    ckpt_pth = "/home/confetti/e5_workspace/deepcluster4brain/runs/test14_8192_batch4096_nview2_pos_weight_2_shuffle_every50/model_epoch_34699.pth"
    device ='cuda'
    mlp = MLP().to(device)
    mlp.eval()
    print(f"begin loading ckpt")
    ckpt= torch.load(ckpt_pth)
    mlp.load_state_dict(ckpt)
    print(f"After loading ckpt")
    return mlp

def generate_feas_map(feats,img_shape,stride):
    mlp = load_mlp()
    feats = torch.from_numpy(feats).float().to('cuda')
    encoded = mlp(feats) #N*C
    encoded = encoded.detach().cpu().numpy()
    encoded = encoded.reshape(93,93,-1) # n*n*C

    # Only zoom the first two dimensions (height and width), not channels
    # zoom_factors = (stride, stride, 1)  # No zoom on channel dim
    # zoomed = zoom(encoded, zoom=zoom_factors, order=1)

    # Only pad the first two dimensions (height and width)
    # padded = np.pad(zoomed, pad_width=((24, 24), (24, 24), (0, 0)), mode='constant')

    return encoded 


def draw_pair_wise_cosine(X,N,title):
    # Normalize vectors
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Compute cosine similarity matrix
    cos_sim_matrix = X_norm @ X_norm.T  # Shape: (N, N)

    # Extract upper triangle (excluding diagonal) to avoid duplicate/self-similarity
    i_upper = np.triu_indices(N, k=1)
    pairwise_cosines = cos_sim_matrix[i_upper]

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(pairwise_cosines, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Cosine Sim_{title}")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


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



sample_num = 0
eval_data = get_eval_data(img_no_list=[1,2,3],ncc_seed_point=False)
data_dic = eval_data[sample_num]
feats=data_dic['feats']
z_slice = data_dic['z_slice']
z_slice = z_slice[24:-24,24:-24]
img_shape = z_slice.shape
stride = 16
feature_map = generate_feas_map(feats=feats,img_shape=img_shape,stride=stride)

viewer = napari.Viewer(ndisplay=2)
viewer.add_image(z_slice,name ='img')

label_data = np.zeros((img_shape),dtype=np.uint8)
label_layer = viewer.add_labels(label_data,name ='Label')
label_layer.brush_size = 30
label_layer.mode = 'PAINT'
print(f"img_shape type :{type(img_shape),img_shape}")

#current only suit for 2d
scaled_img_shape = tuple( int(x//stride)  for x in img_shape)
print(f"scaled_img_shaep :{type(scaled_img_shape), {scaled_img_shape}}")
loc_lst = list(np.ndindex(scaled_img_shape))
d_sigma = 8 
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

