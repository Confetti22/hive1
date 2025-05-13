#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
from lib.arch.ae import build_final_model,load_compose_encoder_dict,build_encoder_model,load_cnnencoder_dict
from config.load_config import load_cfg
from torchsummary import summary
from confettii.plot_helper import grid_plot_list_imgs
device ='cuda'
print(f'{os.getcwd()}=')
args = load_cfg('../config/t11_3d.yaml')
args.avg_pool_size = (8,8,8) 

cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)
cnn_ckpt_pth = '/home/confetti/data/weights/t11_3d_ae_best2.pth'
mlp_ckpt_pth ='/home/confetti/data/weights/t11_3d_mlp_best_new_format.pth'
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth,dims=args.dims)

encoder_model = build_encoder_model(args,dims=3) 
encoder_model.eval().to(device)
load_cnnencoder_dict(encoder_model,cnn_ckpt_pth)

#%%
import tifffile as tif
import torch
from lib.utils.preprocess_img import pad_to_multiple_of_unit
import numpy as np
import matplotlib.pyplot as plt
vol = tif.imread('/home/confetti/data/t1779/test_data_part_brain/0003.tif')
zoom_factor= 8
print(f"{vol.shape= }")
vol = pad_to_multiple_of_unit(vol,unit=zoom_factor) 
print(f"{vol.shape= }")

input = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(device) #add batch and channel dim B*C*H*W
mlp_out = cmpsd_model(input).cpu().detach().squeeze().numpy()
cnn_out = encoder_model(input).cpu().detach().squeeze().numpy()
print(f"{mlp_out.shape= }")
print(f"{cnn_out.shape= }")
C,D,H,W = mlp_out.shape


# %%
import napari
import numpy as np
import matplotlib.pyplot as plt
from helper.one_dim_statis import OneDimStatis
from scipy.ndimage import gaussian_filter


blurred_image = gaussian_filter(vol[32,], sigma=8,mode='reflect')
print(f"after blurr: {blurred_image.shape= }")

viewer = napari.Viewer()
img_layer = viewer.add_image(blurred_image,contrast_limits=[0,4000])

featsmap_dict = {}
featsmap_dict['gray'] = blurred_image[:,:,np.newaxis]
featsmap_dict['mlp'] = np.moveaxis(mlp_out,0,-1)


# Add widget to napari
# mlp_feats_onedims = OneDimStatis(viewer=viewer,image_layer=img_layer,feats_map= np.moveaxis(out,0,-1),zoom_factor=zoom_factor,metric='cos')
gray_value_onedims= OneDimStatis(viewer=viewer,image_layer=img_layer,featsmap_dict = featsmap_dict )
viewer.window.add_dock_widget(gray_value_onedims, area='right')

napari.run()

#%%
# from skimage import io, segmentation, color, measure
from skimage import io,  color,  exposure
from skimage.segmentation import slic, mark_boundaries
from skimage import graph
import numpy as np
import matplotlib.pyplot as plt

image = vol[32,:,:] 
print(f"Image dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
#%%
compactness=30 
n_segments=100 

# Normalize to [0,1] for processing
image = exposure.rescale_intensity(image, in_range='image', out_range=(0, 1))
print(f"Image dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

# Step 2: Gaussian blur (still in float)
blurred = gaussian_filter(image, sigma=12)
blurred = image

# Step 3: Superpixel segmentation (SLIC expects RGB-like input)
image_rgb = np.stack([blurred]*3, axis=-1)
segments = slic(image_rgb, n_segments=200, compactness=20, start_label=1)

# Step 4: Normalized graph cut on superpixels
g = graph.rag_mean_color(image_rgb, segments,mode='similarity',sigma=0.001, connectivity=2)
labels = graph.cut_normalized(segments, g,)

# Display results
grid_plot_list_imgs(
    images=[image,
            blurred,
            mark_boundaries(image_rgb, segments),
            color.label2rgb(labels, image_rgb, kind='avg')],
            ncols=2,
            fig_size=5)

# %%
from scipy.ndimage import zoom
from skimage import graph
import numpy as np

mlp_feats_map = mlp_out[:,3,:,:]
mlp_feats_map = np.moveaxis(mlp_feats_map, 0,-1)
zoom_factor = (raw/feat for raw , feat in zip (image.shape, mlp_feats_map.shape[:-1]))
mlp_feats_map = zoom(mlp_feats_map,zoom=(*zoom_factor,1),order=1) 
print(mlp_feats_map.shape)

cnn_feats_map = cnn_out[:,3,:,:]
cnn_feats_map = np.moveaxis(cnn_feats_map, 0,-1)
zoom_factor = (raw/feat for raw , feat in zip (image.shape, cnn_feats_map.shape[:-1]))
cnn_feats_map = zoom(cnn_feats_map,zoom=(*zoom_factor,1),order=1) 
print(cnn_feats_map.shape)

import numpy as np
from skimage.graph import RAG
import math

def rag_mean_feature(image, labels, connectivity=2, mode='similarity', sigma=0.001):
    # Initialize RAG
    rag = RAG(labels, connectivity=connectivity)

    h, w, c = image.shape  # C is number of feature channels
    for n in rag:
        rag.nodes[n].update({
            'labels': [n],
            'pixel count': 0,
            'total feature': np.zeros((c,), dtype=np.float64),
        })

    # Accumulate feature sums and counts
    for index in np.ndindex(labels.shape):
        current = labels[index]
        rag.nodes[current]['pixel count'] += 1
        rag.nodes[current]['total feature'] += image[index]

    # Compute mean feature vector per region
    for n in rag:
        rag.nodes[n]['mean feature'] = (
            rag.nodes[n]['total feature'] / rag.nodes[n]['pixel count']
        )

    # Compute weights based on feature vector distance
    for x, y, d in rag.edges(data=True):
        diff = rag.nodes[x]['mean feature'] - rag.nodes[y]['mean feature']
        diff_norm = np.linalg.norm(diff)  # Euclidean distance
        if mode == 'similarity':
            d['weight'] = math.e ** (-(diff_norm**2) / sigma)
        elif mode == 'distance':
            d['weight'] = diff_norm
        else:
            raise ValueError(f"The mode '{mode}' is not recognised")

    return rag

#%%
mlp_feats_map = (mlp_feats_map - np.min(mlp_feats_map))/(np.max(mlp_feats_map) - np.min(mlp_feats_map))
print(f"{np.min(mlp_feats_map)},{np.max(mlp_feats_map)}")
segments = slic(mlp_feats_map, n_segments=200, compactness=0.1, start_label=1,channel_axis=-1)

rag = rag_mean_feature(mlp_feats_map, segments, mode='similarity', sigma=0.01)

labels = graph.cut_normalized(segments, rag)

from sklearn.decomposition import PCA

# Reshape to (H*W, 12)
h, w, c = mlp_feats_map.shape
flat = mlp_feats_map.reshape(-1, c)
# PCA to 3 components
pca = PCA(n_components=3)
rgb_vis = pca.fit_transform(flat).reshape(h, w, 3)
# Normalize for display
rgb_vis = (rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min())

# Display results
grid_plot_list_imgs(
    images=[
        rgb_vis,
        mark_boundaries(rgb_vis, segments),
        color.label2rgb(labels, rgb_vis, kind='avg'),
        ],
        col_labels=['pca_feats','superpixel','n_cut'],
    ncols=3,
    fig_size=5)

# %%

# %%
