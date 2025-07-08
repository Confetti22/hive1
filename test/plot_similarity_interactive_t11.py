#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
#%%
from lib.arch.ae import build_final_model,load_compose_encoder_dict,build_encoder_model,load_encoder2encoder
from config.load_config import load_cfg
from torchsummary import summary
from confettii.plot_helper import grid_plot_list_imgs
import time
device ='cuda'
print(f'{os.getcwd()}=')
args = load_cfg('config/rm009.yaml')

avg_pool =8 
args.avg_pool_size = [avg_pool]*3
args.avg_pool_padding =  False

# the old result on rm009 did not load the correct encoder feats
cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)
# cnn_ckpt_pth = '/home/confetti/e5_workspace/hive/rm009_ae_out/weights/test_rm009/Epoch_1451.pth'
cnn_ckpt_pth = '/home/confetti/data/weights/rm009_3d_ae_best.pth'
mlp_ckpt_pth ='/home/confetti/e5_workspace/hive/contrastive_run_rm009/batch4096_nview2_pos_weight_2_mlp[96, 48, 24, 12]_d_near1/model_epoch_2349.pth'
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth,dims=args.dims)

encoder_model = build_encoder_model(args,dims=3) 
encoder_model.eval().to(device)
load_encoder2encoder(encoder_model,cnn_ckpt_pth)


#%%
import tifffile as tif
import torch
from lib.utils.preprocess_img import pad_to_multiple_of_unit
import numpy as np
import matplotlib.pyplot as plt
vol = tif.imread('/home/confetti/data/rm009/seg_valid/_0001.tif')
zoom_factor= 8
print(f"{vol.shape= }")
# vol = pad_to_multiple_of_unit(vol,unit=zoom_factor) 
print(f"after padding: {vol.shape= }")
mask = tif.imread('/home/confetti/data/rm009/seg_valid/_0001_human_mask_3d.tif')

vol = vol[:,432:1100,206:1329]
mask = mask[:,432:1100,206:1329]

mask_slice = mask[32]

batch_size = 512 

#%%
import torch
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from torchvision import transforms
from PIL import Image
from confettii.feat_extract import get_feature_list 

device = 'cuda'
weights = Inception_V3_Weights.DEFAULT
incep_model = models.inception_v3(weights = weights, progress =True)
incep_model.eval()
incep_model.to(device)


# Load and preprocess an image
img = vol[32]
normalized_img = ((img/ img.max()) * 255).astype(np.uint8)
rgb_img = np.stack([normalized_img] * 3, axis=-1)
print(f"{rgb_img.shape=}")

#%%

from confettii.feat_extract import get_feature_list, TraverseDataset2d
from confettii.plot_helper import three_pca_as_rgb_image
from torch.utils.data import DataLoader

#get feats_map via input the whole image 
# !(not feasible, will reduce spatial 30 times!, 1*3*1536*1536 -> 1*2048*46*46
# rgb_input = np.moveaxis(rgb_img,-1,0)
# rgb_input = rgb_input[np.newaxis,:]
# print(f"{rgb_input.shape= }")

# activation = {}
# def getActivation(name):
#     # the hook signature
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook
# extract_layer_name = 'mixed7c'
# incep_model.Mixed_7c.register_forward_hook(getActivation(extract_layer_name))
# _ = incep_model(torch.from_numpy(rgb_input).float().to(device))
# feats_map = activation[extract_layer_name].cpu().detach().numpy()
# print(f"{feats_map.shape= }")

# get feats_map via traverse on small roi
dataset = TraverseDataset2d(rgb_img,stride=8,win_size=128)
out_shape = dataset._get_sample_shape()
print(out_shape)
loader = DataLoader(dataset,batch_size= batch_size,shuffle=None,drop_last=False) 
current = time.time()
extract_layer_name ='avgpool'
feats_list = get_feature_list(device,incep_model,loader,extract_layer_name=extract_layer_name)
print(f"extracting feats: {time.time()-current:.3f}")
incept_feats_map = feats_list.reshape((*out_shape,-1))
print(f"{incept_feats_map.shape= }, {out_shape= }")


#%%

input = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(device) #add batch and channel dim B*C*H*W
mlp_out = cmpsd_model(input).cpu().detach().squeeze().numpy()
cnn_out = encoder_model(input).cpu().detach().squeeze().numpy()
print(f"{mlp_out.shape= }")
print(f"{cnn_out.shape= }")
# C,D,H,W = mlp_out.shape


# %%
import napari
import numpy as np
import matplotlib.pyplot as plt
from helper.one_dim_statis import OneDimStatis
from scipy.ndimage import gaussian_filter
from confettii.plot_helper import get_smooth_contours, get_boundary_via_erosion

smoothed_contours = get_smooth_contours(mask_slice,min_contour_point_num=400)

viewer = napari.Viewer()
label_layer = viewer.add_labels(mask_slice,name='gt')
img_layer = viewer.add_image(vol[32],contrast_limits=[0,3000])
# Add smoothed contours as paths
viewer.add_shapes(
    smoothed_contours,
    shape_type='path',  # use 'polygon' if contours are closed loops
    edge_color='red',
    edge_width=1,
    name='Contours'
)

blurred_image = gaussian_filter(vol[32,], sigma=12,mode='reflect')
middle_mlp_feat = np.moveaxis(mlp_out,0,-1)
middle_ae_feat = np.moveaxis(cnn_out,0,-1)

featsmap_dict = {}
featsmap_dict['gray'] = blurred_image[:,:,np.newaxis]
featsmap_dict['mlp'] = middle_mlp_feat 
featsmap_dict['ae'] = middle_ae_feat 
featsmap_dict['incept']= incept_feats_map


# Add widget to napari
# mlp_feats_onedims = OneDimStatis(viewer=viewer,image_layer=img_layer,feats_map= np.moveaxis(out,0,-1),zoom_factor=zoom_factor,metric='cos')
gray_value_onedims= OneDimStatis(viewer=viewer,image_layer=img_layer,featsmap_dict = featsmap_dict )
viewer.window.add_dock_widget(gray_value_onedims, area='right')

napari.run()
