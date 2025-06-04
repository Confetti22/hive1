#%%
import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

import torch
from lib.arch.ae import modify_key, delete_key
from config.load_config import load_cfg
from lib.arch.ae import build_final_model,build_encoder_model,load_compose_encoder_dict, load_encoder2encoder
from torchsummary import summary
import tifffile as tif
from confettii.plot_helper import kmeans_grid_results

import matplotlib.pyplot as plt
import numpy as np

device ='cuda'
args = load_cfg('../config/t11_3d.yaml')
# avg_pool is applied after three cnn layer
print(f"{args.avg_pool_size= }")
cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)
print(cmpsd_model)
summary(cmpsd_model,(1,*args.input_size))
#%%


cnn_ckpt_pth = '/home/confetti/data/weights/t11_3d_ae_best.pth'
mlp_ckpt_pth ='/home/confetti/data/weights/t11_3d_mlp_best_new_format.pth'
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth,dims=3)


vol = tif.imread('/home/confetti/data/t1779/test_data_part_brain/0003.tif')
vol = vol[:16,:512,:512]
print(f"{vol.shape= }")

input = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(device) #add batch and channel dim B*C*H*W
mlp_out = cmpsd_model(input).cpu().detach().squeeze().numpy()
C,H,W= mlp_out.shape
print(f"{mlp_out.shape= }") 

plt.imshow(mlp_out.max(axis = 0))
mlp_encoded = np.moveaxis(mlp_out,0,-1).reshape(-1,C)
#%%
from sklearn.cluster import KMeans
# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(mlp_encoded)
# Display label image
plt.imshow(labels.reshape(H,W), cmap='tab20')

# %%

args.avg_pool_size = None
encoder_model = build_encoder_model(args,dims=3) 
encoder_model.eval().to(device)
load_encoder2encoder(encoder_model,cnn_ckpt_pth)
cnn_out = encoder_model(input).cpu().detach().squeeze().numpy()
print(cnn_out.shape)
cnn_z0_slice = cnn_out[:,4,:,:]

cnn_z0_slice = cnn_out[:,4,:,:]

C = cnn_z0_slice.shape[0]
print(cnn_z0_slice.shape)
plt.imshow(cnn_z0_slice.max(axis = 0))

cnn_encoded = np.moveaxis(cnn_z0_slice,0,-1).reshape(-1,C)
kmeans_grid_results(cnn_encoded,img_shape=(192,192),K_values=[4,8])

# %%
