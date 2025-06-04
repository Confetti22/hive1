
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

os.environ["NAPARI_ASYNC"] = "1"

from lib.arch.ae import build_final_model,load_compose_encoder_dict,build_encoder_model,load_encoder2encoder
from config.load_config import load_cfg
import numpy as np
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from confettii.plot_helper import three_pca_as_rgb_image
from scipy.ndimage import zoom

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

#prepare the pretrained inception_v3 model
weights = Inception_V3_Weights.DEFAULT
incep_model = models.inception_v3(weights = weights, progress =True)
incep_model.eval()
incep_model.to(device)

model_dict={}
model_dict['mlp']=cmpsd_model
model_dict['ae']  = encoder_model
model_dict['inception']=incep_model

import napari
from helper.image_reader import Ims_Image
import time
import gc
from confettii.feat_extract import get_feature_list,get_feature_map,TraverseDataset3d,TraverseDataset3d_overlap

from torch.utils.data import DataLoader
ims_vol = Ims_Image("/home/confetti/e5_data/t1779/t1779.ims",channel=2)
offset = [7000,3250,5176]
roi_size = [64,2048,2048]
vol = ims_vol.from_roi(coords=(*offset,*roi_size),level=0)
viewer = napari.Viewer()
image_layer = viewer.add_image(vol,contrast_limits=[0,4000])

D,H,W = image_layer.data.shape
z = viewer.dims.current_step[0]

print(f"begin prepare for mlp")
mlp_model = model_dict['mlp']
current = time.time()

z_dim = D 
if z_dim >= 64:
    z_center = z_dim // 2
    mlp_roi = image_layer.data[z_center - 32 : z_center + 32, :, :]
else:
    raise ValueError(f"Z-dimension is too small: {z_dim}. Must be at least 64.")
print(f"read roi data :{time.time()-current:.3f}")

#%% extract feats_map
# current = time.time()
# overlap =0
# dataset = TraverseDataset3d_overlap(mlp_roi,overlap=overlap,win_size=(64,1024,1024),verbose=True) 
# roi_nums = dataset.get_roi_nums()
# print(f"{roi_nums= }")
# loader = DataLoader(dataset,batch_size=1,shuffle=None,drop_last=False) 
# #shpae of feats_map should be (h,w,c)
# mlp_feats_map = get_feature_map('cuda',mlp_model,loader=loader,overlap_i=overlap,roi_nums=roi_nums) 
# print(f"extract_feats:{time.time()-current:.3f}")
# print(f"end prepare for mlp")
# feats_map = mlp_feats_map
# print(f"{feats_map.shape = }")


#%% extract via travers samlle roi
current = time.time()
dataset = TraverseDataset3d(mlp_roi,stride=16,win_size=(64,64,64),verbose=True) 
roi_nums = dataset.get_sample_shape()
print(f"{roi_nums= }")
loader = DataLoader(dataset,batch_size=512,shuffle=None,drop_last=False) 
#shpae of feats_map should be (h,w,c)
mlp_feats_list = get_feature_list('cuda',mlp_model,loader) 
print(f"extract_feats:{time.time()-current:.3f}")
print(f"end prepare for mlp")
feats_map = mlp_feats_list.reshape(roi_nums[1],roi_nums[2],-1)
print(f"{feats_map.shape = }")


del mlp_roi, dataset, loader, mlp_model
gc.collect()

h_f, w_f,c_f = feats_map.shape
current = time.time()
rgb_feats_map = three_pca_as_rgb_image(feats_map.reshape(-1,c_f),final_image_shape=(h_f,w_f))
zoomed_rgb_feats_map = zoom(rgb_feats_map,(H/h_f,W/w_f,1),order=1)

vsi_feats_layer = viewer.add_image(zoomed_rgb_feats_map,name='feats_3pca',rgb=True,opacity=0.6)
napari.run()