#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

os.environ["NAPARI_ASYNC"] = "1"

from lib.arch.ae import build_final_model,load_compose_encoder_dict, build_encoder_model,load_encoder2encoder
from config.load_config import load_cfg
import numpy as np
import torchvision.models as models
from torchvision.models import Inception_V3_Weights


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
model_dict['mlp'] = cmpsd_model
model_dict['ae']  = encoder_model
model_dict['inception']=incep_model





# %%
import napari
import numpy as np
import matplotlib.pyplot as plt
from helper.one_dim_statis import OneDimStatis_dask_array
from dask_image.imread import imread

#read whole slice from stack of tif, utilizing lazing loading
stack = imread("/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Mouse_Brain/20210131_ZSS_USTC_THY1-YFP_1779_1/Reconstruction_1.0/Reconstruction/BrainImage/1.0/*C3.tif")
viewer = napari.Viewer()
img_layer = viewer.add_image(stack,contrast_limits=[0,4000],multiscale=False)
vsi_feats_layer = viewer.add_image(np.zeros(shape=(3,3,3)),name='feats_3pca',rgb=True,opacity=0.6)
viewer.layers.selection = [img_layer]


statis_widget= OneDimStatis_dask_array(viewer=viewer,image_layer=img_layer,vsi_feats_layer=vsi_feats_layer,model_dict=model_dict)
viewer.window.add_dock_widget(statis_widget, area='right')

napari.run()
