#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
#%%

from lib.arch.ae import build_final_model,build_encoder_model,load_compose_encoder_dict
from config.load_config import load_cfg
from torchsummary import summary
device ='cuda'
print(f'{os.getcwd()}=')
args = load_cfg('config/vsi_ae_2d.yaml')
args.avg_pool_size = (5,5) 

cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)
cnn_ckpt_pth = '/home/confetti/data/weights/vsi_2d_ae_best.pth'
mlp_ckpt_pth = f'/home/confetti/e5_workspace/hive/contrastive_run_vsi/avg5_batch2048_nview2_pos_weight_2/model_final.pth'
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth)
print(cmpsd_model)
summary(cmpsd_model,(1,*args.input_size))


#%%
#%%
import tifffile as tif
import torch
from lib.utils.preprocess_img import pad_to_multiple_of_unit
import numpy as np
import matplotlib.pyplot as plt
img = tif.imread('/home/confetti/e5_data/wide_filed/nuclei_channel/063.tif')
# plt.imshow(img)
#%%
# img = img[2080:2080+1536,3180:3180+1536]
zoom_factor= 8

img = pad_to_multiple_of_unit(img,unit=zoom_factor) 
print(f"{img.shape= }")

input = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device) #add batch and channel dim B*C*H*W
mlp_out = cmpsd_model(input).cpu().detach().squeeze().numpy()
C,H,W = mlp_out.shape
from scipy.ndimage import gaussian_filter
blurred_image = gaussian_filter(img, sigma=8,mode='reflect')
print(f"after blurr: {blurred_image.shape= }")

# %%
import napari
import numpy as np
import matplotlib.pyplot as plt
from helper.one_dim_statis import OneDimStatis



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
# %%