#%%
import torch.nn as nn
import torch

from extrac_feats_helper import Encoder ,TraverseDataset3d,get_feature_list
import torch.nn as nn
from extrac_feats_helper import load_cfg
from torchsummary import summary
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

from image_reader import wrap_image, Ims_Image
level = 0
stride = 16
ims_vol = Ims_Image("/home/confetti/e5_data/t1779/t1779.ims",channel=2)
raw_volume_size =ims_vol.rois[level][3:] #the data shape at r3 for test
print(f"raw_volume_size{raw_volume_size}")
whole_volume_size = [int(element//2) for element in raw_volume_size]
whole_volume_offset = [int(element//4) for element in raw_volume_size]
valid_offset = [ int(x + int((3/2) *stride)) for x in whole_volume_offset]
valid_size = [ int(x - int((3/2) *stride)) for x in whole_volume_size]
lb = valid_offset
hb = [ x+ y for x, y in zip(valid_offset,valid_size)] 


roi_offset = [7075,5600,4850]
roi_size =[1075,825,900]
raw_vol = ims_vol.from_roi(coords=[*roi_offset,*roi_size],level=0)
z_slice = raw_vol[461,:,:]

slice_start_idx_2d = [ stride - (offset - lb)%stride for  offset,lb in zip(roi_offset[1:],lb[1:] )]

raw_shape = z_slice.shape
lyp = slice_start_idx_2d[0] 
ret =( raw_shape[0] -  slice_start_idx_2d[0] ) % stride 
hyp = ret if  ret else stride

lxp = slice_start_idx_2d[1] 
ret =( raw_shape[1] -  slice_start_idx_2d[1] ) % stride 
hxp = ret if  ret else stride



#%%
print(f"slice_start_idx :{slice_start_idx_2d}, hyp: {hyp}, hxp:{hxp}")
# cropped_img_vol =raw_vol[461-32:461+32, slice_start_idx_2d[0]:-hyp,slice_start_idx_2d[1]:-hxp]
# print(f"cropped_vol:{cropped_img_vol.shape}")
cropped_img_vol = ims_vol.from_roi(coords=[7075 + 461 -32 ,  
                                           5600 -24 +slice_start_idx_2d[0]  ,  
                                           4850-24 +slice_start_idx_2d[1] ,  
                                            64, 
                                           800+48 ,
                                           880 +48
                                           ],level=0)

print(f"cropped_vol:{cropped_img_vol.shape}")
#%%
#fist crop a image with z_size =64
feats_lst,outs_shape = extract_feats(cropped_img_vol,win_size=64,cnn=cnn_model,mlp=mlp)

#%%
print(outs_shape)
print(feats_lst.shape)
#%%

def cos_theta_plot(idx, encoded, img_shape, C =12):
    att = encoded @ encoded.reshape(-1,C)[idx]
    img = att.reshape(img_shape) 
    return img

feats_slice = feats_lst.reshape(outs_shape[1],outs_shape[2],-1)
print(feats_slice.shape)
#%%

H,W,C = feats_slice.shape
ncc_feats = cos_theta_plot(idx=int(H*W//2),encoded=feats_slice,img_shape=(H,W))
zoomed_labels = np.kron(ncc_feats,np.ones((stride,stride)))

padded_labels =np.pad(zoomed_labels,pad_width=((lyp,hyp),(lxp,hxp)),constant_values=0)

#%%

import napari
viewer = napari.Viewer(ndisplay=2)
# viewer.add_image(cropped_img_vol,name = 'raw_roi')
viewer.add_image(z_slice, name = 'z_slice')
viewer.add_image(padded_labels,name='seg_out')
napari.run()


# %%
print(f"z_slice.shape{z_slice.shape}")
print(f"seg_out.sahpe {padded_labels.shape}")
# %%

###### test pre computed_feats#######
import zarr
from image_reader import Ims_Image
level = 0
stride = 16
ims_vol = Ims_Image(ims_path="/home/confetti/e5_data/t1779/t1779.ims", channel=2)
raw_volume_size =ims_vol.rois[level][3:] #the data shape at r3 for test
print(f"raw_volume_size{raw_volume_size}")
#%%
whole_volume_size = [int(element//2) for element in raw_volume_size]
whole_volume_offset = [int(element//4) for element in raw_volume_size]
valid_offset = [ int(x + int((3/2) *stride)) for x in whole_volume_offset]
valid_size = [ int(x - int((3/2) *stride)) for x in whole_volume_size]
lb = valid_offset
hb = [ x+ y for x, y in zip(valid_offset,valid_size)] 

"""
for point outsie lb and hb, the correspond feats is zero

for point within lb and hb, the correspond feats :
feats_coordinates = (raw_coordinates - lb)//stride 

"""
# roi_offset is 3d
roi_offset = [7075,5600,4850]
roi_size =[1075,825,900]
raw_img = ims_vol.from_roi(coords=[*roi_offset,*roi_size],level=0)
print(f"reading from ims, raw_img shape {raw_img.shape}")
#%%
z_idx = 461 
slice_z_idx = roi_offset[0] + z_idx 
z_slice = raw_img[z_idx,:,:]

slice_start_idx_2d = [ stride - (offset - lb)%stride for  offset,lb in zip(roi_offset[1:],lb[1:] )]

#label is 2d  at z_idx:(y,x)
dummy_label = np.random.randint(low=0,high=5,size=(825,900))

# skipped the first stride cube (both imcompele and complete cube , cube is the roi when extrating feats)
zoomed_label = dummy_label[ slice_start_idx_2d[0]::stride, slice_start_idx_2d[1]::stride] 
zoomed_label = zoomed_label[:-1,:-1]
feats_z_offset = int((slice_z_idx  - lb[-1])//stride)
feats_yx_offset = [ int( (start + offset - lb)//stride) for start, offset, lb in zip(slice_start_idx_2d,roi_offset[1:],lb[1:] )]

feats_map = zarr.open_array('/home/confetti/data/t1779/mlp_feats.zarr',mode='a')
print(f"feats_map.shape{feats_map.shape}")
#%%
(ly,lx) = feats_yx_offset
(hy,hx) =[ l + s for l,s in zip(feats_yx_offset , zoomed_label.shape)]
index = (slice(None),
         slice(feats_z_offset,feats_z_offset+1),
         slice(ly,hy),
         slice(lx,hx)
         )

feats_slice = feats_map[index]
C,D,H,W = feats_slice.shape
feats_lst = np.moveaxis(feats_slice,0,-1).reshape(-1,C) 
print(f"coorepond_feats_slice.shape {feats_slice.shape}")
print(f"slice_start_idx :{slice_start_idx_2d}")
print(f"zoomed_label.shape {zoomed_label.shape}")
print(f"feats_lst.shape {feats_lst.shape}")

def seg_out(feats_slice,label):
    # feats_lst = np.moveaxis(feats_slice,0,-1).reshape(-1,C) 
    feats_slice = np.squeeze(feats_slice)
    feats_slice = np.moveaxis(feats_slice,0,-1)
    H,W,C = feats_slice.shape
    ncc_feats = cos_theta_plot(idx=int(H*W//2 +16),encoded=feats_slice,img_shape=(H,W))

    zoomed_labels = np.kron(ncc_feats,np.ones((stride,stride),dtype = int))

    raw_shape = dummy_label.shape
    lyp = slice_start_idx_2d[0] 
    ret =( raw_shape[0] -  slice_start_idx_2d[0] ) % stride 
    hyp = ret if  ret else stride

    lxp = slice_start_idx_2d[1] 
    ret =( raw_shape[1] -  slice_start_idx_2d[1] ) % stride 
    hxp = ret if  ret else stride
    padded_labels =np.pad(zoomed_labels,pad_width=((lyp,hyp),(lxp,hxp)),constant_values=0)

    return padded_labels 
pre_computed_feats = seg_out(feats_slice,dummy_label)

viewer.add_image(z_slice, name = 'z_slice')
viewer.add_image(pre_computed_feats,name='seg_out_pre_computed')
napari.run()
# %%
