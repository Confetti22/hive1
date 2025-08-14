import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from lib.datasets.simple_dataset import get_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import numpy as np
import tifffile as tif
import pickle
import os
from config.load_config import load_cfg
from lib.arch.ae import  load_compose_encoder_dict,build_final_model

device ='cuda'
args = load_cfg('config/seghead.yaml')
args.e5 = False
args.data_path_dir = "/home/confetti/data/rm009/v1_roi1_seg_valid/rois"
args.mask_path_dir = "/home/confetti/data/rm009/v1_roi1_seg_valid/masks"
args.valid_data_path_dir = "/home/confetti/data/rm009/v1_roi1_seg_valid/rois"
args.valid_mask_path_dir = "/home/confetti/data/rm009/v1_roi1_seg_valid/masks"

args.filters = [32,64]
args.kernel_size = [5,5]
args.mlp_filters =[64,32,24,12]
args.last_encoder = False
args.avg_pool_size = [8,8,8] 

exp_name ='aemlpv1_continue_e2500'
feats_save_dir = f"/home/confetti/data/rm009/v1_roi1_seg_valid/l2_pool8_{exp_name}"
rgb_img_save_dir = f"/home/confetti/data/rm009/v1_roi1_seg_valid/l2_pool8_{exp_name}/rgb_feats"
os.makedirs(feats_save_dir,exist_ok=True)
os.makedirs(rgb_img_save_dir,exist_ok=True)

dataset = get_dataset(args)
loader = DataLoader(dataset,batch_size=1,drop_last=False,shuffle=False,num_workers=0)
E5 = False

if E5:
    data_prefix = "/share/home/shiqiz/data"
    workspace_prefix = "/share/home/shiqiz/workspace/hive"
else:
    data_prefix = "/home/confetti/data"
    workspace_prefix = '/home/confetti/e5_workspace/hive'

cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)

# the latter conv_layer parameters will not be loaded
cnn_ckpt_pth = f'{data_prefix}/weights/ae_feats_nissel_v1_roi1_decaylr_e1600.pth'
mlp_ckpt_pth =f'/home/confetti/e5_workspace/hive1/outs/contrastive_run_rm009/ae_mlp_rm009_v1/continute_FEATl2_avg8_LOSSpostopk_numparis16384_batch4096_nview4_d_near6_shuffle20_cosdecay_valide_with_avgpool/checkpoints/epoch_2500.pth'
mlp_weights_dict = torch.load(mlp_ckpt_pth)['model']
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_weight_dict=mlp_weights_dict,dims=args.dims)
#%%

from confettii.plot_helper import three_pca_as_rgb_image

for idx, img in enumerate(tqdm(loader)):
    #check avgpool is applied after two layer conv
    feats = cmpsd_model(img.to(device))
    feats = np.squeeze(feats.cpu().detach().numpy())
    spatial_shape = feats.shape[1:]
    feats_lst = np.moveaxis(feats,0,-1)
    feats_lst = feats_lst.reshape(-1,feats.shape[0])
    rgb_img=three_pca_as_rgb_image(feats_lst,spatial_shape) 
    with open(f"{feats_save_dir}/{idx:04d}_feats.pkl",'wb') as f:
        pickle.dump(feats,f)
    tif.imwrite(f"{rgb_img_save_dir}/{idx:04d}_rgb_feats.tif",rgb_img)
