#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
from lib.arch.ae import build_encoder_model,load_encoder2encoder
from config.load_config import load_cfg
from torchsummary import summary

device ='cuda'
args = load_cfg('config/rm009.yaml')

avg_pool = None 
args.avg_pool_size = [avg_pool]*3
args.avg_pool_padding =  False
E5 = False 

if E5:
    data_prefix = "/share/home/shiqiz/data"
    workspace_prefix = "/share/home/shiqiz/workspace/hive"
else:
    data_prefix = "/home/confetti/data"
    workspace_prefix = '/home/confetti/e5_workspace/hive'

cnn_ckpt_pth = f'{data_prefix}/weights/rm009_3d_ae_best.pth'

encoder_model = build_encoder_model(args,dims=3) 
encoder_model.eval().to(device)
load_encoder2encoder(encoder_model,cnn_ckpt_pth)
summary(encoder_model,(1,*args.input_size))
print(encoder_model)


#%%
import torch 
import torch.nn as nn
import numpy as np 
from scipy.spatial.distance import pdist, squareform
import tifffile as tif
import numpy as np
vol = tif.imread(f'{data_prefix}/rm009/seg_valid/_0001.tif')
mask = tif.imread(f'{data_prefix}/rm009/seg_valid/_0001_human_mask_3d.tif')
print(f"{vol.shape= },{mask.shape= }")
# vol = pad_to_multiple_of_unit(vol,unit=zoom_factor) 
print(f"after padding: {vol.shape= }")

#crop the cortex roi 
# vol = vol[:,:77,:]
vol  = np.random.randint(0,100,(76,1024,1024))
# mask = mask[:,432:1100,206:1329]
# mask_slice = mask[32]

#%% get different_resolution feature_map
activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

hook1 = encoder_model.conv_in.register_forward_hook(getActivation('layer1'))
hook2 = encoder_model.down_layers[0].register_forward_hook(getActivation('layer2'))

inputs = torch.from_numpy(vol).float().unsqueeze(0).to(device)
# outs = np.squeeze(encoder_model(inputs).cpu().detach().numpy())
l3_feats = encoder_model(inputs)
l1_feats =  activation['layer1']
l2_feats =  activation['layer2']

hook1.remove()
hook2.remove()

# feat_map list structure: avg_pooling (None, 2,4,8) , each avg_pooling is an dict contains l1_l2_l3 feats  
feat_map=[]
for k in (2,4,8):
    feats_dic={}
    # pool = nn.AvgPool3d(kernel_size=[k]*3, stride=1,padding=int((k-1)//2))
    pool = nn.AvgPool3d(kernel_size=[k]*3, stride=1,padding=0)
    l1_feats_avg  = pool(l1_feats)
    l2_feats_avg  = pool(l2_feats)
    l3_feats_avg  = pool(l3_feats)

    feats_dic['l1']=l1_feats_avg.cpu().detach().numpy()
    feats_dic['l2']=l2_feats_avg.cpu().detach().numpy()
    feats_dic['l3']=l3_feats_avg.cpu().detach().numpy()

    feat_map.append(feats_dic)

feats_dic = {}
feats_dic['l1']=l1_feats.cpu().detach().numpy()
feats_dic['l2']=l2_feats.cpu().detach().numpy()
feats_dic['l3']=l3_feats.cpu().detach().numpy()
feat_map.insert(0,feats_dic)

for k_dict in feat_map:
    print(k_dict['l1'].shape)
    print(k_dict['l2'].shape)
    print(k_dict['l3'].shape, "\n")
#%%

# extract the z_slice feat_map from each 4Dfeature_map
one_third_zslice_feats_map = {} 
all_feats_map_for_visal =[]
pooling_index = [0,2,4,8]
for idx,k_dict in enumerate(feat_map):
    # create a dict to hold the central slice for each level
    slice_dict = {}
    for level, volume in k_dict.items():
        # volume.shape == (C, D, H, W)
        C, target_dist, H, W = volume.shape
        z_idx = target_dist // 3     # integer division: picks the “1/3” slice
        zslice = volume[:, z_idx, :, :]  # shape (C, H, W)
        slice_dict[level] = zslice
        all_feats_map_for_visal.append(np.std(zslice,axis=0))
    one_third_zslice_feats_map[pooling_index[idx]]=slice_dict


del feats_dic

for key,k_dict in one_third_zslice_feats_map.items():
    print(k_dict['l1'].shape)
    print(k_dict['l2'].shape)
    print(k_dict['l3'].shape, "\n")

# from confettii.plot_helper import grid_plot_list_imgs
# ae_feats_fig = grid_plot_list_imgs(all_feats_map_for_visal,col_labels=['l1','l2','l3'],row_labels=['k0','k2','k4','k8'],ncols=3,show=False)

# Now zslice_feats_map[i]['l1'], ['l2'], ['l3'] are each (C, H, W) arrays
#%%
from lib.arch.ae import MLP
from distance_contrast_helper import train_demo ,HTMLFigureLogger
from torch.utils.tensorboard import SummaryWriter

exp_save_dir='multi_percentile'
# ae_writer = SummaryWriter(log_dir=f'{exp_save_dir}/whole')
# ae_writer.add_figure('all_ae_feats', ae_feats_fig, global_step=0)

reduce_factor ={'l1':2,'l2':4,'l3':8}
filters_map={'l1':[32,24,12,12],'l2':[64,32,24,12],'l3':[96,64,32,12]}


num_pairs=10000
shuffle_pairs_epoch_num = 50 
num_epochs = 10000 
valid_very_epoch = 10

ori_d_far = 64
for percentile in [None,0.5,0.3,0.1]:
    for pool_key in [8]:
        for level_key in ['l2']:

            d_near= int(16/reduce_factor[level_key]) # 16 pixels in raw img
            d_far= int(ori_d_far/reduce_factor[level_key])  #64 pixels in raw img

            exp_name = f"_dfar_{ori_d_far}level_{level_key}_pool_kernel{pool_key}_proloss{percentile}_shuffle{shuffle_pairs_epoch_num}"
            writer = SummaryWriter(log_dir=f'{exp_save_dir}/{exp_name}')

            img_logger = HTMLFigureLogger(log_dir=f'{exp_save_dir}/{exp_name}',html_name="pca_plot.html",comment=f'shuffle_every_{shuffle_pairs_epoch_num}')

            _feats_map = one_third_zslice_feats_map[pool_key][level_key]
            # _feats_map = _feats_dict[level_key] #C,H,W (H equals to W)
            C,H,W = _feats_map.shape

            img_shape= (H,W)
            features = np.moveaxis(_feats_map,0,-1) #H,W,C
            features = features.reshape(-1,C)

            #the features here is a feature_list
            if len(features.shape)==2:
                features=features[np.newaxis,:]
            agg_feats = features #B*N*D

            locations = np.indices((H, W))          # shape = (2, M, N)
            locations = locations.reshape(2, -1).T        # shape = (M*N, 2)

            print(f"feats.shape :{agg_feats.shape}")
            print(f"locations.shape :{locations.shape}")
            mlp_filters = filters_map[level_key]

            mlp_model = MLP(filters=mlp_filters).to(device)
            mlp_model.train()

            train_demo(agg_feats,locations,d_near,d_far,mlp_model,
                        num_pairs,writer,img_logger,img_shape,num_epochs,
                        valid_very_epoch,shuffle_pairs_epoch_num,percentile=percentile,model_save_dir= f'{exp_save_dir}/{exp_name}')


# %%
#visualize target_dist ratio in all pari_wise distance
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import pdist

# H, W = 77,134 
# target_dist = 4 

# # Generate all (i, j) coordinates in the matrix
# coords = np.array([(i, j) for i in range(H) for j in range(W)])

# # Compute pairwise Euclidean distances (each pair once,only upper triangle or lower traingle of distance matrix)
# dists = pdist(coords, metric='euclidean')

# # Compute proportions
# left_mask = dists <= target_dist
# right_mask = dists > target_dist
# left_pct = np.sum(left_mask) / len(dists) * 100
# right_pct = np.sum(right_mask) / len(dists) * 100

# # Plot histogram
# plt.figure(figsize=(8, 5))
# plt.hist(dists, bins=100, edgecolor='black', alpha=0.7)
# plt.axvline(target_dist, color='red', linestyle='--', label=f'D = {target_dist}')
# plt.title(f'Pairwise Distance Histogram ({H}x{W} grid)')
# plt.xlabel('Distance')
# plt.ylabel('Count')
# plt.legend()

# # Add text annotation for percentages
# plt.text(target_dist + 1, plt.ylim()[1] * 0.9, f'Right: {right_pct:.2f}%', color='red')
# plt.text(target_dist - target_dist * 0.7, plt.ylim()[1] * 0.9, f'Left: {left_pct:.2f}%', color='red')

# plt.grid(True)
# plt.show()

# # Optional: print values
# # Print numerical summary
# total_samples = len(dists)
# print(f"mean distance is {np.mean(dists):.3f}, ={16*np.mean(dists):.3f}um")
# print(f"Total pairwise samples: {total_samples}")
# print(f"Left of D={target_dist}: {left_pct:.2f}%")
# print(f"Right of D={target_dist}: {right_pct:.2f}%")
# %%
