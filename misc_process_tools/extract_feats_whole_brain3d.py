#%%
import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
from lib.arch.ae import build_encoder_model,load_encoder2encoder
from config.load_config import load_cfg
#%%
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import time
import zarr
from confettii.feat_extract import TraverseDataset3d, get_feature_list

device ='cuda'
args = load_cfg('../config/rm009.yaml')
"try avg on None, 5, 9"
avg_pool = 3
args.avg_pool_size = [avg_pool]*3
print('\n\n\n\n')
print(f"{args.avg_pool_size= }")

E5 = False 
cnn_ckpt_pth = '/home/confetti/data/weights/rm009_3d_ae_best.pth'
cluster_ckpt_pth = '/share/home/shiqiz/data/rm009/ae.pth'
cnn_ckpt_pth = cluster_ckpt_pth if E5 else cnn_ckpt_pth
#%%
log_pth = f'../outs/extract_feats/roi_level3_avg{avg_pool}.log'
import sys
from contextlib import redirect_stdout
with open(log_pth, 'w') as f:
    with redirect_stdout(f):
        print("This goes to the log file.")
        # Other code whose output you want to capture

#%%
encoder_model = build_encoder_model(args, dims=3)
encoder_model.eval().to(device)
before_max = encoder_model.conv_in[0].weight.max().item()
load_encoder2encoder(encoder_model, cnn_ckpt_pth)
after_max = encoder_model.conv_in[0].weight.max().item()
print("Before loading:", before_max)
print("After loading:", after_max)

# Assert to ensure they are the same
assert not torch.isclose(torch.tensor(before_max), torch.tensor(after_max), atol=1e-6), \
    f"weight dict load failed"

from helper.image_reader import Ims_Image

#%%

local_img_pth ="/home/confetti/e5_data/rm009/rm009.ims"
local_img_pth = '/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/Analysis/ROIReconstruction/ROIImage/z13750_c1.ims'
cluster_img_pth ="/share/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/RM009_all_/009.ims"  

raw_img_pth = cluster_img_pth if E5 else local_img_pth 
ims_vol=Ims_Image(raw_img_pth,channel=0)

level = 0
raw_volume_size =ims_vol.rois[level][3:] 
print(f"raw_volume_size{raw_volume_size}")
#%%
# whole_volume_size = [int(element//2) for element in raw_volume_size]
# whole_volume_offset = [int(element//4) for element in raw_volume_size]
whole_volume_size = raw_volume_size 
# whole_volume_size = [64,2048,2048] 
whole_volume_offset = [0,0,0]

#%% check for the roi to be extracted feats on
import numpy as np
z_offset,y_offset,x_offset = whole_volume_offset 
z_span     = whole_volume_size[0]   # distance from first to last (your "+3750")
z_last     = z_offset + z_span   # = 5125  (you used this for last_z_slice)

y_size , x_size = whole_volume_size[1:]

n_slices = 6

# Evenly spaced (including first & last)
z_indices = np.round(np.linspace(z_offset, z_last, n_slices)).astype(int)

# Pull the slices
slices = [
    ims_vol.from_roi(
        coords=(z, y_offset, x_offset, 1, y_size, x_size),level=0)
    for z in z_indices]

# Optionally stack into an array of shape (6, Y, X)
import numpy as np
slices_ary = np.stack(slices, axis=0)
import napari
viewer = napari.Viewer()
viewer.add_image(slices_ary)
napari.run()

print(f"whole_volume_size{whole_volume_size}")
print(f"whole_volume_offset{whole_volume_offset}")
#%%
batch_size = 256 

# when extracting features in t11(resol=1um), the roi_size =64, roi_stride=16, 
# so when processing the 4 um rm009, the roi_size = 16, roi_stride=4
cnn_feature_dim =96
mlp_feature_dim =12
region_size=[64,1536,1536]
roi_size =[16,16,16]
roi_stride =[4,4,4]
step = [int(2*(1/2)*r_size/r_stride -1) for r_size, r_stride in zip(roi_size,roi_stride) ]
step_size = roi_stride
margin = [int(s * s_size) for s,s_size in zip(step,step_size)]
region_stride  = [int(r_size - m) for r_size, m in zip(region_size,margin)]
print(f"step: {step}")
print(f"margin: {margin}")
print(f"step_size :{step_size}")
print(f"region_size:{region_size}")
print(f"region_stride:{region_stride}")


#the image can be extended at higher bound to satisfy one roi, so np.ceil is used in final_feats_shape computation
raw_region_num =[ np.ceil((i - k)/s )+1 for i, k ,s in zip(whole_volume_size,region_size,region_stride) ]
raw_chunks_shape =[ np.floor((i - k)/s )+1 for i, k ,s in zip(region_size,roi_size,roi_stride) ]
#subtract one line of higher bound features to insure the raw featuremap correspond with the raw image by strde

#add the channel dim to form the raw 4D data
raw_region_num.append(1)
raw_chunks_shape.append(cnn_feature_dim)
raw_region_num = np.array(raw_region_num, dtype=int)
raw_chunks_shape = np.array(raw_chunks_shape, dtype=int)

#raw is the same as final
final_chunks_shape = raw_chunks_shape.copy()
final_feats_block_num = raw_region_num.copy() 

zarr_block_num = final_feats_block_num
zarr_chunk_shape = final_chunks_shape
zarr_shape =[int(n*s) for n,s in zip(zarr_block_num,zarr_chunk_shape)]
zarr_shape = tuple(int(elem) for elem in zarr_shape)
zarr_chunk_shape = tuple( int(elem) for elem in zarr_chunk_shape)

print(f"zarr_blcok_num{zarr_block_num}")
print(f"zarr_chunk_shape{zarr_chunk_shape}")
print(f"zarr_shape{zarr_shape}")

save_zarr_file_name = f'half_brain_cnn_feats_avg{avg_pool}'
cluster_zarr_path = f"/share/home/shiqiz/data/rm009/{save_zarr_file_name}_r{level}.zarr"
local_zarr_path = f"/home/confetti/data/rm009/{save_zarr_file_name}_r{level}.zarr"
save_zarr_path =  cluster_zarr_path if E5 else local_zarr_path 
#%%
z_arr = zarr.create_array(store = save_zarr_path, shape=zarr_shape, chunks=zarr_chunk_shape, dtype="float32")
print(z_arr.shape)

#%%
# Calculate the total number of iterations (product of all blocks)
total_iterations = zarr_block_num[0] * zarr_block_num[1] * zarr_block_num[2]
print(f"total_iterations {total_iterations}")
print(f"raw chunk shape {raw_chunks_shape}")
#%%

# Create the total progress bar
current = time.time()
with tqdm(total=total_iterations, desc="Processing", position=0) as pbar:
    for z in range(zarr_block_num[0]):
        for y in range(zarr_block_num[1]):
            for x in range(zarr_block_num[2]):            #sliding offset            #sliding offset
                z_off = z*region_stride[0] +whole_volume_offset[0]
                y_off = y*region_stride[1] +whole_volume_offset[1]
                x_off = x*region_stride[2] +whole_volume_offset[2]

                print(f"traverse region offset at {z_off}_{y_off}_{x_off}")
                coords = np.concatenate(([z_off,y_off,x_off], np.array(region_size)))
                # for the last roi, will be padded at the higher bound to satisfy a region_size

                roi=ims_vol.from_roi(coords=coords,level=level)
                # extract features
                draw_border_dataset = TraverseDataset3d(img=roi,stride=roi_stride[0],win_size=roi_size[0])  
                border_draw_loader = DataLoader(draw_border_dataset,batch_size,shuffle=False,drop_last=False)
                current = time.time()
                feats_lst = get_feature_list('cuda',encoder_model,border_draw_loader,adaptive_pool=True)
                #correted reshap version
                H,W = raw_chunks_shape[1:3]
                print(f"{H= },{W= }")
                C = cnn_feature_dim 
                feats_map = feats_lst.reshape(H,W,C) # N*C -> H*W*C
                feats_map = np.expand_dims(feats_map,axis=0)

                # print(f"feats_lst shape is {feats_lst.shape}")
                print(f"extracting chunk{z}_{y}_{x} consume {time.time()-current} seconds")
                index = (
                    slice(z * zarr_chunk_shape[0], (z + 1) * zarr_chunk_shape[0]),
                    slice(y * zarr_chunk_shape[1], (y + 1) * zarr_chunk_shape[1]),
                    slice(x * zarr_chunk_shape[1], (x + 1) * zarr_chunk_shape[2]),
                    slice(None))

                print("zarr_chunk_shape:", zarr_chunk_shape)
                print("Index:", index)
                print("feats_map.shape:", feats_map.shape)
                print("Expected shape:", z_arr[index].shape)
                z_arr[index] = feats_map  # Write to Zarr
                pbar.update(1)
print(f"all finined!! in extract1 at {save_zarr_path}")
print(f"total time: {time.time() - current}")
#%%

#%%
# exam the feature_extration 
import zarr
zarr_path = "/home/confetti/data/rm009/half_brain_cnn_feats_avg8_r0.zarr"
z_arr = zarr.open_array(zarr_path,mode='r')
D,H,W,C = z_arr.shape
# print(type(z_arr))
print("ori Shape:", z_arr.shape)
print("Chunk Shape:", z_arr.chunks)
print("Data Type:", z_arr.dtype)

#%%
import numpy as np
import matplotlib.pyplot as plt
z_slice = z_arr[1,:,:,:]
from confettii.plot_helper import three_pca_as_rgb_image
rgb_img = three_pca_as_rgb_image(z_slice.reshape(-1,C),(H,W))

plt.imshow(np.max(z_slice,axis = -1))
plt.figure(figsize=(12,12))
plt.imshow(rgb_img)




# %%
import zarr
import numpy as np
import matplotlib.pyplot as plt

# Load Zarr array
zarr_path = "/home/confetti/data/rm009/half_brain_cnn_feats_avg8_r0.zarr"
z_arr = zarr.open_array(zarr_path, mode='r')
D, H, W, C = z_arr.shape

print("ori Shape:", z_arr.shape)
print("Chunk Shape:", z_arr.chunks)
print("Data Type:", z_arr.dtype)

# Prepare slice indices
z_indices = np.linspace(0, D - 1, 6, dtype=int)
y_indices = np.linspace(0, H - 1, 6, dtype=int)
x_indices = np.linspace(0, W - 1, 6, dtype=int)

# Create plot
fig, axes = plt.subplots(3, 6, figsize=(18, 9))

# Z slices (along depth, view as XY)
for i, idx in enumerate(z_indices):
    slice_img = z_arr[idx, :, :, :]  # shape: (H, W, C)
    axes[0, i].imshow(np.max(slice_img, axis=-1), cmap='gray')
    axes[0, i].set_title(f'Z slice {idx}')
    axes[0, i].axis('off')

# Y slices (along height, view as XZ)
for i, idx in enumerate(y_indices):
    slice_img = z_arr[:, idx, :, :]  # shape: (D, W, C)
    axes[1, i].imshow(np.max(slice_img, axis=-1), cmap='gray')
    axes[1, i].set_title(f'Y slice {idx}')
    axes[1, i].axis('off')

# X slices (along width, view as ZY)
for i, idx in enumerate(x_indices):
    slice_img = z_arr[:, :, idx, :]  # shape: (D, H, C)
    axes[2, i].imshow(np.max(slice_img, axis=-1).T, cmap='gray')  # transpose for proper orientation
    axes[2, i].set_title(f'X slice {idx}')
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()
# %%
