import itk
import numpy as np
import tifffile as tif
# image = itk.imread("/home/confetti/data/rm009/rm009_roi/all-z64800-65104/All-Z64800-65104.mha")
# size = itk.size(image)
# print("Image size:", size)

# array = itk.array_from_image(image)
# print("Array shape:", array.shape)  # (z, y, x)
# tif.imwrite("/home/confetti/data/rm009/rm009_roi/mask.tiff",array)

#%%
######## padding the roi_with 24(96um) at both z slice######
# import os
# import tifffile
# import numpy as np

# # Directory containing the TIFF files
# folder = "/home/confetti/data/rm009/rm009_roi/4"

# # Define Z range
# z_start = 16176
# z_end = 16299

# # Generate expected filenames
# filenames = [f"Z{z:05d}_C4.tif" for z in range(z_start, z_end + 1)]

# # Load and stack into a 3D array
# volume_slices = []
# for fname in filenames:
#     fpath = os.path.join(folder, fname)
#     if os.path.exists(fpath):
#         img = tifffile.imread(fpath)
#         volume_slices.append(img)
#     else:
#         raise FileNotFoundError(f"{fname} not found in directory.")

# # Stack into a 3D volume (Z, Y, X)
# volume = np.stack(volume_slices, axis=0)
# print("Volume shape:", volume.shape)  # (Z, Y, X)
# print("Volume dtype:", volume.dtype)
# tifffile.imwrite(f"/home/confetti/data/rm009/rm009_roi/z{z_start}_z{z_end}C4.tif",volume)
# import tifffile as tif
# old_mask = tif.imread("/home/confetti/data/rm009/rm009_roi/mask_77_3500_5250.tiff")
# mask = np.pad(old_mask,((24, 23), (0, 0), (0, 0)), mode='constant', constant_values=0)
# tif.imwrite("/home/confetti/data/rm009/rm009_roi/mask_124_3500_5250.tiff",mask)


#%%
############# crop the big roi and mask into chunks, ready for dataloader training ############

# import tifffile as tif
# import numpy as np
# import os
# import tifffile


# def crop_3d_with_stride_and_filter(image, mask, crop_size=(124, 1024, 1024), stride=512, threshold=0.1):
#     D, H, W = image.shape
#     d, h, w = crop_size

#     assert D == d, "Depth of crop must match image depth"

#     roi_list = []
#     mask_list = []

#     for y in range(0, H - h + 1, stride):
#         for x in range(0, W - w + 1, stride):
#             img_crop = image[:, y:y+h, x:x+w]
#             mask_crop = mask[:, y:y+h, x:x+w]

#             # Calculate ratio of non-zero voxels in the mask
#             nonzero_ratio = np.count_nonzero(mask_crop) / mask_crop.size

#             # Keep only if non-zero region > threshold
#             if nonzero_ratio > threshold:
#                 roi_list.append(img_crop)
#                 mask_list.append(mask_crop)

#     if len(roi_list) == 0:
#         return np.empty((0, d, h, w), dtype=image.dtype), np.empty((0, d, h, w), dtype=mask.dtype)

#     return np.stack(roi_list), np.stack(mask_list)


# def save_rois_to_dirs(roi_images, roi_masks, roi_dir, mask_dir):
#     os.makedirs(roi_dir, exist_ok=True)
#     os.makedirs(mask_dir, exist_ok=True)

#     for i in range(roi_images.shape[0]):
#         roi = roi_images[i]   # shape: (124, 1024, 1024)
#         mask = roi_masks[i]   # shape: (124, 1024, 1024)

#         roi_path = os.path.join(roi_dir, f"{i:04d}.tiff")
#         mask_path = os.path.join(mask_dir, f"{i:04d}_mask.tiff")

#         tifffile.imwrite(roi_path, roi)    # or uint16/float32 based on your data
#         tifffile.imwrite(mask_path, mask.astype('uint8'))  # adjust dtype if needed

# # Example usage:

# roi = tif.imread("/home/confetti/data/rm009/rm009_roi/z16176_z16299C4.tif")
# mask = tif.imread("/home/confetti/data/rm009/rm009_roi/mask_124_3500_5250.tif")

# # image and mask shapes: (124, 3500, 5250)
# roi_images, roi_masks = crop_3d_with_stride_and_filter(roi, mask)
# print("Filtered ROI image shape:", roi_images.shape)
# print("Filtered ROI mask shape:", roi_masks.shape)

# # Save to directories
# save_rois_to_dirs(
#     roi_images, roi_masks,
#     roi_dir="/home/confetti/data/rm009/v1_roi1_seg/rois",
#     mask_dir="/home/confetti/data/rm009/v1_roi1_seg/masks",
# )
# %%
###### extract feature ready for training the segmentation head
import sys
sys.path.append("/home/confetti/e5_workspace/hive1")

# %%
from lib.datasets.simple_dataset import get_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pickle
import os
from config.load_config import load_cfg
from lib.arch.ae import  load_compose_encoder_dict,build_final_model
device ='cuda'
args = load_cfg('config/seghead.yaml')
args.filters = [32,64,96]
args.mlp_filters =[96,48,24,12]

exp_name ='postopk_1000'
feats_save_dir = f"/home/confetti/data/rm009/v1_roi1_seg_valid/l3_pool7_{exp_name}"
os.makedirs(feats_save_dir,exist_ok=True)

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
cnn_ckpt_pth = f'{data_prefix}/weights/rm009_3d_ae_best.pth'
mlp_ckpt_pth =f'{data_prefix}/weights/rm009_postopk_1000.pth'
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth,dims=args.dims)

from confettii.plot_helper import three_pca_as_rgb_image

for idx, img in enumerate(tqdm(loader)):
    feats = cmpsd_model(img.to(device))
    feats = np.squeeze(feats.cpu().detach().numpy())
    spatial_shape = feats.shape[1:]
    feats_lst = np.moveaxis(feats,0,-1)
    feats_lst = feats_lst.reshape(-1,feats.shape[0])
    rgb_img=three_pca_as_rgb_image(feats_lst,spatial_shape) 
    with open(f"{feats_save_dir}/{idx:04d}_feats.pkl",'wb') as f:
        pickle.dump(feats,f)
    tif.imwrite(f"{feats_save_dir}/{idx:04d}_rgb_feats.tif",rgb_img)






