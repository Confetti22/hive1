import sys
sys.path.append("/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder")

import numpy as np
import tifffile as tif
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import os
import random
import pickle
from scipy.ndimage import zoom
from pathlib import Path

class SegDataset(Dataset):
    def __init__(self, args,valid=False,use_ratio = 1):
        """
        for 3d feats volume and corresponding mask
        amount : control the amount of data for training
        """
        #filepath,trans,evalue_img=None,evalue_mode=False,amount=0.5
        self.e5 = args.e5
        # Set data paths based on configuration and mode
        self.data_path = args.e5_data_path_dir if self.e5 else args.data_path_dir
        self.mask_path = args.e5_mask_path_dir if self.e5 else args.mask_path_dir
        self.valid_data_path = args.e5_valid_data_path_dir if self.e5 else args.valid_data_path_dir
        self.valid_mask_path = args.e5_valid_mask_path_dir if self.e5 else args.valid_mask_path_dir
        self.valid = valid
        self.feats_level = args.feats_level
        self.feats_avg_kernel = args.feats_avg_kernel

        # Determine the file path to use (training or validation)
        current_data_path = self.valid_data_path if self.valid else self.data_path
        current_mask_path = self.valid_mask_path if self.valid else self.mask_path

        # Collect files ending with `.tif`

        valid_exts = ('.tif', '.tiff', '.pkl')          # any you want to keep

        self.files = sorted(
            [os.path.join(current_data_path, fname) 
            for fname in os.listdir(current_data_path) 
            if fname.endswith(valid_exts)],
            key=lambda x: int(os.path.basename(x)[:4])
        )

        self.masks_files = sorted(
            [os.path.join(current_mask_path, fname) 
            for fname in os.listdir(current_mask_path) 
            if fname.endswith('.tiff')],
            key=lambda x: int(os.path.basename(x)[:4])
        )


        self.files  = self.files[:int(use_ratio*len(self.files))]
        self.mask_files  = self.masks_files[:int(use_ratio*len(self.masks_files))]
        print(f"######init simple_dataset with amount ={use_ratio}, len of datset is {len(self.files)}#####")

    def __len__(self):
        return len(self.files) 


    def __getitem__(self,idx) :
        fname = Path(self.files[idx])
        suffix = fname.suffix.lower()
        # ---------- load ---------------------------------------------------------
        if suffix == ".pkl":
            with fname.open("rb") as f:
                arr = pickle.load(f)            # numpy array shape (C,D,H,W)
        elif suffix in {".tif", ".tiff"}:
            arr = tif.imread(fname)            # (D,H,W) or (Z,H,W)
            arr = np.expand_dims(arr,0)
        else:
            raise ValueError(f"Unsupported file type: {fname}")
        feats  = torch.from_numpy(arr).float()


        #taylor the mask with the same shape as the feats
        mask = tif.imread(self.mask_files[idx])
        mask = zoom(mask,zoom=1/(2**self.feats_level),order=0)
        crop_size = int((self.feats_avg_kernel -1)/2)
        if self.feats_avg_kernel %2 ==0: #for even kernel size, the total crop_size is an odd number
            mask = mask[crop_size+1:-crop_size,crop_size+1:-crop_size,crop_size+1:-crop_size]
        else:
            mask = mask[crop_size:-crop_size,crop_size:-crop_size,crop_size:-crop_size]
    
        mask = torch.from_numpy(mask).long() #shape (D,H,W)
        mask = mask -1 # class number rearrange from 0 to N-1

        return feats,mask

def get_dataset(args):

    # === Get Dataset === #
    train_dataset = SegDataset(args, use_ratio=1)

    return train_dataset

def get_valid_dataset(args):

    # === Get Dataset === #
    train_dataset = SegDataset(args,valid=True,use_ratio =1)

    return train_dataset

if __name__ =="__main__":
    import yaml
    cfg_pth="/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder/config/3dunet_brain_region_contrastive_learing.yaml"
    with open(cfg_pth,"r") as file:
        cfg=yaml.safe_load(file)

    tran_dataset=get_dataset(cfg)
    print(f"success")
