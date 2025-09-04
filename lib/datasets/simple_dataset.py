import sys
sys.path.append("/home/confetti/e5_workspace/hive1")

import numpy as np
import tifffile as tif
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import os
import random

class SimpleDataset(Dataset):
    def __init__(self, args,valid=False,use_ratio = 1):
        """
        amount : control the amount of data for training
        """
        
        #filepath,trans,evalue_img=None,evalue_mode=False,amount=0.5
        self.e5 = args.e5
        # Set data paths based on configuration and mode
        self.data_path = args.e5_data_path_dir if self.e5 else args.data_path_dir
        self.valid_data_path = args.e5_valid_data_path_dir if self.e5 else args.valid_data_path_dir

        self.valid = valid

        # Determine the file path to use (training or validation)
        current_path = self.valid_data_path if self.valid else self.data_path

        # Collect files ending with `.tif`
        # self.files = [os.path.join(current_path, fname) for fname in os.listdir(current_path) if fname.endswith('.tif')]

        self.files = sorted(
                            [os.path.join(current_path, fname) 
                            for fname in os.listdir(current_path) 
                            if fname.endswith(('.tif', '.tiff'))],
                            key=lambda x: int(os.path.basename(x)[:4])
                        )
        # random.shuffle(self.files)
        self.files  = self.files[:int(use_ratio*len(self.files))]
        print(f"######init simple_dataset with amount ={use_ratio}, len of datset is {len(self.files)}#####")

    def __len__(self):
 
        return len(self.files) 


    def __getitem__(self,idx) :

        roi = tif.imread(self.files[idx])
        roi = np.array(roi).astype(np.float32) 

        roi=torch.from_numpy(roi)
        roi=torch.unsqueeze(roi,0)

        return roi

def get_dataset(args):

    # === Get Dataset === #
    train_dataset = SimpleDataset(args, use_ratio=1)

    return train_dataset

def get_valid_dataset(args):

    # === Get Dataset === #
    train_dataset = SimpleDataset(args,valid=True,use_ratio=1)

    return train_dataset




if __name__ =="__main__":
    import yaml
    cfg_pth="/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder/config/3dunet_brain_region_contrastive_learing.yaml"
    with open(cfg_pth,"r") as file:
        cfg=yaml.safe_load(file)

    tran_dataset=get_dataset(cfg)
    print(f"success")
