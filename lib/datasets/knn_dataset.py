import sys
sys.path.append("/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder")

import numpy as np
import tifffile as tif
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import os
import random

class KnnDataset(Dataset):
    def __init__(self, args,knn=2,use_ratio = 1):
        """
        amount : control the amount of data for training
        """
        
        #filepath,trans,evalue_img=None,evalue_mode=False,amount=0.5
        self.e5 = args.e5
        # Set data paths based on configuration and mode
        self.data_path = args.e5_data_path_dir if self.e5 else args.data_path_dir
        self.knn = knn


        self.dirs = [os.path.join(self.data_path,d) for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        self.files = []
        for dir in self.dirs:
            # Collect files ending with `.tif`
            files = [os.path.join(dir, fname) for fname in os.listdir(dir) if fname.endswith('.tif')]
            # files  = files[:int(use_ratio*len(self.files))]
            self.files.append(files)
        print(f"######init simple_dataset with amount ={use_ratio}, len of datset is {len(self.files[0])}#####")

    def __len__(self):
 
        # tif files in the first knn as data_length
        return len(self.files[0]) 

    def __getitem__(self,idx) :


        res = []
        for i in range(self.knn):
            roi = tif.imread(self.files[i][idx]).astype(np.float32) 
            roi = torch.from_numpy(roi)
            roi = torch.unsqueeze(roi,0)
            res.append(roi)

        return res 
    




def get_dataset(args):

    # === Get Dataset === #
    train_dataset = KnnDataset(args, use_ratio=1)

    return train_dataset


if __name__ =="__main__":
    import yaml
    cfg_pth="/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder/config/3dunet_brain_region_contrastive_learing.yaml"
    with open(cfg_pth,"r") as file:
        cfg=yaml.safe_load(file)

    tran_dataset=get_dataset(cfg)
    print(f"success")
