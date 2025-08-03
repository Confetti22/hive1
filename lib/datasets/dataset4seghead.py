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

import tifffile as tif
from scipy.ndimage import zoom
import torch
import numpy as np
from typing import Tuple

class SegDataset(Dataset):
    def __init__(self, args,valid=False,use_ratio = 1,bnd=False,bool_mask=False):
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

        self.bnd =bnd
        self.bool_mask = bool_mask
        if bnd:
            self.bnd_path = args.e5_bnd_path_dir if self.e5 else args.bnd_path_dir
            self.valid_bnd_path = args.e5_valid_bnd_path_dir if self.e5 else args.valid_bnd_path_dir
            current_bnd_path = self.valid_bnd_path if self.valid else self.bnd_path

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
            if fname.endswith('.tif')],
            key=lambda x: int(os.path.basename(x)[:4])
        )

        if self.bnd:
            self.bnd_files = sorted(
                [os.path.join(current_bnd_path, fname) 
                for fname in os.listdir(current_bnd_path) 
                if fname.endswith('.tif')],
                key=lambda x: int(os.path.basename(x)[:4])
            )
            self.bnd_files  = self.bnd_files[:int(use_ratio*len(self.bnd_files))]


        self.files  = self.files[:int(use_ratio*len(self.files))]
        self.mask_files  = self.masks_files[:int(use_ratio*len(self.masks_files))]
        print(f"######init simple_dataset with amount ={use_ratio}, len of datset is {len(self.files)}#####")
    

    def crop_border(self,mask: np.ndarray, crop_size: int, even_kernel: bool) -> np.ndarray:
        """
        Crop `crop_size` (odd kernel) or `crop_size+1` (even kernel) voxels
        from every *croppable* border of `mask`.

        A dimension is croppable iff its length > 2*crop_size (+1 for even kernels).

        Parameters
        ----------
        mask : np.ndarray
            Array with shape (H, W), (D, H, W), or (1, H, W).
        crop_size : int
            Half the kernel size.
        even_kernel : bool
            True if the kernel size is even.

        Returns
        -------
        np.ndarray
            Cropped mask.
        """
        extra = 1 if even_kernel else 0          # the “+1” for even kernels
        start = crop_size + extra
        stop  = -crop_size if crop_size else None   # keep ':' when crop_size == 0

        slicer: Tuple[slice, ...] = tuple(
            slice(start, stop) if dim_len > start + crop_size else slice(None)
            for dim_len in mask.shape
        )
        return mask[slicer]




    def _load_vol(self,path: str,) -> torch.Tensor:

        """Read, down-sample, crop, return a torch tensor."""
        arr = tif.imread(path)                              # np.ndarray
        arr = np.squeeze(arr)
        arr = zoom(arr, self._down, order=0)                # nearest-neighbour
        arr = self.crop_border(arr, self._crop, self._even)      # keeps depth if 1
        return torch.as_tensor(arr, dtype=torch.float32)            # (D,H,W) or (H,W)

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


        self._down   = 1 / (2 ** self.feats_level)
        self._crop   = (self.feats_avg_kernel - 1) // 2
        self._even   = (self.feats_avg_kernel % 2 == 0)

        mask = self._load_vol(self.mask_files[idx])    # labels
        if self.bool_mask:
            mask = mask
        else: # for numerical mask
            mask = mask -1 # class number rearrange from 0 to N-1

        if self.bnd:
            bnd  = self._load_vol(self.bnd_files[idx])   # binary boundary
            return feats,mask,bnd
        else:
            return feats,mask


def get_dataset(args,bnd=False,bool_mask=False):

    # === Get Dataset === #
    train_dataset = SegDataset(args, use_ratio=1,bnd=bnd,bool_mask=bool_mask)

    return train_dataset

def get_valid_dataset(args,bnd=False,bool_mask = False):

    # === Get Dataset === #
    train_dataset = SegDataset(args,valid=True,use_ratio =1,bnd=bnd,bool_mask=bool_mask)

    return train_dataset

if __name__ =="__main__":
    import yaml
    cfg_pth="/home/confetti/e5_workspace/brain_region_unet_contrastive_encoder/config/3dunet_brain_region_contrastive_learing.yaml"
    with open(cfg_pth,"r") as file:
        cfg=yaml.safe_load(file)

    tran_dataset=get_dataset(cfg)
    print(f"success")
