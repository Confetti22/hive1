import sys
sys.path.append("/home/confetti/e5_workspace/hive1")

import numpy as np
import tifffile as tif
from torch.utils.data import Dataset
import torch
import os
class FixedDataset(Dataset):
    def __init__(self, data_path_dir,  use_ratio=1):
        """
        A simplified dataset that does not rely on an `e5` flag or
        `args` namespace. Provide explicit paths instead.

        """


        self.files = sorted(
            [os.path.join(data_path_dir, fname)
             for fname in os.listdir(data_path_dir)
             if fname.endswith((".tif", ".tiff"))],
            key=lambda x: int(os.path.basename(x)[:4]) if os.path.basename(x)[:4].isdigit() else os.path.basename(x)
        )

        self.files = self.files[: int(use_ratio * len(self.files))]
        print(f"######init simple_dataset_noe5 with amount ={use_ratio}, len of dataset is {len(self.files)}#####")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        roi = tif.imread(self.files[idx])
        roi = np.array(roi).astype(np.float32)
        roi = torch.from_numpy(roi)
        roi = torch.unsqueeze(roi, 0)
        return roi


def get_dataset(data_path_dir, use_ratio=1):
    return FixedDataset(data_path_dir=data_path_dir, use_ratio=use_ratio)

