#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
import itertools                      
from pathlib import Path
import math, os
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config.load_config import load_cfg
from lib.arch.ae import build_semantic_seg_model
from distance_contrast_helper import HTMLFigureLogger
from lib.datasets.dataset4seghead import get_dataset, get_valid_dataset
from helper.contrastive_train_helper import log_layer_embeddings

#%%
from skimage.restoration import denoise_tv_chambolle
from skimage import data, img_as_float
from skimage.util import random_noise
import matplotlib.pyplot as plt
import numpy as np
# Load a sample grayscale image (e.g., 'camera' from skimage)
image = img_as_float(data.camera())

# Add Gaussian noise (optional)
noisy_image = random_noise(image, var=0.01)  # Adjust variance as needed

denoised_image = denoise_tv_chambolle(noisy_image, weight=0.1, channel_axis=None)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title('Noisy')
axes[2].imshow(denoised_image, cmap='gray')
axes[2].set_title('TV Denoised')
plt.show()

#%%

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        # Input shape: (B, C, H, W) or (B, C, D, H, W)
        # avoid D==1
        batch_size = x.size(0)
        spatial_dims = x.dim() - 2  # Number of spatial dims (2 for 2D, 3 for 3D)
        tv = 0.0

        if spatial_dims == 2:
            # (B, C, H, W)
            h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
            w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
            count_h = self._tensor_size(x[:, :, 1:, :])
            count_w = self._tensor_size(x[:, :, :, 1:])
            tv = (h_tv / count_h + w_tv / count_w)
        elif spatial_dims == 3:
            # (B, C, D, H, W)
            d_tv = torch.pow(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], 2).sum()
            h_tv = torch.pow(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], 2).sum()
            w_tv = torch.pow(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], 2).sum()
            count_d = self._tensor_size(x[:, :, 1:, :, :])
            count_h = self._tensor_size(x[:, :, :, 1:, :])
            count_w = self._tensor_size(x[:, :, :, :, 1:])
            tv = (d_tv / count_d + h_tv / count_h + w_tv / count_w)
        else:
            raise ValueError("Unsupported input dimensions. Expected 4D or 5D input.")

        return self.TVLoss_weight * 2 * tv / batch_size

    def _tensor_size(self, t):
        return t.numel()



        print("⚠️ Some weights were not loaded exactly:")
        if missing:
            print(f"   • Missing keys ({len(missing)}):\n     {missing}")
        if unexpected:
            print(f"   • Unexpected keys ({len(unexpected)}):\n     {unexpected}")


args = load_cfg('../config/semisupervised.yaml') 

device = 'cpu'
seg_model= build_semantic_seg_model(args).to(device)
seg_model = seg_model

cpkt = torch.load('/home/confetti/e5_workspace/hive1/outs/seg_bnd/semantic_seg/model_epoch_100.pth')
load_result = seg_model.load_state_dict(cpkt['seg_model'])
print("\n\n")
print(f"{load_result=}")

seg_model.eval()

dataset        = get_dataset(args,bnd=False,bool_mask=False,crop_roi=True)
idx = 33

inputs, targets = dataset[idx]
inputs, targets = inputs.to(device), targets
cnn_out,logits = seg_model(inputs)          # NEW
print(f"{logits.shape}")

mask = targets >=0
mask = mask.numpy()
#targets is the middle slice of the 3d output, so take the middle slice of the output
logits_channel_last =  logits.permute(1,2,3,0) # D,H,W,C
logits_middle_slice = logits_channel_last[int(logits_channel_last.shape[0]//2),:,:,:]
print(f"{logits_middle_slice.shape}")

probs = F.softmax(logits_middle_slice, dim=-1).detach().cpu().numpy()               # softmax over channel K
#apply tv_denosing to probs map
print(f"{probs[:,:,0].shape}")
#%%

for p_idx in range(8):
    denoised_probs = denoise_tv_chambolle(probs[:,:,p_idx], weight=0.1, channel_axis=None)
    #predict via argmax
    denoised_pred  = np.argmax(denoised_probs, axis=-1) + 1         # [B,D,H,W], +1 keeps 0 for ignore
    denoised_pred  = denoised_pred * mask
                # ---------- move to CPU once ----------
    denoised_pred_np  = denoised_pred
    label_np = (targets + 1).cpu().numpy()

    pred  = np.argmax(probs, axis=-1) + 1         # [B,D,H,W], +1 keeps 0 for ignore
    pred  = pred * mask

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(label_np, )
    axes[0].set_title('label')
    axes[1].imshow(probs[:,:,p_idx], )
    axes[1].set_title('Noisy')
    axes[2].imshow(denoised_pred_np)
    axes[2].set_title('TV Denoised')
    plt.show()

# %%
