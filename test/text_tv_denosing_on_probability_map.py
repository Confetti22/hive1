#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
import math, os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from config.load_config import load_cfg
from lib.arch.ae import build_semantic_seg_model
from lib.datasets.dataset4seghead import get_dataset

from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
import numpy as np



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
    



def compute_tv_denoised_pred(probs, denosing_weight =1):
    """
    probs: H*W*C 
    """
    denoised_probs = denoise_tv_chambolle(probs, weight=denosing_weight, channel_axis= -1)
    #predict via argmax
    denoised_pred  = np.argmax(denoised_probs, axis=-1) + 1         # [B,D,H,W], +1 keeps 0 for ignore
    denoised_pred  = denoised_pred * mask
    return denoised_pred


def get_tv_loss_input(probs):
    channel_first_probs = np.moveaxis(probs,-1,0)
    loss_input_probs = channel_first_probs[np.newaxis,:]
    loss_input_probs = torch.tensor(loss_input_probs)
    # print(f"{probs.shape= }")
    # print(f"{loss_input_probs.shape= }")
    return loss_input_probs


args = load_cfg('../config/semisupervised.yaml') 
args.e5 = False 
# args.filters = [32,64,96]

device = 'cpu'
seg_model= build_semantic_seg_model(args).to(device)
seg_model = seg_model

cpkt = torch.load('/home/confetti/e5_workspace/hive1/outs/cortex_semantic_seg4/supervised3/model_epoch_100.pth')
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
print(f"{targets.shape= }")
mask = targets >=0
mask = mask.numpy()
#targets is the middle slice of the 3d output, so take the middle slice of the output
logits_channel_last =  logits.permute(1,2,3,0) # D,H,W,C
logits_middle_slice = logits_channel_last[int(logits_channel_last.shape[0]//2),:,:,:]
print(f"{logits_middle_slice.shape}")

probs = F.softmax(logits_middle_slice, dim=-1).detach().cpu().numpy()  # H*W*C,  softmax over channel K

#apply tv_denosing to probs map
print(f"{probs[:,:,0].shape}")
three_probs = probs[:,:,:3]
#%%
weight = 10
all_channel_denosed_pred = compute_tv_denoised_pred(probs,weight)
three_channel_denosed_pred = compute_tv_denoised_pred(three_probs,weight)


fig, axes = plt.subplots(1, 4, figsize=(15, 5))

label_np = (targets + 1).cpu().numpy()

axes[0].imshow(label_np, )
axes[0].set_title('label')

noisy_pred  = np.argmax(probs, axis=-1) + 1         # [B,D,H,W], +1 keeps 0 for ignore
noisy_pred  = noisy_pred * mask
axes[1].imshow(noisy_pred )
axes[1].set_title('Noisy')

axes[2].imshow(all_channel_denosed_pred)
axes[2].set_title('all channel_TV Denoised')

axes[3].imshow(three_channel_denosed_pred)
axes[3].set_title('three channel_TV Denoised')
plt.show()

all_channel_input = get_tv_loss_input(probs)
three_channel_input = denoise_tv_chambolle(three_probs,weight=weight,channel_axis=-1)
three_channel_input = get_tv_loss_input(three_channel_input)
denoised_probs_input = denoise_tv_chambolle(probs, weight= weight, channel_axis= -1)
denoised_probs_input = get_tv_loss_input(denoised_probs_input)

label_input = get_tv_loss_input(label_np[...,np.newaxis])

#%%
tv_loss_fn= TVLoss(TVLoss_weight=1)
tv_loss_noisy = tv_loss_fn(all_channel_input)
print(f"{tv_loss_noisy= :.6f}")
tv_loss_denoised_all_ch = tv_loss_fn(denoised_probs_input)
print(f"{tv_loss_denoised_all_ch= :.6f}")
tv_loss_denoised_three_ch = tv_loss_fn(three_channel_input)
print(f"{tv_loss_denoised_three_ch= :.6f}")

# %%
