#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

"""
input: ae weight dict trained using dl template
output: simplified encoder weight dict 
"""
from lib.arch.ae import modify_key,delete_key
import torch

ckpt_pth ='/home/confetti/data/weights/t11_3d_ae_best2.pth'
ckpt = torch.load(ckpt_pth)
print(ckpt.keys())
#%%
removed_module_dict = modify_key(ckpt['model'],source='module.',target='')
print(removed_module_dict.keys())

# %%

from lib.arch.ae import modify_key,delete_key
removed_decoder_dict = delete_key(removed_module_dict,pattern_lst=('up_layers','conv_out'))
print(removed_decoder_dict.keys())
# %%
torch.save(removed_decoder_dict,ckpt_pth)
# %%
ckpt = torch.load(ckpt_pth)
print(ckpt.keys())
# %%
