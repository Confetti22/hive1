import torch
from glob import glob
import re
import os

def modify_key(weight_dict,source,target):
    new_weight_dict = {}
    for key, value in weight_dict.items():
        new_key = key.replace(source,target)
        new_weight_dict[new_key] = value
    return new_weight_dict

def delete_key(weight_dict,pattern_lst):
    new_weight_dict = {k: v for k, v in weight_dict.items() if not k.startswith(pattern_lst)}
    return new_weight_dict 

def load_encoder_dict(model,exp_name):
    ckpts = sorted(glob(f'out/weights/{exp_name}/Epoch_*.pth'))
    ckpts = sorted(ckpts,key=lambda x: int(re.search(r'Epoch_(\d+).pth', os.path.basename(x)).group(1)))
    ckpt = torch.load(ckpts[-1])
    removed_module_dict = modify_key(ckpt['model'],source='module.',target='')
    deleted_unwanted_dict = delete_key(removed_module_dict,('fc1', 'fc2','contrastive_projt','up_layers','conv_out'))

    model.encoder.load_state_dict(deleted_unwanted_dict,strict=False)

def load_cnn_encoder_dict(model,exp_name):
    ckpts = sorted(glob(f'out/weights/{exp_name}/Epoch_*.pth'))
    ckpts = sorted(ckpts,key=lambda x: int(re.search(r'Epoch_(\d+).pth', os.path.basename(x)).group(1)))
    ckpt = torch.load(ckpts[-1])
    removed_module_dict = modify_key(ckpt['model'],source='module.',target='')
    deleted_unwanted_dict = delete_key(removed_module_dict,('fc1', 'fc2','contrastive_projt','up_layers','conv_out'))

    model.encoder.load_state_dict(deleted_unwanted_dict,strict=False)

def load_cnn_encoder_dict(model,exp_name):
    ckpts = sorted(glob(f'autoencoder_out/weights/{exp_name}/Epoch_*.pth'))
    ckpts = sorted(ckpts,key=lambda x: int(re.search(r'Epoch_(\d+).pth', os.path.basename(x)).group(1)))
    ckpt = torch.load(ckpts[-1])
    removed_module_dict = modify_key(ckpt['model'],source='module.',target='')
    deleted_unwanted_dict = delete_key(removed_module_dict,('fc1', 'fc2','contrastive_projt','up_layers','conv_out'))

    model.cnn.load_state_dict(deleted_unwanted_dict,strict=False)

