import torch.nn as nn
from torch.utils.data import Dataset
from model_basic import *

import argparse
from yacs.config import CfgNode
from cfg_helper import get_cfg_defaults
import numpy as np
import time
import pickle





def load_cfg(args: argparse.Namespace, merge_cmd=False):
    """Load configurations.
    """
    # Set configurations
    cfg = get_cfg_defaults()
    cfg.set_new_allowed(True)
    cfg_path = args if isinstance(args,str) else args.cfg
    cfg.merge_from_file(cfg_path)
    # if merge_cmd:
    #     cfg.merge_from_list(args.opts)

    # # Overwrite options given configs with higher priority.
    # if args.inference:
    #     update_inference_cfg(cfg)
    # overwrite_cfg(cfg, args)
    # cfg.freeze()
    return cfg


class Encoder(nn.Module):

    block_dict = {
        'single': SingleConv3d,
        'double': DoubleConv3d, 
    }

    def __init__(self,
                 in_channel: int = 1,
                 filters: List[int] = [32,64,96],
                 pad_mode: str = 'reflect',
                 act_mode: str = 'elu',
                 norm_mode: str = 'none',
                 kernel_size:List[int] =[5,3,3],
                 init_mode: str = 'orthogonal',
                 block_type: str = 'single',
                 **kwargs):
        super().__init__()

        self.filters = filters
        self.kernel_size =kernel_size
        self.depth = len(filters)


        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        stride = 2
        padding = int(( int( self.kernel_size[0] ) -1)//2)
        print(f"padding ={padding},k{self.kernel_size[0]}")
        # self.conv_in =nn.Sequential( 
        #     nn.Conv3d(in_channel,int(filters[0]//2),kernel_size=self.kernel_size[0],stride =stride,padding = padding),
        #     conv3d_norm_act(int(filters[0]//2), filters[0], kernel_size=self.kernel_size[0],
        #                                stride=1,padding=padding, **self.shared_kwargs)
        #     )
        self.conv_in =conv3d_norm_act(in_channel, filters[0], kernel_size=self.kernel_size[0],
                                       stride=stride,padding=padding, **self.shared_kwargs)
        # encoding path
        self.down_layers = nn.ModuleList()
        for i in range(self.depth -1):
            next = min(self.depth, i+1)
            kernel_size = self.kernel_size[next]
            stride = 2
            padding = int((kernel_size -1)//2) 

            if block_type == 'single':
                self.down_layers.append(
                  nn.Sequential(
                      conv3d_norm_act(filters[i],filters[next],kernel_size=kernel_size,stride = stride, padding=padding,**self.shared_kwargs),
                      conv3d_norm_act(filters[next],filters[next],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs)
                     )
                )

            elif block_type == 'double':
                self.down_layers.append(
                    nn.Sequential(
                        conv3d_norm_act(filters[i],filters[next],kernel_size=kernel_size,stride = stride, padding=padding,**self.shared_kwargs),
                        conv3d_norm_act(filters[next],filters[next],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs),
                        conv3d_norm_act(filters[next],filters[next],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs)
                         )
                )
            else:
                self.down_layers.append(
                    nn.Sequential(
                        conv3d_norm_act(filters[i],filters[next],kernel_size=kernel_size,stride = stride, padding=padding,**self.shared_kwargs),
                          )
                )
 

        #linear projection for embdding
        self.last_encoder_conv=nn.Conv3d(self.filters[-1],self.filters[-1],kernel_size=1,stride=1)

       
        # initialization
        # model_init(self, mode=init_mode)

    def forward(self, x):
        
        #encoder path
        x = self.conv_in(x)
        for i in range(self.depth-1):
            x = self.down_layers[i](x)
        x = self.last_encoder_conv(x) 
        return x


class TraverseDataset3d(Dataset):
    def __init__(self, img, stride:int, win_size:int, ):

        self.img = img
        print(f"init traverseDataset with img of shape {img.shape},stride = {stride}, win_size = {win_size}")
        self.stride = stride
        self.win_size = win_size

        self.patches = self._generate_patches()

    def _generate_patches(self):

        img = self.img

        # Extract patches
        patches = []
        for z in range(0, img.shape[0] - self.win_size + 1, self.stride):
            for y in range(0, img.shape[1] - self.win_size + 1, self.stride):
                for x in range(0, img.shape[2] - self.win_size + 1, self.stride):
                    patch = img[
                        z:z + self.win_size,
                        y:y + self.win_size,
                        x:x + self.win_size,
                    ]
                    patches.append(patch)
        
        

        self.sample_shape = np.array([ int(item//self.stride) +1 for item in [z,y,x]])
        print(f"sample shape = {self.sample_shape}")

        return patches
    def _get_sample_shape(self):
        return self.sample_shape

    def __len__(self):
        return len(self.patches)
    

    def __getitem__(self, idx):
        #TODO: synchronize the preprocess method used in training, right now, did not apply any norm or clip
        # Preprocess and resize the patch
        patch = self.patches[idx]
        # preprocess = v2.Compose([
        #     v2.Resize(size=self.net_input_shape),
        # ])
        patch = torch.tensor(patch,dtype=torch.float32)
        # patch = preprocess(patch)
        patch = torch.unsqueeze(patch,0)
        return patch

def get_feature_list(device,cnn_encoder,mlp,test_loader,save_path=None)->np.ndarray:
    """
    encoder inference on a single input 2d-image
    
    input(numpy)--> test_dataset
    collect feats during inference
    return the feats as shape of N*n_dim

    """
    print(f"device is {device}")

    feats_list=[]
    for i, imgs in enumerate(test_loader):
        outs= cnn_encoder(imgs.to(device)) #B*C*1
        print(f"after cnn: outs.shape {outs.shape}")
        outs = mlp(outs.reshape(outs.shape[0],-1)) #B * C
        print(f"after mlp: outs.shape {outs.shape}")
        feats_list.append(outs.cpu().detach().numpy().reshape(outs.shape[0],-1))
    
    current  = time.time()
    feats_array = np.concatenate([ arr for arr in feats_list], axis=0)
    print(f"concatenating consuming time is {time.time() - current :5f}")
    print(f"fests_arry shape {feats_array.shape}")

    if save_path :
        with open(save_path, 'wb') as file:
            pickle.dump(feats_array, file)
    
    return feats_array
