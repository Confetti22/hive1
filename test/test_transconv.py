#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
from lib.arch.ae import EncoderND_1,DecoderND_1
from torchsummary import summary
import torch
#%%

encoder1 = EncoderND_1(1,[16,32,64],[3,3,3],dimension=3,block_type='single',norm_mode='none',downsample_strategy='conv_stride')
# print(encoder1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder1 = encoder1.to(device)  # Move model to GPU
summary(encoder1,(1,32,1024,1024))
#%%
encoder2 = EncoderND_1(1,[16,32,64],[5,3,3],dimension=3,block_type=None,norm_mode='none',downsample_strategy='max_pool')
print(encoder2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder2 = encoder2.to(device)  # Move model to GPU
summary(encoder2,(1,256,256,256))
#%%
decoder2= DecoderND_1(12,[16,32,64],[3,3,3],dims=3,block_type='single',norm_mode='none',output_padding=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder2= decoder2.to(device)  # Move model to GPU
print(decoder2)
summary(decoder2,(64,3,127,127))
# %%
