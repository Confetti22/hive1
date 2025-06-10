#%%
from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from contrastive_train_helper import cos_loss,valid_one_stage,get_t11_eval_data
from config.load_config import load_cfg
from lib.arch.ae import build_final_model
from torchsummary import summary
import time
import os

device ='cuda'
args = load_cfg('config/t11_3d_one_stage.yaml')
E5 = args.e5

avg_pool = 8
args.avg_pool_size = [avg_pool]*3

model = build_final_model(args).to(device)
#out_shape : B*C*D*H*W ---> need to squeeze
print(model)
summary(model,(1,*args.input_size))

build_dataset_fn = getattr(__import__("lib.datasets.{}".format(args.dataset_name), fromlist=["get_dataset"]), "get_dataset")
train_dataset = build_dataset_fn(args)

#%%

num_epochs = 1000 
num_pairs = 2**20
start_epoch = 0 
batch_size = 128 
# shuffle_very_epoch =50 
valid_very_epoch = 50 
n_views = 2
pos_weight_ratio = 2
# raw_img: 4um  feats_map: 16stride --> 64 um resol in feats_map
d_near = 64 

exp_save_dir = 'contrastive_run_t11'
exp_name =f'onestage_avg_{avg_pool}_batch{batch_size}_nview{n_views}_pos_weight_{pos_weight_ratio}_mlp{args.mlp_filters}_d_near{d_near}'

writer = SummaryWriter(log_dir=f'{exp_save_dir}/{exp_name}')

import os
import shutil
script_path = os.path.abspath(__file__)
shutil.copy2(script_path,f"{exp_save_dir}/{exp_name}/script.py")
print(f"config has been saved")


#%%

#load entire array into memory to accelate indexing feats
current = time.time()
# image only extend to 300 at y axis, and only take the half brain at right

dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle= True,drop_last=False)


device= 'cuda'


optimizer = optim.Adam(model.parameters(), lr=0.0005) 

##load the ckpt
# ckpt_pth = '/share/home/shiqiz/workspace/hive/contrastive_run_rm009/batch4096_nview2_pos_weight_2_mlp[96, 48, 24, 12]_d_near1/model_epoch_1199.pth'
# local_ckpt_pth = '/home/confetti/e5_workspace/hive/contrastive_run_rm009/batch4096_nview2_pos_weight_2_mlp[96, 48, 24, 12]_d_near1/model_epoch_1199.pth'
# ckpt_pth = ckpt_pth if E5 else local_ckpt_pth
# ckpt = torch.load(ckpt_pth, map_location='cpu')
# print(ckpt.keys())
# model.load_state_dict(ckpt)
# print("model dict loaded")
# start_epoch = 1200 
#%%
eval_data = get_t11_eval_data(E5) 
for epoch in range(start_epoch,num_epochs): 
    for it, batch in enumerate(dataloader):
        it = epoch * len(dataloader) +it
        batch = torch.cat(batch,dim=0)
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch) 
        out = out.squeeze()
        loss,pos_cos,neg_cos = cos_loss(features=out,batch_size=batch_size,n_views=n_views,pos_weight_ratio=pos_weight_ratio)

        loss.backward() 
        optimizer.step() 

        # lr_scheduler1.step(loss)

        writer.add_scalar('Loss/train', loss.item(), it)
        writer.add_scalar('lr',optimizer.param_groups[0]["lr"], it)
        writer.add_scalar('pos_cos',pos_cos.item(), it)
        writer.add_scalar('neg_cos',neg_cos.item(), it)

    print(f"epoch:  {epoch}, loss: {loss:.4f}, pos_cos:{pos_cos:.4f}, neg_cos:{neg_cos:.4f}, lr:{optimizer.param_groups[0]["lr"]:.7f}")

    if (epoch) % valid_very_epoch ==0: 
        model.eval()
        valid_one_stage(model,epoch,eval_data,writer)
        model.train()

    if (epoch+1) % valid_very_epoch*4 == 0:
        save_dir = f'{exp_save_dir}/{exp_name}'
        model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at epoch {epoch} to {model_path}")

# Optionally, save the final model
final_model_path = os.path.join(save_dir, 'model_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

writer.close()
                        










