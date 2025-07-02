
#%%

from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from helper.contrastive_train_helper import cos_loss_topk,valid_from_roi,get_rm009_eval_data, MLP,Contrastive_dataset_3d
from config.load_config import load_cfg
from lib.arch.ae import build_final_model,load_compose_encoder_dict
import time
import os
import shutil

device ='cuda'
cfg_path = 'config/rm009.yaml'
args = load_cfg(cfg_path)

avg_pool = 8
args.avg_pool_size = [avg_pool]*3

num_epochs = args.num_epochs
num_pairs = args.num_pairs
start_epoch = args.start_epoch
batch_size = args.batch_size
shuffle_very_epoch = args.shuffle_very_epoch
valid_very_epoch = args.valid_very_epoch
save_very_epoch = 10
n_views = args.n_views
pos_weight_ratio = args.pos_weight_ratio
# raw_img: 4um  feats_map: 16stride --> 64 um resol in feats_map
d_near = args.d_near

exp_save_dir = args.exp_save_dir
only_pos = True 

exp_name =f'__postopk_{avg_pool}_numparis{num_pairs}_batch{batch_size}_nview{n_views}_d_near{d_near}_shuffle{shuffle_very_epoch}'

writer = SummaryWriter(log_dir=f'{exp_save_dir}/{exp_name}')
model_save_dir = f'{exp_save_dir}/{exp_name}'

shutil.copy2(cfg_path,f"{exp_save_dir}/{exp_name}/cfg.yaml")
print(f"config has been saved")

model = MLP(args.mlp_filters).to(device)

E5 = args.e5 

if E5:
    data_prefix = "/share/home/shiqiz/data"
    workspace_prefix = "/share/home/shiqiz/workspace/hive1"
else:
    data_prefix = "/home/confetti/data"
    workspace_prefix = '/home/confetti/e5_workspace/hive1'


feats_name ='half_brain_cnn_feats_avg8_r0.zarr'

zarr_path=f"{data_prefix}/rm009/{feats_name}"

#%%
import zarr
z_arr = zarr.open_array(zarr_path,mode='a')

#load entire array into memory to accelate indexing feats
current = time.time()
# image only extend to 300 at y axis, and only take the half brain at right
feats_map = z_arr
D,H,W,C = feats_map.shape
print(f"read feats_map of shape{feats_map.shape} from zarr consume {time.time() -current}")

dataset = Contrastive_dataset_3d(feats_map,d_near=d_near,num_pairs=num_pairs,n_view=n_views,verbose= False)
dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle= True,drop_last=False)


device= 'cuda'


optimizer = optim.Adam(model.parameters(), lr=0.0005) 

import re
##load the ckpt
if args.re_use:
    ckpt_pth = f"{workspace_prefix}/contrastive_run_rm009/fixed_pro_loss_mean/model_epoch_5.pth" 
    # Extract the epoch number from the filename
    match = re.search(r'model_epoch_(\d+)', ckpt_pth)
    if match:
        start_epoch = int(match.group(1)) + 1
    else:
        raise ValueError("Epoch number not found in checkpoint filename.")

    ckpt = torch.load(ckpt_pth, map_location='cpu')
    print(ckpt.keys())
    res = model.load_state_dict(ckpt)
    print()
#%%
eval_data = get_rm009_eval_data(E5=E5)

cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)


cnn_ckpt_pth = f'{data_prefix}/weights/rm009_3d_ae_best.pth'

load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_weight_dict=model.state_dict(),dims=args.dims)
valid_from_roi(cmpsd_model,0,eval_data,writer)

for epoch in range(start_epoch,num_epochs): 
    for batch_idx, batch in enumerate(dataloader):
        it = epoch * len(dataloader) +batch_idx
        batch = torch.cat(batch,dim=0)
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch) 
        out = out.squeeze()
        loss,pos_cos,neg_cos = cos_loss_topk(features=out,n_views=n_views,pos_weight_ratio=pos_weight_ratio,only_pos=only_pos)

        loss.backward() 
        optimizer.step() 

        # lr_scheduler1.step(loss)

        writer.add_scalar('Loss/train', loss.item(), it)
        writer.add_scalar('lr',optimizer.param_groups[0]["lr"], it)
        writer.add_scalar('pos_cos',pos_cos.item(), it)
        writer.add_scalar('neg_cos',neg_cos.item(), it)

        print(f"epoch|batch:{epoch}|{batch_idx}, loss: {loss:.4f}, pos_cos:{pos_cos:.4f}, neg_cos:{neg_cos:.4f}, lr:{optimizer.param_groups[0]["lr"]:.7f}")

    if (epoch) % valid_very_epoch ==0: 
        model.eval()
        load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_weight_dict=model.state_dict(),dims=args.dims)
        valid_from_roi(cmpsd_model,epoch,eval_data,writer)
        model.train()

    if (epoch+1) % shuffle_very_epoch ==0:
        dataset = Contrastive_dataset_3d(feats_map,d_near=d_near,num_pairs=num_pairs,n_view=n_views,verbose= False)
        dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,drop_last=False)
    # Save the model every 1000 epochs

    if (epoch+1) % save_very_epoch == 0:
        model_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at epoch {epoch} to {model_path}")

# Optionally, save the final model
final_model_path = os.path.join(model_save_dir, 'model_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

writer.close()
                        
    














# %%
