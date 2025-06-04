
#%%

from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from contrastive_train_helper import cos_loss,valid,get_eval_data, MLP,Contrastive_dataset_3d
from config.load_config import load_cfg
import time
import os

device ='cuda'
args = load_cfg('config/rm009.yaml')

num_epochs = 30000 
num_pairs = 8192
start_epoch = 0 
batch_size = 4096 
shuffle_very_epoch =50 
valid_very_epoch = 50 
n_views = 2
pos_weight_ratio = 2
d_near = 64 #um

exp_save_dir = 'contrastive_run_rm009'
exp_name =f'batch{batch_size}_nview{n_views}_pos_weight_{pos_weight_ratio}'

writer = SummaryWriter(log_dir=f'{exp_save_dir}/{exp_name}')

#test whether pin zarr in memory will be faster
device = 'cuda'
model = MLP(args.mlp_filters).to(device)

E5 = False 
if E5:
    zarr_path=""
else:
    zarr_path = f"/home/confetti/data/rm009/half_brain_cnn_feats_r0.zarr"

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

##load the ckpt
# ckpt_pth ='' 
# ckpt = torch.load(ckpt_pth, map_location='cpu')
# print(ckpt.keys())
# model.load_state_dict(ckpt)
# print("model dict loaded")
# start_epoch = 22200 
#%%
# eval_data = get_eval_data()
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

    # if (epoch) % valid_very_epoch ==0: 
    #     model.eval()
    #     valid(model,epoch,eval_data,writer)
    #     model.train()

    if (epoch+1) % shuffle_very_epoch ==0:
        dataset = Contrastive_dataset_3d(feats_map,d_near=d_near,num_pairs=num_pairs,n_view=n_views,verbose= False)
        dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,drop_last=False)
            # Save the model every 1000 epochs

    if (epoch+1) % valid_very_epoch*4 == 0:
        save_dir = f'{exp_save_dir}/{exp_name}'
        model_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at epoch {epoch} to {model_path}")

# Optionally, save the final model
final_model_path = os.path.join(save_dir, 'model_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

writer.close()
                        
    














# %%
