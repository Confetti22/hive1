
#%%

from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from distance_contrast_helper import simple_eval, HTMLFigureLogger
from helper.contrastive_train_helper import cos_loss,cos_loss_topk,valid_from_roi,get_t11_eval_data, MLP,Contrastive_dataset_3d
from config.load_config import load_cfg
from lib.arch.ae import build_final_model,load_compose_encoder_dict
import time
import os
import shutil
import pickle




#%%
device ='cuda'
cfg_path = 'config/rm009.yaml'
args = load_cfg(cfg_path)

num_epochs = args.num_epochs
num_pairs = args.num_pairs
start_epoch = args.start_epoch
batch_size = args.batch_size
shuffle_very_epoch = args.shuffle_very_epoch
valid_very_epoch = args.valid_very_epoch
save_very_epoch = 100 
n_views = args.n_views
pos_weight_ratio = args.pos_weight_ratio
# raw_img: 4um  feats_map: 16stride --> 64 um resol in feats_map
d_near = args.d_near

exp_save_dir = args.exp_save_dir

pool_key =8
level_key = 'l2'

reduce_factor ={'l1':2,'l2':4,'l3':8}
filters_map={'l1':[32,24,12,12],'l2':[64,32,24,12],'l3':[96,64,32,12]}
args.mlp_filters = filters_map[level_key]



exp_name = f"postopk_neview{n_views}_level_{level_key}_pool_kernel{pool_key}_shuffle{shuffle_very_epoch}_batch{batch_size}"
writer = SummaryWriter(log_dir=f'{exp_save_dir}/{exp_name}')
model_save_dir = f'{exp_save_dir}/{exp_name}'

shutil.copy2(cfg_path,f"{exp_save_dir}/{exp_name}/cfg.yaml")
print(f"config has been saved")

model = MLP(filters_map[level_key]).to(device)

E5 = args.e5 

if E5:
    data_prefix = "/share/home/shiqiz/data"
    workspace_prefix = "/share/home/shiqiz/workspace/hive1"
else:
    data_prefix = "/home/confetti/data"
    workspace_prefix = '/home/confetti/e5_workspace/hive1'

with open(f"{data_prefix}/rm009/small_multiscale_zslice_feats_map.pkl",'rb')as f:
    middle_zslice_feats_map  = pickle.load(f)

for key,k_dict in middle_zslice_feats_map.items():
    print(k_dict['l1'].shape)
    print(k_dict['l2'].shape)
    print(k_dict['l3'].shape, "\n")

#%%
device= 'cuda'
feats_map = middle_zslice_feats_map[pool_key][level_key]
C,H,W= feats_map.shape
feats_map = np.moveaxis(feats_map,0,-1) # for contrastive dataset

img_shape= (H,W)
print(H,W)
features = feats_map.reshape(-1,C) #for eval
features = torch.from_numpy(features).float().to(device)
print(f"{features.shape=}")

dataset = Contrastive_dataset_3d(feats_map,d_near=d_near,num_pairs=num_pairs,n_view=n_views,verbose= False,margin=4)
dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle= True,drop_last=False)

optimizer = optim.Adam(model.parameters(), lr=0.0005) 

import re
##load the ckpt
if args.re_use:
    ckpt_pth = f"{workspace_prefix}/outs/contrastive_run_rm009/postopk_neview4_level_l2_pool_kernel8_shuffle50_batch4096/model_epoch_1900.pth" 
    # Extract the epoch number from the filename
    match = re.search(r'model_epoch_(\d+)', ckpt_pth)
    if match:
        start_epoch = int(match.group(1)) + 1
    else:
        raise ValueError("Epoch number not found in checkpoint filename.")

    ckpt = torch.load(ckpt_pth, map_location='cpu')
    print(ckpt.keys())
    model.load_state_dict(ckpt)
    
#%%
img_logger = HTMLFigureLogger(log_dir=f'{exp_save_dir}/{exp_name}',html_name="pca_plot.html",comment=f'shuffle_every_{shuffle_very_epoch}')

for epoch in range(start_epoch,num_epochs): 
    for batch_idx, batch in enumerate(dataloader):
        batch = torch.cat(batch,dim=0)
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch) 
        out = out.squeeze()
        loss,pos_cos,neg_cos = cos_loss_topk(features=out,n_views=n_views,pos_weight_ratio=pos_weight_ratio,only_pos=  True)
        # loss,pos_cos,neg_cos = cos_loss(features=out,n_views=n_views,pos_weight_ratio=pos_weight_ratio)

        loss.backward() 
        optimizer.step() 

        # lr_scheduler1.step(loss)


        print(f"epoch|batch:{epoch}|{batch_idx}, loss: {loss:.4f}, pos_cos:{pos_cos:.4f}, neg_cos:{neg_cos:.4f}, lr:{optimizer.param_groups[0]["lr"]:.7f}")
    
    #record loss each epoch
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('lr',optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar('pos_cos',pos_cos.item(), epoch)
    writer.add_scalar('neg_cos',neg_cos.item(), epoch)


    if (epoch) % valid_very_epoch ==0: 
        model.eval()
        eval_out = model(features)
        simple_eval(eval_out,epoch+1,img_logger,writer,img_shape=img_shape,tag=f'pca')
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

img_logger.finalize()

final_model_path = os.path.join(model_save_dir, 'model_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

writer.close()
                        
    














# %%
