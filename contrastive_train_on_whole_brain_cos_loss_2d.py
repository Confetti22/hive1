#%%
from tqdm.auto import tqdm
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from contrastive_train_helper_2d import cos_loss,valid,get_eval_data, MLP, Contrastive_dataset 
import os
#test whether pin zarr in memory will be faster


num_epochs = 30000 
start_epoch = 0 
batch_size = 4096 
valid_very_epoch = 50 
n_views = 2
pos_weight_ratio = 2
d_near = 16 #um
avg_pool='None'
filters=[24,18,12,8]

exp_save_dir = 'contrastive_run_vsi'
exp_name =f'avg{avg_pool}_batch{batch_size}_nview{n_views}_pos_weight_{pos_weight_ratio}_mlp{filters}'
raw_feats_name=f'ae_fmap_vsi_train_avg{avg_pool}.dat'
writer = SummaryWriter(log_dir=f'{exp_save_dir}/{exp_name}')

device = 'cuda'
model = MLP(filters=filters).to(device)


E5 = False 
if E5:
    raw_feats_pth=f"/home/confetti/data/wide_filed/{raw_feats_name}"
else:
    raw_feats_pth=f"/home/confetti/data/wide_filed/{raw_feats_name}"


fp = np.memmap(raw_feats_pth, dtype='float32', mode='r+', shape=(8322, 24, 64,64))

dataset = Contrastive_dataset(fp,d_near=d_near,n_view=n_views,verbose= True)
dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle= True,drop_last= True)


device= 'cuda'

optimizer = optim.Adam(model.parameters(), lr=0.0005) 
# ckpt_path = '/home/confetti/e5_workspace/hive/contrastive_run_vsi/avg5_batch2048_nview2_pos_weight_2/model_final.pth'
# ckpt = torch.load(ckpt_path,map_location='cpu')
# model.load_state_dict(ckpt)


#%%
eval_data = get_eval_data()
for epoch in range(start_epoch,num_epochs): 
    for it, batch in enumerate(dataloader):
        it = epoch * len(dataloader) +it

        # for lr_scheduler2
        # for i, param_group in enumerate(optimizer.param_groups):
        #         param_group["lr"] = lr_scheduler2[it]

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
        valid(model,epoch,eval_data,writer)
        model.train()

    if (epoch+1) % valid_very_epoch*4 == 0:
        save_dir = f"{exp_save_dir}/{exp_name}"
        model_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at epoch {epoch} to {model_path}")

# Optionally, save the final model
final_model_path = os.path.join(save_dir, 'model_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

writer.close()
                        
    














# %%
