#%%
import torch
import torch.nn as nn
from config.load_config import load_cfg
from helper.image_seger import ConvSegHead  
from train_seghead_helper import ComboLoss, compute_class_weights_from_dataset,seg_valid
from tqdm.auto import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg_path = 'config/seghead.yaml'
args = load_cfg(cfg_path)

num_epochs = args.num_epochs
start_epoch = args.start_epoch
batch_size = args.batch_size
valid_very_epoch = args.valid_very_epoch
save_very_epoch = 50 

#%%

from lib.datasets.dataset4seghead import get_dataset, get_valid_dataset
from distance_contrast_helper import HTMLFigureLogger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os


exp_save_dir='outs/seg_head'
exp_name = f"level{args.feats_level}_avg_pool{args.feats_avg_kernel}_fromepoch1000_focal_combo_loss"
model_save_dir = f"{exp_save_dir}/{exp_name}"

start_epoch = args.start_epoch
dataset = get_dataset(args) 
loader = DataLoader(dataset,batch_size=4,drop_last=False,shuffle=True)

valid_dataset = get_valid_dataset(args)
valid_loader = DataLoader(valid_dataset,batch_size=1,drop_last=False,shuffle=False)

writer = SummaryWriter(log_dir=f'{exp_save_dir}/{exp_name}')
img_logger = HTMLFigureLogger(log_dir=f'{exp_save_dir}/{exp_name}',html_name="seg_valid_result.html")
train_img_logger = HTMLFigureLogger(log_dir=f'{exp_save_dir}/{exp_name}',html_name="train_seg_valid_result.html")


C = 12
num_classes = 8
seg_head = ConvSegHead(C, num_classes).to(device)
seg_head.train()
optimizer = torch.optim.Adam(seg_head.parameters(), lr=args.lr_start)
# loss_fn = nn.CrossEntropyLoss()
class_weights = compute_class_weights_from_dataset(dataset, num_classes)
loss_fn = ComboLoss(class_weights=class_weights,focal=True) 


E5 = args.e5 

if E5:
    data_prefix = "/share/home/shiqiz/data"
    workspace_prefix = "/share/home/shiqiz/workspace/hive1"
else:
    data_prefix = "/home/confetti/data"
    workspace_prefix = '/home/confetti/e5_workspace/hive1'

import re
##load the ckpt
if args.re_use:
    ckpt_pth = f"{workspace_prefix}/outs/seg_head/level3_avg_pool7_fromepoch1000_combo_loss/model_epoch_4000.pth" 
    # Extract the epoch number from the filename
    match = re.search(r'model_epoch_(\d+)', ckpt_pth)
    if match:
        start_epoch = int(match.group(1)) + 1
    else:
        raise ValueError("Epoch number not found in checkpoint filename.")

    ckpt = torch.load(ckpt_pth, map_location='cpu')
    print(ckpt.keys())
    seg_head.load_state_dict(ckpt)



if num_classes < 2:
    raise ValueError("Need at least 2 labeled classes in the mask.")


for epoch in tqdm(range(start_epoch,num_epochs)):
    train_loss = []
    for input, label in loader: 
        input = input.to(device)
        label = label.to(device)

        mask = (label >= 0)

        optimizer.zero_grad()
        logits = seg_head(input)  # [K, D, H, W] 

        # Permute logits to [D, H, W, K] 
        logits_flat = logits.permute(0,2,3,4,1)[mask]  # [N, K]
        labels_flat = label[mask]

        loss = loss_fn(logits_flat, labels_flat)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    
    avg_loss = sum(train_loss)/len(train_loss)
    writer.add_scalar('Loss/train',avg_loss,epoch)
    
    print(f"Conv head epoch {epoch}, loss: {avg_loss:.4f}")
    
    if (epoch) % valid_very_epoch ==0: 
        valid_loss = seg_valid(img_logger,valid_loader,seg_head,epoch,device=device, loss_fn=loss_fn)
        writer.add_scalar('Loss/valid',valid_loss,epoch)
        

    if (epoch) % (4*valid_very_epoch) ==0: 
        valid_loss = seg_valid(train_img_logger,valid_loader=loader,seg_head=seg_head,epoch=epoch,device=device, loss_fn=loss_fn)
        
    if (epoch+1) % save_very_epoch ==0:
        model_save_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(seg_head.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch +1} to {model_save_path}")
img_logger.finalize()
train_img_logger.finalize()



    


        
