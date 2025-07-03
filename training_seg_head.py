#%%
import time
import zarr
import numpy as np
import napari
from magicgui import widgets
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.load_config import load_cfg
from helper.image_seger import ConvSegHead  
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg_path = 'config/seghead.yaml'
args = load_cfg(cfg_path)

num_epochs = args.num_epochs
start_epoch = args.start_epoch
batch_size = args.batch_size
valid_very_epoch = args.valid_very_epoch
save_very_epoch = 100 

#%%

from lib.datasets.dataset4seghead import get_dataset, get_valid_dataset
from distance_contrast_helper import HTMLFigureLogger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt


exp_save_dir='seg_head'
exp_name = f"test_level{args.feats_level}_avg_pool{args.feats_avg_kernel}"
model_save_dir = f"{exp_save_dir}/{exp_name}"

dataset = get_dataset(args) 
loader = DataLoader(dataset,batch_size=1,drop_last=False,shuffle=True)

valid_dataset = get_valid_dataset(args)
valid_loader = DataLoader(valid_dataset,batch_size=1,drop_last=False,shuffle=False)

writer = SummaryWriter(log_dir=f'{exp_save_dir}/{exp_name}')
img_logger = HTMLFigureLogger(log_dir=f'{exp_save_dir}/{exp_name}',html_name="seg_valid_result.html")


C = 12
num_classes = 8
seg_head = ConvSegHead(C, num_classes).to(device)
seg_head.train()
optimizer = torch.optim.Adam(seg_head.parameters(), lr=args.lr_start)
loss_fn = nn.CrossEntropyLoss()



if num_classes < 2:
    raise ValueError("Need at least 2 labeled classes in the mask.")

for epoch in tqdm(range(num_epochs)):
    for input, label in loader: 
        input = input.to(device)
        label = label.to(device)

        mask = (label >= 0)

        optimizer.zero_grad()
        logits = seg_head(input)  # [K, D, H, W] 

        # Permute logits to [D, H, W, K] 
        logits_flat = logits.permute(*range(1, logits.ndim), 0)[mask]  # [N, K]
        labels_flat = label[mask]

        loss = loss_fn(logits_flat, labels_flat)
        loss.backward()
        optimizer.step()
    
    print(f"Conv head epoch {epoch}, loss: {loss.item():.4f}")
    
    if (epoch) % valid_very_epoch ==0: 
        seg_valid(img_logger,writer,valid_loader,seg_head,epoch)
        

    if (epoch+1) % save_very_epoch ==0:
        model_save_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(seg_head.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch +1} to {model_save_path}")


def seg_valid(img_logger,writer,valid_loader,seg_head,epoch):
    seg_head.eval()

    valid_loss = []
    gt_maskes = []
    pred_maskes = []
    for input,label in tqdm(valid_loader):
        input = input.to(device)
        label = label.to(device)
        logits = seg_head(input)

        mask = (label >= 0)
        logits_flat = logits.permute(*range(1, logits.ndim), 0)[mask]  # [N, K]
        labels_flat = label[mask]
        loss = loss_fn(logits_flat, labels_flat)
        valid_loss.append(loss.item())

        probs = F.softmax(logits,dim=0)
        pred = torch.argmax(probs,dim=0)+1
        pred = pred.detach().cpu().numpy()
        gt_maskes.append(label.detach().cpu().numpy()[int(label.shape[0]//2),:])
        pred_maskes.append(pred.detach().cpu().numpy()[int(pred.shape[0]//2),:])
    

    num_classes = 8 
    cmap = plt.get_cmap('nipy_spectral', num_classes)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))  # 2 rows, 3 columns

    # Plot lista images in row 1 (axes[0])
    for i, img in enumerate(gt_maskes):
        axes[0, i].imshow(img, cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[0, i].axis('off')

    # Plot listb images in row 2 (axes[1])
    for i, img in enumerate(pred_maskes):
        axes[1, i].imshow(img, cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[1, i].axis('off')
    plt.show()

    img_logger.add_figure('gt/pred',fig,global_step = epoch)

    valid_loss = sum(valid_loss)/len(valid_loss)
    writer.add_scalar('Loss/valid',valid_loss,epoch)


    


        
