# Copyright (c) Ramy Mounir.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from lib.utils.file import checkdir
from lib.utils.tensorboard import get_writer, TBWriter
from lib.utils.distributed import MetricLogger
from glob import glob
import math
import numpy as np
import tifffile as tif
import sys
import re
from tqdm.auto import tqdm

import os

import torch

def get_three_slice(x):
    radius =int(x.shape[-1]//2)
    x_x = x[:,:,radius]
    x_y = x[:,radius,:]
    x_z = x[radius,:,:]
    return x_x, x_y, x_z

def unnormalize(img):
    clip_low = 96
    clip_high = 2672
    return img *(clip_high - clip_low) + clip_low

class Trainer:

    def __init__(self, args, train_loader, valid_loader,model, loss, optimizer):

        self.args = args
        self.train_gen = train_loader
        self.valid_gen = valid_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.fp16_scaler = torch.GradScaler('cuda') if args.fp16 else None

        self.recon_img_dir = "{}/recon_img/{}".format(args.out, args.exp_name)
        self.valid_recon_img_dir= "{}/valid_recon_img/{}".format(args.out, args.exp_name)
        os.makedirs(self.recon_img_dir,exist_ok=True)
        os.makedirs(self.valid_recon_img_dir,exist_ok=True)

        # === TB writers === #
        if self.args.main:	

            self.writer = get_writer(args)

            self.lr_sched_writer = TBWriter(self.writer, 'scalar', 'Schedules/Learning Rate')			
            self.loss_writer = TBWriter(self.writer, 'scalar', 'Loss/total')
            self.valid_loss_writer = TBWriter(self.writer, 'scalar', 'valid:Loss/total')

            checkdir("{}/weights/{}/".format(args.out, self.args.exp_name), args.reset)


    def train_one_epoch(self, epoch, lr_schedule, save_recon_img_flag, MSE_loss:True):
        self.model.train()


        metric_logger = MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, self.args.epoch_num)

        for it, input_data in enumerate(metric_logger.log_every(self.train_gen, 1, header)):

            # === Global Iteration === #
            it = len(self.train_gen) * epoch + it

            # === Inputs === #
            input_data = input_data.cuda(non_blocking=True) 

            # === Forward pass === #
            if self.args.fp16:
                train_type=torch.float16
            else:
                train_type = torch.float32
            with torch.autocast('cuda',dtype=train_type):
                preds = self.model(input_data)
                loss = self.loss(preds, input_data)

            # Sanity Check
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
            
            # === Backward pass === #
            self.model.zero_grad()
            # for mix precision backward propogation
            if self.args.fp16:
                self.fp16_scaler.scale(loss).backward()
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()
            
            loss.backward()
            
            self.optimizer.step()

            # === Logging === #
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=lr_schedule.get_last_lr()[0])

            if self.args.main:
                self.loss_writer(metric_logger.meters['loss'].value, it)
                self.lr_sched_writer(self.optimizer.param_groups[0]["lr"], it)

                if save_recon_img_flag:
                    preds = preds.detach().cpu().numpy()
                    preds = np.squeeze(preds)

                    input_data = input_data.detach().cpu().numpy()
                    input_data = np.squeeze(input_data)

                    valid_sample_idxes = [0,1,2]
                    for id in valid_sample_idxes:
                        x = input_data[id]
                        re_x = preds[id]

                        x_name = f"{epoch:04d}_{id}_x.tif"
                        re_x_name = f"{epoch:04d}_{id}_re_x.tif"
                        tif.imwrite(os.path.join(self.recon_img_dir,x_name) , x)
                        tif.imwrite(os.path.join(self.recon_img_dir,re_x_name) , re_x)

                        three_dims = (len(preds.shape) ==4)
                        if three_dims:
                            x_x,x_y,x_z=get_three_slice(x)
                            re_x_x, re_x_y, re_x_z = get_three_slice(re_x)
                            merged_x = np.concatenate((x_x, x_y, x_z), axis=1)  
                            merged_re_x = np.concatenate((re_x_x, re_x_y, re_x_z), axis=1)  
                            merged = np.concatenate((merged_x,merged_re_x), axis=0)
                        else:
                            merged = np.concatenate((x,re_x),axis =0)
                        merged = (merged - merged.min()) / (merged.max() - merged.min())

                        self.writer.add_image('x and re_x ',merged,it,dataformats='HW')
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
    
    def valid(self,epoch):
        self.model.eval()

        valid_loss = []
        input_images = []
        pred_images = []

        for input_data in tqdm(self.valid_gen):

            input_data = input_data.to('cuda')

            with torch.no_grad():
                preds = self.model(input_data)

            loss = self.loss(preds, input_data)
            valid_loss.append(loss.item())

            preds = preds.detach().cpu().numpy()
            preds = np.squeeze(preds)
            pred_images.append(preds)
            input_data = input_data.detach().cpu().numpy()
            input_data = np.squeeze(input_data)
            input_images.append(input_data)
            
        valid_loss = sum(valid_loss) / len(valid_loss)
        # === Logging === #
        torch.cuda.synchronize()

        self.valid_loss_writer(valid_loss, epoch)
        input_images = np.concatenate(input_images,axis=0)
        pred_images = np.concatenate(pred_images,axis=0)

        
        three_dims = (len(preds.shape) ==4)
        #B*D*H*W
        x_slices = []
        re_x_slices = []
        for idx in range(3):
            x = input_images[idx]
            re_x = pred_images[idx]

            x_name = f"{epoch:04d}_{idx:02d}_x.tif"
            re_x_name = f"{epoch:04d}_{idx:02d}_re_x.tif"
            tif.imwrite(os.path.join(self.valid_recon_img_dir,x_name) , x)
            tif.imwrite(os.path.join(self.valid_recon_img_dir,re_x_name) , re_x)
            if three_dims:
                x_slices.append(x[int(x.shape[0]//2),:,:])
                re_x_slices.append(re_x[int(re_x.shape[0]//2),:,:])
            else:
                x_slices.append(x)
                re_x_slices.append(re_x)
        x_slices = np.concatenate(x_slices,axis=1)
        re_x_slices = np.concatenate(re_x_slices,axis=1)

        merged = np.concatenate((x_slices,re_x_slices), axis=0)
        merged = (merged - merged.min()) / (merged.max() - merged.min())
        self.writer.add_image('valid: x and re_x',merged,epoch,dataformats='HW')






    def fit(self):

        # === Resume === #
        self.load_if_available()


        from lib.core.scheduler import get_scheduler 
        lr_schedule= get_scheduler(self.optimizer, self.args)

        # === training loop === #
        for epoch in range(self.start_epoch, self.args.epoch_num):

            self.train_gen.sampler.set_epoch(epoch)


            save_recon_img_flag = ( (epoch ) %self.args.save_every==0)
            self.train_one_epoch(epoch, lr_schedule,save_recon_img_flag,MSE_loss=True)
            lr_schedule.step()

            # === eval and save model === #
            if self.args.main and (epoch )% self.args.save_every == 0:
                self.valid(epoch)
                self.save(epoch)
            

    def load_if_available(self):

        ckpts = sorted(glob(f'{self.args.out}/weights/{self.args.exp_name}/Epoch_*.pth'))

        if len(ckpts) >0:
            ckpts = sorted(
                    ckpts,
                    key=lambda x: int(re.search(r'Epoch_(\d+).pth', os.path.basename(x)).group(1))
                    )
            ckpt = torch.load(ckpts[-1], map_location='cpu')
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            print("Loaded ckpt: ", ckpts[-1])

        else:
            self.start_epoch = 0
            print("Starting from scratch")


    def save(self, epoch):

        state = dict(epoch=epoch+1, 
                            model=self.model.state_dict(), 
                            optimizer=self.optimizer.state_dict(),
                        )

        torch.save(state, "{}/weights/{}/Epoch_{}.pth".format(self.args.out, self.args.exp_name, str(epoch+1).zfill(3) ))
