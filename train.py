#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:42:24 2017

@author: ajaver
"""
import pdb
                            
import os
import tqdm
import torch
import shutil
import datetime
import math
from flow import ImgFlowSplitted
from unet import UNet, _crop, unet_loss
from unet_dilated import UNetDilated

from tensorboardX import SummaryWriter

from torch.utils.data import Dataset, DataLoader

from paths import RESULTS_DIR

import torchvision.utils as vutils

class TorchDataset(ImgFlowSplitted, Dataset):
    def __init__(self, **argkws):
        ImgFlowSplitted.__init__(self, **argkws)
        
    def __getitem__(self, index):
        dat = ImgFlowSplitted.__getitem__(self, index)
        
        
        dat = tuple(torch.from_numpy(x) for x in dat)
        return dat
    
    def __len__(self):
        return ImgFlowSplitted.__len__(self)
    
def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)

def main(is_tiny = True, 
         cuda_id = 0,
         n_epochs = 1000,
        batch_size = 1,
        num_workers = 1,
        lr=1e-4,
        model_name = 'unet'
        ):
    
    image2save = 4
    batchs2save = math.ceil(image2save/batch_size)
    
    
    
    if model_name == 'unet':
        model = UNet()
    elif model_name == 'unet_dilated':
        model = UNetDilated()
    else:
        raise ValueError('Invalid model name {}'.format(model_name))

    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
        print(dev_str)
        device = torch.device(dev_str)
    else:
        device = torch.device('cpu')

    
    gen_imgs = TorchDataset(test_split = 0.05,
                               pad_size = 32,
                               add_cnt_weights = False
                               )
    
    gen = DataLoader(gen_imgs, 
                     batch_size = batch_size,
                     num_workers = num_workers)
    
    log_dir_root =  os.path.join(RESULTS_DIR, 'logs')
    if is_tiny:
        print("It's me, tiny-log!!!")
        log_dir_root =  os.path.join(RESULTS_DIR, 'tiny_log')
    
    now = datetime.datetime.now()
    bn = now.strftime('%Y%m%d_%H%M%S') + '_' + model_name
    if is_tiny:
        bn += '_tiny'

    bn = '{}_lr{}_batch{}'.format(bn, lr, batch_size)
    print(bn)

    log_dir = os.path.join(log_dir_root, bn)
    logger = SummaryWriter(log_dir = log_dir)
    
    criterion = unet_loss
    
    model = model.to(device)
       
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    best_loss = 1e10
    n_iter = 0
    image_train = []
    
    for epoch in range(n_epochs):
        train_avg_loss = 0
        
        model.train()
        gen_imgs.train()
        if is_tiny:
            gen_imgs.tiny()
            gen_imgs._is_transform = True
        
        pbar_train = tqdm.tqdm(gen)
        for X, target in pbar_train:
            X = X.to(device)
            target =  target.to(device)
            
            pred = model(X)
            target_cropped = _crop(pred, target)
            loss = criterion(pred, target_cropped)
            
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step() 
        
            if n_iter % 500 <= batchs2save:
                if len(image_train) < batchs2save:
                    dd = X.cpu(), target[:, 0].cpu()[:, None, :, :], pred[:, 0].cpu()[:, None, :, :]
                    image_train.append(dd)
                else:
                    xs = torch.cat([torch.cat(x) for x in image_train])
                    xs = vutils.make_grid(xs, nrow = 3, normalize=True, scale_each = True)
                    logger.add_image('train', xs, n_iter)
                    
                    image_train = []
            
            desc = 'Train Epoch {} , loss={}'.format(epoch+1, loss.item())
            pbar_train.set_description(desc = desc, refresh=False)

            #I prefer to add a point at each iteration since hte epochs are very large
            logger.add_scalar('train_iter_loss', loss.item(), n_iter)
            n_iter+= 1
            
            train_avg_loss += loss.item()
        
        train_avg_loss /= len(gen)
        logger.add_scalar('train_epoch_loss', train_avg_loss, epoch)
        #%%
        test_avg_loss = 0
        with torch.no_grad(): 
            model.eval()
            gen_imgs.test()
            if is_tiny:
                gen_imgs.tiny()
                gen_imgs._is_transform = False
            
            image_test = []
            pbar_test = tqdm.tqdm(gen)
            for X, target in pbar_test:
                X = X.to(device)
                target =  target.to(device)
                pred = model(X)
                
                target_cropped = _crop(pred, target)
                loss = criterion(pred, target_cropped)
                
                if len(image_test) < batchs2save:
                    dd = X.cpu(), target[:, 0].cpu()[:, None, :, :], pred[:, 0].cpu()[:, None, :, :]
                    image_test.append(dd)
                    
                desc = 'Test Epoch {} , loss={}'.format(epoch+1, loss.item())
                pbar_test.set_description(desc = desc, refresh=False)
                
                #I prefer to add a point at each iteration since hte epochs are very large
                logger.add_scalar('test_iter_loss', loss.item(), n_iter)
                n_iter+= 1
                
                test_avg_loss += loss.item()
        
        xs = torch.cat([torch.cat(x) for x in image_test])
        xs = vutils.make_grid(xs, nrow = 3, normalize=True, scale_each = True)
        logger.add_image('test_epoch', xs, n_iter)
        
        test_avg_loss /= len(gen)
        logger.add_scalar('test_epoch_loss', test_avg_loss, epoch)
        #%% save_model
        is_best = test_avg_loss < best_loss
        best_loss = min(test_avg_loss, best_loss)
        
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }
        save_checkpoint(state, is_best, save_dir = log_dir)
        
if __name__ == '__main__':
    import fire
    fire.Fire(main)
