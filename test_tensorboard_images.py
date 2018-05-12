#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:08:25 2018

@author: avelinojaver
"""

import tqdm
import torch

import torchvision.utils as vutils
from torch.utils.data import DataLoader
from train import TorchDataset
import matplotlib.pylab as plt

from tensorboardX import SummaryWriter

if __name__ == '__main__':
    batch_size = 1
    num_workers = 1
    cuda_id  =  0
    
    gen_imgs = TorchDataset(test_split = 0.1,
                               pad_size = 32,
                               add_cnt_weights = False
                               )
    
    gen = DataLoader(gen_imgs, 
                     batch_size = batch_size,
                     num_workers = num_workers)
     
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        device = torch.device("cuda:" + str(cuda_id))
    else:
        device = torch.device('cpu')

    #%%
    writer = SummaryWriter()
    with torch.no_grad(): 
        pbar = tqdm.tqdm(gen)
        
        n_iter = 0
        imgs = []
        for ii, (X, target) in enumerate(pbar):
            
            X = X.to(device)
            target = target.to(device)
            
            
            
            ori = X.cpu()
            
            
            imgs.append(X)
            if len(imgs) == 12:
                imgs = torch.cat(imgs)
                x = vutils.make_grid(imgs, nrow=4, normalize=True)
                writer.add_image('Image', x, n_iter)
                imgs = []
                n_iter += 1
            
            
            if ii == 50:
                break