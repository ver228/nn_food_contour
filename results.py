#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:08:25 2018

@author: avelinojaver
"""
import os
import tqdm
import torch

from torch.utils.data import DataLoader
from train import TorchDataset
from unet import UNet

from paths import RESULTS_DIR

import matplotlib.pylab as plt
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
 
    #saved_model_path = 'logs/food_unet_20180501_192131/model_best.pth.tar'
    #gen_imgs.test()
    
    #saved_model_path = 'tiny_log/unet_tiny_lr0.01_batch1/model_best.pth.tar'
    #saved_model_path = 'tiny_log/unet_tiny_lr0.01_batch4/model_best.pth.tar'
    saved_model_path = 'tiny_log/20180511_121507_unet_dilated_tiny_lr0.0001_batch1/model_best.pth.tar'
    gen_imgs.tiny()
    
    saved_model_path = os.path.join(RESULTS_DIR, saved_model_path)
    checkpoint = torch.load(saved_model_path, map_location='cpu')
    
    model = UNet()
    model.load_state_dict(checkpoint['state_dict'])
     
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        device = torch.device("cuda:" + str(cuda_id))
    else:
        device = torch.device('cpu')

    #%%
    model.to(device)
    model.eval()
    
    with torch.no_grad(): 
        pbar = tqdm.tqdm(gen)
        for ii, (X, target) in enumerate(pbar):
            
            X = X.to(device)
            target = target.to(device)
            
            pred = model(X)
            
            for nn in range(batch_size):
                res = pred.cpu()[nn, 0].numpy()
                ori = X.cpu()[nn, 0].numpy()
                
                fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
                axs[0].imshow(ori)
                axs[1].imshow(res)
                
            if ii > 30:
                break
        plt.show()