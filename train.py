#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:42:24 2017

@author: ajaver
"""
import os
import tqdm
import torch
import shutil
import datetime

from flow import ImgFlowSplitted
from unet import UNet, _crop, unet_loss

import tensorflow as tf #used for tensorbard only

from torch.utils.data import Dataset, DataLoader

from paths import RESULTS_DIR

class TBLogger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

class AugmentedDataset(ImgFlowSplitted, Dataset):
    def __init__(self, **argkws):
        ImgFlowSplitted.__init__(self, **argkws)
        
    def __getitem__(self, index):
        dat = ImgFlowSplitted.__getitem__(self, index)
        dat = tuple(torch.from_numpy(x) for x in dat)
        return dat
    
    def __len__(self):
        return ImgFlowSplitted.__len__(self)
    
    
def _prepare_for_torch(X, cuda_id):
    if cuda_id >= 0:
        X = X.cuda(cuda_id)
    X = torch.autograd.Variable(X)
    return X

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)

def main(is_debug = False, cuda_id = -1):
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        cuda_id = 0
        
    n_epochs = 100
    batch_size = 4
    num_workers = 1
    
    gen_imgs = AugmentedDataset(test_split = 0.1,
                               pad_size = 32,
                               add_cnt_weights = False
                               )
    
    gen = DataLoader(gen_imgs, 
                     batch_size = batch_size,
                     num_workers = num_workers)
    
    log_dir_root =  os.path.join(RESULTS_DIR, 'logs')
    if is_debug:
        print("It's me, tiny-log!!!")
        log_dir_root =  os.path.join(RESULTS_DIR, 'tiny_log')
    
    now = datetime.datetime.now()
    bn = 'food_unet_' + now.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_dir_root, bn)
    logger = TBLogger(log_dir = log_dir)
    
    model = UNet()
    criterion = unet_loss
    
    if cuda_id >= 0:
       model = model.cuda(cuda_id)
       
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    best_loss = 1e10
    n_iter = 0
    for epoch in range(n_epochs):
        gen_imgs.train()
        if is_debug:
            gen_imgs.tiny()
        
        pbar = tqdm.tqdm(gen)
        avg_loss = 0
        for X, target in pbar:
            X = _prepare_for_torch(X, cuda_id)
            target =  _prepare_for_torch(target, cuda_id)
            
            pred = model(X)
            target_cropped = _crop(pred, target)
            
            loss = criterion(pred, target_cropped)
            
            if loss.data[0] < 0:
                import pdb
                pdb.set_trace()

            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step() 
        
            desc = 'Epoch {} , loss={}'.format(epoch+1, loss.data[0])
            pbar.set_description(desc = desc, refresh=False)

            #I prefer to add a point at each iteration since hte epochs are very large
            logger.scalar_summary('iter_loss', loss.data[0], n_iter)
            n_iter+= 1
            
            avg_loss += loss.data[0]
        
        avg_loss /= len(gen)
        logger.scalar_summary('epoch_loss', loss.data[0], epoch)
        
        #%% save_model
        is_best = avg_loss < best_loss
        best_loss = min(avg_loss, best_loss)
        
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