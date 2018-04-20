#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:42:24 2017

@author: ajaver
"""

import math
import tqdm
import numpy as np
import torch
from torch import nn

from flow import ImgFlowSplitted
from unet import UNet, _crop


class TorchBatchifier():
    def __init__(self, 
                 generator, 
                 batch_size,
                 cuda_id = -1):
        
        self.generator = generator
        self.batch_size = batch_size
        self.cuda_id = cuda_id
        
    def __iter__(self):
        batch = []
        for dat in self.generator:
            batch.append(dat)
            
            if len(batch) >= self.batch_size:
                yield self._prepare_batch(batch)
                batch = []
        
        if batch:
            yield self._prepare_batch(batch)
         
    def _prepare_ind(self, X):
        X = np.concatenate([torch.from_numpy(x[None, None, ...]) for x in X])
        X = torch.from_numpy(X)
        
        if self.cuda_id >= 0:
            X = X.cuda(self.cuda_id)
        
        X = torch.autograd.Variable(X)
        return X
    
    def _prepare_batch(self, batch):
        return [self._prepare_ind(x) for x in zip(*batch)]
        
        

    def __len__(self):
        return math.ceil(len(self.generator)/self.batch_size)


if __name__ == '__main__':
    cuda_id = -1
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        cuda_id = 0
        
    n_epochs = 3
    batch_size = 2
    
    gen_imgs = ImgFlowSplitted(test_split = 0.1,
                               pad_size = 32,
                               is_shuffle = False,
                               add_cnt_weights = False
                               )
    
    gen = TorchBatchifier(gen_imgs, 
                          batch_size,
                          cuda_id = cuda_id)
    
    model = UNet()
    criterion = nn.BCELoss()
    if cuda_id >= 0:
       model = model.cuda(cuda_id)
       criterion = criterion.cuda(cuda_id)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(gen)
        for X, target in gen:
            pred = model(X)
            target_cropped = _crop(pred, target)
            
            loss = criterion(pred, target_cropped)
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step() 
            
            desc = 'Epoch {} , loss={}'.format(epoch+1, loss.data[0])
            pbar.set_description(desc = desc, refresh=False)
        
