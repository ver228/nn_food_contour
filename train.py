#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:42:24 2017

@author: ajaver
"""

import tqdm
import torch
from torch import nn

from flow import ImgFlowSplitted
from unet import UNet, _crop

from torch.utils.data import Dataset, DataLoader

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

def unet_loss(target, pred):
    loss = -(target*(pred + 1.e-3).log()).mean()
    return loss

class UnetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, embedding_output, target_classes):
        classification, video_embedding, snps_embedding = embedding_output
        classification_loss = self.classification_loss(classification,
                                                       target_classes)
        
        # Can't use the Loss layer here because it doesn't like - aej, likely due to autograd gradients
        _embedding_loss = self.embedding_loss(snps_embedding, video_embedding)
        
        loss = classification_loss + self.embedding_loss_mixture * _embedding_loss
        return loss

if __name__ == '__main__':
    cuda_id = -1
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        cuda_id = 0
        
    n_epochs = 3
    batch_size = 1
    num_workers = 2
    
    gen_imgs = AugmentedDataset(test_split = 0.1,
                               pad_size = 32,
                               add_cnt_weights = False
                               )
    
    gen = DataLoader(gen_imgs, 
                     batch_size = batch_size,
                     num_workers = num_workers)
    
    model = UNet()
    #criterion = nn.BCEWithLogitsLoss()
    criterion = unet_loss
    if cuda_id >= 0:
       model = model.cuda(cuda_id)
       
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(gen)
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
        
