#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:13:46 2018

@author: avelinojaver

#adapted: from https://github.com/milesial/Pytorch-UNet
#https://github.com/chuckyee/cardiac-segmentation/blob/master/rvseg/models/dilatedunet.py
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

from unet import weights_init_xavier


def _crop(x, x_to_crop):
    c = (x_to_crop.size()[2] - x.size()[2])/2
    c1, c2 =  math.ceil(c), math.floor(c)
    cropped = F.pad(x_to_crop, (-c1, -c2, -c2, -c1)) #negative padding is the same as cropping
    return cropped

class DoubleConv(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_out, 3, padding=1),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_out, n_out, 3, padding=1),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(inplace=True)
            )
        
        
        
    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv_pooled = nn.Sequential(
            nn.MaxPool2d(2),
            
            nn.Conv2d(n_in, n_out, 3, padding=1, dilation=1),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_out, n_out, 3, padding=2, dilation=2),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_pooled(x)
        return x

class Up(nn.Module):
    def __init__(self, n_in, n_out, is_bilinear = True):
        super().__init__()
        if is_bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(n_in, n_out, 2, stride=2)

        self.conv = DoubleConv(n_in, n_out)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = _crop(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNetDilated(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 2):
        super().__init__()
        self.input = DoubleConv(n_channels, 64)
        
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        
        dilated_layers = []
        dilation_rate = 1
        n_feats = 512
        for ii in range(6):
            layer = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=dilation_rate, dilation = dilation_rate),
            nn.BatchNorm2d(n_feats),
            nn.LeakyReLU(inplace=True)
            )
            dilation_rate *= 2
            dilated_layers.append(layer)
        
        "Multi-scale Context Aggregation by Dilated Convolutions"
        self.context_agg = nn.Sequential(*dilated_layers)
        
        
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        
        self.score = nn.Sequential(
                 nn.Conv2d(64, n_classes, 1),
                 nn.Softmax(1)
                 )
        
        for m in self.modules():
            weights_init_xavier(m)
        
    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        xd = self.context_agg(x5)
        
        x = self.up1(xd, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.score(x)
        return x


if __name__ == '__main__':
    from flow import ImgFlow
    
    gen = ImgFlow(pad_size = 32,
                is_shuffle = False,
                  add_cnt_weights = True
                  )
    
    X, Y, Yc = gen[1]
    
    #%%
    Xt = torch.from_numpy(X[None,  ...])
    Xt = torch.autograd.Variable(Xt)
    
    mod = UNetDilated()
    
    if torch.cuda.is_available():
        mod = mod.cuda()    
        Xt = Xt.cuda()
    
    
    out = mod(Xt)
    
    print(out.size())
     
    
