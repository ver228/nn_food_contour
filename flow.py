#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import math

from scipy.ndimage.filters import gaussian_filter
from skimage.io import imread
from skimage.morphology import dilation, disk, binary_erosion


from paths import DATA_DIR
from augmentation import get_random_transform, transform_img

border_weight_params_dlf = dict(sigma=2.5, weigth=10)

transform_params_dfl = dict(
            rotation_range=90, 
             shift_range = 0.1,
             zoom_range = (0.9, 1.5),
             horizontal_flip=True,
             vertical_flip=True,
             elastic_alpha_range = 800,
             elastic_sigma = 10,
             int_alpha = (0.5,2.25)
             )

class BasicImgFlow(object):
    
    def __init__(self, 
                 main_dir = DATA_DIR,
                 is_shuffle = False
                 ):
        
        fnames = os.listdir(main_dir)
        
        #the the name of the correponding y file
        fname_pairs = [(x, 'Y' + x[1:]) for x in fnames if x.endswith('.png') and x.startswith('X')]
        
        #check all files are in fnames
        unpaired_files = (set(sum(fname_pairs, ())) - set(fnames))
        if  len(unpaired_files) > 0: #check all the sets
            raise ValueError('Some X_ files do not have a matching Y_ file.')
        
        #add full path
        fname_pairs = [(os.path.join(main_dir, x), os.path.join(main_dir, y)) for x,y in fname_pairs] 
        
        
        self.main_dir = main_dir
        self.fname_pairs = fname_pairs
        self.is_shuffle = is_shuffle
        
    def _read_img(self, fname):
        X = imread(fname)
        return X
    
    def __getitem__(self,index):
        return tuple(self._read_img(x) for x in self.fname_pairs[index])
    
    def __len__(self):
        return len(self.fname_pairs)
    
    
    def __iter__(self):
        iterator = range(len(self))
        
        if self.is_shuffle:
            iterator = list(iterator)
            random.shuffle(iterator)
        
        for ii in iterator:
            yield self[ii]

class ImgFlow(BasicImgFlow):
    
    def __init__(self,
                 pad_size = 0,
                 add_cnt_weights = False,
                 border_weight_params = border_weight_params_dlf,
                 transform_params = transform_params_dfl,
                 **argkws
                 ):
        super().__init__(**argkws)
        
        
        if 'int_alpha' in transform_params:
            transform_params = transform_params.copy()
            int_alpha = transform_params['int_alpha']
            del transform_params['int_alpha']
        else:
            int_alpha = None
        
        
        
        self.add_cnt_weights = add_cnt_weights
        
        self.transform_params = transform_params
        self.int_alpha = int_alpha
        
        self.border_weight_params = border_weight_params
        self.pad_size = pad_size
        self._is_transform = True
    
    def __getitem__(self,index):
        Xo, Yo = super().__getitem__(index)
        
        if self.pad_size > 0:
            pad_size_s =  (self.pad_size, self.pad_size)
            Xo, Yo = [None if D is None else np.lib.pad(D, pad_size_s, 'reflect') for D in (Xo, Yo)]
        
        
        X = self._img_norm(Xo)
        
        Ym = Yo>0
        Y = self._y_to_weight(Ym)
        
        outputs = [X,Y]
        if self.add_cnt_weights:
            Y_cnt = self._get_contour(Ym)
            Y_cnt = self._y_to_weight(Y_cnt)
            
            outputs.append(Y_cnt)
    
        if self._is_transform:
            outputs = self._transform(outputs)
        
        
        return outputs
        
    def _img_norm(self, _X):
        _X = _X.astype(np.float32)
        _X /= 255
        _X -= np.median(_X)
        return _X
    
    def _y_to_weight(self, _Y):
        
        W = self._normalize_weigths_by_class(_Y)
        if self.border_weight_params:
            W_border = self._increase_border_weight(_Y)
            W += W_border
            
        return W
    
    def _normalize_weigths_by_class(self, _Y):
        #normalize the weights for the classes
        W_label = np.zeros(_Y.shape, np.float32()) 
        lab_w = np.mean(_Y)
        
        dd = _Y>0
        W_label[dd] = 1/lab_w 
        W_label[~dd] = 1/(1-lab_w)
        return W_label
    
    
    def _increase_border_weight(self, _Y):
        #increase the importance of the pixels around the border
        
        sigma = self.border_weight_params['sigma']
        weigth = self.border_weight_params['weigth']
        Yc = _Y ^ binary_erosion(_Y) #xor operator
        
        W_border = gaussian_filter(Yc.astype(np.float32()), sigma=2.5)
        W_border *= (sigma**2)*weigth #normalize weights
        
        return _Y + W_border

    def _get_contour(self, _Y):
        Yc = _Y ^ binary_erosion(_Y) #xor operator
        Yo = dilation(Yc, disk(1))
        return Yo
    

    def _transform(self, imgs):
        
        if len(self.transform_params) > 0:
            expected_size = imgs[0].shape #the expected output size after padding
            transforms = get_random_transform(*expected_size, **self.transform_params)
            
            imgs_t = [transform_img(x, *transforms) for x in imgs]
            
            
            if self.int_alpha is not None:
                #I am assuming the first channel is the image
                alpha = np.random.uniform(*self.int_alpha)
                imgs_t[0] *= alpha
                
        return imgs_t

class ImgFlowSplitted(ImgFlow):
    def __init__(self, 
                 seed = 777,
                 test_split = 0.2,
                 **argkws):
        super().__init__(**argkws)
        self.seed = seed
        self.test_split = test_split
        
        random.seed(seed)
        
        
        bn = [os.path.basename(x)[2:] for x, _ in self.fname_pairs]
        
        #group data by dates 
        group_dates = {}
        for ii, x in enumerate(bn):
            date_str = x.split('_')[-2]
            if not date_str in group_dates:
                group_dates[date_str] = []
            
            group_dates[date_str].append(ii)
        
        self.group_dates = group_dates

        self.train_data = {}
        self.test_data = {}
        
        for k, inds in group_dates.items():
            N = len(inds)
            random.shuffle(inds)
            
            i_split = math.ceil(N*test_split)
            
            self.test_data[k] = inds[:i_split]
            self.train_data[k] = inds[i_split:]
        
        self.train()
        
    def train(self):
        self._is_transform = True
        self._is_train = True
        
        self.sample_inds = []
        dates = list(self.train_data.keys())
        for _ in range(len(self)):
            date_str = random.choice(dates)
            inds = self.train_data[date_str]
            ii = random.choice(inds)
            self.sample_inds.append(ii)
    
    def test(self):
        self._is_transform = False
        self._is_train = False
        self.sample_inds = sum(self.test_data.values(), [])
        
    def __iter__(self):
        for ii in range(len(self)):
            yield self[ii]
    
    
    def __getitem__(self, index):
        ii = self.sample_inds[index]
        return super().__getitem__(ii)
    
    def __len__(self):
        dat = self.train_data if self._is_train else self.test_data
        return sum(len(x) for x in dat.values())




if __name__ == '__main__':
    import matplotlib.pylab as plt
    
    gen = ImgFlowSplitted(
                test_split = 0.1,
                pad_size = 32,
                 add_cnt_weights = True
                 )
    
    for ii, (X, Y, Yc) in enumerate(gen):
    
    
        plt.figure(figsize = (8, 3))
        plt.subplot(1,3,1)
        plt.imshow(X) 
        plt.subplot(1,3,2)
        plt.imshow(Y)
        plt.subplot(1,3,3)
        plt.imshow(Yc)
        
        if ii > 1:
            break
    
    gen.test()
    
    for ii, (X, Y, Yc) in enumerate(gen):
        plt.figure(figsize = (8, 3))
        plt.subplot(1,3,1)
        plt.imshow(X)
        plt.subplot(1,3,2)
        plt.imshow(Y)
        plt.subplot(1,3,3)
        plt.imshow(Yc)
        
        if ii > 1:
            break
    
    