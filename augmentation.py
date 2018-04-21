#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:39:55 2018

@author: avelinojaver
"""
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates, affine_transform

def random_rotation(rg, h, w):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)    
    return transform_matrix


def random_shift(shift_range, h, w):
    tx = np.random.uniform(-shift_range, shift_range) * h
    ty = np.random.uniform(-shift_range, shift_range) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    return translation_matrix


def random_zoom(zoom_range, h, w, same_zoom=False):
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        
    if same_zoom:
        zx = zy
    
    zoom_matrix = np.array([[1/zx, 0, 0],
                            [0, 1/zy, 0],
                            [0, 0, 1]])

    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    
    return transform_matrix

def apply_transform_img(x,
                    transform_matrix,
                    fill_mode='reflect',
                    cval=0.):
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    xt = affine_transform(x,
                        final_affine_matrix,
                        final_offset,
                        order=0,
                        mode=fill_mode,
                        cval=cval
                        )
    return xt


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def elastic_transform(h, w, alpha_range, sigma):
    alpha = np.random.uniform(0, alpha_range)
    random_state = np.random.RandomState(None)
    
    dx = gaussian_filter((random_state.rand(h, w ) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(h, w ) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x,y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x = x + dx 
    y = y + dy 
    
    elastic_inds = np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))
    
    return elastic_inds
    
def get_random_transform(h, 
                     w, 
                     rotation_range, 
                     shift_range,
                     zoom_range,
                     horizontal_flip,
                     vertical_flip,
                     elastic_alpha_range,
                     elastic_sigma,
                     same_zoom = False
                     ):
    
    rot_mat = random_rotation(rotation_range, h, w)
    shift_mat = random_shift(shift_range, h, w)
    zoom_mat = random_zoom(zoom_range, h, w, same_zoom)
    
    
    transform_mat = np.dot(shift_mat, rot_mat)
    transform_mat = np.dot(transform_mat, zoom_mat)
    
    if elastic_alpha_range is not None and elastic_sigma is not None:
        elastic_inds = elastic_transform(h, w, elastic_alpha_range, elastic_sigma)
    else:
        elastic_inds = None
        
    is_h_flip =  horizontal_flip and np.random.random() < 0.5
    is_v_flip =  vertical_flip and np.random.random() < 0.5
        

    return transform_mat, is_h_flip, is_v_flip, elastic_inds

def _transform_img(img, 
                   transform_matrix, 
                   is_h_flip, 
                   is_v_flip, 
                   elastic_inds
                   ):
    
    
    
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    img_aug = affine_transform(
            img,
            final_affine_matrix,
            final_offset,
            order=0,
            mode='reflect',
            output=np.float32,
            cval=0.)
    if is_h_flip:
        img_aug = img_aug[::-1, :] 
    if is_v_flip:
        img_aug = img_aug[:, ::-1] 
    
    if elastic_inds is not None:
        img_aug = map_coordinates(img_aug, elastic_inds, order=1).reshape((img.shape))
    
    return img_aug

def transform_img(D, *args):
    if D is None:
        return None
    
    if D.ndim == 3:
        D_aug = [_transform_img(x, *args) for x in D]
        D_aug = np.array(D_aug)
        return D_aug
    elif D.ndim == 2:
        return _transform_img(D, *args)
    else:
        raise ValueError()