#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:52:42 2018

@author: avelinojaver
"""

import os
import socket

hostname = socket.gethostname()
if 'Avelinos' in hostname:
    ROOT_DIR = '/Volumes/rescomp1/data/WormData/experiments/food/'
else:
    ROOT_DIR = '/well/rittscher/users/avelino/WormData/experiments/food/'
    

if os.path.exists(ROOT_DIR):
    ROOT_DATA_DIR = ROOT_DIR
    ROOT_RESULTS_DIR = ROOT_DIR
    
else:
    #I am using the imperial cluster so i have to change the path
    ROOT_DATA_DIR = os.environ['TMPDIR']
    ROOT_RESULTS_DIR = os.path.join(os.environ['WORK'], 'food')
    
    
DATA_DIR = os.path.join(ROOT_DATA_DIR, 'train_set')
RESULTS_DIR = os.path.join(ROOT_RESULTS_DIR, 'results', 'pytorch')

