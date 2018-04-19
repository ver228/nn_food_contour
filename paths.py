#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:52:42 2018

@author: avelinojaver
"""

import os
import socket

hostname = socket.gethostname()
if hostname == 'Avelinos-MBP':
    ROOT_DIR = os.path.join(os.environ['HOME'], 'OneDrive - Imperial College London/training_data/food/')
else:
    ROOT_DIR = '/well/rittscher/users/avelino/WormData/experiments/food/'

DATA_DIR = os.path.join(ROOT_DIR, 'train_set')