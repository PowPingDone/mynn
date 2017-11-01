#!/usr/bin/python3
#------Config


#------Imports
import os
import numpy as np

#------Seed
np.random.seed(54320)

if os.path.exists('pics.npy') and os.path.exists('preds.npy'):
    x, y = np.load('pics.npy'), np.load('preds.npy')
else:
