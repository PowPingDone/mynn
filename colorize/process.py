#!/usr/bin/python3
# Imports
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from glob import glob
from tqdm import tqdm
from time import sleep

# Config
traindir = 'data'
size = [256, 256]
incmax = 500

# Process input data into pics.npy
print('getting input data')
data = np.zeros([1] + size + [1], dtype=np.float32)
buffer = []
inc = 0
start = True
for x in tqdm(glob('./' + traindir + '/*'), unit='pics', ascii=True):
    #try:
    buffer.append(img_to_array(load_img(x, grayscale=True, target_size=size)))
    inc += 1
    if inc == incmax:
        inc = 0
        data = np.concatenate((data, buffer))
        buffer = []
    if start and inc == incmax:
        start = False
        data = np.delete(data, 0)
    #except:
        #print('couldnt process ' + x.split('/')[-1])
np.save('pics', data)
print('saved input data')
sleep(10)


# Process output data into preds.npy
print('getting expected output data')
data = np.zeros([1] + size + [3], dtype=np.float32)
buffer = []
inc = 0
start = True
for x in tqdm(glob('./' + traindir + '/*'), unit='pics', ascii=True):
    try:
        buffer.append(img_to_array(load_img(x, grayscale=True, target_size=size)))
        inc += 1
        if inc == incmax:
            inc = 0
            data = np.concatenate((data, buffer))
            buffer = []
        if start and inc == incmax:
            start = False
            data = np.delete(data, 0, 0)
    except:
        print('couldnt process ' + x.split('/')[-1])
np.save('preds', data)
print('saved output data')
