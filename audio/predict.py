#!/usr/bin/python3
#------Config
model = "model.h5"
randomdata = False
valuesgen = 900000
output = True
sampling = 16000

#------Imports
import os
import numpy as np
from keras.models import load_model
from sys import exit
from tqdm import tqdm
import librosa

#------RNG Seed
#np.random.seed(22)

#------Load model
print('loading model')
if os.path.isfile(model):
    model = load_model(model)
else:
    print('no model named \"'+model+'\" found. exiting...')
    exit(1)

#------Get first 128 values
if randomdata: #------Generate random data
    print('generating random data')
    data = np.random.random_sample((1,128))
else: #------Grab data from sound cache
    print('getting data from cache.npy')
    tmp = np.load('cache.npy')
    x = np.random.randint(len(tmp)-129)
    data = np.array([list(tmp[int(x):int(x)+128])],dtype=np.float32)
    del tmp,x

#------Predict
out = []
print('predicting values')
for _ in tqdm(range(valuesgen)):
    out += [float(model.predict(data))]
    data = np.array([[float(data[0][x+1]) for x in range(127)]+[out[-1]]],dtype=np.float32)
