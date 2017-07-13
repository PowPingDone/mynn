#!/usr/bin/python3
#------Training Dir
traindir = 'shortdata'

#------Imports
import os,glob,librosa
import numpy as np

#------Read wavfiles/Open numpy mega array
if os.path.isfile('preds.npy') and os.path.isfile('wavfiles.npy'):
    print('load mega array of wavfiles')
    data = np.load('wavfiles.npy')
    Y = np.load('preds.npy')
else:
    import sys
    print('create mega array of wavfiles')
    out = [0]
    for x in glob.glob(os.getcwd()+'/'+traindir+'/*.wav'):
        print('loading file '+x.split('/')[-1])
        tmp = librosa.load(x,mono=True)[0]
        print('proccessing file '+x.split('/')[-1])
        tmp = np.array(librosa.feature.melspectrogram(tmp),dtype=np.float32)
        out += [x for x in tmp[0]]
    print('concatenating tmp array')
    tmp = np.array(out,dtype=np.float32)
    print('create mega array ')
    data = np.array([[tmp[x:x+128]] for x in range(0,len(tmp)-128,4)])
    data = np.array([x[0] for x in data])
    data = np.array(np.array_split(np.array(np.array_split(data,1)),1))
    np.save('wavfiles',data)
    print('create preds')
    Y = [[x] for x in tmp[129:len(tmp):4]]
    Y = np.array(Y,dtype=np.float32)
    Y = np.array_split(np.array(np.array_split(Y,1)),1)
    np.save('preds',Y)
    print('saved mega array as wavfiles.npy and preds as preds.npy, exiting to clear memory')
    sys.exit(1)

#------Create/Load model
if os.path.isfile('model.h5'):
    print('loading model on disk...')
    from keras.models import load_model
    model = load_model('model.h5')
else:
    print('creating new model...')
    from keras.models import Sequential
    from keras.layers import Activation,LSTM,Dense,Dropout,TimeDistributed
    from keras.optimizers import RMSprop
    model = Sequential()
    model.add(LSTM(128, input_shape = data.shape[2:]))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.compile(optimizer = RMSprop(lr = 0.009), loss = 'categorical_crossentropy')

#------♪ AI Train ♪
itermax = 5000
from keras.callbacks import ModelCheckpoint
callback = [ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
model.fit(data[0], Y[0], verbose=1, epochs=itermax, batch_size=1, callbacks=callback)
print('exiting at',str(x),'iterations')
