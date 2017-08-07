#!/usr/bin/python3
#------Training Dir
traindir = 'shortdata'

#------Imports
import os
import numpy as np

#------Seed for reproductability (insert meme numbers here.)
np.random.seed(1337)

#------Read wavfiles/Open numpy mega array
if os.path.isfile('preds.npy') and os.path.isfile('wavfiles.npy'):
    print('load mega array of wavfiles')
    data = np.load('wavfiles.npy')
    Y = np.load('preds.npy')
else:
    import sys,glob,librosa
    print('create mega array of wavfiles')
    if os.path.isfile('cache.npy'):
        print('load wavfile cache')
        out = np.load('cache.npy')
    else:
        print('generating cache.npy')
        out = np.array([])
        for x in glob.glob(os.getcwd()+'/'+traindir+'/*.wav'):
            print('loading file '+x.split('/')[-1])
            tmp = librosa.load(x,mono=True)[0]
            print('proccessing file '+x.split('/')[-1])
            out = np.append(out,tmp)
        print('saving cached wavfiles')
        np.save('cache',out)
    print('create mega array')
    data = np.array([np.arange(128)])
    for x in range(0,len(out)-1,4):
        data= np.concatenate((data,np.array([out[x:x+128]])))
    data = np.array(np.delete(data,0))
    np.save('wavfiles',data)
    print('create preds')
    Y = np.array([[x] for x in out[0][129:len(out):4]])
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
    from keras.layers import LSTM,Dense,Dropout
    from keras.optimizers import RMSprop
    model = Sequential()
    model.add(LSTM(128, input_shape = data.shape, return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.compile(optimizer = RMSprop(lr = 0.009), loss = 'categorical_crossentropy')

#------♪ AI Train ♪
itermax = 100
from keras.callbacks import ModelCheckpoint
callback = [ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
model.fit(data, Y, verbose=1, epochs=itermax, batch_size=1, callbacks=callback)
print('exiting at',str(x),'iterations')
