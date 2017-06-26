#!/usr/bin/python
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
    for x in glob.glob(os.getcwd()+'/'+traindir+'/'+'*.wav'):
        print('loading file '+x.split('/')[-1])
        tmp = librosa.load(x,mono=True)[0]
        print('proccessing file '+x.split('/')[-1])
        tmp = np.array(librosa.feature.melspectrogram(tmp),dtype=np.float32)
        out += [x for x in tmp[0]]
    print('concatenating final array')
    tmp = data = np.array(out,dtype=np.float32)
    data = np.array(np.array_split(np.array(np.array_split(data,len(data)//128)),1))
    np.save('wavfiles',data)
    print('create preds')
    Y = np.array([tmp[x] for x in range(129,len(tmp))] + [tmp[x] for x in range(128)],dtype=np.float32)
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
    from keras.layers import Conv1D,Activation,MaxPooling1D,LSTM,Dense,Dropout,TimeDistributed,Flatten
    from keras.optimizers import RMSprop
    model = Sequential()
    model.add(Conv1D(130,10,padding = 'causal', input_shape = tuple([1]+list(data.shape))))
    model.add(Activation('relu'))
    model.add(Conv1D(130,10))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))
    model.add(Conv1D(130,10,padding = 'causal'))
    model.add(Activation('relu'))
    model.add(Conv1D(130,10))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer = RMSprop(lr = 0.00003), loss = 'categorical_crossentropy')

#------♪ AI Train ♪
itermax = 5000
from keras.callbacks import ModelCheckpoint
callback = [ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
model.fit(data, Y, verbose=1, epochs=itermax, batch_size=1, callbacks=callback)
print('exiting at',str(x),'iterations')
