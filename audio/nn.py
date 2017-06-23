#!/usr/bin/python
#------Training Dir
traindir = 'shortdata'

#------Imports
import os,glob,librosa
import numpy as np

#------Read wavfiles/Open numpy mega array
if os.path.isfile('wavfiles.npy'):
    print('load mega array of wavfiles')
    data = np.load('wavfiles.npy')
    Y = np.load('preds.npy')
else:
    import sys
    print('create mega array of wavfiles')
    out = []
    for x in glob.glob(os.getcwd()+'/'+traindir+'/'+'*.wav'):
        print('loading file '+x.split('/')[-1])
        tmp = librosa.load(x,mono=True)[0]
        print('proccessing file '+x.split('/')[-1])
        tmp = librosa.hz_to_mel(tmp)
        out += list(tmp)
    data = np.array(out)
    np.save('wavfiles',data)
    print('create preds')
    Y = np.array([data[x] for x in range(21,len(data))] + [data[x] for x in range(20)])
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
    from keras.layers import Conv1D,Activation,MaxPooling1D,LSTM,Dense,Dropout,Input
    from keras.optimizers import RMSprop
    model = Sequential()
    model.add(Conv1D(256,10,padding = 'casual',input_shape = (None,data.shape)))
    model.add(Activation('relu'))
    model.add(Conv1D(256,10))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256,10,padding = 'casual'))
    model.add(Activation('relu'))
    model.add(Conv1D(256,10))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer = RMSprop(lr = 0.00003), loss = 'categorical_crossentropy')

#------♪ AI Train ♪
for x in range(5000):
    model.fit(data,Y,verbose=1,batch_size=10)
    try:
        model.save('model.h5')
    except KeyboardInterrupt or EOFError:
        print('let save before exiting')
        model.save('model.h5')
        break
print('exiting at',str(x),'iterations')
