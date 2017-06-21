#------Training Dir
traindir = 'data'

#------Imports
import os,glob
from scipy.io.wavfile import read as wavread
import numpy as np

#------Read wavfiles/Open numpy mega array
if os.path.isfile('wavfiles.npz'):
    print('load mega array of wavfiles')
    data = np.load('wavfiles.npz')
else:
    print('create mega array of wavfiles')
    out = []
    typedef = type(np.array([2,4])) #--used in making mono channels
    currentfile = 0
    for x in glob.glob(os.getcwd()+'/'+traindir+'/'+'*.wav'):
        print('loading file '+x.split('/')[-1])
        tmp = wavread(x)[1]
        print('proccessing file '+x.split('/')[-1])
        for x in tmp:
            if type(x) == typedef:
                out += [np.sum(x)//len(x)]
            else:
                out += x
    data = np.array(out,dtype=np.int16)
    np.save('wavfiles.npz',data)
    print('saved mega array as wavfiles.npz')

#------Create/Load model
if os.path.isfile('model.h5'):
    print('loading model on disk...')
    from keras.models import load_model
    model = load_model('model.h5')
else:
    print('creating new model...')
    from keras.models import Sequential
    from keras.layers import Conv1D,Activation,MaxPooling1D,LSTM,Dense,Dropout
    from keras.optimzers import RMSprop
    model = Sequential()
    model.add(Conv1D(32,10,padding='same',input_shape=(None,1)))
    model.add(Activation('relu'))
    model.add(Conv1D(32,10))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.2))
    model.add(Conv1D(64,10,padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64,10))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(LSTM(128))
    model.add(Dropout(0.35))
    model.add(Dense(1))
    model.compile(optimizer = RMSprop())
