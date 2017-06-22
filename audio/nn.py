#------Training Dir
traindir = 'shortdata'

#------Imports
import os,glob
from scipy.io.wavfile import read as wavread
import numpy as np

#------Read wavfiles/Open numpy mega array
if os.path.isfile('wavfiles.npy'):
    print('load mega array of wavfiles')
    data = np.load('wavfiles.npy')
else:
    print('create mega array of wavfiles')
    out = []
    typedef = type(np.array([2,4])) #--used in making mono channels
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
    np.save('wavfiles',data)
    out=None
    del out,typedef
    print('saved mega array as wavfiles.npy')
Y=np.arange(len(data))

#------Create/Load model
if os.path.isfile('model.h5'):
    print('loading model on disk...')
    from keras.models import load_model
    model = load_model('model.h5')
else:
    print('creating new model...')
    from keras.models import Sequential
    from keras.layers import Conv1D,Activation,MaxPooling1D,LSTM,Dense,Dropout
    from keras.optimizers import RMSprop
    model = Sequential()
    model.add(Conv1D(72,10,padding='casual',input_shape=(None,1)))
    model.add(Activation('relu'))
    model.add(Conv1D(72,10))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.2))
    model.add(Conv1D(36,10,padding='casual'))
    model.add(Activation('relu'))
    model.add(Conv1D(36,10))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.2))
    model.add(LSTM(75))
    model.add(Dropout(0.35))
    model.add(Dense(1))
    model.compile(optimizer = RMSprop(), loss = 'categorical_crossentropy')

#------♪ AI Train ♪
for x in range(5000):
    model.fit(data,Y,verbose=1,batch_size=10)
    try:
        model.save('model.h5')
    except:
        print('let save before exiting')
        model.save('model.h5')
        break
print('exiting at '+str(x)+' iterations')
