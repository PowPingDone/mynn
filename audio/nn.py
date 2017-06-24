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
    out = []
    for x in glob.glob(os.getcwd()+'/'+traindir+'/'+'*.wav'):
        print('loading file '+x.split('/')[-1])
        tmp = librosa.load(x,mono=True)[0]
        print('proccessing file '+x.split('/')[-1])
        tmp = np.array(librosa.feature.melspectrogram(tmp),dtype=np.float32)
        out += [x for x in tmp[0]]
    print('concatenating final array')
    tmp = np.array(out,dtype=np.float32)
    data = np.array(out,dtype=np.float32)
    data = np.array_split(data,len(data)//128)
    np.save('wavfiles',data)
    print('create preds')
    Y = np.array([tmp[x] for x in range(21,len(tmp))] + [tmp[x] for x in range(20)],dtype=np.float32)
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
    from keras.layers import Conv1D,Activation,MaxPooling1D,LSTM,Dense,Dropout,Flatten
    from keras.optimizers import RMSprop
    model = Sequential()
    model.add(Conv1D(256,10,padding = 'causal', input_shape = tuple(list(data.shape)+[1])))
    model.add(Activation('relu'))
    model.add(Conv1D(256,10))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))
    model.add(Conv1D(256,10,padding = 'causal'))
    model.add(Activation('relu'))
    model.add(Conv1D(256,10))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer = RMSprop(lr = 0.00003), loss = 'categorical_crossentropy')

#------♪ AI Train ♪
itermax = 5000
for x in range(itermax):
    print('Iter',x,'out of',itermax,'iterations\n'+('~'*20))
    model.fit(data,Y,verbose=1,nb_epoch=1,batch_size=10)
    try:
        model.save('model.h5')
    except KeyboardInterrupt or EOFError:
        print('let save before exiting')
        model.save('model.h5')
        break
print('exiting at',str(x),'iterations')
