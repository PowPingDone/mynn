#!/usr/bin/python3
#------Config
traindir = 'shortdata'
steps = 70
itermax = 10
size = 256

#------Imports
import os
import numpy as np

#------Seed for reproductability (insert meme numbers here.)
#np.random.seed(1337)

#------Read wavfiles/Open numpy mega array
if os.path.isfile('preds.npy') and os.path.isfile('wavfiles.npy'):
    print('load mega array of wavfiles')
    data = np.load('wavfiles.npy')
    Y = np.load('preds.npy')
else:
    from tqdm import tqdm
    from sys import exit
    print('create mega array of wavfiles')
    if os.path.isfile('cache.npy'):
        print('load wavfile cache')
        out = np.load('cache.npy')
    else:
        import librosa
        from glob import glob
        print('generating cache.npy')
        out = np.array([],dtype=np.float32)
        for x in glob(os.getcwd()+'/'+traindir+'/*.wav'):
            print('loading file '+x.split('/')[-1])
            tmp = librosa.load(x,sr=None)[0]
            out = np.append(out,tmp)
        print('saving cached wavfiles')
        np.save('cache',out)
    if not os.path.isfile('wavfiles.npy'):
        from sklearn.preprocessing import normalize
        print('create mega array')
        tmp = []
        data = np.empty([1,size],dtype=np.float32)
        for x in tqdm(range(0,len(out)-(size+1),steps)):
            tmp.append(list(out[x:x+size]))
            if x%15000==0 and x!=0:
                data = np.concatenate((data,tmp))
                tmp = []
            if x==0:
                data = np.delete(data,0,0)
        data = np.concatenate((data,tmp))
        del tmp
        np.save('wavfiles',normalize(data))
        del data
    print('create preds')
    Y = np.array([[x] for x in tqdm(out[size+1:len(out):steps])],dtype=np.float32)
    np.save('preds',Y)
    print('saved mega array as wavfiles.npy and preds as preds.npy, exiting to clear memory')
    exit(0)

#------Create/Load model
if os.path.isfile('model.h5'):
    print('loading model on disk...')
    from keras.models import load_model
    model = load_model('model.h5')
else:
    print('creating new model...')
    from keras.models import Sequential
    from keras.layers import LSTM,Dense,Dropout,Conv1D,MaxPooling1D,Activation
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.embeddings import Embedding
    from keras.optimizers import RMSprop
    from keras.initializers import RandomUniform
    RandomUniform(minval =-0.1, maxval = 0.1)
    model = Sequential()
    model.add(Embedding(size+1,size*2,input_length=size))
    model.add(Dropout(0.3))
    model.add(Conv1D(128,5,padding='valid',activation='relu',strides=1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.3))
    model.add(LSTM(144,return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

#------♪ AI Train ♪
for x in range(itermax):
    print("Iter:"+str(x)+"/"+str(itermax))
    model.fit(data, Y, verbose=1, epochs=1, batch_size=33**2)
    np.random.seed(np.random.randint(0,2**32-1))
    model.save('model.h5')

print('exiting at',x,'iterations')
