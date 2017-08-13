#!/usr/bin/python3
#------Config
traindir = 'shortdata'
steps = 120
itermax = 48 

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
    import sys,glob
    from tqdm import tqdm
    print('create mega array of wavfiles')
    if os.path.isfile('cache.npy'):
        print('load wavfile cache')
        out = np.load('cache.npy')
    else:
        import librosa,glob
        print('generating cache.npy')
        out = np.array([],dtype=np.float32)
        for x in glob.glob(os.getcwd()+'/'+traindir+'/*.wav'):
            print('loading file '+x.split('/')[-1])
            tmp = librosa.load(x,sr=None)[0]
            out = np.append(out,tmp)
        print('saving cached wavfiles')
        np.save('cache',out)
    if not os.path.isfile('wavfiles.npy'):
        print('create mega array')
        tmp = []
        data = np.empty([1,128],dtype=np.float32)
        for x in tqdm(range(0,len(out)-129,steps)):
            tmp.append(list(out[x:x+128]))
            if x%15000==0 and x!=0:
                data = np.concatenate((data,tmp))
                tmp = []
            if x==0:
                data = np.delete(data,0,0)
        data = np.concatenate((data,tmp))
        del tmp
        np.save('wavfiles',data)
        del data
    print('create preds')
    Y = np.array([[x] for x in tqdm(out[129:len(out):steps])],dtype=np.float32)
    np.save('preds',Y)
    print('saved mega array as wavfiles.npy and preds as preds.npy, exiting to clear memory')
    sys.exit(0)

#------Create/Load model
if os.path.isfile('model.h5'):
    print('loading model on disk...')
    from keras.models import load_model
    model = load_model('model.h5')
else:
    print('creating new model...')
    from keras.models import Sequential
    from keras.layers import LSTM,Dense,Dropout
    from keras.layers.embeddings import Embedding
    model = Sequential()
    model.add(Embedding(129,256,input_length=128))
    model.add(LSTM(256))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.compile(optimizer = 'rmsprop', loss = 'mse')

#------♪ AI Train ♪
for x in range(itermax):
    print("Iter:"+str(x)+"/"+str(itermax))
    model.fit(data, Y, verbose=1, epochs=1, batch_size=60)
    model.save('model.h5')

print('exiting at',x,'iterations')
