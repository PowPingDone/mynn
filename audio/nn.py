#------Training Dir
traindir = 'data'

#------Imports
import os,glob
from scipy.io.wavfile import read as wavread
import numpy as np

#------Read wavfiles
out = []
typedef = type(np.array([2,4])) #--used in making mono channels
currentfile = 0
for x in glob.glob(os.getcwd()+'/'+traindir+'/'+'*.wav'):
    print('loading file '+x)
    tmp = wavread(x)[1]
    print('proccessing file '+x)
    for x in tmp:
        if type(x) == typedef:
            out += [np.sum(x)//len(x)]
        else:
            out += x
data = np.array(out)
