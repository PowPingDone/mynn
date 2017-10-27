#------Config
traindir = 'data'
size = (1024,1024)

#------Imports
from keras.preprocessing.image import img_to_array,load_img
import numpy as np
from glob import glob
from tqdm import tqdm

#------Process input data into indata.npy
data = np.zeros((1)+size,dtype=np.float32)
buffer = []
inc = 0
start = True
for x in tqdm(glob('./'+traindir+'/*')):
  try:
    print('processing '+x.split('/')[-1])
    buffer += img_to_array(load_img(x,grayscale=True,target_size=size))
    inc += 1
    if start:
      data = np.delete(data,0,0)
    if inc==15000:
      inc = 0
      data = np.concatenate((data,buffer))
      buffer = []
  except:
    print('could not process '+x.split('/')[-1])
np.save('indata',data)
del data,buffer
