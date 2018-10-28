
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Convolution1D
from keras.optimizers import SGD

model = Sequential()
model.add(Convolution1D(1, 2, border_mode='valid', input_shape=(6, 1),bias=True,activation='relu'))

l=model.get_layer(index=1)
w1 = l.get_weights()[0]
w2 = l.get_weights()[1]
np.random.seed(0)
x = np.random.random((1,6,1))
y = model.predict(x)


a_tf = np.sum(np.multiply(w1[:,0,0,0],x[0,range(2),0])) # y[0,0] would equal to this with Tensorflow backend
a_th = np.sum(np.multiply(np.flip(w1[:,0,0,0]),x[0,range(2),0])) # y[0,0] would equal to this with Theano backend
