import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import tensorflow as tf
from keras import backend as K

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

def mean_squared_error1(y_true, y_pred, w):
    res = K.mean(K.square(y_pred - y_true)*w, axis=-1)
    return(res)

def mean_squared_error2(w):
  def mean_squared_error3(y_true, y_pred):
    return mean_squared_error1(y_true, y_pred, w)
  return mean_squared_error3

w = tf.constant([1,0,1,0,0,0], dtype=tf.float32)
a=tf.constant([12,12,32,53,45,34], dtype=tf.float32)
b=tf.constant([11,12,12,13,41,14], dtype=tf.float32)

mse1 = mean_squared_error2(w)
xxx = mse1(a,b)
sess = tf.Session()
print(sess.run(xxx))

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)