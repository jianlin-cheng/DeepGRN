from keras.layers import Activation,Dense,Multiply,Flatten,Input,Dropout,Softmax,Bidirectional,Convolution1D,LSTM,Flatten,MaxPooling1D,TimeDistributed,Lambda
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer

import numpy as np

class Attention1D(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention1D, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[2], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(Attention1D, self).build(input_shape)
        
    def call(self,x):
        attention_prob=Softmax(axis=1)(x)
        context= Multiply()([attention_prob,x])
        out=Lambda(lambda x: K.sum(x,axis=1))(context)     
        return Activation('relu')(K.dot(out, self.kernel))
    
    def get_config(self):
        return {"name" : self.__class__.__name__, "output_dim" : self.output_dim}
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def ModelCreate(num_features,num_filters,kernel_size,strides,lstm_dims,dropout_rate,out_dims):
    inputs = Input((None,num_features))
    conv1d_layer = Convolution1D(filters=num_filters,kernel_size=kernel_size, padding='valid',
                                 activation='relu',strides=strides)(inputs)
    conv1d_layer1 = Dropout(dropout_rate)(conv1d_layer)
    conv1d_layer2 = TimeDistributed(Dense(num_filters, activation='relu'))(conv1d_layer1)
    conv1d_layer3 = MaxPooling1D(pool_size=kernel_size//2, strides=kernel_size//2)(conv1d_layer2)
    lstm_layer = Bidirectional(LSTM(lstm_dims,return_sequences=True))(conv1d_layer3)
    lstm_layer1 = Dropout(dropout_rate)(lstm_layer)
    attention_out1 = Attention1D(num_filters)(lstm_layer1)
    attention_out2 = Dropout(dropout_rate)(attention_out1)
    outputs = Dense(out_dims, activation='sigmoid')(attention_out2)
    return Model(inputs=inputs,outputs=outputs)
    

num_features=6
lstm_dims = 16
num_filters = 16
kernel_size = 26
strides = 1  
dropout_rate = 0.1
model = ModelCreate(num_features,num_filters,kernel_size,strides,lstm_dims,dropout_rate,out_dims=5)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])




x_train = np.random.random((1000, 100,6))
y_train = np.random.randint(10, size=(1000, 5))/10
model.fit(x_train, y_train,epochs=5,batch_size=128)

            