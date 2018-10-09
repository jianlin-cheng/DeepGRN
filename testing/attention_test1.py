from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Softmax, Multiply, Lambda, Activation
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Average, Input,Concatenate,RepeatVector,Permute,merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed

units = 64
max_length = 50



_input = Input(shape=[max_length,6], dtype='int32')

# get the embedding layer


activations = LSTM(units, return_sequences=True)(_input)

# compute importance for each step
attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(units)(attention)
attention = Permute([2, 1])(attention)

# apply the attention
sent_representation = merge([activations, attention], mode='mul')
sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

probabilities = Dense(3, activation='softmax')(sent_representation)

model = Model(input=_input, output=probabilities)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])