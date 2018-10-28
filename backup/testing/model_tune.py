from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Softmax, Multiply, Lambda, Activation
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Average, Input
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed


L = 1002
n_channel = 6
num_filters = 128
kernel_size =34
dropout_rate = 0.1
num_recurrent = 128
recurrent_dropout_rate = 0.1
num_dense

def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output

def make_model(hidden_layers,L= 1002,n_channel= 6,num_filters= 128,kernel_size=34,dropout_rate = 0.1,
               num_recurrent = 128,recurrent_dropout_rate = 0.1):
    forward_input = Input(shape=(L, n_channel,))
    reverse_input = Input(shape=(L, n_channel,))
    
    forward_output = get_output(forward_input, hidden_layers)     
    reverse_output = get_output(reverse_input, hidden_layers)
    output = Average()([forward_output, reverse_output])
    model = Model(inputs=[forward_input, reverse_input], outputs=output)
    return model

CTCF_model = [
        Convolution1D(input_shape=(None, n_channel), filters=num_filters,kernel_size=kernel_size, padding='valid', activation='relu',strides=1),
        Dropout(dropout_rate),
        TimeDistributed(Dense(num_filters, activation='relu')),
        MaxPooling1D(pool_size=kernel_size//2, strides=kernel_size//2, padding='valid'),
        Bidirectional(LSTM(num_recurrent, dropout=recurrent_dropout_rate, recurrent_dropout=0.1, return_sequences=True)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(num_dense, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
]