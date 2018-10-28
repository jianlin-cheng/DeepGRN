from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Softmax, Multiply, Lambda, Activation
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Average, Input,Concatenate
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
    
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

    
def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output


# model is one of ['factornet_lstm','factornet_glomax','factornet_max','fullconv_lstm']
def make_model_simple(hidden_layers,L=None,n_channel=6, kernel_size=34,num_filters=128,flanking=0, num_recurrent=64, num_dense=128, dropout_rate=0.5):
        
    forward_input = Input(shape=(L, n_channel,))
    reverse_input = Input(shape=(L, n_channel,))
    forward_output = get_output(forward_input, hidden_layers)     
    reverse_output = get_output(reverse_input, hidden_layers)
    output = Average()([forward_output, reverse_output])
    model = Model(inputs=[forward_input, reverse_input], outputs=output)
    return model

def make_model_complex(hidden_layers,L=None,n_channel=6, kernel_size=34,num_filters=128,flanking=0, num_recurrent=64, num_dense=128, dropout_rate=0.5, num_meta=0):
    forward_input = Input(shape=(L, n_channel,))
    reverse_input = Input(shape=(L, n_channel,))
    meta_input = Input(shape=(num_meta,))
    forward_dense = get_output(forward_input, hidden_layers)     
    reverse_dense = get_output(reverse_input, hidden_layers)
    forward_dense = Concatenate()([forward_dense, meta_input])
    reverse_dense = Concatenate()([reverse_dense, meta_input])
    dropout2_layer = Dropout(0.1)
    forward_dropout2 = dropout2_layer(forward_dense)
    reverse_dropout2 = dropout2_layer(reverse_dense)
    dense2_layer = Dense(num_dense, activation='relu')
    forward_dense2 = dense2_layer(forward_dropout2)
    reverse_dense2 = dense2_layer(reverse_dropout2)
    dropout3_layer = Dropout(dropout_rate)
    forward_dropout3 = dropout3_layer(forward_dense2)
    reverse_dropout3 = dropout3_layer(reverse_dense2)
    sigmoid_layer =  Dense(1, activation='sigmoid')
    forward_output = sigmoid_layer(forward_dropout3)
    reverse_output = sigmoid_layer(reverse_dropout3)
    output = Average()([forward_output, reverse_output])
    model = Model(inputs=[forward_input, reverse_input, meta_input], outputs=output)
    return model

def make_model(model,L=None,n_channel=6, kernel_size=34,num_filters=128,flanking=0, num_recurrent=64, num_dense=128, dropout_rate=0.5, num_meta=8):
    factornet_glomax_layers = [
            Convolution1D(input_shape=(None, n_channel), filters=num_filters,kernel_size=kernel_size, padding='valid', activation='relu',strides=1),
            Dropout(0.1),
            TimeDistributed(Dense(num_filters, activation='relu')),
            GlobalMaxPooling1D(),
            Dropout(dropout_rate),
            Dense(num_dense, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ]
    factornet_max_layers = [
            Convolution1D(input_shape=(None, n_channel), filters=num_filters,kernel_size=kernel_size, padding='valid', activation='relu',strides=1),
            Dropout(0.1),
            TimeDistributed(Dense(num_filters, activation='relu')),
            MaxPooling1D(pool_size=kernel_size//2, strides=kernel_size//2),
            Dropout(dropout_rate),
            Flatten(),
            Dense(num_dense, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ]
    factornet_lstm_layers = [
            Convolution1D(input_shape=(None, n_channel), filters=num_filters,kernel_size=kernel_size, padding='valid', activation='relu',strides=1),
            Dropout(dropout_rate),
            TimeDistributed(Dense(num_filters, activation='relu')),
            MaxPooling1D(pool_size=kernel_size//2, strides=kernel_size//2),
            Bidirectional(LSTM(num_recurrent, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)),
            Dropout(dropout_rate),
            Flatten(),
            Dense(num_dense, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ]
    fullconv_lstm_attention_layers = [
            Convolution1D(input_shape=(None, n_channel), filters=num_filters,kernel_size=kernel_size, padding='valid', activation='relu',strides=1),
            Dropout(dropout_rate),
            TimeDistributed(Dense(num_filters, activation='relu')),
            MaxPooling1D(pool_size=kernel_size//2, strides=kernel_size//2),
            Bidirectional(LSTM(num_recurrent, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)),
            Dropout(dropout_rate),
            Attention1D(num_dense),
            Dense(num_dense, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ]
    fullconv_lstm_layers = [
            Convolution1D(input_shape=(None, n_channel), filters=num_filters,kernel_size=200+2*flanking, padding='valid', activation='relu',strides=50),
            Dropout(dropout_rate),
            TimeDistributed(Dense(num_filters, activation='relu')),
            Bidirectional(LSTM(num_recurrent, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)),
            Dropout(dropout_rate),
            Convolution1D(filters=num_filters,kernel_size=kernel_size, padding='same', activation='relu',strides=1),
            Dropout(dropout_rate),
            Convolution1D(filters=1,kernel_size=kernel_size, padding='same', activation='sigmoid',strides=1),
        ]
    fullconv_simple_layers = [
            Convolution1D(input_shape=(None, n_channel), filters=num_filters,kernel_size=200+2*flanking, padding='valid', activation='relu',strides=50),
            Dropout(dropout_rate),
            Convolution1D(filters=1,kernel_size=kernel_size, padding='same', activation='sigmoid',strides=1)
        ]
    if model=='factornet_lstm':
        hidden_layers = factornet_lstm_layers
    elif model=='factornet_glomax':
        hidden_layers = factornet_glomax_layers
    elif model=='factornet_max':
        hidden_layers = factornet_max_layers
    elif model=='factornet_attention':
        hidden_layers = fullconv_lstm_attention_layers
        L = None
    elif model=='fullconv_lstm':
        hidden_layers = fullconv_lstm_layers
        L = None
    else:
        hidden_layers = fullconv_simple_layers
        L = None
    if num_meta==0:
        return make_model_simple(hidden_layers,L,n_channel, kernel_size,num_filters,flanking, num_recurrent, num_dense, dropout_rate)
    else:
        return make_model_complex(hidden_layers[0:len(hidden_layers)-2],L,n_channel, kernel_size,num_filters,flanking, num_recurrent, num_dense, dropout_rate,num_meta)

#m = make_model('factornet_attention',L=1002,num_meta=8)
#out_file = 'C:/Users/chen/Documents/Dropbox/MU/workspace/encode_dream/plots/factornet_attention_meta.png'
#from keras.utils import plot_model
#plot_model(m, to_file=out_file,show_shapes=True,show_layer_names=False)

           






