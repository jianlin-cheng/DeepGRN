from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Softmax, Multiply, Lambda, Activation
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Average, Input,Concatenate,Maximum
from keras.layers.convolutional import Convolution1D, MaxPooling1D
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

def make_model(model,L=None,n_channel=6, num_conv = 0,num_lstm = 0,num_denselayer=0,kernel_size=34,num_filters=128, num_recurrent=64, num_dense=128, dropout_rate=0.5, num_meta=8,merge='ave'):
    forward_input = Input(shape=(L, n_channel,))
    reverse_input = Input(shape=(L, n_channel,))
    hidden_layers = [Convolution1D(input_shape=(None, n_channel), filters=num_filters,kernel_size=kernel_size, padding='valid', activation='relu',strides=1),
                     Dropout(dropout_rate)]
    for i in range(num_conv):
        hidden_layers.append(Convolution1D(filters=num_filters,kernel_size=kernel_size, padding='valid', activation='relu',strides=1))
        hidden_layers.append(Dropout(dropout_rate))
    hidden_layers.append(TimeDistributed(Dense(num_filters, activation='relu')))
    hidden_layers.append(MaxPooling1D(pool_size=kernel_size//2, strides=kernel_size//2))
    hidden_layers.append(Bidirectional(LSTM(num_recurrent, dropout=dropout_rate, recurrent_dropout=0.1, return_sequences=True)))
    hidden_layers.append(Dropout(dropout_rate))
    for i in range(num_lstm):
        hidden_layers.append(Bidirectional(LSTM(num_recurrent, dropout=dropout_rate, recurrent_dropout=0.1, return_sequences=True)))
        hidden_layers.append(Dropout(dropout_rate))
    if model == 'factornet_attention':
        hidden_layers.append(Attention1D(num_dense))
    else:
        hidden_layers.append(Flatten())
    hidden_layers.append(Dense(num_dense, activation='relu'))
    forward_dense = get_output(forward_input, hidden_layers)     
    reverse_dense = get_output(reverse_input, hidden_layers)
    if not num_meta==0:
        meta_input = Input(shape=(num_meta,))
        forward_dense = Concatenate()([forward_dense, meta_input])
        reverse_dense = Concatenate()([reverse_dense, meta_input])
        input_layers = [forward_input, reverse_input, meta_input]
    else:
        input_layers = [forward_input, reverse_input]
    dropout_layer1 = Dropout(dropout_rate)
    forward_dropout = dropout_layer1(forward_dense)
    reverse_dropout = dropout_layer1(reverse_dense)
    sigmoid_layer =  Dense(1, activation='sigmoid')
    if num_denselayer > 0:
        dense_layers = []
        for i in range(num_denselayer):
            dense_layers.append(Dense(num_dense, activation='relu'))
            dense_layers.append(Dropout(dropout_rate))
            
        forward_dense2 = get_output(forward_dropout, dense_layers)
        reverse_dense2 = get_output(reverse_dropout, dense_layers)
        forward_output = sigmoid_layer(forward_dense2)
        reverse_output = sigmoid_layer(reverse_dense2)
    else:
        forward_output = sigmoid_layer(forward_dropout)
        reverse_output = sigmoid_layer(reverse_dropout)
    
    if merge == 'ave':
        output = Average()([forward_output, reverse_output])
    else:
        output = Maximum()([forward_output, reverse_output])
    model = Model(inputs=input_layers, outputs=output)
    return model


if __name__ == '__main__':
    m = make_model('factornet_attention1',L=1002,num_meta=8,num_conv = 3,num_lstm = 3,num_denselayer=3)
    out_file = 'C:/Users/chen/Documents/Dropbox/MU/workspace/encode_dream/plots/factornet_attention_meta2.png'
    from keras.utils import plot_model
    plot_model(m, to_file=out_file,show_shapes=True,show_layer_names=False)


