from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Softmax, Multiply, Lambda, Activation, BatchNormalization
from keras.models import Model
from keras.layers import Dense,Dropout,Flatten,Average,Input,Concatenate,Maximum,CuDNNLSTM,Permute,RepeatVector
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed

def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output

def get_lstm_output(fw_input,rc_input,num_lstm,lstm_units,rnn_dropout,cudnn,lstm_dropout1 = 0,lstm_dropout2=0):
    lstm_layers = []
    if cudnn:
        for i in range(num_lstm):
            lstm_layers.append(Bidirectional(CuDNNLSTM(lstm_units, return_sequences=True)))
            lstm_layers.append(Dropout(rnn_dropout))
    else:
        for i in range(num_lstm):
            lstm_layers.append(Bidirectional(LSTM(lstm_units, return_sequences=True,
                                                        dropout=lstm_dropout1,recurrent_dropout=lstm_dropout2)))
            lstm_layers.append(Dropout(rnn_dropout))
    fw_lstm = get_output(fw_input,lstm_layers)
    rc_lstm = get_output(rc_input,lstm_layers)   
    return fw_lstm,rc_lstm

def attention_after_lstm_output(fw_input,rc_input,num_lstm,lstm_units,rnn_dropout,time_steps,single_attention_vector,cudnn=False):
    fw_lstm,rc_lstm = get_lstm_output(fw_input,rc_input,num_lstm,lstm_units,rnn_dropout,cudnn)
    prob_layers = []
    permute_layer = Permute((2, 1))
    prob_layers.append(permute_layer)
    alpha_dense_layer = Dense(time_steps, activation='softmax')
    prob_layers.append(alpha_dense_layer)
    if single_attention_vector:
        a2 = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')
        prob_layers.append(a2)
        a3 = RepeatVector(lstm_units*2) #since we use Bidirectional
        prob_layers.append(a3)
    alpha_probs_permute = Permute((2, 1), name='attention_vec')
    prob_layers.append(alpha_probs_permute)
    
    fw_alpha = get_output(fw_lstm,prob_layers)
    rc_alpha = get_output(rc_lstm,prob_layers)
    
    multiply_layer = Multiply()
    fw_output = multiply_layer([fw_lstm,fw_alpha])
    rc_output = multiply_layer([rc_lstm,rc_alpha])
    return fw_output,rc_output



def attention_before_lstm_output(fw_input,rc_input,num_lstm,lstm_units,rnn_dropout,time_steps,single_attention_vector,cudnn=False):
    prob_layers = []
    permute_layer = Permute((2, 1))
    prob_layers.append(permute_layer)
    alpha_dense_layer = Dense(time_steps, activation='softmax')
    prob_layers.append(alpha_dense_layer)
    if single_attention_vector:
        a2 = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')
        prob_layers.append(a2)
        a3 = RepeatVector(int(fw_input.shape[2]))
        prob_layers.append(a3)
    alpha_probs_permute = Permute((2, 1), name='attention_vec')
    prob_layers.append(alpha_probs_permute)
    
    fw_alpha = get_output(fw_input,prob_layers)
    rc_alpha = get_output(rc_input,prob_layers)
    multiply_layer = Multiply()
    fw_output1 = multiply_layer([fw_input,fw_alpha])
    rc_output1 = multiply_layer([rc_input,rc_alpha])

    fw_output,rc_output = get_lstm_output(fw_output1,rc_output1,num_lstm,lstm_units,rnn_dropout,cudnn)
    return fw_output,rc_output


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

#model can bel one of ""
def make_model(attention_position='attention1d_after_lstm',L=None,n_channel=6, num_conv = 0,num_denselayer=0,num_lstm=0,kernel_size=34,
                         num_filters=128, num_recurrent=64, num_dense=128, dropout_rate=0.1,rnn_dropout=[0.1,0.5],num_meta=8,
                         merge='ave',cudnn=False,single_attention_vector=False,batch_norm = False):
    forward_input = Input(shape=(L, n_channel,))
    reverse_input = Input(shape=(L, n_channel,))
    layers_before_lstm = []
    for i in range(num_conv):
        layers_before_lstm.append(Convolution1D(filters=num_filters,kernel_size=kernel_size, padding='valid', activation='relu',strides=1))
        if dropout_rate > 0:
            layers_before_lstm.append(Dropout(dropout_rate))
        if batch_norm:
            layers_before_lstm.append(BatchNormalization())
    layers_before_lstm.append(TimeDistributed(Dense(num_filters, activation='relu')))
    layers_before_lstm.append(MaxPooling1D(pool_size=kernel_size//2, strides=kernel_size//2))
    if batch_norm:
        layers_before_lstm.append(BatchNormalization())
    fw_tensor_before_lstm = get_output(forward_input, layers_before_lstm)     
    rc_tensor_before_lstm = get_output(reverse_input, layers_before_lstm)
    
    lstm_time_steps = int(fw_tensor_before_lstm.shape[1])
    if attention_position == 'lstm':
        fw_tensor_after_lstm,rc_tensor_after_lstm = get_lstm_output(fw_tensor_before_lstm,rc_tensor_before_lstm,num_lstm,num_recurrent,
                                                                    dropout_rate,cudnn)
        layers_lstm_to_meta = [Flatten(),Dense(num_dense, activation='relu')]
    elif attention_position == 'attention_before_lstm':
        fw_tensor_after_lstm,rc_tensor_after_lstm = attention_before_lstm_output(fw_tensor_before_lstm,rc_tensor_before_lstm,num_lstm,
                                                                                lstm_units=num_recurrent,time_steps=lstm_time_steps,
                                                                                rnn_dropout = dropout_rate,
                                                                                single_attention_vector=single_attention_vector,cudnn=cudnn)
        layers_lstm_to_meta = [Flatten(),Dense(num_dense, activation='relu')]
    elif attention_position == 'attention_after_lstm':
        fw_tensor_after_lstm,rc_tensor_after_lstm = attention_after_lstm_output(fw_tensor_before_lstm,rc_tensor_before_lstm,num_lstm,
                                                                                lstm_units=num_recurrent,time_steps=lstm_time_steps,
                                                                                rnn_dropout = dropout_rate,
                                                                                single_attention_vector=single_attention_vector,cudnn=cudnn)         
        layers_lstm_to_meta = [Flatten(),Dense(num_dense, activation='relu')]
    elif attention_position == 'attention1d_after_lstm_original':
        attention1d_layer = Attention1D(output_dim = num_dense)
        fw_tensor_after_lstm1,rc_tensor_after_lstm1 = get_lstm_output(fw_tensor_before_lstm,rc_tensor_before_lstm,num_lstm,num_recurrent,
                                                                    dropout_rate,cudnn,rnn_dropout[0],rnn_dropout[1])
        fw_tensor_after_lstm = get_output(fw_tensor_after_lstm1,[attention1d_layer])
        rc_tensor_after_lstm = get_output(rc_tensor_after_lstm1,[attention1d_layer])
        layers_lstm_to_meta = [Dense(num_dense, activation='relu')]
    elif attention_position == 'attention1d_after_lstm':
        fw_tensor_after_lstm1,rc_tensor_after_lstm1 = get_lstm_output(fw_tensor_before_lstm,rc_tensor_before_lstm,
                                                                      num_lstm,num_recurrent,dropout_rate,cudnn,
                                                                      rnn_dropout[0],rnn_dropout[1])
        dense_layer_att = Dense(num_dense, activation='relu',kernel_initializer='glorot_normal',use_bias=False)
        sum_layer = Lambda(lambda x: K.sum(x,axis=1))
        
        attention_prob_fw = Softmax(axis=1)(fw_tensor_after_lstm1)
        attention_prob_rc = Softmax(axis=1)(rc_tensor_after_lstm1)
        context_fw = Multiply()([attention_prob_fw,fw_tensor_after_lstm1])
        context_rc = Multiply()([attention_prob_rc,rc_tensor_after_lstm1])
        atout_fw = sum_layer(context_fw)
        atout_rc = sum_layer(context_rc)
        fw_tensor_after_lstm = dense_layer_att(atout_fw)
        rc_tensor_after_lstm = dense_layer_att(atout_rc)
        
        layers_lstm_to_meta = [Dense(num_dense, activation='relu')]
        
    else:
        return 0
    if batch_norm:
        layers_lstm_to_meta.append(BatchNormalization())
    fw_tensor_before_meta = get_output(fw_tensor_after_lstm, layers_lstm_to_meta)
    rc_tensor_before_meta = get_output(rc_tensor_after_lstm, layers_lstm_to_meta)
    if not num_meta==0:
        meta_input = Input(shape=(num_meta,))
        meta_merge_layer = Concatenate()
        fw_tensor_before_dense = meta_merge_layer([fw_tensor_before_meta, meta_input])
        rc_tensor_before_dense = meta_merge_layer([rc_tensor_before_meta, meta_input])
        input_layers = [forward_input, reverse_input, meta_input]
    else:
        input_layers = [forward_input, reverse_input]
        fw_tensor_before_dense = fw_tensor_before_meta
        rc_tensor_before_dense = rc_tensor_before_meta
        
    layers_after_meta = []  
    if dropout_rate > 0:
        layers_after_meta.append(Dropout(dropout_rate))     
    layers_after_meta.append(Dense(num_dense, activation='relu'))
    if batch_norm:
        layers_after_meta.append(BatchNormalization())    
    if dropout_rate > 0:
        layers_after_meta.append(Dropout(dropout_rate))  
        
    sigmoid_layer =  Dense(1, activation='sigmoid')
    for i in range(num_denselayer):
        layers_after_meta.append(Dense(num_dense, activation='relu'))
        if batch_norm:
            layers_after_meta.append(BatchNormalization())    
        if dropout_rate > 0:
            layers_after_meta.append(Dropout(dropout_rate))  
    layers_after_meta.append(sigmoid_layer)
    
    fw_tensor_before_merge = get_output(fw_tensor_before_dense, layers_after_meta)
    rc_tensor_before_merge = get_output(rc_tensor_before_dense, layers_after_meta)         

    if merge == 'ave':
        output = Average()([fw_tensor_before_merge, rc_tensor_before_merge])
    else:
        output = Maximum()([fw_tensor_before_merge, rc_tensor_before_merge])
    model = Model(inputs=input_layers, outputs=output)
    return model

def make_model_original(model,L=None,n_channel=6, num_conv = 0,num_lstm = 0,num_denselayer=0,kernel_size=34,num_filters=128,recurrent_dropout=0.1,
               num_recurrent=64, num_dense=128, dropout_rate=0.5, num_meta=8,merge='ave',cudnn=False):
    if cudnn:
        LSTM_in_use = CuDNNLSTM
    else:
        LSTM_in_use = LSTM
    forward_input = Input(shape=(L, n_channel,))
    reverse_input = Input(shape=(L, n_channel,))
    hidden_layers = [Convolution1D(input_shape=(None, n_channel), filters=num_filters,kernel_size=kernel_size, padding='valid',
                                   activation='relu',strides=1),
                     Dropout(dropout_rate)]
    for i in range(num_conv):
        hidden_layers.append(Convolution1D(filters=num_filters,kernel_size=kernel_size, padding='valid', activation='relu',strides=1))
        hidden_layers.append(Dropout(dropout_rate))
    hidden_layers.append(TimeDistributed(Dense(num_filters, activation='relu')))
    hidden_layers.append(MaxPooling1D(pool_size=kernel_size//2, strides=kernel_size//2))
    hidden_layers.append(Bidirectional(LSTM_in_use(num_recurrent, return_sequences=True)))
    hidden_layers.append(Dropout(dropout_rate))
    for i in range(num_lstm):
        hidden_layers.append(Bidirectional(LSTM_in_use(num_recurrent, return_sequences=True)))
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
    
    m = make_model('attention1d',L=1002,num_meta=0,kernel_size=15,n_channel=5,num_dense=64,
                             num_filters=64,num_conv = 1,num_denselayer=1,num_lstm=1)
    from keras.utils import plot_model # require graphviz
    plot_model(m,show_shapes=True,show_layer_names=False,to_file='model.png')



