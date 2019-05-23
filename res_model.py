from __future__ import print_function
from keras.layers import Dense, Conv1D, BatchNormalization, Activation
from keras.layers import AveragePooling1D, Input, Flatten, Add, Average, Softmax, Multiply, Lambda
from keras.layers import Dropout,Concatenate,Maximum,CuDNNLSTM,Permute,RepeatVector
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional

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

def attention_after_lstm_output(fw_input,rc_input,num_lstm,lstm_units,rnn_dropout,time_steps,
                                single_attention_vector,cudnn=False):
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

def get_resnet_layers(num_filters=16,kernel_size=32,strides=1, activation='relu',
                 batch_normalization=True,conv_first=True):
    resnet_layers_list = []
    conv = Conv1D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',
                  kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))
    bn = BatchNormalization()
    acfun = Activation(activation)
    if conv_first:
        resnet_layers_list.append(conv)
        if batch_normalization:
            resnet_layers_list.append(bn)
        if activation is not None:
            resnet_layers_list.append(acfun)
    else:
        if batch_normalization:
            resnet_layers_list.append(bn)
        if activation is not None:
            resnet_layers_list.append(acfun)
        resnet_layers_list.append(conv)
    return resnet_layers_list


def resnet_v1(input_shape, depth,kernel_size=20):

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs1 = Input(shape=input_shape)
    inputs2 = Input(shape=input_shape)
    
    resnet_layers_list1 = get_resnet_layers(kernel_size=kernel_size)
    x1 = get_output(inputs1,resnet_layers_list1)
    x2 = get_output(inputs2,resnet_layers_list1)
    
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
                
            resnet_layers_list2 = get_resnet_layers(num_filters=num_filters,
                                                    strides=strides,kernel_size=kernel_size)
            tensor2_x1 = get_output(x1,resnet_layers_list2)
            tensor2_x2 = get_output(x2,resnet_layers_list2)
            
            
            resnet_layers_list3 = get_resnet_layers(num_filters=num_filters,
                                                    activation=None,kernel_size=kernel_size)
            tensor3_x1 = get_output(tensor2_x1,resnet_layers_list3)
            tensor3_x2 = get_output(tensor2_x2,resnet_layers_list3)
                
            if stack > 0 and res_block == 0:  # first layer but not first stack
                resnet_layers_list4 = get_resnet_layers(num_filters=num_filters,
                                                        kernel_size=1,strides=strides,
                                                        activation=None,batch_normalization=False)
                x1 = get_output(x1,resnet_layers_list4)
                x2 = get_output(x2,resnet_layers_list4)                
            
            add_layer = Add()
            ac_layer = Activation('relu')
            x1 = add_layer([x1, tensor3_x1])
            x2 = add_layer([x2, tensor3_x2])
            x1 = ac_layer(x1)
            x2 = ac_layer(x2)
        num_filters *= 2
        
    resnet_layers_list6 = [AveragePooling1D(pool_size=8),Flatten(),
                           Dense(128,activation='relu',kernel_initializer='he_normal'),
                           Dense(1,activation='sigmoid',kernel_initializer='he_normal')]
    outputs1 = get_output(x1,resnet_layers_list6)
    outputs2 = get_output(x2,resnet_layers_list6)
    
    outputs = Average()([outputs1, outputs2])
    # Instantiate model.
    model = Model(inputs=[inputs1,inputs2], outputs=outputs)
    return model




def resnet_v1_lstm(input_shape = (1000,6), depth = 20,kernel_size=20,attention_position='attention_after_lstm',
                   num_dense = 64, num_recurrent = 16,num_lstm = 1,dropout_rate = 0.1,cudnn = True,
                   single_attention_vector = True, rnn_dropout = [0.1,0.1],batch_norm = False,
                   num_meta = 0, num_denselayer = 0,merge = 'ave'):

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs1 = Input(shape=input_shape)
    inputs2 = Input(shape=input_shape)
    
    resnet_layers_list1 = get_resnet_layers(kernel_size=kernel_size)
    x1 = get_output(inputs1,resnet_layers_list1)
    x2 = get_output(inputs2,resnet_layers_list1)
    
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
                
            resnet_layers_list2 = get_resnet_layers(num_filters=num_filters,
                                                    strides=strides,kernel_size=kernel_size)
            tensor2_x1 = get_output(x1,resnet_layers_list2)
            tensor2_x2 = get_output(x2,resnet_layers_list2)
            
            
            resnet_layers_list3 = get_resnet_layers(num_filters=num_filters,
                                                    activation=None,kernel_size=kernel_size)
            tensor3_x1 = get_output(tensor2_x1,resnet_layers_list3)
            tensor3_x2 = get_output(tensor2_x2,resnet_layers_list3)
                
            if stack > 0 and res_block == 0:  # first layer but not first stack
                resnet_layers_list4 = get_resnet_layers(num_filters=num_filters,
                                                        kernel_size=1,strides=strides,
                                                        activation=None,batch_normalization=False)
                x1 = get_output(x1,resnet_layers_list4)
                x2 = get_output(x2,resnet_layers_list4)                
            
            add_layer = Add()
            ac_layer = Activation('relu')
            x1 = add_layer([x1, tensor3_x1])
            x2 = add_layer([x2, tensor3_x2])
            x1 = ac_layer(x1)
            x2 = ac_layer(x2)
        num_filters *= 2
        

    lstm_time_steps = int(x1.shape[1])
    if attention_position == 'lstm':
        fw_tensor_after_lstm,rc_tensor_after_lstm = get_lstm_output(x1,x2,num_lstm,num_recurrent,
                                                                    dropout_rate,cudnn)
        layers_lstm_to_meta = [Flatten(),Dense(num_dense, activation='relu')]
    elif attention_position == 'attention_before_lstm':
        fw_tensor_after_lstm,rc_tensor_after_lstm = attention_before_lstm_output(x1,x2,num_lstm,
                                                                                lstm_units=num_recurrent,time_steps=lstm_time_steps,
                                                                                rnn_dropout = dropout_rate,
                                                                                single_attention_vector=single_attention_vector,cudnn=cudnn)
        layers_lstm_to_meta = [Flatten(),Dense(num_dense, activation='relu')]
    elif attention_position == 'attention_after_lstm':
        fw_tensor_after_lstm,rc_tensor_after_lstm = attention_after_lstm_output(x1,x2,num_lstm,
                                                                                lstm_units=num_recurrent,time_steps=lstm_time_steps,
                                                                                rnn_dropout = dropout_rate,
                                                                                single_attention_vector=single_attention_vector,cudnn=cudnn)         
        layers_lstm_to_meta = [Flatten(),Dense(num_dense, activation='relu')]
    elif attention_position == 'attention1d_after_lstm_original':
        attention1d_layer = Attention1D(output_dim = num_dense)
        fw_tensor_after_lstm1,rc_tensor_after_lstm1 = get_lstm_output(x1,x2,num_lstm,num_recurrent,
                                                                    dropout_rate,cudnn,rnn_dropout[0],rnn_dropout[1])
        fw_tensor_after_lstm = get_output(fw_tensor_after_lstm1,[attention1d_layer])
        rc_tensor_after_lstm = get_output(rc_tensor_after_lstm1,[attention1d_layer])
        layers_lstm_to_meta = [Dense(num_dense, activation='relu')]
    elif attention_position == 'attention1d_after_lstm':
        fw_tensor_after_lstm1,rc_tensor_after_lstm1 = get_lstm_output(x1,x2,
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
        input_layers = [inputs1, inputs2, meta_input]
    else:
        input_layers = [inputs1, inputs2]
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
    # Instantiate model.
    model = Model(inputs=[inputs1,inputs2], outputs=output)
    return model

if __name__ == '__main__':
    from keras.utils import plot_model # require graphviz
    m = resnet_v1_lstm()
    plot_model(m,show_shapes=True,show_layer_names=False,to_file='model.png')