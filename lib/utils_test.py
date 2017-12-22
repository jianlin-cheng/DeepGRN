#shared functions for data parsing
import re
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from os.path import expanduser
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input,Conv1D, Conv2D, MaxPooling2D, Lambda, BatchNormalization, Reshape, Concatenate

def get_abs_path(input_path):
    home = expanduser("~")
    if re.match(r'^[A-Z]',home) :
        home = home + '\\Documents'
    input_path = re.sub(r'~',home,input_path)
    return(input_path)
    
def crop(start_i, end_i,start_j, end_j):
    end_i += 1
    end_j += 1
    def func(x):
        return x[:, start_i: end_i, start_j: end_j]
    return Lambda(func) 
    
def crop1d(start_i, end_i):
    end_i += 1
    def func(x):
        return x[:, start_i: end_i]
    return Lambda(func) 


def load_features(feature_path,multi_channel=True):
    feature_files = [f for f in listdir(feature_path) if isfile(join(feature_path, f))]
    feature_input = []

    for i in range(len(feature_files)):
        new_mat = pd.read_csv(feature_path+feature_files[i], sep=',',index_col=0)
        new_mat = new_mat.as_matrix()
        if multi_channel==True:
            new_mat = new_mat.reshape(new_mat.shape[0],new_mat.shape[1],1)
        feature_input.append(new_mat)

    feature_input_ndarray = np.array(feature_input,np.float)
    return(feature_input_ndarray)

def load_labels(label_file,feature_path):
    feature_files = [f for f in listdir(feature_path) if isfile(join(feature_path, f))]
    feature_order = list(map(lambda x: x[0:(len(x)-4)],feature_files))
    
    label_mat = pd.read_csv(label_file, sep=',',index_col=0)
    label_mat = label_mat.loc[feature_order].as_matrix()
    return(label_mat)
    
def load_interaction_pairs(pair_file_name):
    new_mat = pd.read_csv(pair_file_name, sep=',',index_col=0,dtype='int32')
    new_mat -= 1
    interaction_pairs = []
    for i in range(new_mat.shape[0]):
        interaction_pairs.append(list(new_mat.loc[i+1]))
    return(interaction_pairs)

def get_multiout_model(input_shape,output_dim,num_filters,num_conv2d_layers,num_cat,subsample = True,batchnorm = False):
    inputs = Input(shape=(input_shape[1],input_shape[2],input_shape[3],))
    x = inputs
    for i in range(0,num_conv2d_layers):
        x = Conv2D(num_filters,(3,3), activation='relu')(x)
        if batchnorm == True:
            x = BatchNormalization()(x)
    if subsample:    
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
    x = Flatten()(x)
    if batchnorm == True:
        x = BatchNormalization()(x)
    out_nodes = []
    for i in range(0,output_dim):
        out_nodes.append(Dense(num_cat, activation='softmax')(x))
    model = Model(inputs=inputs, outputs=out_nodes)
    return(model)

def get_2d_model(input_shape,num_filters,num_conv2d_layers,num_cat,ext_pad,interaction_pairs,batchnorm = False):
    inputs = Input(shape=(input_shape[1],input_shape[2],input_shape[3],))
    x = inputs
    for i in range(num_conv2d_layers):
        x = Conv2D(num_filters,(3,3), activation='relu',padding='same')(x)
        if batchnorm == True:
            x = BatchNormalization()(x)
    x = Conv2D(num_filters,(3,3), activation='relu',padding='same')(x)
    if batchnorm == True:
        x = BatchNormalization()(x)
    out_nodes = []
    for pair in interaction_pairs:
        [i,j] = pair
        start_i = max([0,i- ext_pad])
        end_i   = min([inputs.shape[1].value,i+ext_pad])
        start_j = max([0,j- ext_pad])
        end_j   = min([inputs.shape[2].value,j+ext_pad])
        out_nodes.append(Dense(num_cat, activation='softmax')(Flatten()(crop(start_i, end_i,start_j, end_j)(x)))) 
    model = Model(inputs=inputs, outputs=out_nodes)
    return(model)

def get_pathway_nn_model(input_shape,input_dim_x,input_dim_y,num_cat,ext_pad,interaction_pairs,batchnorm,num_filters,num_conv2d_layers):
    inputs = []
    input_nodes = []
    for i in range(len(input_dim_x)):
        inputs.append(Input(shape=(len(input_dim_x[i]),len(input_dim_y[i]),1)))
    for i in inputs:
        input_nodes.append(Dense(1, activation='relu')(Flatten()(i)))
    x = Concatenate()(input_nodes)
    x = Reshape((input_shape[1],input_shape[2]))(x)
    for i in range(num_conv2d_layers):
        x = Conv2D(num_filters,(3,3), activation='relu',padding='same')(x)
        if batchnorm == True:
            x = BatchNormalization()(x)
    out_nodes = []
    for pair in range(len(interaction_pairs)):
        start_i = max([0,pair- ext_pad])
        end_i   = min([len(interaction_pairs),pair+ext_pad])
        out_nodes.append(Dense(num_cat, activation='softmax')(Flatten()(crop1d(start_i, end_i)(x)))) 
    model = Model(inputs=inputs, outputs=out_nodes)
    return(model)

def viz_model(history,outfile):
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.subplot(1,2,2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    
    plt.savefig(outfile)  
    plt.show()
    plt.close()    
    
def output_model(history):
    model_info = pd.DataFrame({'val_loss':history.history['val_loss'],'val_acc':history.history['val_acc'],'loss':history.history['loss'],'acc':history.history['acc']})
    return(model_info)

def reshape_pathway_data(x,input_dim_x,input_dim_y):
    pathway_data = []
    for i in range(len(input_dim_x)):
        new_data = []
        for j in range(x.shape[0]):
            df = pd.DataFrame(x[j], index=range(x[0].shape[0]), columns=range(x[0].shape[1]))
            new_data.append(df.loc[input_dim_x[i], input_dim_y[i]].values)
        new_data_all = np.asarray(new_data)
        pathway_data.append(new_data_all.reshape(new_data_all.shape[0],new_data_all.shape[1],new_data_all.shape[2],1))
    return(pathway_data)






