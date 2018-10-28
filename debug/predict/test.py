
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import model_from_json

model_file = '/home/ccm3x/test.h5'

model_json = '{"class_name": "Model", "keras_version": "1.2.2", "config": {"layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 1002, 6], "input_dtype": "float32", "sparse": false, "name": "input_1"}, "inbound_nodes": [], "name": "input_1"}, {"class_name": "Convolution1D", "config": {"batch_input_shape": [null, null, 6], "W_constraint": null, "b_constraint": null, "name": "convolution1d_1", "activity_regularizer": null, "trainable": true, "filter_length": 34, "init": "glorot_uniform", "bias": true, "nb_filter": 128, "input_dtype": "float32", "subsample_length": 1, "border_mode": "valid", "input_dim": 6, "b_regularizer": null, "W_regularizer": null, "activation": "relu", "input_length": null}, "inbound_nodes": [[["input_1", 0, 0]]], "name": "convolution1d_1"}], "input_layers": [["input_1", 0, 0]], "output_layers": [["convolution1d_1", 0, 0]], "name": "model_1"}}'
model = model_from_json(model_json)
model.load_weights(model_file)


x = np.loadtxt('/home/ccm3x/Dropbox/MU/workspace/encode_dream/data/test_models/x.csv')
x =np.reshape(x,(1,x.shape[0],x.shape[1]))

#x=np.zeros((1,model.get_input_shape_at(0)[1],model.get_input_shape_at(0)[2]))+1


y = model.predict(x)
y[0,range(100),4]

l=model.get_layer(index=1)
w1 = l.get_weights()[0]
w2 = l.get_weights()[1]


data1 = w1[:,0,:,4]
data2 = x[0,range(34),:]
ans=0
for i in range(6):
    ans += np.sum(np.multiply(data1[:,i],data2[:,i]))


np.sum(np.multiply(data1.flatten(),data2.flatten()))