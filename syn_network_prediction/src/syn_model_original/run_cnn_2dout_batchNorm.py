import sys
sys.path.insert(0, 'C:/Users/chen/Documents/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src')
sys.path.insert(0, '/home/ccm3x/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src')
sys.path.insert(0, 'C:/Users/EBN307/Documents/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src')
import utils_test
import keras

# set parameters
feature_path = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model_original/syn_prediction_model_data/features/'
label_file   = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model_original/syn_prediction_model_data/label_matrix.csv'
performance_file_name =  '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model_original/performance/performance_2dout_0.1_batchNorm.tsv'
pair_file_name = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model_original/syn_prediction_model_data/pairs_selected.csv'
ext_pad = 1
num_cat = 3
num_filters = 1
num_conv2d_layers = 2
output_dim = 'all'

# load model data
feature_input_ndarray = utils_test.load_features(utils_test.get_abs_path(feature_path))
label_mat = utils_test.load_labels(utils_test.get_abs_path(label_file),utils_test.get_abs_path(feature_path))
if output_dim == 'all':
    output_dim = label_mat.shape[1]
y_tr=label_mat[:,0:output_dim]
y_tr = []
for i in range(output_dim):
    y_tr.append(keras.utils.to_categorical(label_mat[:,i], num_cat))
interaction_pairs = utils_test.load_interaction_pairs(utils_test.get_abs_path(pair_file_name))
interaction_pairs = interaction_pairs[0:output_dim]

# construct model
model = utils_test.get_2d_model(feature_input_ndarray.shape,num_filters,num_conv2d_layers,num_cat,ext_pad,interaction_pairs)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(feature_input_ndarray, y_tr, batch_size=32, epochs=120,validation_split=0.1)
utils_test.pd.DataFrame(history.history).to_csv(utils_test.get_abs_path(performance_file_name), sep='\t',index=False)
