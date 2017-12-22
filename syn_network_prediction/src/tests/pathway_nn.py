import sys
sys.path.insert(0, 'C:/Users/chen/Documents/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src')
sys.path.insert(0, '/home/ccm3x/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src')
sys.path.insert(0, 'C:/Users/EBN307/Documents/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src')
import utils_test
import keras

# set parameters
feature_path = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/syn_prediction_model_data/features/'
label_file   = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/syn_prediction_model_data/label_matrix.csv'
performance_file_name =  '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/performance/performance_2dout.tsv'
pair_file_name = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/syn_prediction_model_data/pairs_selected.csv'
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


x_train = np.random.random_sample((100,30, 30))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random_sample((20,30, 30))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)


input_dim_x = [[12,3,4,5,6],[3,4,12,23,2],[1,2,4,5],[23,24,21]]
input_dim_y = [[3,4,5,6,15],[3,4,29,2],[13,1,2,5,5],[23,14,21]]

pathway_data_train = utils_test.reshape_pathway_data(x_train,input_dim_x,input_dim_y)
pathway_data_test = utils_test.reshape_pathway_data(x_test,input_dim_x,input_dim_y)

model = utils_test.get_pathway_nn_model((100,2,2,1),input_dim_x,input_dim_y,10,ext_pad,[[1,1],[1,2],[2,1],[2,2]],batchnorm=False,num_filters=2,num_conv2d_layers=2)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

history = model.fit(pathway_data_train,y_train,epochs=50, batch_size=32)