import sys
sys.path.insert(0, 'C:/Users/chen/Documents/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src')
sys.path.insert(0, '/home/ccm3x/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src')
sys.path.insert(0, 'C:/Users/EBN307/Documents/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src')
import utils_test
import keras

# set parameters
feature_path = '/storage/htc/bdm/ccm3x/syn_network_prediction/syn_model/pathway_nn_model_data/features/'
label_file   = '/storage/htc/bdm/ccm3x/syn_network_prediction/syn_model/pathway_nn_model_data/label_matrix.csv'
performance_file_name =  '/storage/htc/bdm/ccm3x/syn_network_prediction/syn_model/performance/pathway/performance_sub0.2.tsv'
pair_file_name = '/storage/htc/bdm/ccm3x/syn_network_prediction/syn_model/pathway_nn_model_data/pairs_selected.csv'
ext_pad = 1
num_cat = 3
num_filters = 8
num_conv2d_layers = 5
output_dim = 'all'
batchnorm = False

# load model data
feature_input_ndarray = utils_test.load_features(feature_path,multi_channel=False)
label_mat = utils_test.load_labels(label_file,feature_path)

#test code
#feature_input_ndarray = feature_input_ndarray[range(50)]
#label_mat = label_mat[range(50)]

if output_dim == 'all':
    output_dim = label_mat.shape[1]
label_mat=label_mat[:,0:output_dim]
y_tr = []
for i in range(output_dim):
    y_tr.append(keras.utils.to_categorical(label_mat[:,i], num_cat))
interaction_pairs = utils_test.load_interaction_pairs(pair_file_name)
interaction_pairs = interaction_pairs[0:output_dim]

input_dim_df = utils_test.pd.read_csv('/storage/htc/bdm/ccm3x/syn_network_prediction/syn_model/subnetworks/sub_0.2.csv')
input_dim_x = list(map(lambda x: list(map(int,x.split('_'))), input_dim_df['V1'].values))
input_dim_y = list(map(lambda x: list(map(int,x.split('_'))), input_dim_df['V2'].values))

pathway_data_train = utils_test.reshape_pathway_data(feature_input_ndarray,input_dim_x,input_dim_y)

model = utils_test.get_pathway_nn_model(feature_input_ndarray.shape,input_dim_x,input_dim_y,num_cat,ext_pad,interaction_pairs,batchnorm=False,num_filters=2,num_conv2d_layers=2)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(pathway_data_train,y_tr,epochs=100, batch_size=32,validation_split=0.1)
utils_test.pd.DataFrame(history.history).to_csv(performance_file_name, sep='\t',index=False)
