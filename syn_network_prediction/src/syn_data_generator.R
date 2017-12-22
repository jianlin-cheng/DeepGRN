source('~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src/rcode/keras_utils.R')

num_network = 500
num_batch_per_graph = 4
total_nodes = 500
interaction_mut_prob = 0.2
repr_percentage = 0.3
num_sample_per_batch = 10
num_external_genes = 10

# parameters for sgnesR
# (1) Translation rate, (2) RNA degradationrate, (3) Protein degradation rate, (4) Protein binding rate, 
# (5) unbinding rate, (6) transcription rate
rc = c(0.002, 0.005, 0.005, 0.005, 0.01, 0.02)
out_path = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/'
gene_names = read.delim('~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model_original/gene_names_lib.txt',sep = ' ',as.is = T,header = F)$V1


generate_feature(out_path,gene_names)