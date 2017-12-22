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
gene_names = read.delim('~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/gene_names_lib.txt',sep = ' ',as.is = T,header = F)$V1

generate_feature(out_path,gene_names)

# get label indices for 2D model
co_exp1 <- read.csv(paste(out_path,'syn_prediction_model_data/features/expr_model1_rep1.csv',sep = ''),row.names = 1,header = T)
label_mat <- read.csv(paste(out_path,'syn_prediction_model_data/label_matrix.csv',sep = ''),row.names = 1,header = T)
interaction_pairs <- strsplit(colnames(label_mat),split = '_')
interaction_pairs2 <- lapply(interaction_pairs, function(x)return(c(which(rownames(co_exp1)==x[1]),which(colnames(co_exp1)==x[2]))))
interaction_pairs3 <- matrix(unlist(interaction_pairs2), ncol = 2, byrow = TRUE)
write.csv(interaction_pairs3,paste(out_path,'syn_prediction_model_data/pairs_selected.csv',sep = ''),quote = F)


# generate full co-exp matrix for pathway NN
exp_path <- '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/syn_rna_expression/'
out_path <- '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/pathway_nn_model_data/'
for(i in list.files(exp_path)){
  exp_data <- read.csv(paste(exp_path,i,sep = ''),row.names = 1)
  co_exp <- cor(t(exp_data),method = 'spearman')
  co_exp[is.na(co_exp)] <- 0
  co_exp <- round(co_exp,2)
  write.csv(co_exp,paste(out_path,'features/',i,sep = ''))
}
write.csv(label_mat,paste(out_path,'label_matrix.csv',sep = ''))
co_exp1 <- co_exp
interaction_pairs <- strsplit(colnames(label_mat),split = '_')
interaction_pairs2 <- lapply(interaction_pairs, function(x)return(c(which(rownames(co_exp1)==x[1]),which(colnames(co_exp1)==x[2]))))
interaction_pairs3 <- matrix(unlist(interaction_pairs2), ncol = 2, byrow = TRUE)
write.csv(interaction_pairs3,paste(out_path,'/pairs_selected.csv',sep = ''),quote = F)
write.csv(matrix(unlist(interaction_pairs), ncol = 2, byrow = TRUE),paste(out_path,'/pairs_selected_names.csv',sep = ''),quote = F)
