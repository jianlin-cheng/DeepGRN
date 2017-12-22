library(igraph)
network_path = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/random_scale_free_network/network_adj_1.csv'
subnetwork_path = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/subnetworks/'
pair_selected <- read.csv('~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/pathway_nn_model_data/pairs_selected_names.csv',row.names = 1,as.is = T)
set.seed(2017)
network1 <- read.csv(network_path,row.names = 1)
network1[network1!=0] <- 1


co_exp <- read.csv('~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/pathway_nn_model_data/features/expr_model1_rep1.csv',row.names = 1)
gene_mapping <- rownames(co_exp)
names(gene_mapping) <- 1:nrow(co_exp)
pair_selecte_vec <- paste(pair_selected$V1,pair_selected$V2,sep = '_')


el <- NULL
for(i in 1:nrow(network1)){
  for(j in 1:ncol(network1)){
    gene_i <- rownames(network1)[i]
    gene_j <- colnames(network1)[j]
    if(paste(gene_i,gene_j,sep = '_')%in%pair_selecte_vec){
      n_i <- colnames(network1)[network1[gene_i,]==1]
      n_j <- rownames(network1)[network1[,gene_j]==1]
      
      n_i_final <- unique(c(n_i,gene_i,colnames(network1)[j]))
      n_j_final <- unique(c(n_j,gene_j,rownames(network1)[i]))
      n_i_final <- names(gene_mapping[gene_mapping%in%n_i_final])
      n_j_final <- names(gene_mapping[gene_mapping%in%n_j_final])
      
      el <- rbind(el,c(paste(as.numeric(n_i_final)-1,collapse = '_'),paste(as.numeric(n_j_final)-1,collapse = '_')))
    }else{
      ind_i <- as.numeric(names(gene_mapping)[gene_mapping==gene_i])-1
      ind_j <- as.numeric(names(gene_mapping)[gene_mapping==gene_j])-1
      el <- rbind(el,rep(paste(ind_i,ind_j,sep = '_'),2))
    }
  }
}
write.csv(el,paste(subnetwork_path,'pathway_all.csv',sep = ''))


# for(i in seq(0.2,1,0.2)){
#   el_permute <- el
#   el_permute[sample(1:nrow(el),size = nrow(el)*(1-i)),] <- el[sample(1:nrow(el),size = nrow(el)*(1-i)),]
#   write.csv(el_permute,paste(subnetwork_path,'sub_',i,'.csv',sep = ''))
# }
