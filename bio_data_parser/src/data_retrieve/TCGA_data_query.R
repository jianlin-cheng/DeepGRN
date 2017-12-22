source('~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/src/utils.R')
# Download TCGA data
# library(TCGAbiolinks)
# data_mapping <- read.table('~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/kegg_tcga_mapping.txt',as.is = T)
# for(p in data_mapping$V2[-1]){
#   for(q in unlist(strsplit(p,','))){
#     query <- GDCquery(project = q,
#                       data.category = "Transcriptome Profiling",
#                       data.type = "Gene Expression Quantification",
#                       workflow.type = "HTSeq - Counts")
#     data <- GDCdownload(query)
#     cnt_p=GDCprepare(query,summarizedExperiment = F,save = T,save.filename = 'AML.rda')
#     cnt_p <- cnt_p[grep('^ENSG',cnt_p$X1),]
#     write.csv(cnt_p,paste('~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/features/expression_matrix/',q,'.csv',sep = ''))  
#   }
# }

# Get co-expression matrix
library(dplyr)
data_path = '~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/features/expression_matrix/'
out_path = '~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/features/cor_matrix/'
edge_data <- read.csv('~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/kegg_labels.csv',header = T,row.names = 1)
node_data <- read.csv('~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/gene_loci_sort.csv',header = T,row.names = 1,as.is = T)

data_list <- list()
for(i in list.files(data_path)){
  print(i)
  data1 <- read.csv(paste(data_path,i,sep = ''))[,-1]
  data1$X1 <- gsub('\\..+','',data1$X1)
  data2 <- data1 %>% group_by(X1)%>% summarise_all(funs(sum))
  data3 <- data.frame(data2[,-1])
  rownames(data3) <- data2$X1
  data3 <- data3[intersect(rownames(data3),node_data$ensembl_gene_id),]
  data_list[[gsub('.csv','',i)]] <- data3
}
gene_intersect <- Reduce(intersect,lapply(data_list,rownames))
gene_intersect <- node_data$ensembl_gene_id[node_data$ensembl_gene_id %in% gene_intersect]

for(i in names(data_list)){
  grps <- make_groups(ncol(data_list[[i]]),30)
  for(j in 1:length(grps)){
    data_m <- data_list[[i]][gene_intersect,grps[[j]]]
    cor_m <- cor(t(data_m),method = 'spearman')
    cor_m[is.na(cor_m)] <- 0
    cor_m <- round(cor_m,3)
    write.csv(cor_m,paste(out_path,i,'_',j,'.csv',sep = ''))
  }
}

# Get label matrix for prediction
network_data <- read.csv('~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/kegg_labels.csv',header = T,row.names = 1)
kegg_mapping<- read.delim('~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/kegg_tcga_mapping.txt',sep = ' ',header = F,as.is = T)
cor_m <- read.csv('~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/features/cor_matrix/TARGET-AML_1.csv',row.names = 1,header = T)
gene_intersect <- rownames(cor_m)
network_pred <- NULL
network_pred_names <- c()
all_samples <- gsub('.csv','',list.files(out_path))
for(i in 1:nrow(kegg_mapping)){
  network_slt <- kegg_mapping$V1[i]
  project_slt <- unlist(strsplit(kegg_mapping$V2[i],',')[[1]])
  for(p in project_slt){
    samples_slt <- grep(p,all_samples,value = T)
    network_pred_names <- c(network_pred_names, samples_slt)
    network_pred <- rbind(network_pred,matrix(rep(as.numeric(network_data[network_slt,]),length(samples_slt)),nrow = length(samples_slt),byrow = T))
  }
}

pair_list <- strsplit(colnames(network_pred),'_')
network_pred <- network_pred[,unlist(lapply(pair_list,function(x)sum(x%in%gene_intersect)==2))]

pair_list <- strsplit(colnames(network_pred),'_')
genes <- 1:ncol(cor_m)
names(genes) <- colnames(cor_m)
pair_mat <- NULL
for(i in pair_list){
  pair_mat <- rbind(pair_mat,c(genes[i[1]],genes[i[2]]))
}
colnames(pair_mat) <- NULL

write.csv(network_pred,'~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/label_matrix.csv')
write.csv(pair_mat,'~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/pairs_selected.csv')



