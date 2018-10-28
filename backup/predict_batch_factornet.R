# m_file = '/storage/htc/bdm/ccm3x/deepGRN/evaluate_factornet/factornet_best_models.csv'
# dst_path = '/storage/htc/bdm/ccm3x/deepGRN/evaluate_factornet/models/'
# 
# factornet_model_table <- read.csv(m_file,as.is = T)
# factornet_model_table1 <- factornet_model_table[factornet_model_table$multitask==0,]
# for(i in 1:nrow(factornet_model_table1)){
#   tf_name <- gsub('.+/models/(.+)/.+','\\1',factornet_model_table1$model[i])
#   unique35 <- rnaseq <- gencode <- 'False'
#   if(factornet_model_table1$unique35[i]==1)unique35 <- 'True'
#   if(factornet_model_table1$rnaseq[i]==1)rnaseq <- 'True'
#   if(factornet_model_table1$gencode[i]==1)gencode <- 'True'
#   model_file_name <- paste0(gsub('.+/(.+)','\\1',factornet_model_table1$model[i]),'.',tf_name,'.1.401.unique35',
#                             unique35,'.RNAseq',rnaseq,'.Gencode',gencode,'.h5')
#   file.copy(from = paste0(factornet_model_table1$model[i],'/best_model.hdf5'),to = paste0(dst_path,model_file_name))
# }

setwd('/storage/htc/bdm/ccm3x/deepGRN/logs')
library(data.table)
model_path = '/storage/htc/bdm/ccm3x/deepGRN/evaluate_factornet/models'
label_final_path = '/storage/htc/bdm/ccm3x/deepGRN/raw/label/final/'
out_path = '/storage/htc/bdm/ccm3x/deepGRN/evaluate_factornet/predictions/'

model_files = list.files(model_path,full.names = T)
for (i in model_files){
  tf_name <- unlist(strsplit(i,'.',fixed = T))[2]
  label_file <- paste0(label_final_path,tf_name,'.train.labels.tsv')
  cell_names <- colnames(fread(label_file,header = T,nrows = 3))
  cell_names <- cell_names[4:length(cell_names)]
  for(cell_name in cell_names){
    system(paste('sbatch ~/configs/py_sub_compatible.sbatch /storage/htc/bdm/ccm3x/deepGRN/src/model_predict.py',i,cell_name,out_path))
    
  }
}


