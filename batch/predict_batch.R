setwd('/storage/htc/bdm/ccm3x/deepGRN/logs')
library(data.table)

model_path = '/storage/htc/bdm/ccm3x/deepGRN/evaluate/models'
label_final_path = '/storage/htc/bdm/ccm3x/deepGRN/raw/label/final/'
out_path = '/storage/htc/bdm/ccm3x/deepGRN/evaluate/predictions/'

model_files = list.files(model_path,full.names = T)


for (i in model_files){
  tf_name <- unlist(strsplit(i,'.',fixed = T))[2]
  label_file <- paste0(label_final_path,tf_name,'.train.labels.tsv')
  cell_names <- colnames(fread(label_file,header = T,nrows = 3))
  cell_names <- cell_names[4:length(cell_names)]
  for(cell_name in cell_names){
    system(paste('sbatch ~/configs/py_sub.sbatch /storage/htc/bdm/ccm3x/deepGRN/src/model_predict.py',i,cell_name,out_path))
  }
}


