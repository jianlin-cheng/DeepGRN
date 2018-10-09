library(data.table)

model_path = '/home/chen/data/deepGRN/evaluate_chr11/models'
label_final_path = '/home/chen/data/deepGRN/raw/label/final/'

model_files = list.files(model_path,full.names = T)


for (i in model_files){
  tf_name <- unlist(strsplit(i,'.',fixed = T))[2]
  label_file <- paste0(label_final_path,tf_name,'.train.labels.tsv')
  cell_names <- colnames(fread(label_file,header = T,nrows = 3))
  cell_names <- cell_names[4:length(cell_names)]
  for(cell_name in cell_names){
    write(paste('python ~/Dropbox/MU/workspace/encode_dream/src/local/model_predict.py',i,cell_name),
               file = '~/test.sh',append=TRUE)
    
  }
}


