args = commandArgs(T)

tf_name = args[1]
result_path = args[2]

tf_name = 'E2F1'
result_path = '/storage/htc/bdm/ccm3x/deepGRN/results_original/E2F1/factornet_attention.set7'

setwd('/storage/htc/bdm/ccm3x/deepGRN/logs/')
model_file = list.files(result_path,pattern = '.+\\.h5$',full.names = T)

if(length(model_file)>0){
  h <- read.delim(paste0('/storage/htc/bdm/ccm3x/deepGRN/raw/label/final_unzip/',tf_name,'.train.labels.tsv'),
                  header = F,nrows = 1,as.is = T)
  cell_names <- as.character(h)[4:ncol(h)]
  for(cell_name in cell_names){
    system(paste0('sbatch /storage/htc/bdm/ccm3x/deepGRN/src/batch/model_predict.sh ',model_file,' ',
                  cell_name,' ',result_path,'/'))
  }

}

