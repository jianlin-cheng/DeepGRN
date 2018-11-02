args = commandArgs(T)

tf_name = args[1]
output_all_path = args[2]

tf_name = 'CTCF'
output_all_path = '/storage/htc/bdm/ccm3x/deepGRN/results_original/'

setwd('/storage/htc/bdm/ccm3x/deepGRN/logs/')
tf_path <- paste0(output_all_path,tf_name)
for(i in list.dirs(tf_path,recursive = F)){
  pred_file_names <- list.files(i,pattern = '*.tab.gz',full.names = T)
  if(length(pred_file_names)>0){
    for (pred_file_name in pred_file_names) {
      system(paste0('sbatch /storage/htc/bdm/ccm3x/deepGRN/src/batch/model_evaluate.sh ',pred_file_name,
                   ' ',i,'/performance.txt ',tf_path,'/'))
    }
  }
}