setwd('/storage/htc/bdm/ccm3x/deepGRN/logs')

predict_path = '/storage/htc/bdm/ccm3x/deepGRN/evaluate_factornet/predictions/'
out_path  = '/storage/htc/bdm/ccm3x/deepGRN/evaluate_factornet/score/'

for(pred_file in list.files(predict_path,full.names = T)){
  system(paste('sbatch ~/configs/r_sub.sbatch /storage/htc/bdm/ccm3x/deepGRN/src/model_evaluate.R',pred_file,out_path))
  
}


