
predict_path = '/storage/htc/bdm/ccm3x/deepGRN/evaluate/predictions/'
tf_names = unique(unlist(lapply(strsplit(list.files(predict_path),'.',fixed = T),function(x)x[2])))

for(tf_name in tf_names){
  system(paste('sbatch ~/configs/r_sub.sbatch /storage/htc/bdm/ccm3x/deepGRN/src/model_evaluate.R',tf_name))
  
}


