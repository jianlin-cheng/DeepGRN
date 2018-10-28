eval_path = '/storage/htc/bdm/ccm3x/deepGRN/results/set1/'

setwd('/storage/htc/bdm/ccm3x/deepGRN/logs')

out_path  = paste0(eval_path,'score/')
predict_path = paste0(eval_path,'predictions/')
use_blacklist = 'F'

if(use_blacklist){
  final_result = paste0(eval_path,'final_result_blacklist.csv')
}else{
  final_result = paste0(eval_path,'final_result.csv')
}

for(i in list.files(predict_path,full.names = T)){
  system(paste('sbatch ~/configs/interactive_sub.sbatch Rscript /storage/htc/bdm/ccm3x/deepGRN/src/model_evaluate.R',i,out_path,use_blacklist))
}


# wait for all jobs 
res <- NULL
for(i in list.files(out_path)){
  d <- read.csv(paste0(out_path,i))
  res <- rbind(res,c(i,d$auPRC,d$auROC,d$recall_at_fdr_0.1,d$recall_at_fdr_0.5))
}
write.csv(res,final_result)