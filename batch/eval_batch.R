setwd('/storage/htc/bdm/ccm3x/deepGRN/logs')

out_path  = '/storage/htc/bdm/ccm3x/deepGRN/evaluate_chr11/score/'
predict_path = '/storage/htc/bdm/ccm3x/deepGRN/evaluate_chr11/predictions/'
final_result = '/storage/htc/bdm/ccm3x/deepGRN/evaluate_chr11/final_result.csv'

for(i in list.files(predict_path,full.names = T)){
  system(paste('sbatch ~/configs/r_sub.sbatch /storage/htc/bdm/ccm3x/deepGRN/src/model_evaluate.R',i,out_path))
}

res <- NULL
for(i in list.files(out_path)){
  d <- read.csv(paste0(out_path,i))
  res <- rbind(res,c(i,d$auPRC,d$auROC,d$recall_at_fdr_0.1,d$recall_at_fdr_0.5))
}
write.csv(res,final_result)