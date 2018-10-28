score_path = '/storage/htc/bdm/ccm3x/FactorNet/score/'

merged_pred = '/storage/htc/bdm/ccm3x/FactorNet/merged_predictions/'
pred_factornet = '/storage/htc/bdm/ccm3x/FactorNet/score/pred_factornet/'
true_labels = '/storage/htc/bdm/ccm3x/FactorNet/score/true_labels/'

setwd(merged_pred)
for(i in list.files(merged_pred)){
  dst_folder <- paste0(pred_factornet,gsub('.bed','/',i))
  dir.create(dst_folder)
  d <- unlist(strsplit(i,'.',fixed = T))
  out_file <- paste0(dst_folder,'F.',d[1],'.',d[2],'.tab')
  file.copy(i,out_file)
  system(paste0('gzip -c ',out_file,' > ',out_file,'.gz'))
}

for (i in list.files()) {
  print(i)
  system(paste0('head ',i,' -n 1 > ',true_labels,i))
  system(paste0('awk -F"\t" \'$1 == "chr1" { print }\' ',i,' >> ',true_labels,i))
  system(paste0('awk -F"\t" \'$1 == "chr21" { print }\' ',i,' >> ',true_labels,i))
  system(paste0('awk -F"\t" \'$1 == "chr8" { print }\' ',i,' >> ',true_labels,i))
}
setwd(true_labels)
for(i in list.files()){
  system(paste0('gzip -c ',i,' > ',i,'.gz'))
}
setwd(score_path)
for(i in list.dirs(pred_factornet,recursive = F)){
  pred_file <- list.files(i,pattern = '*.gz')
  system(paste0('sbatch ~/configs/py2_cpu_sub.sbatch python score.py ',i,'/',pred_file,' ',true_labels,' ',i,'/',gsub('tab.gz','score',pred_file)))
}

final_score <- NULL
for(i in list.dirs(pred_factornet,recursive = F)){
  exp_name <- gsub('.+/','',i)
  final_score <- rbind(final_score,data.frame(exp_name,read.delim(list.files(i,pattern = '.score'))))
}


colnames(final_score) <- c('model_name','auROC','auPRC','FDR_10','FDR_50')
write.csv(final_score,paste0(score_path,'score_summary.csv'),quote = F,row.names = F,col.names = F)






