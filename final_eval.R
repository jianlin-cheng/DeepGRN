args <- commandArgs(T)
result_path = args[1]

final_result <- NULL
exp_names <- c()
for(i in list.dirs(result_path,recursive = F,full.names = F)){
  fn <- paste0(result_path,i,'/performance.txt')
  if(file.exists(fn)){
    exp_names <- c(exp_names,i)
    d <- as.character(read.delim(fn,header=F,as.is=T))[1:8]
    x <- gsub('.+: ','',d)
    names(x) <- gsub(':.+','',d)
    final_result <- rbind(final_result,x)
  }
}
final_result_blacklisted <- NULL
exp_names_blacklisted <- c()
for(i in list.dirs(result_path,recursive = F,full.names = F)){
  fn <- paste0(result_path,i,'/performance.txt.blacklisted')
  if(file.exists(fn)){
    exp_names_blacklisted <- c(exp_names_blacklisted,i)
    d <- as.character(read.delim(fn,header=F,as.is=T))[1:8]
    x <- gsub('.+: ','',d)
    names(x) <- gsub(':.+','',d)
    final_result_blacklisted <- rbind(final_result_blacklisted,x)
  }
}

rownames(final_result) <- exp_names
rownames(final_result_blacklisted) <- exp_names_blacklisted
hyper_parameters <- read.csv(paste0(result_path,'hyperparameters.csv'),as.is = T,row.names = 1)

get_results_with_parameters <- function(hyper_parameters,final_result){
  hyper_parameters_with_results <- cbind.data.frame(hyper_parameters,'auROC'=NA,'auPRC'=NA,'auROC_att'=NA,'auPRC_att'=NA)
  final_result_att <- final_result[grep('attention',rownames(final_result)),]
  final_result_noatt <- final_result[grep('attention',rownames(final_result),invert = T),]
  
  hyper_parameters_with_results[gsub('^.+set','',rownames(final_result_att)),'auROC_att'] <- final_result_att[,2]
  hyper_parameters_with_results[gsub('^.+set','',rownames(final_result_att)),'auPRC_att'] <- final_result_att[,3]
  hyper_parameters_with_results[gsub('^.+set','',rownames(final_result_noatt)),'auROC'] <- final_result_noatt[,2]
  hyper_parameters_with_results[gsub('^.+set','',rownames(final_result_noatt)),'auPRC'] <- final_result_noatt[,3]
  return(hyper_parameters_with_results)
}
hyper_parameters_with_results <- get_results_with_parameters(hyper_parameters,final_result)
hyper_parameters_with_results_bl <- get_results_with_parameters(hyper_parameters,final_result_blacklisted)


write.csv(final_result,paste0(result_path,'performance_all.csv'))
write.csv(hyper_parameters_with_results,paste0(result_path,'hyper_parameters_with_results.csv'))
write.csv(hyper_parameters_with_results_bl,paste0(result_path,'hyper_parameters_with_results_bl.csv'))
