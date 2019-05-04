performance_file_path = '~/Dropbox/MU/workspace/encode_dream/data/feature_importance_analysis/'


plot_parameter_effect <- function(performance_data1,test_parameter = 'use_rnaseq'){
  library(ggplot2)
  library(reshape2)
  performance_data_all <- NULL
  for(performance_file in list.files(performance_file_path)){
    performance_data <- read.csv(paste0(performance_file_path,performance_file),row.names = 1,as.is = T)
    performance_data[is.na(performance_data)] <- 0
    unique_parameters <- which(apply(performance_data[grep('^au.+',colnames(performance_data),invert = T)], 2, function(x)length(unique(x))>1))
    
    if(test_parameter %in% names(unique_parameters)){
      tf_name = gsub('\\..+','',performance_file)
      performance_data1 <- cbind.data.frame(performance_data[,unique_parameters],'auPRC'=performance_data$auPRC,stringsAsFactors=F)
      unique_parameters <- which(apply(performance_data1[,-ncol(performance_data1)], 2, function(x)length(unique(x))>1))
      control_idx <- rep(T,nrow(performance_data1))
      
      for(unique_parameter in unique_parameters){
        control_idx <- control_idx &( performance_data1[,unique_parameter] == performance_data1[1,unique_parameter] )
      }
      treatment_idx <- performance_data1[,test_parameter] != performance_data1[1,test_parameter]
      performance_data2 <- performance_data1[control_idx | treatment_idx ,]
      performance_data2 <- cbind.data.frame('tf'=tf_name,performance_data2[,c(test_parameter,'auPRC')],stringsAsFactors=F)
      performance_data_all <- rbind(performance_data_all,performance_data2)
    }
    performance_data_all[,test_parameter] <- factor(performance_data_all[,test_parameter] )
  }
  # Change the position : interval between dot plot of the same group
  g <- ggplot(performance_data_all, aes_string(x=test_parameter, y='auPRC', fill=test_parameter))+ geom_dotplot(binaxis='y', stackdir='center') +
    facet_wrap(~tf, scale="free")+ theme(legend.position="none")
  return(g)
}


g <- plot_parameter_effect(performance_data1,test_parameter = 'ratio_negative')
plot(g)
