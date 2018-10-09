aaa=read.csv('~/Dropbox/MU/workspace/encode_dream/performance/data/final_result.csv',as.is = T)


sample_info <- matrix(unlist(strsplit(gsub('.csv','',aaa$V1),'.',fixed = T)),ncol = 4,byrow = T)
colnames(sample_info) <- c('model_name','unique35','RNAseq','gencode')
sample_info[,2] <- as.numeric(sample_info[,2] =='unique35True')
sample_info[,3] <- as.numeric(sample_info[,3] =='RNAseqTrue')
sample_info[,4] <- as.numeric(sample_info[,4] =='GencodeTrue')

idx <- 1:8
xxx <- xxx1 <- cbind.data.frame(sample_info[idx,],aaa[idx,3:6])

for(i in 5:ncol(xxx1)){
  xxx1[,i] <- order(xxx1[,i])
}
xxx2 <- apply(xxx1[,5:8], 1, function(x)sum(log2(x)))
xxx1 <- cbind(xxx1,xxx2)
colnames(xxx)[5:8] <- c('auPRC','auROC','Recall_at_0.1_FDR','Recall_at_0.5_FDR')
colnames(xxx1)[5:9] <- c('auPRC','auROC','Recall_at_0.1_FDR','Recall_at_0.5_FDR','score')
