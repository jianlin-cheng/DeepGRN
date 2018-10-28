library(pheatmap)
library(dplyr)
library(reshape2)

score_path = '/storage/htc/bdm/ccm3x/deepGRN/evaluate/score/'
out_path = '~/Dropbox/MU/workspace/encode_dream/performance/'

score <- NULL
for(i in list.files(score_path)){
  d <- unlist(strsplit(i,split = '.',fixed = T))
  s <- read.csv(paste0(score_path,i),row.names = 1)
  score <- rbind(score,c(d[1],d[2],d[3],s$auPRC,s$auROC,s$accuracy))
}
colnames(score) <- c('model','tf_name','bin_size','auPRC','auROC','accuracy')

score[1:8,2] <- 'CTCF_ipsc'
score[9:16,2] <- 'CTCF_PC-3'

for (i in colnames(s)) {
  for(j in c('fullconv_simple','fullconv_lstm')){
    score1 <- data.frame(score[grep(j,score[,1]),c('tf_name','bin_size',i)],stringsAsFactors=F)
    score1[,3] <- as.numeric(score1[,3])
    score2 <- acast(score1, tf_name~bin_size, value.var=i,fill = 0)
    score2 <- score2[,c('1','5','11','21')]
    write.csv(score2,paste0(out_path,'data/',i,'_',j,'.csv'))
    png(paste0(out_path,'plots/',i,'_',j,'.png'))
    pheatmap(score2,cluster_cols = F,main = i)
    dev.off()
  }

}