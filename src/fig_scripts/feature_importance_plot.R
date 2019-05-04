deepgrn_fi_path = '/mnt/data/chen/encode_dream_data/feature_importance/'
factornet_fi_path = '/mnt/data/chen/encode_dream_data/predictions_merge_fi_factornet/'

d_deepgrn_base <- read.csv('~/Dropbox/MU/workspace/encode_dream/performance/score/deepgrn_score.csv',as.is = T)
d_factornet_base <- read.csv('~/Dropbox/MU/workspace/encode_dream/performance/score/original_factornet.csv',as.is = T,row.names = 1)

###############################
d_deepgrn_fi <- matrix(NA,nrow = nrow(d_deepgrn_base),5)
colnames(d_deepgrn_fi) <- c('dnase','sequence','unique35','rnaseq','gencode')
for(i in 1:nrow(d_deepgrn_base)){
  for(f in colnames(d_deepgrn_fi)){
    filein <- paste0(deepgrn_fi_path,f,'/',d_deepgrn_base$TF[i],'/',d_deepgrn_base$TF[i],'.',d_deepgrn_base$cell[i],'.txt')
    x <- read.delim(filein,header = F,as.is = T)
    d_deepgrn_fi[i,f] <- d_deepgrn_base$auPRC[i] - as.numeric(gsub('auPRC: ','',x$V3))
  }
}
d_deepgrn_fi[d_deepgrn_base$TF=='NANOG',c('rnaseq','gencode','unique35')] <- NA
#####################################
d_factornet_fi <- matrix(NA,nrow = nrow(d_factornet_base),5)
colnames(d_factornet_fi) <- c('dnase','sequence','unique35','rnaseq','gencode')
for(i in 1:nrow(d_factornet_base)){
  for(f in colnames(d_factornet_fi)){
    filein <- paste0(factornet_fi_path,f,'/F.',d_factornet_base$TF[i],'.',d_factornet_base$cell[i],'.txt')
    if(file.exists(filein)){
      x <- read.delim(filein,header = F,as.is = T)
      d_factornet_fi[i,f] <- d_factornet_base$auPRC[i] - as.numeric(gsub('auPRC: ','',x$V3))
    }
  }
}

########################################

colnames(d_deepgrn_fi) <- colnames(d_factornet_fi) <- c('DNase','Unique35','Sequence','RNASeq','Gencode')


library(ggplot2)
library(reshape2)
df_deepgrn <- melt(d_deepgrn_fi,na.rm = T)
df_factornet <- melt(d_factornet_fi,na.rm = T)

df_deepgrn1 <- cbind.data.frame(df_deepgrn,'model'='With_attention',stringsAsFactors=F)
df_factornet1 <- cbind.data.frame(df_factornet,'model'='Without_attention',stringsAsFactors=F)
colnames(df_deepgrn1)[1] <- colnames(df_factornet1)[1] <- 'TF'
colnames(df_deepgrn1)[2] <- colnames(df_factornet1)[2] <- 'Feature'
colnames(df_deepgrn1)[3] <- colnames(df_factornet1)[3] <- 'd_auPRC'

df_all <- rbind.data.frame(df_deepgrn1,df_factornet1)


ggplot(df_all, aes_string(x='Feature', y='d_auPRC', fill='model')) + geom_boxplot()
