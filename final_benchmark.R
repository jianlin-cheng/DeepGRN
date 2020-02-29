auprc <- read.csv('~/Dropbox/MU/workspace/encode_dream/performance/all_metric/auPRC.csv')
auroc <- read.csv('~/Dropbox/MU/workspace/encode_dream/performance/all_metric/auROC.csv')
fdr50 <- read.csv('~/Dropbox/MU/workspace/encode_dream/performance/all_metric/fdr50.csv')
fdr10 <- read.csv('~/Dropbox/MU/workspace/encode_dream/performance/all_metric/fdr10.csv')

dg <- read.csv('~/Dropbox/MU/workspace/encode_dream/performance/score/deepgrn_score.csv')
dg2 <- dg
dg2[1:2,] <- dg[c(2,1),]


score_mat <- matrix(0,nrow = 13,ncol = 5)

for(i in 1:nrow(score_mat)){
    auroc1 <- as.numeric(unlist(c(auroc[i,3:6],dg$auROC[i])))
    auprc1 <- as.numeric(unlist(c(auprc[i,3:6],dg$auPRC[i])))
    
    fdr501 <- as.numeric(unlist(c(fdr50[i,3:6],dg$Re.0.50.FDR[i])))
    fdr101 <- as.numeric(unlist(c(fdr10[i,3:6],dg$Re.0.10.FDR[i])))
    
    score_mat[i,] <- log(order(auroc1,decreasing = T)/6)+log(order(auprc1,decreasing = T)/6)+log(order(fdr501,decreasing = T)/6)+log(order(fdr101,decreasing = T)/6)
    
}
aaa <- t(apply(score_mat, 1, order))
colnames(aaa) <- c('Anchor','FactorNet','Cheburashka','Catchitt','DeepGRN')
aaa <- cbind(auprc[,1:2],aaa)
write.csv(aaa,'~/Dropbox/MU/workspace/encode_dream/performance/all_metric/rank.csv')
