d_factornet <- read.csv('~/Dropbox/MU/workspace/encode_dream/performance/score/original_factornet.csv',header = T,row.names = 1,as.is = T)
d_deepgrn <- read.csv('~/Dropbox/MU/workspace/encode_dream/performance/score/deepgrn_score.csv',header = T,as.is = T)
# d_deepgrn = d_deepgrn[1:11,]
par(mfrow=c(2,2))

aaa <- c('auROC','auPRC','Recall at 0.5','Recall at 0.1')
for(i in 3:6){
  v1 <- as.numeric(d_factornet[,i+1])
  v2 <- as.numeric(d_deepgrn[,i])
  x_min <- c(0.95,0,0,0)
  x_max <- 1
  y_min <- c(0.95,0,0,0)
  y_max <- 1
  
  plot(v1,v2,xlab = 'Without attention',ylab = 'With attention',main = aaa[i-2],
       xlim = c(x_min[i-2],x_max), ylim = c(y_min[i-2],y_max),col='blue',pch=16)
  # text(v1,v2,label=d$V1, cex=0.6, pos=4)
  lines(x = c(x_min[i-2],x_max), y = c(y_min[i-2],y_max),lty=2)
}

auroc <- d_factornet$auROC
auroc_att <- d_deepgrn$auROC
auprc <- d_factornet$auPRC
auprc_att <- d_deepgrn$auPRC
    
avg_auroc <- mean(auroc)
avg_auroc_att <- mean(auroc_att)

avg_auprc <- mean(auprc)
avg_auprc_att <- mean(auprc_att)


# percentage of incerease
(avg_auroc_att-avg_auroc)/avg_auroc

(avg_auprc_att-avg_auprc)/avg_auprc

auprc_improve <- auprc_att-auprc
names(auprc_improve) <- d_factornet$TF
sort(auprc_improve,T)

auprc_merge <- auprc[2:nrow(d_factornet)]
auprc_merge[1] <- mean(auprc[1:2])

auprc_merge_att <- auprc_att[2:nrow(d_factornet)]
auprc_merge_att[1] <- mean(auprc_att[1:2])

improve_auprc_merge <- auprc_merge_att-auprc_merge
theta <- round(cor(auprc_merge,improve_auprc_merge),4)
rho <- round(cor(auprc_merge,improve_auprc_merge,method = 'spearman'),4)

par(mfrow=c(1,1))
plot(auprc_merge,improve_auprc_merge,col='blue',pch=16,xlab = 'auPRC without attention',ylab = 'Improvements in auPRC')
lines(x = c(min(auprc_merge),max(auprc_merge)), y = c(max(improve_auprc_merge),min(improve_auprc_merge)),lty=2)
# text('topright',paste0('theta=',theta,'\nrho=',rho))

#ensemble
d <- read.csv('~/Dropbox/MU/workspace/encode_dream/performance/score/ensemble.csv',header = T,as.is = T,row.names = 1)
o <- order(apply(d[,3:5], 1, mean))
plot(1:nrow(d),d$without_attention[o],pch=1,col=3,ylim = c(0,max(c(d$without_attention,d$with_attention,d$ensemble))),
     ylab = 'auPRC',xlab = '',xaxt="n")
points(1:nrow(d),d$with_attention[o],pch=2,col=2)
points(1:nrow(d),d$ensemble[o],pch=3,col=4)
x_names <- d$TF[o]
x_names[12] <- paste0(x_names[12],'\n(PC-3)')
x_names[13] <- paste0(x_names[13],'\n(iPSC)')
axis(1, at=1:nrow(d),labels=x_names, las=2)
legend('topleft',legend=c('Without Attention','With Attention','Ensemble'),pch=c(1,2,3),col=c(3,2,4),bty = "n")



