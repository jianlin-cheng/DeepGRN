d_single <- read.csv('~/Dropbox/MU/workspace/encode_dream/data/revision202001/deepgrn_single.csv',header = T,as.is = T)
d_pair <- read.csv('~/Dropbox/MU/workspace/encode_dream/data/revision202001/deepgrn_pair.csv',header = T,as.is = T)
# d_deepgrn = d_deepgrn[1:11,]
par(mfrow=c(2,2))

aaa <- c('auROC','auPRC','Recall at 0.5','Recall at 0.1')

for(i in 3:6){
  v1 <- as.numeric(d_single[,i])
  v2 <- as.numeric(d_pair[,i])
  x_min <- c(0.95,0,0,0)
  x_max <- 1
  y_min <- c(0.95,0,0,0)
  y_max <- 1
  
  plot(v1,v2,xlab = 'Single attention',ylab = 'Pairwise attention',main = aaa[i-2],
       xlim = c(x_min[i-2],x_max), ylim = c(y_min[i-2],y_max),col='blue',pch=16)
  # text(v1,v2,label=d$V1, cex=0.6, pos=4)
  lines(x = c(x_min[i-2],x_max), y = c(y_min[i-2],y_max),lty=2)
  pea_cor <- format(cor(v1,v2,method = 'pearson'),digits = 2)
  spr_cor <- format(cor(v1,v2,method = 'spearman'),digits = 2)
  
  legend_texts = c(as.expression(bquote(bold(rho==.(pea_cor)))),as.expression(bquote(bold(sigma==.(spr_cor)))))
  legend(
    "topleft"
    ,legend = legend_texts,
    cex=1.3, pt.cex = 1,bty = "n"
  )
}

