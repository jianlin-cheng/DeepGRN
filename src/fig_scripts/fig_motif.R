library(ggplot2)

mean_data <- read.csv('~/Dropbox/MU/workspace/encode_dream/data/figs_data/fig_motif/motif2.csv',header = F)
df <- cbind.data.frame('x'=1:nrow(mean_data),'y'=mean_data$V1)
ggplot(df,aes(x,y)) + geom_point() + geom_smooth() +xlab('Subsequence coordinate')+ylab('DNase score')
