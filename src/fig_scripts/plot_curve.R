library(PRROC)
library(data.table)

### CTCF IPSC
score1_file = '~/data/deepGRN/best_pred/F.CTCF.induced_pluripotent_stem_cell.tab.gz'
score2_file = '~/FactorNet/predictions/meta_RNAseq_Unique35_DGF.F.CTCF.induced_pluripotent_stem_cell.tab.gz'
label_file = '~/data/deepGRN/raw/label/final/CTCF.train.labels.tsv.gz'

label_data <- fread(label_file,data.table = F,header = T,sep = '\t')
label_data1 <- label_data[label_data$chr %in% c('chr1','chr8','chr21'),5] == 'B'

score1 <- fread(score1_file,data.table = F,header = F,sep = '\t')
score2 <- fread(score2_file,data.table = F,header = F,sep = '\t')

roc <- roc.curve(scores.class0 = score1$V4[label_data1],scores.class1 = score1$V4[!label_data1],curve=TRUE)
roc1 <- roc.curve(scores.class0 = score2$V4[label_data1],scores.class1 = score2$V4[!label_data1],curve=TRUE)
prc <- pr.curve(scores.class0 = score1$V4[label_data1],scores.class1 = score1$V4[!label_data1],curve=TRUE)
prc1 <- pr.curve(scores.class0 = score2$V4[label_data1],scores.class1 = score2$V4[!label_data1],curve=TRUE)

par(mfrow=c(1,2))

plot( roc, color = "red", auc.main=FALSE )
plot( roc1, color = "blue", add = TRUE )
legend('bottomright',col=c('red','blue'),lty=1,legend = c('attention','original'))

plot( prc, color = "red", auc.main=FALSE )
plot( prc1, color = "blue", add = TRUE )
legend('topright',col=c('red','blue'),lty=1,legend = c('attention','original'))

###JUND liver
score1_file = '~/data/deepGRN/best_pred/F.JUND.liver.tab.gz'
score2_file = '~/FactorNet/predictions/meta_Unique35_DGF_2.F.JUND.liver.tab.gz'
label_file = '~/data/deepGRN/raw/label/final/JUND.train.labels.tsv.gz'

label_data <- fread(label_file,data.table = F,header = T,sep = '\t')
label_data1 <- label_data[label_data$chr %in% c('chr1','chr8','chr21'),4] == 'B'

score1 <- fread(score1_file,data.table = F,header = F,sep = '\t')
score2 <- fread(score2_file,data.table = F,header = F,sep = '\t')

roc <- roc.curve(scores.class0 = score1$V4[label_data1],scores.class1 = score1$V4[!label_data1],curve=TRUE)
roc1 <- roc.curve(scores.class0 = score2$V4[label_data1],scores.class1 = score2$V4[!label_data1],curve=TRUE)
prc <- pr.curve(scores.class0 = score1$V4[label_data1],scores.class1 = score1$V4[!label_data1],curve=TRUE)
prc1 <- pr.curve(scores.class0 = score2$V4[label_data1],scores.class1 = score2$V4[!label_data1],curve=TRUE)

par(mfrow=c(1,2))

plot( roc, color = "red", auc.main=FALSE )
plot( roc1, color = "blue", add = TRUE )
legend('bottomright',col=c('red','blue'),lty=1,legend = c('attention','original'))

# prc$curve = prc$curve[prc$curve[,1]>0.005,]
# prc1$curve = prc1$curve[prc1$curve[,1]>0.005,]
plot( prc, color = "red", auc.main=FALSE )
plot( prc1, color = "blue", add = TRUE )
legend('topright',col=c('red','blue'),lty=1,legend = c('attention','original'))