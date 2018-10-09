args=commandArgs(T)
pred_file = args[1]
out_path = args[2]
# pred_file='/storage/htc/bdm/ccm3x/deepGRN/evaluate_chr11/predictions/TAF1_liver_factornet_lstm.unique35True.RNAseqTrue.GencodeTrue.csv'
# out_path  = '/storage/htc/bdm/ccm3x/deepGRN/evaluate_chr11/score/'

library(data.table)
library(PRROC)
library(ROCR)

true_path = '/storage/htc/bdm/ccm3x/deepGRN/raw/label/final/'
val_chr = c('chr1','chr8','chr21')

pred_info <- unlist(strsplit(gsub('.+/','',pred_file),'_'))
tf_name <- pred_info[1]
cell_name <- pred_info[2]
label_file <- paste0(true_path,tf_name,'.train.labels.tsv')
true <- fread(label_file,header = T)
true1 <- true[true$chr %in% val_chr,grep(cell_name,colnames(true)),with=F]
true1 <- as.numeric(true1[[1]]=='B')
pred <- fread(pred_file,header=F)[[1]]
pred <- c(pred,rep(pred[length(pred)],length(true1)-length(pred)))

recall_at_fdr <- function(pred, true, fdr_cutoff=c(0.1,0.5)){
  rp <- performance(prediction(pred, true), "prec", "rec")
  prec <- rp@y.values[[1]]
  prec <- prec[!is.nan(prec)]
  rec <- rp@x.values[[1]][!is.nan(prec)]
  fdr = 1- prec
  
  recall_at_fdr <- c()
  for(i in fdr_cutoff){
    fdr1 <- which(fdr<=i)
    cutoff_index = min(length(fdr1)+1,length(fdr))
    recall_at_fdr <- c(recall_at_fdr,rec[cutoff_index])
  }
  return(recall_at_fdr)
}


model_eval <- function(pred,true){
  if(length(pred)!=length(true))warning('predictions and true labels ar not equal length')
  au_prc <- pr.curve(pred[true==1],pred[true==0])$auc.davis.goadrich
  au_roc <- roc.curve(pred[true==1],pred[true==0])$auc
  rf <- recall_at_fdr(pred,true)
  
  return(data.frame('auPRC'=au_prc,'auROC'=au_roc,'recall_at_fdr_0.1'=rf[1],'recall_at_fdr_0.5'=rf[2]))
}
res <- model_eval(pred,true1)

write.csv(res,paste0(out_path,gsub('.+/','',pred_file)))