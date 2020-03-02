results_path = '/home/chen/data/deepGRN/results/results_revision202001/score/'
out_path = '~/Dropbox/MU/workspace/encode_dream/data/revision202001/'
feature_importance_res_path = '/home/chen/data/deepGRN/results/results_revision202001/feature_importance/'
feature_list = c('dnase','sequence','unique35','rnaseq','gencode')

# summary_all_tf <- function(results_path,res_file,prefix_num=0){
#   perf_all <- NULL
#   for (i in res_file) {
#     res_info <- unlist(strsplit(gsub('.+/','',i),split = '.',fixed = T))
#     tf_name <- res_info[2+prefix_num]
#     cell_name <- res_info[3+prefix_num]
#     res_i <- read.delim(paste0(results_path,i),header = F,as.is = T)
#     auROC <- as.numeric(gsub('auROC: ','',res_i$V2[1]))
#     auPRC <- as.numeric(gsub('auPRC: ','',res_i$V3[1]))
#     fdr50 <- as.numeric(gsub('Re@0.50 FDR: ','',res_i$V5[1]))
#     fdr10 <- as.numeric(gsub('Re@0.10 FDR: ','',res_i$V7[1]))
#     if(is.null(perf_all)){
#       perf_all <- data.frame(tf_name,cell_name,auROC,auPRC,fdr50,fdr10,stringsAsFactors = F)
#     }else{
#       perf_all <- rbind.data.frame(perf_all,c(tf_name,cell_name,auROC,auPRC,fdr50,fdr10),stringsAsFactors = F)
# 
#     }
# 
#   }
#   colnames(perf_all) <- c("TF","cell","auROC","auPRC","Re.0.50.FDR","Re.0.10.FDR")
#   perf_all <- perf_all[c(2,1,3:nrow(perf_all)),]
#   return(perf_all)
# }
# 
# perf_single =summary_all_tf(results_path,list.files(results_path,pattern = '^single.+'),1)
# perf_pair =summary_all_tf(results_path,list.files(results_path,pattern = '^pair.+'),1)
# 
# perf_importance_single <- cbind.data.frame('feature'='full',perf_single,stringsAsFactors=F)
# perf_importance_pair <- cbind.data.frame('feature'='full',perf_pair,stringsAsFactors=F)
# 
# perf_single_path <- paste0(feature_importance_res_path,'single/')
# perf_pair_path <- paste0(feature_importance_res_path,'pair/')
# 
# for (f in feature_list) {
#   perf_single_f <- paste0(perf_single_path,f,'/')
#   perf_all <- summary_all_tf(perf_single_f,list.files(perf_single_f,pattern = '.+.txt'))
#   perf_importance_single <- rbind.data.frame(perf_importance_single,
#                                              cbind.data.frame('feature'=f,perf_all,stringsAsFactors=F),stringsAsFactors = F)
#   
#   perf_pair_f <- paste0(perf_pair_path,f,'/')
#   perf_all <- summary_all_tf(perf_pair_f,list.files(perf_pair_f,pattern = '.+.txt'))
#   perf_importance_pair <- rbind.data.frame(perf_importance_pair,
#                                            cbind.data.frame('feature'=f,perf_all,stringsAsFactors=F),stringsAsFactors = F) 
# }
# write.csv(perf_importance_single,paste0(out_path,'perf_importance_single.csv'),row.names = F)
# write.csv(perf_importance_pair,paste0(out_path,'perf_importance_pair.csv'),row.names = F)

# res_file <- list.files(paste0(feature_importance_res_path,'ens'))
# perf_all <- NULL
# for (i in res_file) {
#   res_info <- unlist(strsplit(gsub('.+/','',i),split = '.',fixed = T))
#   tf_name <- res_info[3]
#   cell_name <- res_info[4]
#   res_i <- read.delim(paste0(feature_importance_res_path,'ens/',i),header = F,as.is = T)
#   feature_name <- res_info[1]
#   auROC <- as.numeric(gsub('auROC: ','',res_i$V2[1]))
#   auPRC <- as.numeric(gsub('auPRC: ','',res_i$V3[1]))
#   fdr50 <- as.numeric(gsub('Re@0.50 FDR: ','',res_i$V5[1]))
#   fdr10 <- as.numeric(gsub('Re@0.10 FDR: ','',res_i$V7[1]))
#   if(is.null(perf_all)){
#     perf_all <- data.frame(feature_name,tf_name,cell_name,auROC,auPRC,fdr50,fdr10,stringsAsFactors = F)
#   }else{
#     perf_all <- rbind.data.frame(perf_all,c(feature_name,tf_name,cell_name,auROC,auPRC,fdr50,fdr10),stringsAsFactors = F)
# 
#   }
# 
# }
# colnames(perf_all) <- c("feature","TF","cell","auROC","auPRC","Re.0.50.FDR","Re.0.10.FDR")
# perf_all <- perf_all[c(2,1,3:nrow(perf_all)),]
# perf_all <- rbind.data.frame(perf_importance_pair[perf_importance_pair$feature=='full',],perf_all,stringsAsFactors = F)
# rownames(perf_all) <- paste0(perf_all$feature,perf_all$TF,perf_all$cell,sep='_')
# perf_all <- perf_all[paste0(perf_importance_pair$feature,perf_importance_pair$TF,perf_importance_pair$cell,sep='_'),]
# write.csv(perf_all,paste0(out_path,'perf_importance_average.csv'),row.names = F)

library(ggplot2)

plot_feature_importance <- function(perf_importance_single,metric,feature_list){
  auprc_loss_f_single <- NULL
  for (f in feature_list) {
    score_without_f_single <- perf_importance_single[perf_importance_single$feature==f,]
    rownames(score_without_f_single) <- paste(score_without_f_single$TF,score_without_f_single$cell,sep = ',')
    score_with_f_single <- perf_importance_single[perf_importance_single$feature=='full',]
    rownames(score_with_f_single) <- paste(score_with_f_single$TF,score_with_f_single$cell,sep = ',')
    score_without_f_single <- score_without_f_single[rownames(score_with_f_single),]
    
    auprc_loss_f_single <- rbind.data.frame(auprc_loss_f_single,
                                            cbind.data.frame('feature'=f,
                                                             'target'=rownames(score_with_f_single),
                                                             'value'=as.numeric(score_with_f_single[,metric])-as.numeric(score_without_f_single[,metric]),
                                                             stringsAsFactors=F),
                                            stringsAsFactors = F)
  }
  
  
  auprc_loss_f_single$value <- as.numeric(auprc_loss_f_single$value)
  auprc_loss_f_single$feature <- factor(auprc_loss_f_single$feature)
  auprc_loss_f_single$target <- factor(auprc_loss_f_single$target)
  auprc_loss_f_single$value [is.na(auprc_loss_f_single$value)] <- 0
  g <- ggplot(auprc_loss_f_single, aes(x=target, y=value, fill=feature)) +
    geom_bar(stat='identity', position=position_dodge(width = 0.5)) +
    coord_flip() + theme(axis.text=element_text(size=14,face="bold"),legend.text=element_text(size=12,face="bold"),
                         axis.title=element_text(size=14,face="bold"),legend.title=element_text(size=12,face="bold"))
  
  
}
perf_importance_single <- read.csv(paste0(out_path,'perf_importance_single.csv'),as.is = T)
perf_importance_pair <- read.csv(paste0(out_path,'perf_importance_pair.csv'),as.is = T)
perf_importance_average <- read.csv(paste0(out_path,'perf_importance_average.csv'),as.is = T)

feature_importance_mat <- matrix(0,nrow = length(feature_list),ncol=3)
rownames(feature_importance_mat) <- feature_list
colnames(feature_importance_mat) <- c('single','pair','final')

for (i in feature_list) {
  perf_importance_tf <- perf_importance_single$TF[perf_importance_single$feature==i]

  
  feature_importance_mat[i,1] <- mean(perf_importance_single[perf_importance_single$feature=='full'&perf_importance_single$TF%in%perf_importance_tf,'auPRC'] - perf_importance_single[perf_importance_single$feature==i,'auPRC'])
  feature_importance_mat[i,2] <- mean(perf_importance_pair[perf_importance_pair$feature=='full'&perf_importance_pair$TF%in%perf_importance_tf,'auPRC']- perf_importance_pair[perf_importance_pair$feature==i,'auPRC'])
  feature_importance_mat[i,3] <- mean(perf_importance_average[perf_importance_average$feature=='full'&perf_importance_average$TF%in%perf_importance_tf,'auPRC']- perf_importance_average[perf_importance_average$feature==i,'auPRC'])
  
  
}




metric = 'auPRC'

g_auprc_single <- plot_feature_importance(perf_importance_single,metric,feature_list)
g_auprc_pair <- plot_feature_importance(perf_importance_pair,metric,feature_list)
g_auprc_average <- plot_feature_importance(perf_importance_average,metric,feature_list)

o1 <- order(perf_importance_single[perf_importance_single$feature=='full','auPRC'] - perf_importance_single[perf_importance_single$feature=='sequence','auPRC'],decreasing = T)
tf_seq_single <- perf_importance_single$TF[perf_importance_single$feature=='full'][o1]

o2 <- order(perf_importance_pair[perf_importance_pair$feature=='full','auPRC'] - perf_importance_pair[perf_importance_pair$feature=='sequence','auPRC'],decreasing = T)
tf_seq_pair <- perf_importance_pair$TF[perf_importance_pair$feature=='full'][o2]


write.csv(data.frame('single'=unique(tf_seq_single)[1:6],'pair'=unique(tf_seq_pair)[1:6]),
          paste0(feature_importance_res_path,'sequence_sensitive_tfs.csv'))


