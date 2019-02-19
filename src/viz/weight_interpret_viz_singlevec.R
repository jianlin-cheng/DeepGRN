library(pheatmap)
library(ggplot2)
library(reshape2)
library(seqLogo)
data_path = '~/Dropbox/MU/workspace/encode_dream/data/figs_data/fig6/JUND.HCT116.chr10.179450.179650/'

args <- commandArgs(T)
data_path_all = args[1]
fc_max = as.numeric(args[2])
dnase_max = as.numeric(args[3])
att_max = as.numeric(args[4])

for(data_path in list.dirs(data_path_all,recursive = F)){
  if(!file.exists(paste0(data_path,'fc.tiff'))){
    data_path <- paste0(data_path,'/')
    prob_fw <- read.csv(paste0(data_path,'prob_fw.csv'),header = F)
    prob_rc <- read.csv(paste0(data_path,'prob_rc.csv'),header = F)
    
    prob_fw <- t(prob_fw)
    prob_rc <- t(prob_rc)
    
    xy_fw <- read.csv(paste0(data_path,'xy_fw.csv'),header = F)
    xy_rc <- read.csv(paste0(data_path,'xy_rc.csv'),header = F)
    
    tiff(paste0(data_path,'fw_att.tiff'),width = 840,height = 280)
    pheatmap(t(prob_fw),cluster_cols = F,scale = 'none',show_rownames = F,show_colnames = F)
    dev.off()
    tiff(paste0(data_path,'rc_att.tiff'),width = 840,height = 280)
    pheatmap(t(prob_rc),cluster_cols = F,scale = 'none',show_rownames = F,show_colnames = F)
    dev.off()
    
    prob_fw_var <- apply(prob_fw, 2, var)
    prob_fw_var_slt <- prob_fw_var>=quantile(prob_fw_var,0)
    prob_rc_var <- apply(prob_rc, 2, var)
    prob_rc_var_slt <- prob_rc_var>=quantile(prob_rc_var,0)
    
    ave_prob_fw <- apply(prob_fw[,prob_fw_var_slt], 1, mean)
    ave_prob_rc <- apply(prob_rc[,prob_rc_var_slt], 1, mean)
    
    g_prob_fw <- ggplot() + geom_point(aes(x = 1:length(ave_prob_fw), y = ave_prob_fw, colour = ave_prob_fw), size = 3) +xlab('')+ylab('')+ylim(0,att_max)+ theme(legend.position="none")
    # +stat_smooth(aes(x = 1:length(ave_prob_fw), y = ave_prob_fw), method = "lm",formula = y ~ poly(x, 23), se = FALSE) 
    g_prob_rc <- ggplot() + geom_point(aes(x = 1:length(ave_prob_rc), y = ave_prob_rc, colour = ave_prob_rc), size = 3) +xlab('')+ylab('')+ylim(0,att_max)+ theme(legend.position="none")
    # +stat_smooth(aes(x = 1:length(ave_prob_rc), y = ave_prob_rc), method = "lm",formula = y ~ poly(x, 23), se = FALSE) 
    
    tiff(paste0(data_path,'g_prob_fw.tiff'),width = 840,height = 280)
    plot(g_prob_fw)
    dev.off()
    
    tiff(paste0(data_path,'g_prob_rc.tiff'),width = 840,height = 280)
    plot(g_prob_rc)
    dev.off()
    
    # pheatmap(t(xy_fw)[5:7,],cluster_cols = F,cluster_rows = F,scale = 'row',show_rownames = F)
    # pheatmap(t(xy_rc)[5:7,],cluster_cols = F,cluster_rows = F,scale = 'row',show_rownames = F)
    
    
    val_uq35 <- scale(xy_fw$V5,T,T)
    val_dnase <- scale(xy_fw$V6,T,T)
    fold_change <- xy_fw$V7
    
    g_uq35 <- ggplot() + geom_point(aes(x = 1:nrow(xy_fw), y = val_uq35, colour = val_uq35), size = 3) + xlab('')+ylab('')+ theme(legend.position="none")
    # +stat_smooth(aes(x = 1:nrow(xy_fw), y = val_uq35), method = "lm",formula = y ~ poly(x, 23), se = FALSE) 
    
    g_dnase <- ggplot() + geom_point(aes(x = 1:nrow(xy_fw), y = val_dnase, colour = val_dnase), size = 3) +xlab('')+ylab('')+ylim(-0.3,dnase_max)+ theme(legend.position="none")
    # +stat_smooth(aes(x = 1:nrow(xy_fw), y = val_dnase), method = "lm",formula = y ~ poly(x, 23), se = FALSE)
    
    g <- ggplot() + geom_point(aes(x = 1:nrow(xy_fw), y = fold_change, colour = fold_change), size = 3) +xlab('')+ylab('')+ylim(0,fc_max)+ theme(legend.position="none")
    # +stat_smooth(aes(x = 1:nrow(xy_fw), y = fold_change), method = "lm",formula = y ~ poly(x, 23), se = FALSE)
    
    tiff(paste0(data_path,'fc.tiff'),width = 840,height = 280)
    plot(g)
    dev.off()
    
    tiff(paste0(data_path,'g_uq35.tiff'),width = 840,height = 280)
    plot(g_uq35)
    dev.off()
    
    tiff(paste0(data_path,'g_dnase.tiff'),width = 840,height = 280)
    plot(g_dnase)
    dev.off()
  }
}




# 
# freqs<- t(xy_fw[,1:4])
# rownames(freqs) <- c('A','T','G','C')
# freqs <- freqs[c('A','C','G','T'),]
# p <- makePWM(freqs[,500:600])
# seqLogo(p,yaxis = F,xaxis = F)
# 
# 
# seq_all <- rep('A',ncol(freqs))
# seq_all[freqs[,2]== 1] <- 'C'
# seq_all[freqs[,3]== 1] <- 'G'
# seq_all[freqs[,4]== 1] <- 'T'
# seq_all1 <- paste0(seq_all,collapse = '')
# 
