# tf_name = 'CTCF'
# result_path = '/home/chen/data/deepGRN/results/results_revision202001/motif/'
# fimo_path = '/home/chen/meme/bin'
# model_type = 'single'


args <- commandArgs(T)
tf_name = args[1]
result_path = args[2]
fimo_path = args[3]
model_type = args[4]

motif_file = paste0(result_path,'JASPAR/',tf_name,'.txt')
seq_file = paste0(result_path,model_type,'/fa/',tf_name,'_fw.fa')
gff_file = paste0(result_path,model_type,'/fimo_out/',tf_name,'/fimo.gff')
wfile = paste0(result_path,model_type,'/attention_score/',tf_name,'_fw.csv')
dnase_file = paste0(result_path,model_type,'/dnase/',tf_name,'_fw.csv')
plot_path = paste0(result_path,model_type,'/plots/',tf_name,'/')
dir.create(plot_path,showWarnings = F)

library(pheatmap)
require(ggplot2)
require(ggseqlogo)
library(RColorBrewer)
library(viridis)

flanking = 10
setwd(fimo_path)
system(paste0(fimo_path,'/fimo --thresh 1e-3 --o ',result_path,model_type,'/fimo_out/',tf_name,'/ ',motif_file, ' ',seq_file))

attw <- read.csv(wfile,header = F,sep = ' ')
gff_data <- read.delim(gff_file,as.is = T,comment.char = '#',header = F)
dnase_data  <- read.delim(dnase_file,sep = ' ',header = F)
seq_index <- as.numeric(unlist(lapply(strsplit(gff_data$V1,'_'), function(x)x[1])))
motif_names <- gsub('Name=(.+)_\\d+_chr.+','\\1',gff_data$V9)
seq_all <- gsub('.+sequence=(.+);','\\1',gff_data$V9)

seq_start <- as.numeric(unlist(lapply(strsplit(gff_data$V1,'_'), function(x)x[3])))+1
att_center <- as.numeric(unlist(lapply(strsplit(gff_data$V1,'_'), function(x)x[4])))+1
motif_start <- att_center-flanking+gff_data$V4-1
motif_end <- motif_start+gff_data$V5-gff_data$V4

motif_start_in_1002 <- motif_start-(seq_start-401)
motif_len <- gff_data$V5-gff_data$V4
# motif_end_in_1002 <- motif_start_in_1002+motif_end-motif_start

attw_rowmax_idx <- apply(attw, 1, which.max)
attw1 <- attw[order(attw_rowmax_idx),]
# pheatmap(attw1/nrow(attw1),cluster_rows = F,cluster_cols = F,show_rownames = F,show_colnames = F)
span_mask <- motif_start_in_1002>0 & motif_start_in_1002 < (1002-6)

for (m in unique(motif_names)) {
  print(m)
  m_flag <- (motif_names==m) & span_mask

  bin_m_idx <- seq_index[m_flag]
  
  attw_m <- attw[bin_m_idx,]
  
  attw_rowmax_idx <- apply(attw_m, 1, which.max)
  attw1 <- attw_m[order(attw_rowmax_idx),]
  
  
  png(paste0(plot_path,tf_name,'_',m,'_att.png'),height = 800,width = 600)
  pheatmap(attw1/nrow(attw1),cluster_rows = F,cluster_cols = F,show_rownames = F,show_colnames = F)
  dev.off()  
  motif_start_in_1002_m <- motif_start_in_1002[m_flag]
  # motif_end_in_1002_m <- motif_end_in_1002[m_flag]
  
  mofit_loc_mat <- matrix(0,nrow = nrow(attw1),ncol=1002)
  for (i in 1:nrow(mofit_loc_mat)) {
    mofit_loc_mat[i,motif_start_in_1002_m[i]:min(1002,motif_start_in_1002_m[i]+motif_len-1)] <- 1
  }
  png(paste0(plot_path,tf_name,'_',m,'_motif_span.png'),height = 800,width = 600)
  pheatmap(mofit_loc_mat[order(attw_rowmax_idx),],cluster_rows = F,cluster_cols = F,show_rownames = F,show_colnames = F,color = c("white", "black"))
  dev.off()
  
  png(paste0(plot_path,tf_name,'_',m,'_dnase.png'),height = 800,width = 600)
  pheatmap(log2(dnase_data[bin_m_idx[order(attw_rowmax_idx)],]+1),cluster_rows = F,cluster_cols = F,show_rownames = F,show_colnames = F)
  # pheatmap(dnase_data[bin_m_idx[order(attw_rowmax_idx)],],cluster_rows = F,cluster_cols = F,show_rownames = F,show_colnames = F)
  dev.off()
  
  seq_m <- seq_all[m_flag]
  p1 = ggseqlogo( seq_m, method = 'bits' )
  png(paste0(plot_path,tf_name,'_',m,'_seqlogo.png'),height = 188*2,width = 376*2)
  plot(p1)
  dev.off()
  
}

