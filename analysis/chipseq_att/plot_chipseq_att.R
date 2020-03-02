library(pheatmap)
library(ggplot2)
library(reshape2)
library(gridExtra)
# tf_name = 'CTCF'
data_path = '~/data/deepGRN/results/results_revision202001/chipseq_att/pair/data/'

out_path <- gsub('data/$','plots/',data_path)

multiplot1 <- function(..., plotlist=NULL, cols) {
  require(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # Make the panel
  plotCols = cols                          # Number of columns of plots
  plotRows = ceiling(numPlots/plotCols) # Number of rows needed, calculated from # of cols
  
  # Set up the page
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(plotRows, plotCols)))
  vplayout <- function(x, y)
    viewport(layout.pos.row = x, layout.pos.col = y)
  
  # Make each plot, in the correct location
  for (i in 1:numPlots) {
    curRow = ceiling(i/plotCols)
    curCol = (i-1) %% plotCols + 1
    print(plots[[i]], vp = vplayout(curRow, curCol ))
  }
  
}

for (i in list.files(data_path,pattern = '.+_att.csv$')) {
  att_score <- read.delim(paste0(data_path,i),sep = ' ',as.is = T,header = F)
  chipseq_score <- read.delim(paste0(data_path,gsub('_att.csv','_chipseq.csv',i)),sep = ' ',as.is = T,header = F)
  pos_data <- read.delim(paste0(data_path,gsub('_att.csv','_pos.csv',i)),sep = '\t',as.is = T,header = F)
  
  att_score_max_pos <- apply(att_score,1,which.max)/ncol(att_score)
  chipseq_max_pos <- apply(chipseq_score,1,which.max)/ncol(chipseq_score)
  
  att_chip_dist <- abs(att_score_max_pos-chipseq_max_pos)
  
  fn <- unlist(strsplit(i,'_'))
  tf_name = fn[1]
  cell_name = fn[2]
  if(cell_name=='induced'){cell_name='induced_pluripotent_stem_cell'}
  g_max = 7
  att_max = 0
  att_score_list <- seq(min(att_score_max_pos),max(att_score_max_pos),length.out = g_max)
  g_list <- list()
  for (j in 1:(g_max-1)) {
    att_score_max_pos_select <- att_score_max_pos>=att_score_list[j] & att_score_max_pos<att_score_list[j+1]
    
    best_att_match <- which.min(abs(att_score_max_pos[att_score_max_pos_select]-chipseq_max_pos[att_score_max_pos_select]))
    
    
    
    if(abs(att_score_max_pos[best_att_match]-chipseq_max_pos[best_att_match])<0.5){
      print(j)
      select_att_val <- as.numeric(att_score[att_score_max_pos_select,][best_att_match,])
      select_chipseq_val <- as.numeric(chipseq_score[att_score_max_pos_select,][best_att_match,])
      pos_val <- pos_data[att_score_max_pos_select,][best_att_match,]
      att_max <- max(select_att_val+0.1,att_max)
      df1 <-  data.frame('x'=1:length(select_att_val),y = select_att_val)
      df2 <-  data.frame('x'=1:length(select_chipseq_val),y = select_chipseq_val)
      
      g_prob_fw <- ggplot(df1) + geom_point(aes(x, y, colour = y), size = 3) +xlab('')+ylab('')+ theme(legend.position="none") +
        ggtitle(paste(pos_val$V1,pos_val$V2-401,'-',pos_val$V2+601))
      g_chipseq_fw <- ggplot(df2) + geom_point(aes(x, y, colour = y), size = 3) +xlab('')+ylab('')+ theme(legend.position="none") + 
        ggtitle(' ')
      
      # plot(g_prob_fw)
      # plot(g_chipseq_fw)
      
      g_list[[length(g_list)+1]] <- g_prob_fw
      g_list[[length(g_list)+1]] <- g_chipseq_fw
    }
  }
  g_list[[length(g_list)-1]] <- g_list[[length(g_list)-1]] +xlab('Attention weights')
  g_list[[length(g_list)]] <- g_list[[length(g_list)]] +xlab('ChIP-Seq scores')
  
  # for(i in 1:(length(g_list)/2)){
  #   g1 <- g_list[[2*i-1]]+ylim(c(0,att_max))
  #   g_list[[2*i-1]] <-g1 
  #   
  # }
  
  
  tiff(paste0(out_path,tf_name,'.',cell_name,'.tiff'),width = 1200,height = 900)
  g_list <- g_list[c(3:10)]
  grid.arrange(grobs=g_list, nrow = length(g_list)/2)
  dev.off()
}








