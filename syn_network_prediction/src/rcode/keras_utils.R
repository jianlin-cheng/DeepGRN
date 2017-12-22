library(keras)

get_array <- function(features_list){
  array_num <- NULL
  array_list <- lapply(features_list,function(x)as.numeric(unlist(x)))
  for(i in 1:length(array_list)){
    array_num <- rbind(array_num,array_list[[i]])
  }
  array_final <- array(as.numeric(array_num),dim = c(length(features_list),dim(features_list[[1]])[1],dim(features_list[[1]])[2],1))
}

get_array_y <- function(label_matrix){
  label_matrix2 <- as.matrix(label_matrix)
  y_train <- split(label_matrix2, rep(1:ncol(label_matrix2), each = nrow(label_matrix2)))
  y_train <- lapply(y_train,function(x)to_categorical(x,num_classes = 3))
}

pred_edge <- function(x,cutoff){
  x[x<-cutoff] <- -1
  x[x>cutoff] <-1
  x[abs(x)!=1] <- 0
  return(x)
}


get_acuracy <- function(y_pred,y_true,cutoff=0.5){
  y_pred2 <- lapply(y_pred,pred_edge,cutoff=cutoff)
  
  edge_pred <- unlist(lapply(y_pred2,function(x)x[,-1]))
  edge_true <- unlist(lapply(y_true,function(x)x[,-1]))
  tp <- sum(edge_true[edge_pred!=0]!=0)
  fp <- sum(edge_true[edge_pred!=0]==0)
  fn <- sum(edge_true[edge_pred==0]!=0)
  
  if((tp+fn)==0){
    recall_res <- 1
  }else{
    recall_res <- tp/(tp+fn)
  }
  if((tp+fp)==0){
    perc_res <- 1
  }else{
    perc_res <- tp/(tp+fp)
  }
  return(list('recall'=recall_res,'precision'=perc_res))
}


get_softmax <- function(train_ind,test_ind,model,features_list,y_all,get_plot = F,epochs = 20,batch_size = 32,validation_split = 0.2){
  
  x_train <- get_array(features_list[train_ind])
  y_train <- lapply(y_all,function(x)x[train_ind,])
  x_test <- get_array(features_list[test_ind])
  y_test <- lapply(y_all,function(x)x[test_ind,])
  names(y_train) <- NULL
  history1 <- model %>% fit(
    x = x_train,
    y = y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = validation_split
  )
  if(get_plot)return(history1)
  y_pred <- model %>% predict(x_test)
  return(y_pred)
}

get_performance_plot <- function(x){
  x1=NULL
  for(i in 1:ncol(x)){
    x1 <- rbind(x1,cbind('epoch'=(1:nrow(x))-1,'val'=x[,i],'metric'=colnames(x)[i]))
  }
  x1 <- data.frame(x1)
  x1$epoch <- as.numeric(as.character(x1$epoch))
  x1$val <- as.numeric(as.character(x1$val))
  x1$metric <- as.factor(x1$metric)
  p1 <- ggplot(x1, aes(x = epoch, y = val, colour = metric)) + geom_line()
  return(p1)
}


generate_feature <- function(out_path,gene_names,num_network = 5,num_batch_per_graph = 30,total_nodes = 500,interaction_mut_prob = 0.2,repr_percentage = 0.3,num_sample_per_batch = 10,num_external_genes = 10,rc = c(0.002, 0.005, 0.005, 0.005, 0.01, 0.02)){
  library(igraph)
  library(sgnesR)
  set.seed(11)
  
  out_path_network = paste(out_path,'random_scale_free_network/',sep = '')
  out_path_exprs = paste(out_path,'syn_rna_expression/',sep = '')
  out_path_coexprs = paste(out_path,'syn_prediction_model_data/features/',sep = '')
  out_path_label = paste(out_path,'syn_prediction_model_data/',sep = '')
  
  dir.create(out_path_network,showWarnings = F)
  dir.create(out_path_exprs,showWarnings = F)
  dir.create(out_path_coexprs,showWarnings = F)
  dir.create(out_path_label,showWarnings = F)
  
  
  # generate random scale-free network
  g <- barabasi.game(total_nodes,directed = T)
  adjmat_base <- as.matrix(as_adj(g))
  adjmat_base[adjmat_base==1] <- sample(c(-1,1),sum(adjmat_base==1),replace = T,prob = c(repr_percentage,1-repr_percentage))
  adjmat_base_dense <- adjmat_base[apply(adjmat_base,1,sum)>0,apply(adjmat_base,2,sum)>0]
  adjmat_base_dense <- t(adjmat_base_dense)
  tf_names <- sample(gene_names,nrow(adjmat_base_dense))
  target_names <- sample(gene_names,ncol(adjmat_base_dense))
  all_genes <- unique(c(tf_names,target_names))
  rownames(adjmat_base_dense) <- tf_names
  colnames(adjmat_base_dense) <- target_names
  base_counts <- sample(5:100,length(all_genes),replace = T)
  
  adj2el <- function(adj){
    el <- NULL
    for(i in rownames(adj)){
      for(j in colnames(adj)){
        if(adj[i,j]==1){
          el <- rbind(el,c(i,j,'1'))
        }else if(adj[i,j]==-1){
          el <- rbind(el,c(i,j,'-1'))
        }
      }
    }
    return(el)
  }
  
  # generate expression profile from random scale-free network
  all_comb_names <- c()
  for(i in rownames(adjmat_base_dense)){
    for(j in colnames(adjmat_base_dense)){
      all_comb_names <- c(all_comb_names,paste(i,j,sep = '_'))
    }
  }
  el_list <- list()
  adj_all <- NULL
  adj_rownames <- c()
  for(i in 1:num_network){
    adj_new <- adjmat_base_dense
    mut_ind <- sample(which(adj_new!=0),round(sum(adj_new!=0)*interaction_mut_prob))
    adj_new[mut_ind] <- sample(c(-1,1,0),length(mut_ind),replace = T)
    for(j in 1:num_batch_per_graph){
      adj_all <- rbind(adj_all,as.numeric(t(adj_new)))
      adj_rownames <- c(adj_rownames,paste('expr_model',i,'_rep',j,sep = ''))
    }
    write.csv(adj_new,paste(out_path_network,'network_adj_',i,'.csv',sep = ''))
    el_list[[i]]<- adj2el(adj_new)
  }
  colnames(adj_all) <- all_comb_names
  rownames(adj_all) <- adj_rownames
  exprs_list <- list()
  for(i in 1:length(el_list)){
    for(j in 1:num_batch_per_graph){
      g <- graph.edgelist(el_list[[i]][,1:2],directed = T)
      nodes_add <- all_genes[!(all_genes%in%V(g)$name)]
      g <- add_vertices(g,length(nodes_add), name = nodes_add)
      new_count <- base_counts
      if(num_external_genes>0){
        new_count[sample(1:length(new_count),num_external_genes)] <- sample(5:100,num_external_genes,replace = T)
      }
      V(g)$Rpop <- new_count
      E(g)$op <- el_list[[i]][get.edge.ids(g,as.character(t(el_list[[i]][,1:2]))),3]
      rsg <- new("rsgns.data",network=g, rconst=rc)
      rp<-new('rsgns.param',time=0,stop_time=1000,readout_interval=500)
      xx <- rsgns.rn(rsg, rp, timeseries=F, sample=num_sample_per_batch)
      exprs_mat <- xx$expression
      rownames(exprs_mat) <- gsub('^R','',rownames(exprs_mat))
      exprs_name <- paste('expr_model',i,'_rep',j,sep = '')
      exprs_list[[exprs_name]] <- exprs_mat
      write.csv(exprs_mat,paste(out_path_exprs,exprs_name,'.csv',sep = ''))
    }
  }
  
  # generate co-expression profile(features) from random scale-free network
  coexprs_list <- list()
  for(i in names(exprs_list)){
    new_cor_mat <- cor(t(exprs_list[[i]]),method = 'spearman')
    new_cor_mat[is.na(new_cor_mat)] <- 0
    new_cor_mat <- round(new_cor_mat[tf_names,target_names],2)
    coexprs_list[[i]] <- new_cor_mat
    write.csv(new_cor_mat,paste(out_path_coexprs,i,'.csv',sep = ''))
  }
  
  # generate labels from random scale-free network
  all_interactions <- unique(unlist(lapply(el_list,function(x)paste(x[,1],x[,2],sep = '_'))))
  label_mat <- matrix(0,nrow = 0,ncol = length(all_interactions))
  colnames(label_mat) <- all_interactions
  for(i in 1:num_network){
    new_el <- rep(0,ncol(label_mat))
    names(new_el) <- all_interactions
    el1 <- el_list[[i]]
    int_type <- el1[,3]
    names(int_type) <- paste(el1[,1],el1[,2],sep = '_')
    new_el[names(int_type)] <- int_type
    for(j in 1:num_batch_per_graph){
      label_mat <- rbind(label_mat,new_el)
    }
  }
  rownames(label_mat) <- adj_rownames
  write.csv(label_mat,paste(out_path_label,'label_matrix.csv',sep = ''))
  write.csv(adj_all,paste(out_path_label,'label_mat_all_comb.csv',sep = ''))
  write.csv(adj_all[,1:5000],paste(out_path_label,'label_mat_all_comb_test.csv',sep = ''))
}

plot_epoch <- function(model_info){
  library(ggplot2)
  library(reshape2)
  model_info <- cbind.data.frame(model_info,'epoch'=1:nrow(model_info))
  data1 <- melt(model_info,id.vars = 'epoch')
  g <- ggplot(data1, aes(x=epoch, y=value, color=variable)) + geom_line(size=1)
  return(g)
}

plot_performance <- function(performance_file,output_dir,plot_heatmap=T){
  library(pheatmap)
  library(ggplot2)
  
  dir.create(output_dir,showWarnings = F)
  performance <- read.table(performance_file,header = T,sep = '\t')
  acc <- performance[,grep('^dense.+_acc$',colnames(performance))]
  los <- performance[,grep('^dense.+_loss$',colnames(performance))]
  val_acc <- performance[,grep('^val.+_acc$',colnames(performance))]
  val_los <- performance[,grep('^val.+_loss$',colnames(performance))]
  
  png(paste(output_dir,'performance_by_epoch.png',sep = ''))
  avg_res <- cbind('acc' = apply(acc, 1, mean),'loss' = apply(los, 1, mean),'val_acc' = apply(val_acc, 1, mean),'val_loss' = apply(val_los, 1, mean))
  plot(get_performance_plot(avg_res))
  dev.off()
  
  if(plot_heatmap){
    png(paste(output_dir,'acc.png',sep = ''))
    pheatmap(t(acc),cluster_cols = F,show_rownames = F,show_colnames = T)
    dev.off()
    png(paste(output_dir,'loss.png',sep = ''))
    pheatmap(t(los),cluster_cols = F,show_rownames = F,show_colnames = T)
    dev.off()
    png(paste(output_dir,'val_acc.png',sep = ''))
    pheatmap(t(val_acc),cluster_cols = F,show_rownames = F,show_colnames = T)
    dev.off()
    png(paste(output_dir,'val_loss.png',sep = ''))
    pheatmap(t(val_los),cluster_cols = F,show_rownames = F,show_colnames = T)
    dev.off()
  }

  
}

summary_grid_search <- function(performance_file_folder){
  avg_res_final <- NULL
  for(i in 1:10){
    for(j in 1:10){
      performance_file <- paste(performance_file_folder,'performance_2dout_',i,'_',j,'_rmsprop.tsv',sep = '')
      performance <- read.table(performance_file,header = T,sep = '\t')
      acc <- performance[,grep('^dense.+_acc$',colnames(performance))]
      los <- performance[,grep('^dense.+_loss$',colnames(performance))]
      val_acc <- performance[,grep('^val.+_acc$',colnames(performance))]
      val_los <- performance[,grep('^val.+_loss$',colnames(performance))]
      avg_res <- cbind('acc' = apply(acc, 1, mean),'loss' = apply(los, 1, mean),'val_acc' = apply(val_acc, 1, mean),'val_loss' = apply(val_los, 1, mean))
      avg_res_final <- rbind(avg_res_final,avg_res[nrow(avg_res),])
    }
  }
  return(avg_res_final)
}



