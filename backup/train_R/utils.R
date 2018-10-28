library(keras)
library(abind)
library(Biostrings)
library(data.table)
library(rtracklayer)
library(Matrix)
library(pryr)
library(PRROC)

get_seg_bin_idx <- function(blacklist_merge){
  blacklist_merge$V4 <- (blacklist_merge$V3-blacklist_merge$V2-150)/50
  seg_bin_idx <- matrix(c(1,blacklist_merge$V4[1]),nrow = 1)
  for(i in 2:nrow(blacklist_merge)){
    seg_bin_idx <- rbind(seg_bin_idx,c(seg_bin_idx[i-1,2]+1,seg_bin_idx[i-1,2]+blacklist_merge$V4[i]))
  }
  seg_bin_idx <- cbind.data.frame(blacklist_merge,seg_bin_idx,stringsAsFactors=F)
  colnames(seg_bin_idx) <- NULL
  return(seg_bin_idx)
}

import_bw <- function(bw_file,blacklist_merge){
  data_bw <- list()
  for(i in unique(blacklist_merge$V1)){
    print(paste('importing',i))
    range_select <- GRanges(seqnames = i,ranges = IRanges(start = 1,end =max(blacklist_merge$V3[blacklist_merge$V1==i])))
    bw_data <- import(bw_file,format = gsub('.+\\.','',bw_file),as='NumericList',selection = BigWigSelection(range_select))
    data_bw[[i]] <- Matrix(bw_data[[1]],  ncol = 1, sparse = TRUE)
  }
  data_bw
}

import_bw1 <- function(bw_file,blacklist_merge){
  data_bw <- list()
  for(i in unique(blacklist_merge$V1)){
    print(paste('importing',i))
    range_select <- GRanges(seqnames = i,ranges = IRanges(start = 1,end =max(blacklist_merge$V3[blacklist_merge$V1==i])))
    bw_data <- import(bw_file,format = gsub('.+\\.','',bw_file),as='NumericList',selection = BigWigSelection(range_select))
    data_bw[[i]] <- bw_data[[1]]
  }
  data_bw
}


get_one_hot_seq <- function(genome,chr,seq_start,seq_stop){
  seq_range <- subseq(genome[[chr]], start = seq_start,end = seq_stop)
  d <- NULL
  for(i in c('A','T','G','C')){
    d <- cbind(d,as.numeric(unlist(strsplit(toString(seq_range),split = ''))==i))
    
  }
  return(d)
}

get_rc_x <- function(x){
  x_rc <- x
  x_rc[,1] <- x[,2]==1
  x_rc[,2] <- x[,1]==1
  x_rc[,3] <- x[,4]==1
  x_rc[,4] <- x[,3]==1
  x_rc[rev(1:nrow(x_rc)),]
}

get_x_from_bin_range <- function(bin_start,bin_end,genome,cell_bw,unique35_bw,flanking_length,seg_bin_idx){
  # bin_range must be continous integers within same segment
  bin_info <- seg_bin_idx[seg_bin_idx[,5]<=bin_start & seg_bin_idx[,6]>=bin_end,]
  if(length(bin_info)==0){
    warning(paste('bin',bin_start,bin_end,'crossing blacklisted regions'))
  }
  chr <- as.character(bin_info[1,1])
  seq_start <- as.numeric(bin_info[1,2])+1+50*(bin_start-as.numeric(bin_info[1,5]))-flanking_length
  seq_stop <- as.numeric(bin_info[1,2])+200+50*(bin_end-as.numeric(bin_info[1,5]))+flanking_length
  cell_bw_bin <- cell_bw[[chr]][seq_start:seq_stop,1]
  as.matrix(cbind(get_one_hot_seq(genome,chr,seq_start,seq_stop),unique35_bw[[chr]][seq_start:seq_stop],cell_bw_bin))
}

get_y_from_bin_range <- function(bin_start,bin_end,label_data){
  if(bin_start==bin_end){
    return(as.numeric(label_data[bin_start]))
  }else{
    return(as.numeric(unlist(label_data[bin_start:bin_end])))
  }
}

get_x_from_bin_range_table <- function(bin_range_table,genome,cell_bw,unique35_bw,flanking_length,seg_bin_idx){
  x1_batch <- x2_batch <- NULL
  for(i in 1:nrow(bin_range_table)){
    bin_start <- bin_range_table[i,1]
    bin_end <- bin_range_table[i,2]
    
    x1_idx <- get_x_from_bin_range(bin_start,bin_end,genome,cell_bw,unique35_bw,flanking_length,seg_bin_idx)
    x1_idx[is.na(x1_idx)] <- 0
    x2_idx <- get_rc_x(x1_idx)
    
    dim(x1_idx) <- dim(x2_idx) <- c(1,nrow(x1_idx),ncol(x1_idx))
    
    if(is.null(x1_batch)){
      x1_batch <- x1_idx
      x2_batch <- x2_idx
    }else{
      x1_batch <- abind(x1_batch,x1_idx,along = 1)
      x2_batch <- abind(x2_batch,x2_idx,along = 1)
    }
  }
  return(list(x1_batch,x2_batch))
}

get_y_from_bin_range_table <- function(bin_range_table,label_data,multiout){
  y_batch <- NULL
  for(i in 1:nrow(bin_range_table)){
    bin_start <- bin_range_table[i,1]
    bin_end <- bin_range_table[i,2]
    
    y_idx  <- get_y_from_bin_range(bin_start,bin_end,label_data)
    dim(y_idx) <- c(1,length(y_idx),1)
    
    if(is.null(y_batch)){
      y_batch <- y_idx
    }else{
      y_batch <- abind(y_batch,y_idx,along = 1)
    }
  }
  if(!multiout){
    y_batch <- y_batch[,,1]
  }
  return(y_batch)
}

get_feature_from_bin_range_table <- function(bin_range_table,genome,cell_bw,unique35_bw,label_data,flanking_length,multiout,seg_bin_idx){
  return(list(get_x_from_bin_range_table(bin_range_table,genome,cell_bw,unique35_bw,flanking_length,seg_bin_idx),
              get_y_from_bin_range_table(bin_range_table,label_data,multiout)))
}

get_min_bin_num <- function(bin_idx,seg_bin_idx){
  bin_idx_row <- as.numeric(seg_bin_idx[seg_bin_idx[,5]<=bin_idx & seg_bin_idx[,6]>=bin_idx,][1,5:6])
  return(min(c(bin_idx-bin_idx_row[1],bin_idx_row[2]-bin_idx)))
}

train_data_generator <- function(positive_bin_idx,negative_bin_idx,genome,cell_bw,unique35_bw,label_data,bin_num,batch_size,flanking_length,multiout,seg_bin_idx) {
  start_idx <- 1
  batch_size <- round(batch_size/2)
  function() {
    if (start_idx > length(positive_bin_idx)){
      start_idx <<- 1
      positive_bin_idx <<- sample(positive_bin_idx,length(positive_bin_idx),replace = F)
    }
    end_idx <- min(length(positive_bin_idx),start_idx+batch_size-1)
    positive_bin_idx_batch <- positive_bin_idx[start_idx:end_idx]
    negative_bin_idx_batch <- sample(negative_bin_idx,length(positive_bin_idx_batch),replace = F)
    
    # get min bin_num possible for positive_bin_idx_batch as bin_num_batch
    bin_num_batch <- min(c(bin_num,unlist(lapply(positive_bin_idx_batch, function(x)get_min_bin_num(x,seg_bin_idx)))))
    
    # remove negative_bin_idx_batch which with bin_num_batch will be invalid
    negative_bin_idx_batch <- negative_bin_idx_batch[unlist(lapply(negative_bin_idx_batch, function(x)get_min_bin_num(x,seg_bin_idx)))>=bin_num_batch]
    
    bin_idx_list <- sample(c(positive_bin_idx_batch,negative_bin_idx_batch),length(positive_bin_idx_batch)+length(negative_bin_idx_batch),replace = F)
    start_idx <<- start_idx+batch_size
    bin_range_table <- NULL
    for(i in bin_idx_list){
      bin_range_table <- rbind(bin_range_table,c(i-bin_num_batch,i+bin_num_batch))
    }
    
    get_feature_from_bin_range_table(bin_range_table,genome,cell_bw,unique35_bw,label_data,flanking_length,multiout,seg_bin_idx)
  }
}

train_single <- function(model,genome,cell_bw,unique35_bw,label_data_train,label_data_val,
                         blacklist_merge,bin_num,batch_size,epochs,multiout,flanking_length,
                         seg_bin_idx_train,seg_bin_idx_val,out_path_history,out_path_model){
  
  bin_idx_train_pos <- which(label_data_train==T)
  bin_idx_train_neg <- which(label_data_train==F)
  bin_idx_val_pos <- which(label_data_val==T)
  bin_idx_val_neg <- which(label_data_val==F)
  
  train_gen <- train_data_generator(bin_idx_train_pos,bin_idx_train_neg,genome,cell_bw,unique35_bw,
                               label_data_train,bin_num,batch_size,flanking_length,multiout,seg_bin_idx_train)
  val_gen <- train_data_generator(bin_idx_val_pos,bin_idx_val_neg,genome,cell_bw,unique35_bw,
                             label_data_val,bin_num,batch_size,flanking_length,multiout,seg_bin_idx_val)
  history <- model %>% fit_generator(train_gen,steps_per_epoch = 2*length(bin_idx_train_pos)/batch_size, epochs = epochs,
                                     validation_data=val_gen,validation_steps=2*length(bin_idx_val_pos)/batch_size,
                                     callbacks = list(callback_csv_logger(out_path_history, separator = ",", append = T),
                                                      callback_model_checkpoint(out_path_model, monitor = "val_loss")))
  return(list('model'=model,'history'=history$metrics))
}

train <- function(model,genome,feature_dnase_dir,unique35_bw,label_data,blacklist_merge,bin_num,
                  batch_size,epochs,validate_chr,out_path_history,out_path_model){
  multiout <- length(model$output$get_shape()$dims)==3
  flanking_length = (model$input_shape[[1]][[2]]-200)/2
  if(length(flanking_length)==0) flanking_length <- 0
  
  seg_bin_idx <- get_seg_bin_idx(blacklist_merge)
  seg_bin_idx_val <- seg_bin_idx[seg_bin_idx[,1]==validate_chr,]
  validate_bins <- as.numeric(seg_bin_idx_val[1,5]):as.numeric(seg_bin_idx_val[nrow(seg_bin_idx_val),6])
  seg_bin_idx_train <- get_seg_bin_idx(blacklist_merge[!blacklist_merge$V1%in%validate_chr,])
  seg_bin_idx_val <- get_seg_bin_idx(blacklist_merge[blacklist_merge$V1%in%validate_chr,])
  label_data_train_all <- label_data[-validate_bins,,drop=F]
  label_data_val_all <- label_data[validate_bins,,drop=F]
  
  #import cell bw data
  cell_names <- colnames(label_data)
  cell_bw_all <- list()
  for(cell_name in cell_names){
    print(paste('importing DNase data for cell',cell_name))
    cell_bw_all[[cell_name]] <- import_bw(paste0(feature_dnase_dir,cell_name,'.1x.bw'),blacklist_merge)
  }
  
  cell_bw <- lapply(cell_bw_all[[1]], function(x)data.table(as.numeric(x)))
  label_data_train <- data.table(label_data_train_all[,1])
  label_data_val <- data.table(label_data_val_all[,1])
  
  # if there is only one cell line in training data
  if(ncol(label_data)==1){
    train_res <- train_single(model,genome,cell_bw,unique35_bw,label_data_train,label_data_val,
                              blacklist_merge,bin_num,batch_size,epochs,multiout,flanking_length,
                              seg_bin_idx_train,seg_bin_idx_val,out_path_history,out_path_model)
    return(train_res)
  }
  history_all <- NULL
  for(i in 1:epochs){
    acc <- loss<- val_acc <- val_loss <- 0
    print(paste('training on epoch',i))
    cell_idx_order <- sample(1:ncol(label_data),ncol(label_data),replace = F)
    
    for(cell_idx in cell_idx_order){
      print(paste('training on cell',colnames(label_data)[cell_idx]))
      for(i in 1:length(cell_bw)){
        cell_bw[[i]][1:nrow(cell_bw[[i]]),V1:=as.numeric(cell_bw_all[[cell_idx]][[i]])] 
      }
      label_data_train[1:nrow(label_data_train),V1:= label_data_train_all[,cell_idx]]
      label_data_val[1:nrow(label_data_val),V1:=label_data_val_all[,cell_idx]]
      train_res <- train_single(model,genome,cell_bw,unique35_bw,label_data_train,label_data_val,
                                blacklist_merge,bin_num,batch_size,1,multiout,flanking_length,
                                seg_bin_idx_train,seg_bin_idx_val,out_path_history,out_path_model)
      
      model <- train_res$model
      acc <- acc+ train_res$history$acc
      val_acc <- val_acc+ train_res$history$val_acc
      loss <- loss+ train_res$history$loss
      val_loss <- val_loss+ train_res$history$val_loss
    }
    history_all <- rbind(history_all,c('acc'=acc/length(cell_idx_order),'loss'=loss/length(cell_idx_order),
                                       'val_acc'=val_acc/length(cell_idx_order),'val_loss'=val_loss/length(cell_idx_order)))
    print(mem_used())
  }
  return(list('model'=model,'history'=history_all))
}


get_steps_per_segment <- function(seg_bin_num,bin_num,batch_size){
  bin_num <- 2*bin_num+1
  seg_bin_num%/%(bin_num*batch_size)+
    as.numeric((seg_bin_num%%(bin_num*batch_size))%/%bin_num!=0)+
    as.numeric((seg_bin_num%%(bin_num*batch_size))%%bin_num!=0)
}

predict_data_generator <- function(genome,cell_bw,unique35_bw,bin_num,batch_size,flanking_length,seg_bin_idx) {
  start_idx <- 1
  function() {
    idx_row <- which(as.numeric(seg_bin_idx[,5])<=start_idx & as.numeric(seg_bin_idx[,6])>=start_idx)
    current_seg_end <- as.numeric(seg_bin_idx[idx_row,6])
    # if first sample with length 2*bin_num+1 exceeds current_seg_end, make this batch as batch size 1
    if(start_idx+2*bin_num > current_seg_end){
      bin_range_table <- matrix(c(start_idx,current_seg_end),ncol = 2)
      start_idx <<- current_seg_end+1
    }else{
      bin_range_table <- NULL
      for(i in 1:batch_size){
        if(start_idx+2*bin_num > current_seg_end){
          break
        }else{
          bin_range_table <- rbind(bin_range_table,c(start_idx,start_idx+bin_num*2))
          start_idx <<- as.numeric(bin_range_table[nrow(bin_range_table),2])+1
        }
      }
    }
    get_x_from_bin_range_table(bin_range_table,genome,cell_bw,unique35_bw,flanking_length,seg_bin_idx)
  }
}


predict_single <- function(cell_name,model,genome,cell_bw,unique35_bw,blacklist_merge,bin_num,batch_size){
  seg_bin_idx <- get_seg_bin_idx(blacklist_merge)
  total_steps <- sum(apply(seg_bin_idx, 1, function(x)get_steps_per_segment(as.numeric(x[6])-as.numeric(x[5])+1,bin_num,batch_size)))
  multiout <- length(model$output$get_shape()$dims)==3
  flanking_length <- (model$input_shape[[1]][[2]]-200)/2
  if(length(flanking_length)==0) flanking_length <- 0
  
  predict_gen <- predict_data_generator(genome,cell_bw,unique35_bw,bin_num,batch_size,flanking_length,seg_bin_idx)
  pred <- predict_generator(model,predict_gen, total_steps,verbose = 1)
  return(as.numeric(pred))
  
}

predict_multiple <- function(cell_name,model,genome,cell_bw,unique35_bw,blacklist_merge,bin_num,batch_size){
  seg_bin_idx <- get_seg_bin_idx(blacklist_merge)
  total_steps <- sum(apply(seg_bin_idx, 1, function(x)get_steps_per_segment(as.numeric(x[6])-as.numeric(x[5])+1,bin_num,batch_size)))
  multiout <- length(model$output$get_shape()$dims)==3
  flanking_length <- (model$input_shape[[1]][[2]]-200)/2
  if(length(flanking_length)==0) flanking_length <- 0
  pred <- c()
  
  predict_gen <- predict_data_generator(genome,cell_bw,unique35_bw,bin_num,batch_size,flanking_length,seg_bin_idx)
  for(i in 1:total_steps){
    print(paste(i,'/',total_steps))
    new_data <- predict_gen()
    pred_new <- predict_proba(model,x = new_data,batch_size = dim(new_data)[1])
    pred <- c(pred,as.numeric(t(pred_new[,,1])))
  }
  return(pred)
}

model_eval <- function(pred,true){
  if(length(pred)!=length(true))warning('predictions and true labels ar not equal length')
  au_prc <- pr.curve(pred[true],pred[!true])$auc.davis.goadrich
  au_roc <- roc.curve(pred[true],pred[!true])$auc
  return(data.frame('type'=c('auPRC','auROC'),'auc'=c(au_prc,au_roc)))
}



