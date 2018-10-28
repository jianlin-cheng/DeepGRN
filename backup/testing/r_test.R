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

bin_idx_train_pos <- which(label_data_train==T)
bin_idx_train_neg <- which(label_data_train==F)
bin_idx_val_pos <- which(label_data_val==T)
bin_idx_val_neg <- which(label_data_val==F)

positive_bin_idx = bin_idx_train_pos
negative_bin_idx = bin_idx_train_neg

start_idx <- 1
batch_size <- round(batch_size/2)

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
write.csv(bin_range_table,'~/test.csv',row.names = F)
datag = get_feature_from_bin_range_table(bin_range_table,genome,cell_bw,unique35_bw,label_data,flanking_length,multiout,seg_bin_idx)
xfw = datag[[1]][[1]]
xrc = datag[[1]][[2]]
y = datag[[2]]