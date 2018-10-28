library(keras)
library(abind)

seq_len = 1000000
n_channel = 6

window_size = 200
stride = 50
batch_size = 32
epochs = 10
bin_num = 2

n_test = 10

x1 <- rnorm(seq_len*n_channel)
x2 <- rnorm(seq_len*n_channel)
dim(x1) <- dim(x2) <- c(seq_len,n_channel)
y <- sample(0:1,(seq_len-window_size)/stride+1,replace = T,prob = c(0.9,0.1))
dim(y) <- c((seq_len-window_size)/stride+1,1)

get_x_from_bin <- function(x,bin_idx,bin_num,window_size,stride){
  bin_start <- (bin_idx-bin_num-1)*stride+1
  bin_end <- (bin_idx+bin_num)*stride+window_size-stride
  x[bin_start:bin_end,]
}

get_y_from_bin <- function(y,bin_idx,bin_num){
  y[(bin_idx-bin_num):(bin_idx+bin_num),,drop=F]
}

get_rc_x <- function(x){
  x_rc <- x
  x_rc[,1] <- x[,2]==1
  x_rc[,2] <- x[,1]==1
  x_rc[,3] <- x[,4]==1
  x_rc[,4] <- x[,3]==1
  x_rc[rev(1:nrow(x_rc)),]
}

get_feature_from_bin_idx <- function(bin_idx_list,x1,x2,y,bin_num,window_size,stride){
  x1_batch <- x2_batch <- y_batch <- NULL
  
  for(bin_idx in bin_idx_list){
    x1_idx <- get_x_from_bin(x1,bin_idx,bin_num,window_size,stride)
    x2_idx <- get_rc_x(x1_idx)
    y_idx  <- get_y_from_bin(y,bin_idx,bin_num)
    
    dim(x1_idx) <- dim(x2_idx) <- c(1,nrow(x1_idx),ncol(x1_idx))
    dim(y_idx) <- c(1,length(y_idx),1)
    
    if(is.null(x1_batch)){
      x1_batch <- x1_idx
      x2_batch <- x2_idx
      y_batch <- y_idx
    }else{
      x1_batch <- abind(x1_batch,x1_idx,along = 1)
      x2_batch <- abind(x2_batch,x2_idx,along = 1)
      y_batch <- abind(y_batch,y_idx,along = 1)
    }
  }
  return(list(list(x1_batch,x2_batch),y_batch))
}

sequence_generator <- function(positive_bin_idx,negative_bin_idx,x1,x2,y,bin_num,window_size,stride,batch_size) {
  start_idx <- 1
  batch_size <- round(batch_size/2)
  function() {
    if (start_idx > length(positive_bin_idx)){
      start_idx <<- 1
      positive_bin_idx <<- sample(positive_bin_idx,length(positive_bin_idx),replace = F)
    }
    end_idx <<- min(length(positive_bin_idx),start_idx+batch_size-1)
    negative_bin_idx <- sample(negative_bin_idx,size = length(positive_bin_idx),replace = F)
    positive_bin_idx_batch <- positive_bin_idx[start_idx:end_idx]
    negative_bin_idx_batch <- sample(negative_bin_idx,length(positive_bin_idx),replace = F)
    bin_idx_list <- sample(c(positive_bin_idx_batch,negative_bin_idx_batch),length(positive_bin_idx_batch)*2,replace = F)
    start_idx <<- start_idx+batch_size
    return(get_feature_from_bin_idx(bin_idx_list,x1,x2,y,bin_num,window_size,stride))
  }
}

random_generator <- function(a) {
  w <- round(rnorm(1)*10)
  batch_size <- sample(31:32,1,replace = F)
  function() {
    d <- rnorm(batch_size*w*1)
    dim(d) <- c(batch_size,w,1)
    return(d)
  }
}

# convert the function to a python iterator
positive_bin_idx <- which(y==1)
negative_bin_idx <- which(y==0)

positive_bin_idx <- positive_bin_idx[positive_bin_idx>bin_num & positive_bin_idx<=dim(y)[1]-bin_num]
negative_bin_idx <- negative_bin_idx[negative_bin_idx>bin_num & negative_bin_idx<=dim(y)[1]-bin_num]

# val_data
x1_test <- rnorm(n_test*((bin_num*2)*stride+window_size)*n_channel)
x2_test <- rnorm(n_test*((bin_num*2)*stride+window_size)*n_channel)
dim(x1_test) <- dim(x2_test) <- c(n_test,(bin_num*2)*stride+window_size,n_channel)
y_test <- sample(0:1,n_test*(bin_num*2+1),prob = c(0.9,0.1),replace = T)
dim(y_test) <- c(n_test,bin_num*2+1,1)

m <- load_model_hdf5('~/Dropbox/MU/workspace/encode_dream/models/fullconv_simple.h5')
m %>% compile(loss = 'binary_crossentropy',optimizer = optimizer_rmsprop(),metrics = c('accuracy'))
history <- m %>% fit_generator(sequence_generator(positive_bin_idx,negative_bin_idx,x1,x2,y,bin_num,window_size,stride,batch_size), 
                steps_per_epoch = length(positive_bin_idx) / batch_size, epochs = epochs,validation_data = list(list(x1_test,x2_test),y_test))

pred <- predict(m,x=list(x1_test,x2_test))
eval <- evaluate(m,x=list(x1_test,x2_test),y = y_test)
mean(as.numeric(pred[,,1]>=0.5) == as.numeric(y_test[,,1]))


model <- keras_model_sequential()
model%>%layer_conv_1d(filters = 1,kernel_size =  8, padding = "same", activation = "sigmoid", strides = 1,input_shape =list(NULL,1))
model %>% compile(loss = 'binary_crossentropy',optimizer = optimizer_rmsprop(),metrics = c('accuracy'))

random_generator <- function() {
  w <- round(rnorm(1)*100)
  batch_size <- sample(31:32,1,replace = F)
  function() {
    d <- rnorm(batch_size*w*1)
    dim(d) <- c(batch_size,w,1)
    y <- sample(0:1,batch_size*w,replace = T)
    dim(y) <- c(batch_size,w,1)
    return(list(d,y))
  }
}
random_generator1 <- function() {
  w <- round(rnorm(1)*100)
  batch_size <- sample(31:32,1,replace = F)
  # w=100
  function() {
    d <- rnorm(batch_size*w*1)
    dim(d) <- c(batch_size,w,1)
    y <- sample(0:1,batch_size*w,replace = T)
    dim(y) <- c(batch_size,w,1)
    return(list(d))
  }
}
history <- model %>% fit_generator(random_generator(), steps_per_epoch = 5, epochs = 5)
pred <- predict_generator(model,generator = random_generator1(),steps = 1)

eval <- evaluate(m,x=list(x1_test,x2_test),y = y_test)
mean(as.numeric(pred[,,1]>=0.5) == as.numeric(y_test[,,1]))

