library(keras)
source('~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src/rcode/keras_utils.R')
label_file = "~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/syn_prediction_model_data/label_matrix.csv"
feature_path = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/syn_prediction_model_data/features/'
edge_select = 'all'  # if "all" then use all interactions

label_matrix <- read.csv(label_file,row.names = 1,as.is = T)
label_matrix <- label_matrix[,apply(label_matrix,2,function(x)length(unique(x)))==3]

features_list <- list()
for(i in list.files(feature_path)){
  features_list[[gsub('.csv$','',i)]] <- read.csv(paste(feature_path,i,sep = ''),row.names = 1,as.is = T)
}
label_matrix <- label_matrix[names(features_list),]

main_input <- layer_input(shape = c(dim(features_list[[1]])[1],dim(features_list[[1]])[2],1), dtype = 'float32', name = 'main_input')

cnn_out <- main_input %>% layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', padding='same') %>% 
  layer_conv_2d(filters = 2, kernel_size = c(3,3), activation = 'relu', padding='same') %>% layer_flatten()

interactions_out <- c()
for(i in 1:length(edge_select)){
  interactions_out <- c(interactions_out,cnn_out %>% 
                          layer_dense(units = 3, activation = 'softmax', name = paste('output',i,sep='_')))
}

model <- keras_model(
  inputs = main_input, 
  outputs = interactions_out
)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = 'categorical_crossentropy',
  loss_weights = rep(1,length(edge_select))
)

if(edge_select == 'all')edge_select <- 1:ncol(label_matrix)
label_matrix2 <- as.matrix(label_matrix)[,edge_select]
y_all <- get_array_y(label_matrix2)


y_pred <- get_softmax(1:600,1:600,model,features_list,y_all,get_plot = T,epochs = 12,batch_size = 32,validation_split = 0.2)
png(filename = '~/performance.png',width = 800,height = 21600)
plot(y_pred)
dev.off()



rand_ind <- sample(1:75,75)
y_pred_all <- y_true_all <- list()
for(i in 1:length(edge_select)){
  y_pred_all[[i]] <- y_true_all[[i]] <- NA
}

for(i in seq(1,75,5)){
  test_ind <- rand_ind[i:(i+4)]
  train_ind <- rand_ind[!(rand_ind %in% test_ind)]
  
  y_pred <- get_softmax(train_ind,test_ind,model,features_list,y_all)
  y_true <- lapply(y_all,function(x)x[test_ind,])
  
  for(i in 1:length(edge_select)){
    if(is.na( y_pred_all[[i]][1])){
      y_pred_all[[i]] <- y_pred[[i]]
      y_true_all[[i]] <- y_true[[i]]
    }else{
      y_pred_all[[i]] <- rbind(y_pred_all[[i]],y_pred[[i]])
      y_true_all[[i]] <- rbind(y_true_all[[i]],y_true[[i]])
    }
  }
}

get_acuracy(y_pred_all,y_true_all)






# 
# library(ggplot2)
# ggplot(df,aes(recall,precision,color=GeneSet))+geom_line(size = 2, alpha = 0.7)+
#   labs(title= "ROC curve", 
#        x = "False Positive Rate (1-Specificity)", 
#        y = "True Positive Rate (Sensitivity)")
# 









