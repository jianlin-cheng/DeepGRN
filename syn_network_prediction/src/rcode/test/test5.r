library(keras)
source('~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src/rcode/keras_utils.R')
label_file = "~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model_original/syn_prediction_model_data/label_matrix.csv"
feature_path = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model_original/syn_prediction_model_data/features/'
edge_select = 1:156

label_matrix <- read.csv(label_file,row.names = 1,as.is = T)
label_matrix <- label_matrix[,apply(label_matrix,2,var)>0]

features_list <- list()
for(i in list.files(feature_path)){
  features_list[[gsub('.csv$','',i)]] <- read.csv(paste(feature_path,i,sep = ''),row.names = 1,as.is = T)
}
label_matrix <- label_matrix[names(features_list),]

main_input <- layer_input(shape = c(dim(features_list[[1]])[1],dim(features_list[[1]])[2],1), dtype = 'float32', name = 'main_input')

cnn_out <- main_input %>% layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', padding='same') %>% 
  layer_conv_2d(filters = 2, kernel_size = c(3,3), activation = 'relu', padding='same') %>%
  layer_flatten()

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
  optimizer = optimizer_nadam(),
  loss = 'categorical_crossentropy',
  loss_weights = rep(1,length(edge_select))
)


label_matrix2 <- as.matrix(label_matrix)[,edge_select]
y_all <- get_array_y(label_matrix2)


y_pred <- get_softmax(1:nrow(label_matrix),1:nrow(label_matrix),model,features_list,y_all,get_plot = T,epochs = 12,batch_size = 32,validation_split = 0.2)
png(filename = '~/performance_nadam.png',width = 1080,height = 25600)
plot(y_pred)
dev.off()





