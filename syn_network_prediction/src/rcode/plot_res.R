source('~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src/rcode/keras_utils.R')
output_dir = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/plots/plots_syndata_opt/performance_2dout_1_2_sgd/'
performance_file = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/performance/opt_test/performance_2dout_1_2_sgd.tsv'

plot_performance(performance_file,output_dir)


avg_res_final <- read.csv('~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/performance/grid_search.csv',row.names = 1)
acc = avg_res_final$acc
loss = avg_res_final$loss
dim(acc) <- c(10,10)
dim(loss) <- c(10,10)

library(plotly)
g1 <- plot_ly(showscale = FALSE) %>% add_surface(z = ~acc ) %>% add_surface(z = ~loss, opacity = 0.98) %>% layout(
  title = "Training",
  scene = list(
    xaxis = list(title = "depth"),
    yaxis = list(title = "filters"),
    zaxis = list(title = "metric")
  ))

acc = avg_res_final$val_acc
loss = avg_res_final$val_loss
dim(acc) <- c(10,10)
dim(loss) <- c(10,10)
g2 <- plot_ly(showscale = FALSE) %>% add_surface(z = ~acc ) %>% add_surface(z = ~loss, opacity = 0.98) %>% layout(
  title = "Validation",
  scene = list(
    xaxis = list(title = "depth"),
    yaxis = list(title = "filters"),
    zaxis = list(title = "metric")
  ))
