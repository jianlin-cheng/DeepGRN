source('~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/src/rcode/keras_utils.R')

performance_path = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/syn_model/performance/pathway/deep/'
out_path = '~/Dropbox/MU/workspace/DeepGRN/syn_network_prediction/plots/pathway_model/deep/'

for(i in list.files(performance_path,pattern = '*.tsv')){
  plot_performance(paste(performance_path,i,sep = ''),paste(out_path,gsub('.tsv','',i),'/',sep = ''))
}