data_path = '/home/chen/data/deepGRN/results/results_revision202001/feature_importance/'
out_path = '/home/chen/data/deepGRN/results/results_revision202001/feature_importance/ens/'

run_command = ''
for (i in c('dnase','rnaseq','gencode','sequence','unique35')) {
  for (j  in list.files(paste0(data_path,'single/',i),pattern = '.+.gz')) {
    single_pred_file <- paste0(data_path,'single/',i,'/',j)
    pair_pred_file <- paste0(data_path,'pair/',i,'/',j)
    run_command <- paste0(run_command,'python /home/chen/Dropbox/MU/workspace/encode_dream/src/score_ensumble.py ',
                          single_pred_file,',',pair_pred_file,' /home/chen/data/deepGRN/raw/label/final ' ,
                          out_path,i,'.',gsub('.tab.gz','.txt',j),'\n')
  }
}


writeLines(run_command,'~/Dropbox/MU/workspace/encode_dream/src/revision202001/feature_importance/feature_importance_ens.sh')