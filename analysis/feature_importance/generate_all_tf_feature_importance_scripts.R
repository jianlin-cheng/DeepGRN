model_folder = '/home/chen/data/deepGRN/results/results_revision202001/best_pair/best_models/'
out_dir = '/home/chen/data/deepGRN/results/results_revision202001/feature_importance/pair/'

slurm_config = ""
pred_python_command = 'python /home/chen/Dropbox/MU/workspace/encode_dream/src/revision202001/feature_importance/predict_feature_importance.py -i /home/chen/data/deepGRN/raw/ -p /home/chen/data/deepGRN/raw/label/predict_region.bed -m '
score_python_command = 'python /home/chen/Dropbox/MU/workspace/encode_dream/src/score_ensumble.py '
label_dir = '/home/chen/data/deepGRN/raw/label/final/ '
d_deepgrn_base_file = '~/Dropbox/MU/workspace/encode_dream/data/revision202001/deepgrn_ens.csv'
train_par_file = '~/Dropbox/MU/workspace/encode_dream/data/revision202001/train_parameters.csv'

feature_list = c('dnase','sequence','unique35','rnaseq','gencode')

d_deepgrn_base <- read.csv(d_deepgrn_base_file,as.is = T)
train_par <- read.csv(train_par_file,as.is = T)

for (f in feature_list) {
  run_file <- ''
  out_dir_f <- paste0(out_dir,f,'/')
  dir.create(out_dir_f,showWarnings = F,recursive = T)
  for (i in list.files(model_folder)) {
    tf_name <- gsub('_.+','',i)
    cell_list <- d_deepgrn_base$cell[d_deepgrn_base$TF==tf_name]
    for (cell_name in cell_list) {
      out_file_f <- paste0(out_dir_f,'F.',tf_name,'.',cell_name,'.tab.gz')
      # pred_command <- paste0(pred_python_command,model_folder,i,' -c ',cell_name,' -o ',out_file_f,' -f ',f)
      pred_command = ''
      score_command <- paste0(score_python_command,out_file_f,' ',label_dir,gsub('.tab.gz','.txt',out_file_f))
      run_file <- paste0(slurm_config,'\n',pred_command,'\n',score_command)
      if(!f %in% colnames(train_par) ){
        writeLines(run_file,paste0(out_dir_f,f,'_',tf_name,'_',cell_name,'.sh'))
      }else if(train_par[train_par$tf_name==tf_name,f]=='TRUE'){
        writeLines(run_file,paste0(out_dir_f,f,'_',tf_name,'_',cell_name,'.sh'))
      }
      
    }
  }
  
}

run_ctcf = ''
for (i in list.files('/home/chen/data/deepGRN/results/results_revision202001/feature_importance',
                     full.names = T,pattern = '.+CTCF.+.sh',recursive = T)) {
  run_ctcf <- paste0(run_ctcf,'bash ',i,'\n')
}
writeLines(run_ctcf,'/home/chen/data/deepGRN/results/results_revision202001/feature_importance/run_ctcf.sh')
