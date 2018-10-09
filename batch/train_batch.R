setwd('/storage/htc/bdm/ccm3x/deepGRN/logs')

tf_names <- gsub('.train.labels.tsv','',list.files('/storage/htc/bdm/ccm3x/deepGRN/raw/label/final'))
model_names = c('factornet_lstm','factornet_attention')
output_path = '/storage/htc/bdm/ccm3x/deepGRN/evaluate/'
unique35='True'

for(model_name in model_names){
  for (i in tf_names){
    for(rnaseq in c('False','True')){
      for (gencode in c('False','True')) {
        system(paste('sbatch ~/configs/py_sub.sbatch /storage/htc/bdm/ccm3x/deepGRN/src/model_train.py',
                     i,model_name,401,1,rnaseq,gencode,unique35,output_path))
      }
    }
  }
}