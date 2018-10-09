tf_names <- gsub('.train.labels.tsv','',list.files('/home/chen/data/deepGRN/raw/label/final'))
tf_names <- c('CTCF','JUND','MAX')
model_names = c('factornet_attention')
unique35='False'

for(model_name in model_names){
  for (i in tf_names){
    for(rnaseq in c('False')){
      for (gencode in c('False')) {
        write(paste('python ~/Dropbox/MU/workspace/encode_dream/src/local/model_train.py',i,model_name,401,1,rnaseq,gencode,unique35),
              file = '~/train.sh',append=TRUE)
      }
    }
  }
}

