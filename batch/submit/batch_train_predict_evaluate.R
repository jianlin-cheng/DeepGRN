args = commandArgs(T)

tf_name = args[1]
output_all_path = args[2]
use_peak = as.numeric(args[3])
use_cudnn = as.numeric(args[4])

# tf_name= 'CTCF'
# output_all_path = '/storage/htc/bdm/ccm3x/deepGRN/results_new/'
# use_peak = 0
# use_cudnn = 1

num_hyperparameter_sets = 60

set.seed(2018)

setwd('/storage/htc/bdm/ccm3x/deepGRN/logs')
final_label_path = '/storage/htc/bdm/ccm3x/deepGRN/raw/label/final/'

model_names = c('factornet_attention','factornet_lstm')
flanking = 401
bin_num = 1
use_rnaseq = c('True')
use_gencode = c('True')
use_unique35='True'

epochs  = 60
patience = 5
batch_size = c(32,64)
learningrate = c(0.01,0.001,0.0001)
kernel_size=34
num_filters=c(64,128)
num_recurrent=c(32,64)
num_dense=c(64,128)
dropout_rate= c(0.1,0.5)
merge = c('ave','max')

num_conv = 0
num_lstm = 0
num_denselayer = 1
ratio_negative = c(1,5,10)


output_path = paste0(output_all_path,tf_name,'/')
dir.create(output_path,recursive = T,showWarnings = F)

d = expand.grid(tf_name,flanking,bin_num,use_rnaseq,use_gencode,use_unique35,epochs,patience,
                batch_size, learningrate, kernel_size,num_filters,num_recurrent,num_dense,dropout_rate,merge,
                num_conv,num_lstm,num_denselayer,ratio_negative,use_peak,stringsAsFactors = F)
colnames(d) <- c('tf_name','flanking','bin_num','use_rnaseq','use_gencode','use_unique35',
                 'epochs','patience','batch_size','learningrate','kernel_size','num_filters','num_recurrent',
                 'num_dense','dropout_rate','merge','num_conv','num_lstm','num_denselayer','ratio_negative','use_peak')
d1 <- d[sample(1:nrow(d),num_hyperparameter_sets),]
rownames(d1) <- NULL
write.csv(d1,paste0(output_path,'hyperparameters.csv'))

label_file <- paste0(final_label_path,tf_name,'.train.labels.tsv')
h <- read.delim(label_file,header = F,nrows = 1,as.is = T)
cell_names <- as.character(h)[4:ncol(h)]
final_cell_name <- cell_names[1]

for(i in 1:nrow(d1)){
  for(model_name in model_names){
    output_path_set <- paste0(output_path,model_name,'.set',i,'/')
    dir.create(output_path_set,showWarnings = F)
    out_model_file <- paste0(output_path_set,model_name,'.',tf_name,'.',d1[i,3],'.',d1[i,2],'.unique35',d1[i,6],
                             '.RNAseq',d1[i,4],'.Gencode',d1[i,5],'.h5')
    pred_file <- paste0(output_path_set,'F.',tf_name,'.',final_cell_name,'.tab.gz')
    score_file <- paste0(output_path_set,'performance.txt')
    
    all_args <- paste(tf_name,model_name,paste(d1[i,2:6],collapse =' '),
                      output_path_set,paste(d1[i,7:16],collapse =' '),
                      out_model_file,final_cell_name,output_path_set,
                      pred_file,score_file,
                      paste(d1[i,17:21],collapse =' '))
    setwd(output_path_set)
    if(use_cudnn == 0){
      system(paste('sbatch /storage/htc/bdm/ccm3x/deepGRN/src/batch/model_train_predict_evaluate.sh',
                  all_args,paste0(output_all_path,tf_name,'/'),collapse = ' '))
    }else{
      system(paste('sbatch /storage/htc/bdm/ccm3x/deepGRN/src/batch/cudnn_model_train_predict_evaluate.sh',
                   all_args,paste0(output_all_path,tf_name,'/'),collapse = ' '))
    }

  }
}

