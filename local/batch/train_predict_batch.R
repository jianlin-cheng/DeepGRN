cell_num_control =T

data_path = '/home/chen/data/deepGRN/raw/'
script_path_train = '/home/chen/py_env/py3_tf/bin/python /home/chen/Dropbox/MU/workspace/encode_dream/src/local/train.py'
script_path_predict = '/home/chen/py_env/py3_tf/bin/python /home/chen/Dropbox/MU/workspace/encode_dream/src/local/predict.py'
script_path_score = '/home/chen/py_env/py3_tf/bin/python /home/chen/Dropbox/MU/workspace/encode_dream/src/local/score.py'

tf_name= 'CTCF'
output_all_path = '/home/chen/data/deepGRN/test/'
use_peak = 1
use_cudnn = 1
ratio_negative_upper_bound = 9
num_hyperparameter_sets = 60

set.seed(2018)
setwd(output_all_path)
final_label_path = paste0(data_path,'label/final/')
pred_region_file <- paste0(data_path,'label/predict_region.bed')

# model_names = c('factornet_attention','factornet_lstm')
model_names = c('factornet_attention')
flanking = 401
bin_num = 1
use_rnaseq = c('True')
use_gencode = c('True')
use_unique35= c('True')

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
par_sets_all <- d[sample(1:nrow(d),num_hyperparameter_sets),]
rownames(par_sets_all) <- NULL
write.csv(par_sets_all,paste0(output_path,'hyperparameters.csv'))

label_file <- paste0(final_label_path,tf_name,'.train.labels.tsv.gz')
h <- read.delim(label_file,header = F,nrows = 1,as.is = T)
cell_names <- as.character(h)[4:ncol(h)]
if(cell_num_control){
  cell_names <- cell_names[1]
}
black_list_bool <- paste0(data_path,'nonblacklist_bools.csv')
final_label_all_path <- paste0(data_path,'label/final/')
  
for(i in 1:nrow(par_sets_all)){
  par_sets <- par_sets_all[i,]
  for(model_name in model_names){
    output_path_set <- paste0(output_path,model_name,'.set',i,'/')
    dir.create(output_path_set,showWarnings = F)
    setwd(output_path_set)
    
    out_model_file <- paste0(output_path_set,model_name,'.',tf_name,'.',par_sets$bin_num,'.',par_sets$flanking,'.unique35',par_sets$use_unique35,
                             '.RNAseq',par_sets$use_rnaseq,'.Gencode',par_sets$use_gencode,'.h5')
    all_args <- paste0('-i ',data_path,' -t ',tf_name,' -m ',model_name,' -f ',par_sets$flanking,' -o ',output_path_set,
                       ' -e ',par_sets$epochs,' -p ',par_sets$patience,' -s ',par_sets$batch_size,' -l ',par_sets$learningrate,
                       ' -k ',par_sets$kernel_size,' -nf ',par_sets$num_filters,' -nr ',par_sets$num_recurrent,' -nd ',par_sets$num_dense,
                       ' -d ',par_sets$dropout_rate,' -me ',par_sets$merge,' -nc ',par_sets$num_conv,' -nl ',par_sets$num_lstm,
                       ' -dl ',par_sets$num_denselayer,' -rn ',par_sets$ratio_negative)
    if(par_sets$use_rnaseq=='True'){
      all_args <- paste(all_args,'--rnaseq')
    }
    if(par_sets$use_gencode=='True'){
      all_args <- paste(all_args,'--gencode')
    }
    if(par_sets$use_unique35=='True'){
      all_args <- paste(all_args,'--unique35')
    }
    if(use_peak == 1){
      all_args <- paste(all_args,'--use_peak')
    }
    if(use_cudnn == 1){
      all_args <- paste(all_args,'--use_cudnn')
    }
    command_run <- paste(script_path_train,all_args)
    
    #predict
    model_file <- paste0(output_path_set,model_name,'.',tf_name,'.1.',par_sets$flanking,
                         '.unique35',par_sets$use_unique35,'RNAseq',par_sets$use_rnaseq,'.Gencode',par_sets$use_gencode,'.h5')
    for(final_cell_name in cell_names){
      all_args_predict <- paste0('-i ',data_path,' -m ',model_file,' -c ',final_cell_name,
                                 ' -p ',pred_region_file,' -o ',output_path_set)
      command_run <- paste0(command_run,';',script_path_predict,' ',all_args_predict)
      pred_result_file <- paste0(output_path_set,'F.',tf_name,'.',final_cell_name,'.tab.gz')
      
      all_args_score <- paste0(pred_result_file,' ',final_label_all_path,' ',output_path_set,final_cell_name,'.performance.txt')
      all_args_score_bl <- paste0(pred_result_file,' ',final_label_all_path,' ',output_path_set,final_cell_name,'.performance.txt.blacklisted -b ',black_list_bool)
      command_run <- paste0(command_run,';',script_path_score,' ',all_args_score,';',
                            script_path_score,' ',all_args_score_bl)
    }
    if(par_sets$ratio_negative < ratio_negative_upper_bound){
      system(command_run)
    }
  }
}
