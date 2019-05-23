train_command = '/storage/hpc/data/ccm3x/py_env/tf_gpu_112/bin/python /home/ccm3x/data/deepGRN/src/train_relay.py'
pred_command = '/storage/hpc/data/ccm3x/py_env/tf_gpu_112/bin/python /home/ccm3x/data/deepGRN/src/predict.py'
score_commnad = '/storage/hpc/data/ccm3x/py_env/tf_cpu/bin/python /home/ccm3x/data/deepGRN/src/score.py '
data_dir = '/home/ccm3x/data/deepGRN/raw/'
output_path = '/home/ccm3x/data/deepGRN/results/results_res_fullbin/'
info_file = '/home/ccm3x/data/deepGRN/deepgrn_score.csv'
sbatch_file = '/home/ccm3x/data/configs/gpu_sub_1d22h.sh'
train_parameters_file = '/home/ccm3x/data/deepGRN/train_parameters.csv'

black_list_file = paste0(data_dir,'nonblacklist_bools.csv')
pred_region_file = paste0(data_dir,'label/predict_region.bed')
label_path = paste0(data_dir,'label/final/')

slurm_prefix <- readLines(sbatch_file)
slurm_prefix <- slurm_prefix[1:(length(slurm_prefix)-2)]
slurm_prefix <- paste0(slurm_prefix,collapse = '\n')

d <- read.csv(train_parameters_file,as.is = T)
info <- read.csv(info_file,as.is = T)

for(model_type in c('resnetv1','resnetv1_lstm')){
    for(rs in c(1,2,5,2018,2019)){
        for (i in 1) {
            tf_name = d$TF[i]
            output_path_i <- paste0(output_path,'/',tf_name,'/',model_type,'_',rs,'/')
            dir.create(output_path_i,recursive = T,showWarnings = F)
            ts_file <- paste0(output_path_i,'train_script.sh')
            ps_file <- paste0(output_path_i,'predict_script.sh')
            train_command1 <- paste0(slurm_prefix,'\n',train_command,' -i ',data_dir,' -t ',tf_name,' -o ',output_path_i,
                                    ' -s ',d$batch_size[i],' -l ',d$learningrate[i],' -k ',d$kernel_size[i],' -ap ',model_type,
                                    ' -nf ',d$num_filters[i],' -nr ',d$num_recurrent[i],' -nd ',d$num_dense[i],' -rs ',rs,
                                    ' -d ',d$dropout_rate[i],' -rd1 ',d$rnn_dropout1[i],' -rd2 ',d$rnn_dropout2[i],
                                    ' -me ',d$merge[i],' -nc ',d$num_conv[i],' -dl ',d$num_denselayer[i],' -ts ',ts_file,
                                    ' -ps ',ps_file,' -rn ',d$ratio_negative[i],' -nl 1 -e 5 --use_cudnn --unique35')
            
            model_file <- paste0(output_path_i,'/best_model.h5')
            cells <- info$cell[info$TF==tf_name]
            pred_command1 <- ''
            score_commnad1 <- ''
            for(cell in cells){
                out_file_name <- paste0(output_path_i,'/F.',tf_name,'.',cell,'.tab.gz')
                pred_command1 <- paste0(pred_command1,pred_command,' -i ',data_dir,' -p ',pred_region_file,' -l ',
                                        black_list_file,' -m ',model_file, ' -c ',cell,' -o ',out_file_name,'\n')
                score_commnad1 <- paste0(score_commnad1,score_commnad,' ',out_file_name,' ',label_path,' ',
                                         gsub('tab.gz','txt',out_file_name),'\n')
            }
            

            
            write(train_command1, file = ts_file,append = F)
            write(paste0(slurm_prefix,'\n',pred_command1,'\n',score_commnad1), file = ps_file,append = F)
        }
    }
}

for(i in list.files(output_path,pattern = 'train_script.sh',recursive = T,full.names = T)){
    system(paste('sbatch',i))
}
