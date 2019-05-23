pred_command = '/storage/hpc/data/ccm3x/py_env/tf_cpu/bin/python /home/ccm3x/data/deepGRN/src/predict.py'
score_commnad = '/storage/hpc/data/ccm3x/py_env/tf_cpu/bin/python /home/ccm3x/data/deepGRN/src/score.py '
data_dir = '/home/ccm3x/data/deepGRN/raw/'
models_path = '/home/ccm3x/data/deepGRN/results/reproduce3/'
info_file = '/home/ccm3x/data/deepGRN/deepgrn_score.csv'
sbatch_file_pred = '/home/ccm3x/data/configs/cpu_sub_1d22h.sh'

patience = 5

black_list_file = paste0(data_dir,'nonblacklist_bools.csv')
pred_region_file = paste0(data_dir,'label/predict_region.bed')
# pred_region_file1 = paste0(data_dir,'label/predict_region_sub1.bed')
# pred_region_file2 = paste0(data_dir,'label/predict_region_sub2.bed')
label_path = paste0(data_dir,'label/final/')
d <- read.csv(info_file,as.is = T)



slurm_prefix <- readLines(sbatch_file_pred)
slurm_prefix <- slurm_prefix[1:(length(slurm_prefix)-1)]
slurm_prefix <- paste0(slurm_prefix,collapse = '\n')

for(i in list.files(models_path,pattern = '.+\\.csv',recursive = T,full.names = T)){
    out_dir <- gsub('(.+/).+','\\1',i)
    tf_name <- gsub('.+/(.+)/','\\1',out_dir)
    cell_names <- d$cell[d$TF==tf_name]
    model_file <- list.files(out_dir,pattern = '.+\\.h5',full.names = T)
    
        
    for(pred_cell_name in cell_names){
        out_file_name <- paste0(out_dir,'/F.',tf_name,'.',pred_cell_name,'.tab.gz')
        pred_command1 <- paste0(slurm_prefix,'\n',pred_command,' -i ',data_dir,' -p ',pred_region_file,' -l ',black_list_file,
                               ' -m ',model_file, ' -c ',pred_cell_name,' -o ',out_file_name,'\n')
        
        score_commnad1 <- paste0(score_commnad,' ',out_file_name,' ',label_path,' ',
                                 gsub('tab.gz','txt',out_file_name))
        write(paste0(pred_command1,'\n',score_commnad1), file =paste0(out_dir,'/predict.sh'),append = F)
        
        # out_file_name <- paste0(out_dir,'/F.',tf_name,'.',pred_cell_name,'.tab.gz.seg2')
        # pred_command2 <- paste0(slurm_prefix,'\n',pred_command,' -i ',data_dir,' -p ',pred_region_file2,
        #                        ' -m ',model_file, ' -c ',pred_cell_name,' -o ',out_file_name,' -l ',black_list_file,'\n')
        # write(pred_command2, file =paste0(out_dir,'/predict2.sh'),append = F)

    }
}

###################after submit the training jobs###########################
for(i in list.files(models_path,pattern = '.+\\.csv',recursive = T,full.names = T)){
    out_dir <- gsub('(.+/).+','\\1',i)
    his <- read.csv(i)
    if(((nrow(his)-which.max(his$val_acc))>=0)){
        # training finished
        print(i)
        if(!file.exists(paste0(out_dir,'/','already_predict'))){
            # file.create(paste0(out_dir,'/','already_predict'))
            print(paste0('sbatch ',out_dir,'/predict.sh'))
        }
        # if(!file.exists(paste0(out_dir,'/','already_predict2'))){
        #     file.create(paste0(out_dir,'/','already_predict2'))
        #     system(paste0('sbatch ',out_dir,'/predict2.sh'))
        # }
        
    }
}

###################after submit the prediction jobs###########################
# for(i in list.files(models_path,pattern = '.+\\.csv',recursive = T,full.names = T)){
#     out_dir <- gsub('(.+/).+','\\1',i)
#     seg_files <- list.files(out_dir,pattern = '.+\\.seg\\d',full.names = T)
#     if(length(seg_files)==2){
#         out_file_name <- gsub('\\.seg.+','',seg_files[1])
#         system(paste0('cat ',seg_files[1],' ',seg_files[2],' > ',out_file_name))
#         
#         print(paste0('sbatch ',sbatch_file_score,' ',score_commnad,' ',out_file_name,' ',label_path,' ',
#                      gsub('tab.gz','txt',out_file_name)))
#         
#     }
# }




