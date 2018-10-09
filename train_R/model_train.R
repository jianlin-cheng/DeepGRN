args <- commandArgs(T)

# tf_name = 'NANOG'
# bin_num = 0
# model_file = '/storage/htc/bdm/ccm3x/deepGRN/model_templates/fullconv_simple.h5'

tf_name = args[1]
bin_num = as.numeric(args[2])
model_file = args[3]

source('/storage/htc/bdm/ccm3x/deepGRN/src/train_R/utils.R')

window_size = 200
stride = 50
batch_size = 32
epochs = 12
flanking_length = 0
validate_chr = c('chr11')

unique35_file = '/storage/htc/bdm/ccm3x/deepGRN/raw/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
genome_fasta_file = '/storage/htc/bdm/ccm3x/deepGRN/raw/hg19.genome.fa'
feature_dnase_dir = '/storage/htc/bdm/ccm3x/deepGRN/raw/DNase/'
label_path = '/storage/htc/bdm/ccm3x/deepGRN/raw/label/train/'
label_validate_path = '/storage/htc/bdm/ccm3x/deepGRN/raw/label/final/'
blacklist_merge_file = '/storage/htc/bdm/ccm3x/deepGRN/raw/label/train_regions.blacklistfiltered.merged.bed'
feature_common_file = '/storage/htc/bdm/ccm3x/deepGRN/raw/feature_common.rda'
out_path = '/storage/htc/bdm/ccm3x/deepGRN/tmp/performance/'
out_path_model = '/storage/htc/bdm/ccm3x/deepGRN/tmp/models/'

model_name = gsub('.+/(.+).h5','\\1',model_file)
print(paste('TF name:',tf_name,'bin_num:',bin_num,'model name:',model_name))
out_path_history <- paste0(out_path,model_name,'_',tf_name,'_bin_',bin_num,'.csv')
out_path_model <- paste0(out_path_model,model_name,'_',tf_name,'_bin_',bin_num,'.h5')
if(file.exists(out_path_model)) model_file <- out_path_model
genome <- readDNAStringSet(genome_fasta_file)
blacklist_merge <- fread(blacklist_merge_file,sep = '\t',header = F,data.table = F)
unique35_bw <- import_bw1(unique35_file,blacklist_merge)


# save(list = c('genome','blacklist_merge','unique35_bw'),file = feature_common_file)

# load(feature_common_file)
label_data <- fread(paste0(label_path,tf_name,'.train.labels.tsv'),sep = '\t',header = T,data.table = F,drop = 1:3)=='B'

model <- load_model_hdf5(model_file)
model %>% compile(loss = 'binary_crossentropy',optimizer = optimizer_adam(lr = 0.001),metrics = c('accuracy'))

if(length(model$output$get_shape()$dims)==2){
  bin_num = 0
}
train_result <- train(model,genome,feature_dnase_dir,unique35_bw,label_data,blacklist_merge,bin_num,batch_size,
                      epochs,validate_chr,out_path_history,out_path_model)

