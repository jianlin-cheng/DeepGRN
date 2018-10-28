args <- commandArgs(T)

# tf_name = 'NANOG'
# bin_num = 0
# model_file = '/storage/htc/bdm/ccm3x/deepGRN/models/fullconv_simple_NANOG_bin_0.h5'
# cell_name = 'induced_pluripotent_stem_cell'

tf_name = args[1]
bin_num = as.numeric(args[2])
model_file = args[3]
cell_name = args[4]

source('/storage/htc/bdm/ccm3x/deepGRN/src/utils.R')

window_size = 200
stride = 50
batch_size = 16
flanking_length = 0
# predict_chr = c('chr1','chr8','chr21')
predict_chr = c('chr21')

unique35_file = '/storage/htc/bdm/ccm3x/deepGRN/raw/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
genome_fasta_file = '/storage/htc/bdm/ccm3x/deepGRN/raw/hg19.genome.fa'
feature_dnase_dir = '/storage/htc/bdm/ccm3x/deepGRN/raw/DNase/'
label_path = '/storage/htc/bdm/ccm3x/deepGRN/raw/label/final/'
blacklist_merge_file = '/storage/htc/bdm/ccm3x/deepGRN/raw/label/test_regions.blacklistfiltered.merged.bed'
out_path = '/storage/htc/bdm/ccm3x/deepGRN/predictions/'

model_name = gsub('.+/(.+).h5','\\1',model_file)
print(paste('TF name:',tf_name,'bin_num:',bin_num,'model name:',model_name))
out_path <- paste0(out_path,model_name)
dir.create(out_path,showWarnings = F)

genome <- readDNAStringSet(genome_fasta_file)

blacklist_merge <- fread(blacklist_merge_file,sep = '\t',header = F,data.table = F)
blacklist_merge <- blacklist_merge[blacklist_merge[,1]%in%predict_chr,]
label_data <- fread(paste0(label_path,tf_name,'.train.labels.tsv'),sep = '\t',header = T,data.table = F)
label_data <- label_data[label_data$chr %in% predict_chr,cell_name]=='B'
cell_bw <- lapply(import_bw(paste0(feature_dnase_dir,cell_name,'.1x.bw'),blacklist_merge), function(x)data.table(as.numeric(x)))
unique35_bw <- lapply(import_bw(unique35_file,blacklist_merge), function(x)data.table(as.numeric(x)))

model <- load_model_hdf5(model_file)
model %>% compile(loss = 'binary_crossentropy',optimizer = optimizer_adam(lr = 0.001),metrics = c('accuracy'))

if(length(model$output$get_shape()$dims)==2){
  bin_num = 0
}

if(bin_num==0){
  pred <- predict_single(cell_name,model,genome,cell_bw,unique35_bw,blacklist_merge,bin_num,batch_size)
}else{
  pred <- predict_multiple(cell_name,model,genome,cell_bw,unique35_bw,blacklist_merge,bin_num,batch_size)
}

res <- model_eval(pred,label_data)
write.csv(pred,paste0(out_path,'/',tf_name,'_',cell_name,'.csv'),row.names = F,quote = F)
write.csv(res,paste0(out_path,'/',tf_name,'_',cell_name,'_eval.csv'))

