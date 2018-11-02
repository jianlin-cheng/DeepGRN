model_file = '/home/chen/data/deepGRN/test/CTCF/factornet_attention.set1/factornet_attention.CTCF.1.401.unique35True.RNAseqTrue.GencodeTrue.h5'
output_all_path = '/home/chen/data/deepGRN/test/CTCF/factornet_attention.set1/'

data_path = '/home/chen/data/deepGRN/raw/'
script_path = '/home/chen/py_env/py3_tf/bin/python /home/chen/Dropbox/MU/workspace/encode_dream/src/local/predict.py'

tf_name <- unlist(strsplit(gsub('.+/','',model_file),'.',fixed = T))[2]
final_label_path <- paste0(data_path,'label/final/',tf_name,'.train.labels.tsv.gz')
pred_region_file <- paste0(data_path,'label/predict_region.bed')
h <- read.delim(final_label_path,header = F,nrows = 1,as.is = T)
cell_names <- as.character(h)[4:ncol(h)]
final_cell_name <- cell_names[1]

all_args <- paste0('-i ',data_path,' -m ',model_file,' -c ',final_cell_name,' -p ',pred_region_file,' -o ',output_all_path)

print(paste(script_path,all_args))