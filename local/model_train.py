import sys
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.models import load_model
from pathlib import Path

import get_model
import utils

tf_name = sys.argv[1]
model_type = sys.argv[2]
flanking = int(sys.argv[3])
bin_num  = int(sys.argv[4])
rnaseq= sys.argv[5]=='True'
gencode= sys.argv[6]=='True'
unique35= sys.argv[7]=='True'

#For testing purposes
#tf_name = 'CTCF'
#model_type = 'factornet_lstm'
#flanking = 401
#bin_num = 1
#rnaseq = True
#gencode = True
#unique35 = True

print(tf_name,model_type,flanking,bin_num,rnaseq,gencode,unique35)

data_dir = '/home/chen/data/deepGRN/'
genome_fasta_file = data_dir+'raw/hg19.genome.fa'
DNase_path =data_dir+ 'raw/DNase'
bigwig_file_unique35 = data_dir + 'raw/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
rnaseq_data_file = data_dir + 'raw/rnaseq_data.csv'
gencode_train_file = data_dir + 'raw/gencode_feature_train.tsv'

train_label_path = data_dir + 'raw/label/train/'
final_label_path = data_dir + 'raw/label/train/'

output_model_path = data_dir + 'evaluate_test/models/'
output_history_path = data_dir + 'evaluate_test/history/'

label_data_file = train_label_path+tf_name+'.train.labels.tsv'
output_model = output_model_path + model_type + '.' + tf_name + '.'+str(bin_num)+ '.'+str(flanking)+'.unique35'+str(unique35)+ '.RNAseq'+str(rnaseq)+ '.Gencode'+str(gencode)+'.h5'
output_history = output_history_path + model_type + '.' + tf_name + '.'+str(bin_num)+ '.'+str(flanking)+'.unique35'+str(unique35)+ '.RNAseq'+str(rnaseq)+ '.Gencode'+str(gencode)+'.csv'

val_chr = ['chr11']
epochs  = 60
patience = 5
batch_size = 32
learningrate = 0.001

all_chr,cell_list,train_region,val_region,train_label_data,val_label_data,train_label_data_unbind,val_label_data_unbind,train_idx,val_idx = utils.import_label(label_data_file)
genome = utils.import_genome(genome_fasta_file,all_chr)
bw_dict_unique35 = bigwig_file_unique35

num_meta = rnaseq_train = rnaseq_val = 0
if rnaseq:
    rnaseq_data = pd.read_csv(rnaseq_data_file)
    rnaseq_train = rnaseq_data[cell_list].values
    rnaseq_val = rnaseq_data[cell_list].values
    num_meta = num_meta + 8
else:
    rnaseq_train = rnaseq_val = 0

if gencode:
    num_meta = num_meta + 6
    gencode_train,gencode_val = utils.import_gencode(gencode_train_file,train_idx,val_idx)
else:
    gencode_train = gencode_val = 0

#If model already exist, continue training
module_outfile = Path(output_model)
if module_outfile.is_file():
    model = load_model(output_model,custom_objects={'Attention1D': get_model.Attention1D})
else:
    n_channel=6
    if not unique35:
        n_channel = 5
    model = get_model.make_model(model_type,L=150+50*bin_num+flanking*2,flanking=flanking,n_channel=n_channel,num_meta=num_meta)
    model.compile(Adam(lr=learningrate),loss='binary_crossentropy',metrics=['accuracy'])

datagen_train = utils.DataGeneratorSingle(genome,bw_dict_unique35,DNase_path,train_region,train_label_data,train_label_data_unbind,cell_list,rnaseq_train,gencode_train,unique35,rnaseq,gencode,flanking,batch_size,)
datagen_val = utils.DataGeneratorSingle(genome,bw_dict_unique35,DNase_path,val_region,val_label_data,val_label_data_unbind,cell_list,rnaseq_val,gencode_val,unique35,rnaseq,gencode,flanking,batch_size)

checkpointer = ModelCheckpoint(filepath=output_model,verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
csv_logger = CSVLogger(output_history,append=True)

history = model.fit_generator(datagen_train,epochs=epochs,validation_data=datagen_val,callbacks=[checkpointer, earlystopper, csv_logger])
