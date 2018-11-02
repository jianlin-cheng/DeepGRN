import sys
import numpy as np
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
output_dir = sys.argv[8]

epochs  = int(sys.argv[9])
patience = int(sys.argv[10])
batch_size = int(sys.argv[11])
learningrate = float(sys.argv[12])
kernel_size = int(sys.argv[13])
num_filters = int(sys.argv[14])
num_recurrent = int(sys.argv[15])
num_dense = int(sys.argv[16])
dropout_rate = float(sys.argv[17])
merge = sys.argv[18]

num_conv = int(sys.argv[19])
num_lstm = int(sys.argv[20])
num_denselayer = int(sys.argv[21])
ratio_negative = int(sys.argv[22])
use_peak = int(sys.argv[23])

#For testing purposes
#tf_name = 'CTCF'
#model_type = 'factornet_attention'
#flanking = 401
#bin_num = 1
#rnaseq = True
#gencode = True
#unique35 = True
#output_dir = '/home/ccm3x/test/'
#
#epochs = 3
#patience = 3
#batch_size = 64
#learningrate = 0.01
#kernel_size  = 34
#num_filters = 64
#num_recurrent = 32
#num_dense = 64
#dropout_rate = 0.1
#merge = 'ave'
#
#num_conv = 0
#num_lstm = 0
#num_denselayer = 1
#ratio_negative = 5
#use_peak = 0

np.random.seed(2018)

print(tf_name,model_type,flanking,bin_num,rnaseq,gencode,unique35,output_dir)
print(epochs,patience,batch_size,learningrate,kernel_size,num_filters,num_recurrent,num_dense,dropout_rate,merge)
print(num_conv,num_lstm,num_denselayer,ratio_negative,use_peak)

data_dir = '/storage/htc/bdm/ccm3x/deepGRN/'
genome_fasta_file = data_dir+'raw/hg19.genome.fa'
DNase_path =data_dir+ 'raw/DNase'
bigwig_file_unique35 = data_dir + 'raw/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
rnaseq_data_file = data_dir + 'raw/rnaseq_data.csv'
gencode_train_file = data_dir + 'raw/gencode_feature_train.tsv'

train_label_path = data_dir + 'raw/label/train/'
train_peaks_path = data_dir + 'raw/label/train_positive/'

output_model_path = output_dir
output_history_path = output_dir

label_data_file = train_label_path+tf_name+'.train.labels.tsv'
positive_peak_file = train_peaks_path+tf_name+'.train.peak.tsv'
output_model = output_model_path + model_type + '.' + tf_name + '.'+str(bin_num)+ '.'+str(flanking)+'.unique35'+str(unique35)+ '.RNAseq'+str(rnaseq)+ '.Gencode'+str(gencode)+'.h5'
output_history = output_history_path + model_type + '.' + tf_name + '.'+str(bin_num)+ '.'+str(flanking)+'.unique35'+str(unique35)+ '.RNAseq'+str(rnaseq)+ '.Gencode'+str(gencode)+'.csv'

window_size = 200
val_chr = ['chr11']
all_chr,cell_list,train_region,val_region,train_label_data,val_label_data,train_label_unbind,val_label_unbind,train_idx,val_idx = utils.import_label(label_data_file)
genome = utils.import_genome(genome_fasta_file,all_chr)

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
    model = get_model.make_model(model_type,L=window_size+2*flanking,n_channel=n_channel, 
                                 num_conv = num_conv,num_lstm = num_lstm,num_denselayer=num_denselayer,
                                 kernel_size=kernel_size,num_filters=num_filters, num_recurrent=num_recurrent, num_dense=num_dense, 
                                 dropout_rate=dropout_rate, num_meta=num_meta,merge=merge)
    model.compile(Adam(lr=learningrate),loss='binary_crossentropy',metrics=['accuracy'])

if use_peak:
    label_bind = pd.read_csv(positive_peak_file,sep='\t')
    train_label_bind = label_bind[label_bind.chrom.isin(set(all_chr)-set(val_chr))]
    val_label_bind = label_bind[label_bind.chrom.isin(val_chr)]
    datagen_train = utils.TrainGeneratorSingle(genome,bigwig_file_unique35,DNase_path,train_region,train_label_bind,train_label_unbind,cell_list,rnaseq_train,gencode_train,True,unique35,rnaseq,gencode,flanking,batch_size,ratio_negative)
    datagen_val = utils.TrainGeneratorSingle(genome,bigwig_file_unique35,DNase_path,val_region,val_label_bind,val_label_unbind,cell_list,rnaseq_val,gencode_val,False,unique35,rnaseq,gencode,flanking,batch_size,ratio_negative)
else:
    
    datagen_train = utils.DataGeneratorSingle(genome=genome,bw_dict_unique35=bigwig_file_unique35,DNase_path=DNase_path,
                                              label_region=train_region,label_data=train_label_data,label_data_unbind=train_label_unbind,
                                              cell_list=cell_list,rnaseq_data = rnaseq_train,gencode_data=gencode_train,
                                              unique35=unique35,rnaseq=rnaseq,gencode=gencode,flanking=flanking,batch_size=batch_size,ratio_negative=ratio_negative)
    datagen_val = utils.DataGeneratorSingle(genome,bigwig_file_unique35,DNase_path,val_region,val_label_data,val_label_unbind,cell_list,rnaseq_val,gencode_val,unique35,rnaseq,gencode,flanking,batch_size,ratio_negative)

checkpointer = ModelCheckpoint(filepath=output_model,verbose=1, save_best_only=True, monitor='val_acc')
earlystopper = EarlyStopping(monitor='val_acc', patience=patience, verbose=1)
csv_logger = CSVLogger(output_history,append=True)

history = model.fit_generator(datagen_train,epochs=epochs,validation_data=datagen_val,callbacks=[checkpointer, earlystopper, csv_logger])
