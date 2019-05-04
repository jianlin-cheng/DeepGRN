import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.models import load_model
import os,sys,argparse
import utils,get_model,res_model
from keras.utils import plot_model # require graphviz
from keras.models import Model
import pyBigWig

best_model_file = 'attention_after_lstm.JUND.1.401.unique35True.RNAseqTrue.GencodeTrue.h5'
label_file = '/home/chen/data/deepGRN/raw/label/final/JUND.train.labels.tsv.gz'
unique35 = True
kernel_size = 20

double_strand = True
batch_size = 32
flanking = 401
rnaseq = True
gencode = True

data_dir = '/home/chen/data/deepGRN/raw/'
output_dir='/home/chen/Dropbox/MU/workspace/encode_dream/DeepGRN/data/models/'
genome_fasta_file = data_dir+'/hg19.genome.fa'
DNase_path =data_dir+ '/DNase/'
bigwig_file_unique35 = data_dir + '/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
rnaseq_data_file = data_dir+'rnaseq_data.csv'

model = load_model(output_dir+best_model_file,custom_objects={'Attention1D': get_model.Attention1D})
plot_model(model,show_shapes=True,show_layer_names=False,to_file='model.png')


input_data = pd.read_csv(label_file,sep='\t')
pred_cell_name = list(input_data.columns)[3]
DNase_file = DNase_path+pred_cell_name+'.1x.bw'
predict_region_file = data_dir+'label/predict_region.bed'
contig_mask = ( (input_data['chr'].values == 'chr1') 
                | (input_data['chr'].values == 'chr8') 
                | (input_data['chr'].values == 'chr21') )
input_data = input_data.loc[contig_mask]    

predict_region = pd.read_csv(predict_region_file, sep='\t', header=None)
positive_flag = input_data['liver']=='B'
predict_region_positive = predict_region.loc[positive_flag.values]
pred_chr = set(predict_region_positive[0])
genome = utils.import_genome(genome_fasta_file,pred_chr)

conv1dout_model1 = Model(inputs=model.input,outputs=model.get_layer('dim_reduction').get_output_at(0))
conv1dout_model2 = Model(inputs=model.input,outputs=model.get_layer('dim_reduction').get_output_at(1))

rnaseq_pred = gencode_pred = 0
if rnaseq:
    rnaseq_data = pd.read_csv(rnaseq_data_file)
    pred_cell_list = [pred_cell_name]
    rnaseq_pred = rnaseq_data[pred_cell_list].values

if gencode:
    gencode_file = data_dir + '/gencode_feature_test.tsv'
    gencode_pred = pd.read_csv(gencode_file, sep='\t', header=None,dtype=utils.np.bool_).values  


#aviod OOM
predict_region_positive1 = predict_region_positive

datagen_pred = utils.PredictionGeneratorSingle(genome,bigwig_file_unique35,DNase_file,
                                       predict_region_positive1,rnaseq_pred,gencode_pred,
                                       unique35,rnaseq,gencode,flanking,batch_size,double_strand)

print('start prediction')
pred = conv1dout_model1.predict_generator(datagen_pred,verbose=1, workers=1, use_multiprocessing=False)

pred_max_probs = np.max(pred,axis=1)
pred_argmax_probs = np.argmax(pred,axis=1)


all_activate_subseq = []

for i in range(len(pred_argmax_probs)):
    current_chr = predict_region_positive1.iloc[i,0]
    current_start = predict_region_positive1.iloc[i,1]-flanking
    current_end = current_start+1002
    current_middle =current_start+int(round(1002*pred_argmax_probs[i]/98))
    seq_str = genome[current_chr][(current_middle-10):(current_middle+10)].seq.upper()
    all_activate_subseq.append(seq_str)
    


#pred_keep_bool = pred_keep> np.median(pred_keep)*1.5
#pred_keep_bool = pred_max_probs > np.quantile(pred,0.5,axis=1)*1,5
#query_seq = 175
#result_bins = []
#i=0
#for bin_idx in range(pred_keep_bool.shape[0]):
#    for conv_pos in range(pred_keep_bool.shape[1]):
#        if pred_keep_bool[bin_idx][conv_pos]:
#            current_chr = predict_region_positive1.iloc[bin_idx,0]
#            current_start = predict_region_positive1.iloc[bin_idx,1]-flanking+conv_pos
#            current_end = current_start+kernel_size
#            seq_str = genome[current_chr][current_start:current_end].seq.upper()
#            all_activate_subseq.append(seq_str)
#            if i == query_seq:
#                result_bins.append(bin_idx)
#            i=i+1
            

f = open("JUND_75q_attention.fasta", "w")
for i in range(len(all_activate_subseq)):
    f.write(">seq"+str(i)+'\n')
    f.write(all_activate_subseq[i]+'\n')
f.close()

out_dir_name = '/home/chen/Dropbox/MU/workspace/encode_dream/data/figs_data/fig_motif/'

import re

motif1_regex = 'AGGTAAACA[AC]ATG[CG]C'
motif2_regex = 'TGGTGAGG'
motif3_regex = 'TGACTCAC'
motif4_regex = 'TCACC[AT]GC[CT][AC]A[AG]'



motif1_seq = []
motif1_seq_dnase = np.zeros(20)
for i in range(len(pred_argmax_probs)):
    current_chr = predict_region_positive1.iloc[i,0]
    current_start = predict_region_positive1.iloc[i,1]-flanking
    current_end = current_start+1002
    current_middle =current_start+int(round(1002*pred_argmax_probs[i]/98))
    seq_str = genome[current_chr][(current_middle-10):(current_middle+10)].seq.upper()

    if bool(re.match('.*'+motif2_regex+'.*',seq_str)):
        bigwig = pyBigWig.open(DNase_file)
        bw_val = np.array(bigwig.values(current_chr, current_middle-10, current_middle+10))
        bigwig.close()
        bw_val[np.isnan(bw_val)] = 0
        motif1_seq.append(seq_str)
        motif1_seq_dnase = motif1_seq_dnase + bw_val

motif1_seq_dnase_mean = motif1_seq_dnase/len(motif1_seq)
np.savetxt(out_dir_name+'motif2.csv',motif1_seq_dnase_mean, fmt='%.8e', delimiter=',')
















