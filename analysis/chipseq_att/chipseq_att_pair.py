import argparse
import numpy as np
import pandas as pd
from keras.models import load_model
from keras_pos_embd import PositionEmbedding, TrigPosEmbedding
import tensorflow as tf


import utils
import get_model

dnase_ratio_cutoff = 20
batch_size = 2000

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Predict a model.",formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--tf_name', '-t', type=str, required=True,help='tf_name')

    return parser

def make_predict_input(genome,bigwig_file_unique35,DNase_file,pred_region,
                       flanking,unique35):
    #pred_region format: pd.Dataframe([chr_id,start_id,end_id])
    x_batch = np.zeros((pred_region.shape[0],1002,6))
    for i in range(pred_region.shape[0]):
        chr_id = pred_region.iloc[i,0]
        bin_start = int(pred_region.iloc[i,1])
        bin_stop = int(pred_region.iloc[i,2])
        seq_start = int(bin_start-flanking)
        seq_end = int(bin_stop+flanking)
        x_batch[i,:,:] = utils.make_x(genome,bigwig_file_unique35,DNase_file,
               seq_start,seq_end,flanking,chr_id)        
        
    x_rc = utils.make_rc_x(x_batch)
    if not unique35:
        x_batch = np.delete(x_batch,4,2)
        x_rc = np.delete(x_rc,4,2)
    X = [x_batch,x_rc]
    return X

def get_bigwig(chipseq_file,chr_id,start,stop):
    bigwig = utils.pyBigWig.open(chipseq_file)
    bw_val = np.array(bigwig.values(chr_id, start, stop))
    bw_val[np.isnan(bw_val)] = 0
    bigwig.close()
    return bw_val
    


tf_name = 'TAF1'
model_type = 'pair'

data_dir = '/home/chen/data/deepGRN/raw/'
model_file = '/home/chen/data/deepGRN/results/results_revision202001/best_'+model_type+'/best_models/'+tf_name+'_'+model_type+'.h5'
label_file = '/home/chen/data/deepGRN/raw/label/final/'+tf_name+'.train.labels.tsv.gz'


model = load_model(model_file,custom_objects={'Attention1D': get_model.Attention1D,
                                              'PositionEmbedding':PositionEmbedding,
                                              'TrigPosEmbedding':TrigPosEmbedding,
                                              'tf':tf})
model_inputs = model.get_input_shape_at(0)
#########################################
predict_region = pd.read_csv(label_file, sep='\t')
cell_name = predict_region.columns.values[-1]
att_avg_out = '/home/chen/data/deepGRN/results/results_revision202001/chipseq_att/pair/data/'+tf_name+'_'+cell_name+'_fw_att.csv'
chipseq_out = '/home/chen/data/deepGRN/results/results_revision202001/chipseq_att/pair/data/'+tf_name+'_'+cell_name+'_fw_chipseq.csv'
pos_out = '/home/chen/data/deepGRN/results/results_revision202001/chipseq_att/pair/data/'+tf_name+'_'+cell_name+'_fw_pos.csv'

contig_mask = ( (predict_region['chr'].values == 'chr1') 
                | (predict_region['chr'].values == 'chr8') 
                | (predict_region['chr'].values == 'chr21') )

predict_region1 = predict_region.loc[contig_mask]

pred_file = '/home/chen/data/deepGRN/results/results_revision202001/best_pair/best_pred/F.'+tf_name+'.'+cell_name+'.tab.gz'
pred_data = pd.read_csv(pred_file, sep='\t',names=['chr','start','stop','val'])
bind_labels = np.logical_and(predict_region1[cell_name]=='B', pred_data['val']>0.5)

predict_region2 = predict_region1.loc[bind_labels,:]
#    #########################################
#    predict_region3 = predict_region2.iloc[15908,:]
#    
pred_chr = set(['chr1','chr8','chr21'])
#    predict_region.index = range(predict_region.shape[0])

if model_inputs[0][2] == 6:
    unique35 = True
else:
    unique35 = False

flanking = 401



genome_fasta_file = data_dir+'/hg19.genome.fa'
DNase_path =data_dir+ '/DNase/'
bigwig_file_unique35 = data_dir + '/wgEncodeDukeMapabilityUniqueness35bp.bigWig'


DNase_file = DNase_path+cell_name+'.1x.bw'
genome = utils.import_genome(genome_fasta_file,pred_chr)

chipseq_file = data_dir+'/chipseq_peak_final/ChIPseq.'+cell_name+'.'+tf_name+'.fc.signal.test.bw'


#    att_avg_layer = get_model.Lambda(lambda x:get_model.K.sum(x,axis=2,keepdims=False),name='avg_output')
fw_output = model.get_layer('attention_score').get_output_at(0)
#    fw_avg_output = att_avg_layer(fw_output)

#    bin_len = model.input[0].shape[1].value
#    att_len = fw_output.shape[1].value

att_model_fw = get_model.Model(inputs=model.input[0:2],outputs=fw_output)
#    att_model_rc = get_model.Model(inputs=model.input[0:2],outputs=model.get_layer('attention_score').get_output_at(1))

#    pred_region = pd.DataFrame(columns=['chr_id', 'start_id', 'end_id'])
#    for i in range(1):
#         pred_region.loc[i] = [chr_id,start_id,end_id]

print(predict_region2.shape[0]//batch_size +1)
seq_idx = 1
   


f1 = open(att_avg_out,'ab')
f2 = open(chipseq_out,'ab')
f3 = open(pos_out,'a')
for s in range(predict_region2.shape[0]//batch_size +1 ):
    
    print(s)
    start_idx_s = s*batch_size
    end_idx_s = min(start_idx_s+batch_size,predict_region2.shape[0])
            
    predict_region3 = predict_region2.iloc[start_idx_s:end_idx_s]
    datagen_pred = make_predict_input(genome,bigwig_file_unique35,DNase_file,
                                      predict_region3,flanking,unique35)
    
    dnase_val = datagen_pred[0][:,:,-1]
    dnase_ratio = np.max(dnase_val,axis=1)/np.mean(dnase_val,axis=1)
    
    datagen_pred1 =[datagen_pred[0][dnase_ratio>dnase_ratio_cutoff,:,:],datagen_pred[1][dnase_ratio>dnase_ratio_cutoff,:,:]]
#        dnase_val1 = dnase_val[dnase_ratio>dnase_ratio_cutoff,:]
    predict_region4 = predict_region3.loc[dnase_ratio>dnase_ratio_cutoff]

    pred_fw = att_model_fw.predict(datagen_pred1)
    if len(pred_fw)>0:
        pred_fw1 = np.zeros((pred_fw.shape[0],pred_fw.shape[1]))
        
        for r in range(pred_fw1.shape[0]):
            ind = np.unravel_index(np.argmax(pred_fw[r,:,:], axis=None), pred_fw[r,:,:].shape)
            pred_fw1[r,:] = pred_fw[r,:,ind[1]]
            
        
                
        chipseq_arr = np.zeros((pred_fw.shape[0],1002))
        for r in range(predict_region4.shape[0]):
            seq_chr = predict_region4['chr'].iloc[r]
            seq_start = predict_region4['start'].iloc[r]-flanking
            chipseq_arr[r,:] = get_bigwig(chipseq_file,seq_chr,seq_start,seq_start+1002)
            f3.write(seq_chr+'\t'+str(seq_start)+'\n')
    
            seq_idx += 1
            
            
        np.savetxt(f1,pred_fw1)
        np.savetxt(f2,chipseq_arr, fmt='%1.2f')

f1.close()
f2.close()
f3.close()
