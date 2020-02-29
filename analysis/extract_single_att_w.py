import argparse
import numpy as np
import pandas as pd
from keras.models import load_model
from keras_pos_embd import PositionEmbedding, TrigPosEmbedding
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

import utils
import get_model

dnase_ratio_cutoff = 15

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


def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    
    tf_name = args.tf_name
    
#    tf_name = 'CTCF'
    model_type = 'single'
    
    data_dir = '/home/chen/data/deepGRN/raw/'
    model_file = '/home/chen/data/deepGRN/results/results_revision202001/best_'+model_type+'/best_models/'+tf_name+'_'+model_type+'.h5'
    label_file = '/home/chen/data/deepGRN/raw/label/final/'+tf_name+'.train.labels.tsv.gz'
    
    out_fa_file = '/home/chen/data/deepGRN/results/results_revision202001/motif/'+model_type+'/fa/'+tf_name+'_fw.fa'
    att_avg_out = '/home/chen/data/deepGRN/results/results_revision202001/motif/'+model_type+'/attention_score/'+tf_name+'_fw.csv'
    dnase_out = '/home/chen/data/deepGRN/results/results_revision202001/motif/'+model_type+'/dnase/'+tf_name+'_fw.csv'

    
    model = load_model(model_file,custom_objects={'Attention1D': get_model.Attention1D,
                                                  'PositionEmbedding':PositionEmbedding,
                                                  'TrigPosEmbedding':TrigPosEmbedding,
                                                  'tf':tf})
    model_inputs = model.get_input_shape_at(0)
    #########################################
    predict_region = pd.read_csv(label_file, sep='\t')
    cell_name = predict_region.columns.values[-1]
    
    contig_mask = ( (predict_region['chr'].values == 'chr1') 
                    | (predict_region['chr'].values == 'chr8') 
                    | (predict_region['chr'].values == 'chr21') )
    
    predict_region1 = predict_region.loc[contig_mask]
    
    bind_labels = predict_region1[cell_name]=='B'
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
    if len(model_inputs) == 2:
        rnaseq = False
        gencode = False
    elif model_inputs[2][1] == 6:
        rnaseq = False
        gencode = True
    elif model_inputs[2][1] == 8:
        rnaseq = True
        gencode = False
    elif model_inputs[2][1] == 14:
        rnaseq = True
        gencode = True
        
    window_size = 200
    flanking = 401

    

    genome_fasta_file = data_dir+'/hg19.genome.fa'
    DNase_path =data_dir+ '/DNase/'
    bigwig_file_unique35 = data_dir + '/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
    rnaseq_data_file = data_dir + '/rnaseq_data.csv'
    gencode_file = data_dir + '/gencode_feature_test.tsv'
        
    DNase_file = DNase_path+cell_name+'.1x.bw'
    genome = utils.import_genome(genome_fasta_file,pred_chr)
    
#    att_avg_layer = get_model.Lambda(lambda x:get_model.K.sum(x,axis=2,keepdims=False),name='avg_output')
    fw_output = model.get_layer('attention_vec').get_output_at(0)
#    fw_avg_output = att_avg_layer(fw_output)
    
    bin_len = model.input[0].shape[1].value
    att_len = fw_output.shape[1].value
    
    att_model_fw = get_model.Model(inputs=model.input[0:2],outputs=fw_output)
#    att_model_rc = get_model.Model(inputs=model.input[0:2],outputs=model.get_layer('attention_score').get_output_at(1))
    
#    pred_region = pd.DataFrame(columns=['chr_id', 'start_id', 'end_id'])
#    for i in range(1):
#         pred_region.loc[i] = [chr_id,start_id,end_id]
    
    print(predict_region2.shape[0]//10000 +1)
    outF = open(out_fa_file, "a")
    pred_fw_all = None
    seq_idx = 1
    for s in range(predict_region2.shape[0]//10000 +1 ):
        print(s)
        start_idx_s = s*10000
        end_idx_s = min(start_idx_s+10000,predict_region2.shape[0])
                
        predict_region3 = predict_region2.iloc[start_idx_s:end_idx_s]
        datagen_pred = make_predict_input(genome,bigwig_file_unique35,DNase_file,
                                          predict_region3,flanking,unique35)
        
        dnase_val = datagen_pred[0][:,:,-1]
        dnase_ratio = np.max(dnase_val,axis=1)/np.mean(dnase_val,axis=1)
        
        datagen_pred1 =[datagen_pred[0][dnase_ratio<dnase_ratio_cutoff,:,:],datagen_pred[1][dnase_ratio<dnase_ratio_cutoff,:,:]]
        dnase_val1 = dnase_val[dnase_ratio<dnase_ratio_cutoff,:]
        predict_region4 = predict_region3.loc[dnase_ratio<dnase_ratio_cutoff]
    
        pred_fw = att_model_fw.predict(datagen_pred1)
        pred_fw1 = np.zeros((pred_fw.shape[0],pred_fw.shape[1]))
        for r in range(pred_fw1.shape[0]):
            ind = np.unravel_index(np.argmax(pred_fw[r,:,:], axis=None), pred_fw[r,:,:].shape)
            pred_fw1[r,:] = pred_fw[r,:,ind[1]]
        
                
        att_fw_max_idx = np.argmax(pred_fw1,axis = 1)
        
        att_max_center_pos = np.round((att_fw_max_idx+1)/att_len*bin_len)
        
        for i in range(predict_region4.shape[0]):
            seq_chr = predict_region4['chr'].iloc[i]
            seq_start = predict_region4['start'].iloc[i]
            seq_att_center = seq_start-400+int(att_max_center_pos[i])
            chr_str = genome[seq_chr][(seq_att_center-10):(seq_att_center+10)].seq.upper()
            outF.write('>'+str(seq_idx)+'_'+str(seq_chr)+'_'+str(seq_start)+'_'+str(seq_att_center)+'\n')
            outF.write(chr_str+'\n')
            seq_idx += 1
        if pred_fw_all is None:
            pred_fw_all = pred_fw1
            dnase_fw_all = dnase_val1
        else:
            pred_fw_all = np.concatenate((pred_fw_all,pred_fw1))
            dnase_fw_all = np.concatenate((dnase_fw_all,dnase_val1))
    outF.close()
    
#    att2d_fw = get_model.Model(inputs=model.input[0:2],outputs=model.get_layer('attention_score').get_output_at(0))
#    pred2d_fw = att2d_fw.predict(datagen_pred1)
#    ax = sns.heatmap(pred2d_fw[1500], cmap="YlGnBu")
    
    np.savetxt(att_avg_out,pred_fw_all)
    np.savetxt(dnase_out,dnase_fw_all, fmt='%1.2f')
#    print(pred_fw_all.shape)
#    pred_rc = att_model_rc.predict(datagen_pred1)
##    pred = model.predict(datagen_pred1)
#    
#    dnase_i = dnase_val1[0,:]
#    att_i = pred_fw[0,:,:]
#    np.fill_diagonal(att_i, 'nan')
#    
#    ax = sns.heatmap(att_i, cmap="YlGnBu")
#    plt.plot(dnase_i)
#    
#    
#    datagen_pred2 =[datagen_pred[0][dnase_ratio>63,:,:],datagen_pred[1][dnase_ratio>63,:,:]]
#    dnase_val2 = dnase_val[dnase_ratio>63,:]
#    
#
#    pred_fw2 = att_model_fw.predict(datagen_pred2)
#    pred_rc2 = att_model_rc.predict(datagen_pred2)
#    pred = model.predict(datagen_pred2)
#    
#    dnase_i = dnase_val2[100,:]
#    ax = sns.heatmap(pred_fw2[100,:,:], cmap="YlGnBu")
#    plt.plot(dnase_i)
    
    
if __name__ == '__main__':
    main()
