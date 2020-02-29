import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K
import pyBigWig

import os
import re
import utils
import get_model
import sys

#tf_name = sys.argv[1]
#cell_name = sys.argv[2]
#type1 = sys.argv[3]

tf_name = 'CTCF'
cell_name = 'PC-3'
type1 = 'tp'

if type1 == 'tp':
    label_file_path = '/home/chen/data/deepGRN/analysis_s/tp_false_in_factornet/'
    out_dir = '/home/chen/data/deepGRN/analysis_revision_TP/'
if type1 == 'tn':
    label_file_path = '/home/chen/data/deepGRN/analysis_s/tn_true_in_factornet/'
    out_dir = '/home/chen/data/deepGRN/analysis_revision_TN/'
if type1 == 'tp_both':
    label_file_path = '/home/chen/data/deepGRN/analysis_s/tp_true_in_both/'
    out_dir = '/home/chen/data/deepGRN/analysis_revision_tp_both/'
if type1 == 'tn_both':
    label_file_path = '/home/chen/data/deepGRN/analysis_s/tn_false_in_both/'
    out_dir = '/home/chen/data/deepGRN/analysis_revision_tn_both/'
model_summary_file = '/home/chen/Dropbox/MU/workspace/encode_dream/performance/score/train_parameters.csv'
model_path = '/home/chen/Dropbox/MU/workspace/encode_dream/data/best_models/'
data_dir = '/home/chen/data/deepGRN/raw'
fc_data_dir = '/mnt/data/chen/encode_dream_data/chipseq_peak_final/'


label_files = [f for f in os.listdir(label_file_path) if re.match(tf_name+r'.+', f)]
chr_all = ['chr8','chr1','chr21']

#label_file = label_files[0]

label_file = [f for f in label_files if re.match(r'.+'+cell_name+r'.+', f)][0]

label_table = pd.read_csv(label_file_path+label_file, header=0)
model_file1 = [f for f in os.listdir(model_path) if re.match(r'.+'+tf_name+r'.+\.h5', f)][0]
model_file = model_path+model_file1
model_type = model_file1.split('.')[0]
model_summary = pd.read_csv(model_summary_file, header=0)
model_tf = model_summary.loc[model_summary['TF']==tf_name]


bigwig_file_unique35 = data_dir + '/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
rnaseq_data_file = data_dir + '/rnaseq_data.csv'
gencode_file = data_dir + '/gencode_feature_test.tsv'
predict_region_file = data_dir + '/label/predict_region.bed'
DNase_path =data_dir+ '/DNase/'
label_file = data_dir+'/label/final/'+tf_name+'.train.labels.tsv.gz'
label_region = pd.read_csv(label_file, sep='\t', header=0)


#cell_names = label_region.columns.values[3:]
#cell_name = cell_names[0]


flanking = 401
bigwig_file = fc_data_dir+'ChIPseq.'+cell_name+'.'+tf_name+'.fc.signal.test.bw'
DNase_file = data_dir+'/DNase/'+cell_name+'.1x.bw'
genome_fasta_file = data_dir+'/hg19.genome.fa'
genome = utils.import_genome(genome_fasta_file,chr_all)
model = load_model(model_file,custom_objects={'Attention1D': get_model.Attention1D})
out_dir2 = out_dir+tf_name+'.'+cell_name+'/'

if not os.path.exists(out_dir2):
    os.mkdir(out_dir2)
    print("Directory " , out_dir2 ,  " Created ")
else:    
    print("Directory " , out_dir2 ,  " already exists")

rnaseq_data = pd.read_csv(rnaseq_data_file)[cell_name].values
predict_region = pd.read_csv(predict_region_file, sep='\t', header=None)
gencode_data = pd.read_csv(gencode_file, sep='\t', header=None,dtype=np.bool_).values
    
seq_input_dim = model.input[0].shape.as_list()[2]
if len(model.input)==3:
    dense_input_dim =  model.input[2].shape.as_list()[1]
else:
    dense_input_dim = 0

inc_flag = 0
fw_meta_idx = 0
rc_meta_idx = 0
model_grad = get_model.load_model_from_config(model_tf,get_model.make_model_fix_fusion)
for i in range(len(model.layers)):
    if i >= len(model_grad.layers):
        break
    
    if not model.layers[i].get_weights() == []:
        if  model_grad.layers[i].get_weights() == []:
            inc_flag = 1
        else:
            print('load weights for layer '+str(i)+' '+model_grad.layers[i+inc_flag].name)
            model_grad.layers[i+inc_flag].set_weights(model.layers[i].get_weights())            
        
    if model_grad.layers[i].name == 'meta_fuse_fw':
        fw_meta_idx = i
    if model_grad.layers[i].name == 'meta_fuse_rc':
        rc_meta_idx = i

grads_func = K.gradients(model_grad.output,model_grad.input)
iterate = K.function(model_grad.input, grads_func)

model_att = get_model.load_model_from_config(model_tf,get_model.make_model_extract_att)
for i in range(len(model_att.layers)):
    if not model_att.layers[i].get_weights() == []:
        print('load weights for layer '+str(i)+' '+model_att.layers[i].name)
        model_att.layers[i].set_weights(model.layers[i].get_weights())

bin_num = label_table.shape[0]
if label_table.shape[0] > 1000:
    # sample 1000 bins from all
    bin_num = 1000
    label_table = label_table.sample(n=1000, random_state=1)
    label_table = label_table.reset_index()
        

input_len = model_att.input_shape[0][1]

saliency_fw = np.zeros((bin_num,input_len))
saliency_rc = np.zeros((bin_num,input_len))

dnase_all = np.zeros((bin_num,input_len))
fc_all = np.zeros((bin_num,input_len))

downsample_fc = None

att_mean_fw = None
att_mean_rc = None

att_best_fw = None
att_best_rc = None

cor_att_fc_fw_all = np.zeros(bin_num)
cor_att_fc_rc_all = np.zeros(bin_num)

cor_slcy_fc_fw_all = np.zeros(bin_num)
cor_slcy_fc_rc_all = np.zeros(bin_num)

slcy_best_fw = np.zeros((bin_num,input_len))
slcy_best_rc = np.zeros((bin_num,input_len))

for i in range(bin_num):
    chr_id = label_table['V1'][i]
    start = label_table['V2'][i]
    print(chr_id+' '+str(start))
    
    stop = start+200
    X_all = utils.make_feature(genome,bigwig_file_unique35,DNase_file,chr_id,start,stop,
                     flanking,seq_input_dim,predict_region,rnaseq_data,gencode_data,
                     cell_name,dense_input_dim)
    
    if fw_meta_idx != 0:
        model_grad.layers[fw_meta_idx].set_weights([X_all[2]])
        model_grad.layers[rc_meta_idx].set_weights([X_all[2]])
    
    dnase_bw = pyBigWig.open(DNase_file)
    dnase_val = np.array(dnase_bw.values(chr_id, start-flanking, stop+flanking))
    dnase_bw.close()
    dnase_val[np.isnan(dnase_val)] = 0
    
    bigwig = pyBigWig.open(bigwig_file)
    bw_val = np.array(bigwig.values(chr_id, start-flanking, stop+flanking))
    bigwig.close()
    bw_val[np.isnan(bw_val)] = 0
    grad = iterate([X_all[0],X_all[1]])
    grad[0] = np.absolute(grad[0])
    grad[1] = np.absolute(grad[1])
    
    att_score = model_att.predict([X_all[0],X_all[1]])
    dnase_score = np.reshape(dnase_val,(len(dnase_val),1))
    bw_val = np.reshape(bw_val,(len(bw_val),1))
    
    saliency_fw[i,:] = np.mean(grad[0],2)[0,:]
    saliency_rc[i,:] = np.mean(grad[1],2)[0,:]
    
    dnase_all[i,:] = dnase_score[:,0]
    fc_all[i,:] = bw_val[:,0]
    output_len = att_score[0].shape[1]
    
    if downsample_fc is None:
        downsample_fc= np.zeros((bin_num,output_len))
        
        att_mean_fw = np.zeros((bin_num,output_len))
        att_mean_rc = np.zeros((bin_num,output_len))
        
        att_best_fw = np.zeros((bin_num,output_len))
        att_best_rc = np.zeros((bin_num,output_len))
    
    att_mean_fw[i,:] = np.mean(att_score[0],2)[0,:]
    att_mean_rc[i,:] = np.mean(att_score[1],2)[0,:]
    
    att_len = att_score[0].shape[1]
    split_arr = np.linspace(0, len(bw_val[:,0]), num=att_len+1, dtype=int)
    dwnsmpl_subarr = np.split(bw_val[:,0], split_arr[1:])
    dwnsmpl_arr = np.array( list( np.mean(item) for item in dwnsmpl_subarr[:-1] ))
    best_cor_fw = -1
    best_cor_rc = -1
    
    best_cor_fw_slcy = -1
    best_cor_rc_slcy = -1
    
    best_att_fw = np.zeros(output_len)
    best_att_rc = np.zeros(output_len)
    
    best_slcy_fw = np.zeros(grad[0].shape[1])
    best_slcy_rc = np.zeros(grad[0].shape[1])
    
    downsample_fc[i,:] = dwnsmpl_arr
    
    for j in range(att_score[0].shape[2]):
        cor_att_fc_fw = np.corrcoef(dwnsmpl_arr,att_score[0][0,:,j])[0,1]
        if cor_att_fc_fw>best_cor_fw:
            best_cor_fw = cor_att_fc_fw
            best_att_fw = att_score[0][0,:,j]
            
        cor_att_fc_rc = np.corrcoef(np.flip(dwnsmpl_arr),att_score[1][0,:,j])[0,1]
        if cor_att_fc_rc>best_cor_rc:
            best_cor_rc = cor_att_fc_rc
            best_att_rc = att_score[1][0,:,j]

    for j in range(grad[0].shape[2]):
        cor_slcy_fc_fw = np.corrcoef(bw_val[:,0],grad[0][0,:,j])[0,1]
        if cor_slcy_fc_fw>best_cor_fw_slcy:
            best_cor_fw_slcy = cor_slcy_fc_fw
            best_slcy_fw = grad[0][0,:,j]
            
        cor_slcy_fc_rc = np.corrcoef(np.flip(bw_val[:,0]),grad[1][0,:,j])[0,1]
        if cor_slcy_fc_rc>best_cor_rc_slcy:
            best_cor_rc_slcy = cor_slcy_fc_rc
            best_slcy_rc = grad[1][0,:,j]

            
    att_best_fw[i,:] = best_att_fw
    att_best_rc[i,:] = best_att_rc
    
    slcy_best_fw[i,:] = best_slcy_fw
    slcy_best_rc[i,:] = best_slcy_rc
    
    cor_att_fc_fw_all[i] = best_cor_fw
    cor_att_fc_rc_all[i] = best_cor_rc
    
    
for i in range(bin_num):
    cor_slcy_fc_fw_all[i] = np.corrcoef(saliency_fw[i,:],fc_all[i,:])[0,1]
    cor_slcy_fc_rc_all[i] = np.corrcoef(saliency_rc[i,:],fc_all[i,:])[0,1]

np.savetxt(out_dir2+'saliency_fw.csv',saliency_fw)
np.savetxt(out_dir2+'saliency_rc.csv',saliency_rc)

np.savetxt(out_dir2+'att_mean_fw.csv',att_mean_fw)
np.savetxt(out_dir2+'att_mean_rc.csv',att_mean_rc)

np.savetxt(out_dir2+'dnase_all.csv',dnase_all)
np.savetxt(out_dir2+'fc_all.csv',fc_all)
np.savetxt(out_dir2+'downsample_fc.csv',downsample_fc)

np.savetxt(out_dir2+'pred_score.csv',label_table['V4'][0:bin_num])

np.savetxt(out_dir2+'att_best_fw.csv',att_best_fw)
np.savetxt(out_dir2+'att_best_rc.csv',att_best_rc)

np.savetxt(out_dir2+'slcy_best_fw.csv',slcy_best_fw)
np.savetxt(out_dir2+'slcy_best_rc.csv',slcy_best_rc)

np.savetxt(out_dir2+'cor_att_fc_fw_all.csv',cor_att_fc_fw_all)
np.savetxt(out_dir2+'cor_att_fc_rc_all.csv',cor_att_fc_rc_all)

np.savetxt(out_dir2+'cor_slcy_fc_fw_all.csv',cor_slcy_fc_fw_all)
np.savetxt(out_dir2+'cor_slcy_fc_rc_all.csv',cor_slcy_fc_rc_all)



