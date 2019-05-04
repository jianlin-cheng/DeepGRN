import pyBigWig
import numpy as np
import pandas as pd
from keras.models import load_model,Model

import os
import utils
import get_model

tf_name = 'JUND'
model_file = '/home/chen/Dropbox/MU/workspace/encode_dream/data/best_models/attention_after_lstm.JUND.1.401.unique35True.RNAseqTrue.GencodeTrue.h5'

label_file = '/home/chen/data/deepGRN/raw/label/train/'+tf_name+'.train.labels.tsv.gz'
flanking = 401
chr_id = 'chr1'

label_region = pd.read_csv(label_file, sep='\t', header=0)
label_region = label_region[label_region['chr']=='chr10']
label_region = label_region[label_region['HepG2']!='U']

data_dir = '/home/chen/data/deepGRN/raw'
genome_fasta_file = data_dir+'/hg19.genome.fa'
genome = utils.import_genome(genome_fasta_file,[chr_id])
model = load_model(model_file,custom_objects={'Attention1D': get_model.Attention1D})

start1 = 179450
out_dir = '/home/chen/Dropbox/MU/workspace/encode_dream/data/analysis4_region4/'

os.mkdir(out_dir)
#########################################################
for cell_name in label_region.columns.values[3:]:
    print(cell_name)

    bigwig_file = '/mnt/data/chen/encode_dream_data/data/ChIPseq.'+cell_name+'.'+tf_name+'.fc.signal.train.bw'
    DNase_file = '/home/chen/data/deepGRN/raw/DNase/'+cell_name+'.1x.bw'
    
    bigwig_file_unique35 = data_dir + '/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
    rnaseq_data_file = data_dir + '/rnaseq_data.csv'
    gencode_file = data_dir + '/gencode_feature_test.tsv'
    DNase_path =data_dir+ '/DNase/'
    
    for i in [-300, -200,-100,0,100,200]:
        start = start1+i
        stop = start+200
        bigwig = pyBigWig.open(bigwig_file)
        bw_val = np.array(bigwig.values(chr_id, start-flanking, stop+flanking))
        bigwig.close()
        bw_val[np.isnan(bw_val)] = 0
        bw_val = np.reshape(bw_val,(len(bw_val),1))
        
        x_fw = utils.make_x(genome,bigwig_file_unique35,DNase_file,start-flanking,stop+flanking,flanking,chr_id)
        x_fw1 = np.array([x_fw])
        x_rc1 = utils.make_rc_x(x_fw1)
        
        xy_fw = np.concatenate((x_fw1[0],bw_val),axis=1)
        xy_rc = np.concatenate((x_rc1[0],bw_val),axis=1)
        
        
        intermediate_layer_model1 = Model(inputs=model.input,outputs=model.get_layer('dense_2').get_output_at(0))
        intermediate_layer_model2 = Model(inputs=model.input,outputs=model.get_layer('dense_2').get_output_at(1))
        
        o1 = intermediate_layer_model1.predict([x_fw1,x_rc1,np.zeros((1,14))])
        o2 = intermediate_layer_model2.predict([x_fw1,x_rc1,np.zeros((1,14))])
        
        out_dir_name = out_dir+tf_name+'.'+cell_name+'.'+chr_id+'.'+str(start)+'.'+str(stop)+'/'
        os.mkdir(out_dir_name)
        np.savetxt(out_dir_name+'prob_fw.csv',o1[0], fmt='%.8e', delimiter=',')
        np.savetxt(out_dir_name+'prob_rc.csv',o2[0], fmt='%.8e', delimiter=',')
        
        np.savetxt(out_dir_name+'xy_fw.csv',xy_fw, fmt='%.8e', delimiter=',')
        np.savetxt(out_dir_name+'xy_rc.csv',xy_rc, fmt='%.8e', delimiter=',')

os.system("Rscript /home/chen/Dropbox/MU/workspace/encode_dream/src/viz/weight_interpret_viz_singlevec.R "+out_dir)