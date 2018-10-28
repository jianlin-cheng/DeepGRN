import re
import numpy as np
import pandas as pd
import os
os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import model_from_json

import utils
#import get_model

model_file = '/storage/htc/bdm/ccm3x/deepGRN/results/evaluate_factornet/models/metaGENCODE_RNAseq_Unique35_DGF.CTCF.1.401.unique35True.RNAseqTrue.GencodeTrue.h5'
model_json_file = '/storage/htc/bdm/ccm3x/deepGRN/results/evaluate_factornet/models/metaGENCODE_RNAseq_Unique35_DGF.CTCF.1.401.unique35True.RNAseqTrue.GencodeTrue.json'
cell_name = 'PC-3'
output_predict_path = '/storage/htc/bdm/ccm3x/deepGRN/results/evaluate_factornet/predictions/'
data_dir = '/storage/htc/bdm/ccm3x/deepGRN/'


model_name = re.sub(r'.+/(.+).h5','\\1',model_file)
model_name1 = model_name.split('.')
tf_name = model_name1[1]
bin_num = int(model_name1[2])
flanking = int(model_name1[3])
unique35= model_name1[4]=='unique35True'
rnaseq= model_name1[5]=='RNAseqTrue'
gencode= model_name1[6]=='GencodeTrue'

print(tf_name,model_name,bin_num,cell_name,unique35,rnaseq,gencode)

genome_fasta_file = data_dir+'raw/hg19.genome.fa'
DNase_path =data_dir+ 'raw/DNase/'
bigwig_file_unique35 = data_dir + 'raw/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
rnaseq_data_file = data_dir + 'raw/rnaseq_data.csv'
gencode_file = data_dir + 'raw/gencode_feature_val.tsv'


predict_region_file = data_dir + 'raw/label/test_regions.blacklistfiltered.bed'


pred_chr = ['chr1','chr8','chr21']
batch_size = 32

predict_region = pd.read_csv(predict_region_file, sep='\t', header=None)
pred_idx = predict_region[0].isin(pred_chr)
predict_region = predict_region[pred_idx]
predict_region.index = range(predict_region.shape[0])


DNase_file = DNase_path+cell_name+'.1x.bw'

genome = utils.import_genome(genome_fasta_file,pred_chr)
bw_dict_unique35 = bigwig_file_unique35

rnaseq_train = rnaseq_pred = gencode_pred = 0
if rnaseq:
    rnaseq_data = pd.read_csv(rnaseq_data_file)
    pred_cell_list = [cell_name]
    rnaseq_pred = rnaseq_data[pred_cell_list].values

if gencode:
    gencode_pred = pd.read_csv(gencode_file, sep='\t', header=None,dtype=utils.np.bool_)
    gencode_pred = gencode_pred[pred_idx].values
    
#model = load_model(model_file,custom_objects={'Attention1D': get_model.Attention1D})
model_json_fp = open(model_json_file, 'r')
model_json = model_json_fp.read()
model = model_from_json(model_json)
model.load_weights(model_file)

datagen_pred = utils.PredictionGeneratorSingle(genome,bw_dict_unique35,DNase_file,predict_region,rnaseq_pred,gencode_pred,unique35,rnaseq,gencode,flanking,batch_size)
pred = model.predict_generator(datagen_pred,val_samples=300000)
pred_final = pred.flatten()
np.savetxt(output_predict_path+tf_name+'.'+cell_name+'.'+model_name1[0]+ '_batch32_test.unique35'+str(unique35)+ '.RNAseq'+str(rnaseq)+ '.Gencode'+str(gencode)+'.csv', pred_final, delimiter=",")



