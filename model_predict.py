import sys
import re
import numpy as np
import pandas as pd
from keras.models import load_model
import os
import utils
import get_model

model_file = sys.argv[1]
cell_name = sys.argv[2]
output_predict_path = sys.argv[3]

#model_file = '/storage/htc/bdm/ccm3x/deepGRN/results/evaluate_factornet/models/GENCODE_Unique35_DGF_2.REST.1.401.unique35True.RNAseqFalse.GencodeTrue.h5'
#cell_name = 'liver'
#output_predict_path = '/storage/htc/bdm/ccm3x/deepGRN/results/evaluate_factornet/predictions/'

model_name = re.sub(r'.+/(.+).h5','\\1',model_file)
model_name1 = model_name.split('.')
tf_name = model_name1[1]
bin_num = int(model_name1[2])
flanking = int(model_name1[3])
unique35= model_name1[4]=='unique35True'
rnaseq= model_name1[5]=='RNAseqTrue'
gencode= model_name1[6]=='GencodeTrue'

print(tf_name,model_name,bin_num,cell_name,unique35,rnaseq,gencode)

data_dir = '/storage/htc/bdm/ccm3x/deepGRN/'
genome_fasta_file = data_dir+'raw/hg19.genome.fa'
DNase_path =data_dir+ 'raw/DNase/'
bigwig_file_unique35 = data_dir + 'raw/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
rnaseq_data_file = data_dir + 'raw/rnaseq_data.csv'
gencode_file = data_dir + 'raw/gencode_feature_test.tsv'


predict_region_file = data_dir + 'raw/label/predict_region.bed'

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
    

model = load_model(model_file,custom_objects={'Attention1D': get_model.Attention1D})

datagen_pred = utils.PredictionGeneratorSingle(genome,bw_dict_unique35,DNase_file,predict_region,rnaseq_pred,gencode_pred,unique35,rnaseq,gencode,flanking,batch_size)
pred = model.predict_generator(datagen_pred,verbose=1)
pred_final = pred.flatten()

pred_out = pd.concat([predict_region,pd.DataFrame(pred_final)],axis=1)
#np.savetxt(, pred_final, delimiter=",")
pred_out.to_csv(output_predict_path+'/F.'+tf_name+'.'+cell_name+'.tab.gz',sep='\t',header=False,index=False,compression='gzip')



