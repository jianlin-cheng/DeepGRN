import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import pylab
import matplotlib
import pandas

import utils
import pickle

# Standard library imports
import sys
import errno
from keras.models import load_model,model_from_json


input_dir = '/storage/htc/bdm/ccm3x/FactorNet/data/PC-3'
model_dir = '/storage/htc/bdm/ccm3x/FactorNet/models/CTCF/metaGENCODE_RNAseq_Unique35_DGF'
tf = 'CTCF'
bed_file = '/storage/htc/bdm/ccm3x/FactorNet/pred_regions/seg00predict_region.bed'
output_file = '/storage/htc/bdm/ccm3x/FactorNet/test.tab.gz'


print 'Loading genome'
genome = utils.load_genome()
print 'Loading model'
model_tfs, model_bigwig_names, features, model = utils.load_model(model_dir)
L = model.input_shape[0][1]
utils.L = L
assert tf in model_tfs
assert 'bigwig' in features
use_meta = 'meta' in features
use_gencode = 'gencode' in features
use_unique35 = 'Unique35' in model_bigwig_names

print 'Loading test data'
is_sorted = True
bigwig_names, meta_names, datagen_bed, nonblacklist_bools = utils.load_beddata(genome, bed_file,use_unique35, use_meta, use_gencode, input_dir, is_sorted)

test_idx = 10891

datagen_bed1 = datagen_bed
datagen_bed1.data_list = datagen_bed.data_list[(test_idx-9):(test_idx-9+2000)]
x = datagen_bed1.next()

model_file = '/storage/htc/bdm/ccm3x/deepGRN/results/evaluate_factornet/models/metaGENCODE_RNAseq_Unique35_DGF.CTCF.1.401.unique35True.RNAseqTrue.GencodeTrue.h5'
model_json_file = '/storage/htc/bdm/ccm3x/deepGRN/results/evaluate_factornet/models/metaGENCODE_RNAseq_Unique35_DGF.CTCF.1.401.unique35True.RNAseqTrue.GencodeTrue.json'
model_json_fp = open(model_json_file, 'r')
model_json = model_json_fp.read()
model = model_from_json(model_json)
model.load_weights(model_file)


x1=x[0][[0],:,:]
x2=x[1][[0],:,:]
x3=x[2][[0],:]

from keras.models import Model
intermediate_model = Model(input=[model.layers[0].input,model.layers[1].input,model.layers[10].input],
                              output=model.layers[2].get_output_at(0))
y1 = intermediate_model.predict([x1,x2,x3])
y1[0,range(230),4]
        
assert bigwig_names == model_bigwig_names
if use_meta:
    model_meta_file = model_dir + '/meta.txt'
    assert os.path.isfile(model_meta_file)
    model_meta_names = np.loadtxt(model_meta_file, dtype=str)
    if len(model_meta_names.shape) == 0:
        model_meta_names = [str(model_meta_names)]
    else:
        model_meta_names = list(model_meta_names)
    assert meta_names == model_meta_names
print 'Generating predictions'
model_tf_index = model_tfs.index(tf)
model_predicts = model.predict_generator(datagen_bed, val_samples=len(datagen_bed), pickle_safe=True)
if len(model_tfs) > 1:
    model_tf_predicts = model_predicts[:, model_tf_index]
else:
    model_tf_predicts = model_predicts
final_scores = np.zeros(len(nonblacklist_bools))
final_scores[nonblacklist_bools] = model_tf_predicts.reshape(len(model_tf_predicts),)
print 'Saving predictions'
df = pandas.read_csv(bed_file, sep='\t', header=None)
df[3] = final_scores
df.to_csv(output_file, sep='\t', compression='gzip', float_format='%.3e', header=False, index=False)
    
