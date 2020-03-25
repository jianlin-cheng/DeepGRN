import re
import argparse
import numpy as np
import pandas as pd
from keras.models import load_model
from keras_pos_embd import PositionEmbedding, TrigPosEmbedding
import tensorflow as tf

import utils
import get_model

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Predict a model.",formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--data_dir', '-i', type=str, required=True,help='path to the input data')
    parser.add_argument('--model_file', '-m', type=str, required=True,help='path to model file')
    parser.add_argument('--cell_name', '-c', type=str, required=True,help='cell name')
    parser.add_argument('--predict_region_file', '-p', type=str, required=True,help='predict region file')
    
    parser.add_argument('--bigwig_file_unique35', '-bf', type=str, required=False,help='35bp uniqueness file',default='')
    parser.add_argument('--rnaseq_data_file', '-rf', type=str, required=False,help='RNA-Seq PCA data file',default='')
    parser.add_argument('--gencode_file', '-gc', type=str, required=False,help='Genomic annotation file',default='')

    parser.add_argument('--output_predict_path', '-o', type=str, required=True,help='output path of prediction')
    parser.add_argument('--batch_size', '-b', type=int, required=False,help='batch size',default=512)
    parser.add_argument('--blacklist_file', '-l', type=str,required=False,
                        help='blacklist_file to use, no fitering if not provided',default='')
    return parser

def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    
    bigwig_file_unique35 = args.bigwig_file_unique35
    rnaseq_data_file = args.rnaseq_data_file
    gencode_file = args.gencode_file
    
    data_dir = args.data_dir
    model_file = args.model_file
    cell_name = args.cell_name
    predict_region_file = args.predict_region_file
    output_predict_path = args.output_predict_path
    batch_size = args.batch_size
    blacklist_file = args.blacklist_file

    model = load_model(model_file,custom_objects={'Attention1D': get_model.Attention1D,
                                                  'PositionEmbedding':PositionEmbedding,
                                                  'TrigPosEmbedding':TrigPosEmbedding,
                                                  'tf':tf})
    model_inputs = model.get_input_shape_at(0)
    
    predict_region = pd.read_csv(predict_region_file, sep='\t', header=None)
    pred_chr = set(predict_region[0])
    predict_region.index = range(predict_region.shape[0])
    
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
        
    window_size = predict_region[2][0] - predict_region[1][0]
    flanking = (model_inputs[0][1] - window_size) //2
    

    genome_fasta_file = data_dir+'/hg19.genome.fa'
    DNase_path =data_dir+ '/DNase/'
    # bigwig_file_unique35 = data_dir + '/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
    # rnaseq_data_file = data_dir + '/rnaseq_data.csv'
    # gencode_file = data_dir + '/gencode_feature_test.tsv'
        
    DNase_file = DNase_path+cell_name+'.1x.bw'
    genome = utils.import_genome(genome_fasta_file,pred_chr)
    
    rnaseq_pred = gencode_pred = 0
    if rnaseq:
        rnaseq_data = pd.read_csv(rnaseq_data_file)
        pred_cell_list = [cell_name]
        rnaseq_pred = rnaseq_data[pred_cell_list].values
    
    if gencode:
        gencode_pred = pd.read_csv(gencode_file, sep='\t', header=None,dtype=utils.np.bool_).values  
    
    
    datagen_pred = utils.PredictionGeneratorSingle(genome,bigwig_file_unique35,DNase_file,predict_region,rnaseq_pred,gencode_pred,unique35,rnaseq,gencode,flanking,batch_size)
    pred = model.predict_generator(datagen_pred,verbose=1)
    pred_final = pred.flatten()
    if blacklist_file != '':
        blacklist_bool = np.loadtxt(blacklist_file)==0
        pred_final[blacklist_bool] = 0
        
    pred_out = pd.concat([predict_region,pd.DataFrame(pred_final)],axis=1)
    pred_out.to_csv(output_predict_path,sep='\t',header=False,index=False,compression='gzip')


if __name__ == '__main__':
    main()

#For testing purposes
# python train.py -i /home/chen/data/deepGRN/raw/ -t CTCF -ap resnetv1_lstm -o /home/chen/data/deepGRN/results/res_ctcf/ --plot_model --use_cudnn --unique35 -k 20 -nr 32 -dl 0
# python predict.py -i /home/ccm3x/data/deepGRN/raw/ -m /home/ccm3x/data/deepGRN/results/res_ctcf/best_model.h5 -c PC-3 -p /home/chen/data/deepGRN/raw/label/predict_region.bed -o /home/chen/data/deepGRN/results/res_ctcf/F.CTCF.PC-3.tab.gz -l /home/ccm3x/data/deepGRN/raw/blacklist.bed.gz
# python score.py /home/chen/data/deepGRN/results/res_ctcf/F.CTCF.PC-3.tab.gz /home/ccm3x/data/deepGRN/raw/label/final/ /home/chen/data/deepGRN/raw/label/predict_region.bed -o /home/chen/data/deepGRN/results/res_ctcf/F.CTCF.PC-3.txt
