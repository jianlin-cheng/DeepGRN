import re
import argparse
import pandas as pd
from keras.models import load_model

import utils
import get_model

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Predict a model.",formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--data_dir', '-i', type=str, required=True,help='data_dir')
    parser.add_argument('--model_file', '-m', type=str, required=True,help='model_file')
    parser.add_argument('--cell_name', '-c', type=str, required=True,help='cell_name')
    parser.add_argument('--predict_region_file', '-p', type=str, required=True,help='predict_region_file')

    parser.add_argument('--output_predict_path', '-o', type=str, required=True,help='output_predict_path')
    parser.add_argument('--batch_size', '-b', type=int, required=False,help='batch_size',default=32)
    return parser

def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    
    data_dir = args.data_dir
    model_file = args.model_file
    cell_name = args.cell_name
    predict_region_file = args.predict_region_file
    output_predict_path = args.output_predict_path
    batch_size = args.batch_size

    model_name = re.sub(r'.+/(.+).h5','\\1',model_file)
    model_name1 = model_name.split('.')
    tf_name = model_name1[1]
    flanking = int(model_name1[3])
    unique35= model_name1[4]=='unique35True'
    rnaseq= model_name1[5]=='RNAseqTrue'
    gencode= model_name1[6]=='GencodeTrue'
    
    print(tf_name,model_name,cell_name,unique35,rnaseq,gencode)
    
    genome_fasta_file = data_dir+'/hg19.genome.fa'
    DNase_path =data_dir+ '/DNase/'
    bigwig_file_unique35 = data_dir + '/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
    rnaseq_data_file = data_dir + '/rnaseq_data.csv'
    gencode_file = data_dir + '/gencode_feature_test.tsv'
        
    predict_region = pd.read_csv(predict_region_file, sep='\t', header=None)
    pred_chr = set(predict_region[0])
    predict_region.index = range(predict_region.shape[0])
    
    
    DNase_file = DNase_path+cell_name+'.1x.bw'
    genome = utils.import_genome(genome_fasta_file,pred_chr)
    
    rnaseq_pred = gencode_pred = 0
    if rnaseq:
        rnaseq_data = pd.read_csv(rnaseq_data_file)
        pred_cell_list = [cell_name]
        rnaseq_pred = rnaseq_data[pred_cell_list].values
    
    if gencode:
        gencode_pred = pd.read_csv(gencode_file, sep='\t', header=None,dtype=utils.np.bool_).values  
    
    model = load_model(model_file,custom_objects={'Attention1D': get_model.Attention1D})
    
    datagen_pred = utils.PredictionGeneratorSingle(genome,bigwig_file_unique35,DNase_file,predict_region,rnaseq_pred,gencode_pred,unique35,rnaseq,gencode,flanking,batch_size)
    pred = model.predict_generator(datagen_pred,verbose=1)
    pred_final = pred.flatten()
    
    pred_out = pd.concat([predict_region,pd.DataFrame(pred_final)],axis=1)
    #np.savetxt(, pred_final, delimiter=",")
    pred_out.to_csv(output_predict_path+'/F.'+tf_name+'.'+cell_name+'.tab.gz',sep='\t',header=False,index=False,compression='gzip')


if __name__ == '__main__':
    main()

#For testing purposes
#data_dir = '/home/chen/data/deepGRN/raw'
#model_file = '/home/chen/data/deepGRN/test/CTCF/factornet_attention.set1/factornet_attention.CTCF.1.401.unique35True.RNAseqTrue.GencodeTrue.h5'
#batch_size = 32
#cell_name = 'PC-3'
#predict_region_file = '/home/chen/data/deepGRN/raw/label/predict_region.bed'
#output_predict_path = '/home/chen/data/deepGRN/test/CTCF/factornet_attention.set1/'