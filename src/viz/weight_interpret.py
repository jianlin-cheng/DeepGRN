import pyBigWig
import numpy as np
from keras.models import load_model,Model

import os
import utils
import viz_utils
import argparse

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Analysis of attention weights.",formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--data_dir', '-i', type=str, required=True,help='data_dir')
    parser.add_argument('--tf_name', '-t', type=str, required=True,help='tf_name')
    parser.add_argument('--model_file', '-m', type=str, required=True,help='model_file')
    parser.add_argument('--cell_name', '-c', type=str, required=True,help='cell_name')
    parser.add_argument('--fc_file', '-f', type=str, required=True,help='ChIP_fold_change_file')
    parser.add_argument('--chr_id', '-r', type=str, required=True,help='chr_name')
    parser.add_argument('--location', '-l', type=int, required=True,help='location')
    
    parser.add_argument('--flanking', '-a', type=int, required=False,help='flanking',default=401)
    parser.add_argument('--single_vec', '-s', type=bool, required=False,help='if single_vec is used in the model',default=False)
    
    parser.add_argument('--out_dir', '-o', type=str, required=True,help='output_path')

    return parser

def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    
    data_dir = args.data_dir
    tf_name = args.tf_name
    model_file = args.model_file
    cell_name = args.cell_name
    fc_file = args.fc_file
    chr_id = args.chr_id
    location = args.location
    flanking = args.flanking
    single_vec = args.single_vec
    out_dir = args.out_dir
    

    if single_vec:
        aw_layer_name = 'attention_vec'
    else:
        aw_layer_name = 'dense_2'
    
    
    genome_fasta_file = data_dir+'/hg19.genome.fa'
    genome = utils.import_genome(genome_fasta_file,[chr_id])
    model = load_model(model_file,custom_objects={'Attention1D': viz_utils.Attention1D})
    
    os.mkdir(out_dir)
    print(cell_name)
     
    bigwig_file_unique35 = data_dir + '/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
    DNase_file =data_dir+ '/DNase/'+cell_name+'.1x.bw'
    
    start = location
    stop = start+200
    bigwig = pyBigWig.open(fc_file)
    bw_val = np.array(bigwig.values(chr_id, start-flanking, stop+flanking))
    bigwig.close()
    bw_val[np.isnan(bw_val)] = 0
    bw_val = np.reshape(bw_val,(len(bw_val),1))
    
    x_fw = utils.make_x(genome,bigwig_file_unique35,DNase_file,start-flanking,stop+flanking,flanking,chr_id)
    x_fw1 = np.array([x_fw])
    x_rc1 = utils.make_rc_x(x_fw1)
    
    xy_fw = np.concatenate((x_fw1[0],bw_val),axis=1)
    xy_rc = np.concatenate((x_rc1[0],bw_val),axis=1)
    
    
    intermediate_layer_model1 = Model(inputs=model.input,outputs=model.get_layer(aw_layer_name).get_output_at(0))
    intermediate_layer_model2 = Model(inputs=model.input,outputs=model.get_layer(aw_layer_name).get_output_at(1))
    
    o1 = intermediate_layer_model1.predict([x_fw1,x_rc1,np.zeros((1,14))])
    o2 = intermediate_layer_model2.predict([x_fw1,x_rc1,np.zeros((1,14))])
    
    out_dir_name = out_dir+tf_name+'.'+cell_name+'.'+chr_id+'.'+str(start)+'.'+str(stop)+'/'
    os.mkdir(out_dir_name)
    np.savetxt(out_dir_name+'prob_fw.csv',o1[0], fmt='%.8e', delimiter=',')
    np.savetxt(out_dir_name+'prob_rc.csv',o2[0], fmt='%.8e', delimiter=',')
    
    np.savetxt(out_dir_name+'xy_fw.csv',xy_fw, fmt='%.8e', delimiter=',')
    np.savetxt(out_dir_name+'xy_rc.csv',xy_rc, fmt='%.8e', delimiter=',')

    
    os.system("Rscript weight_interpret_viz.R "+out_dir)
    
if __name__ == '__main__':
    main()
