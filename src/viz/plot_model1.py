import argparse
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.models import load_model
from pathlib import Path

import get_model
import utils
from keras.utils import plot_model # require graphviz

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Train a model.",formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--data_dir', '-i', type=str, required=True,help='data_dir')
    parser.add_argument('--tf_name', '-t', type=str, required=True,help='tf_name')
    parser.add_argument('--output_dir', '-o', type=str, required=True,help='output_dir')

    parser.add_argument('--model_type', '-m', type=str, required=False,help='model_type',default='attention_after_lstm')
    parser.add_argument('--flanking', '-f', type=int, required=False,help='flanking',default=401)
    parser.add_argument('--val_chr', '-v', type=str, required=False,help='val_chr',default='chr11')
    parser.add_argument('--epochs', '-e', type=int, required=False,help='epochs',default=60)
    parser.add_argument('--patience', '-p', type=int, required=False,help='patience',default=5)
    parser.add_argument('--batch_size', '-s', type=int, required=False,help='batch_size',default=32)
    parser.add_argument('--learningrate', '-l', type=float, required=False,help='learningrate',default=0.001)
    parser.add_argument('--kernel_size', '-k', type=int, required=False,help='kernel_size',default=34)
    parser.add_argument('--num_filters', '-nf', type=int, required=False,help='num_filters',default=64)
    parser.add_argument('--num_recurrent', '-nr', type=int, required=False,help='num_recurrent',default=64)
    parser.add_argument('--num_dense', '-nd', type=int, required=False,help='num_dense',default=64)
    parser.add_argument('--dropout_rate', '-d', type=float, required=False,help='dropout_rate',default=0.1)
    parser.add_argument('--rnn_dropout', '-rd', type=float, required=False,help='rnn_dropout',default=0.1)
    parser.add_argument('--merge', '-me', type=str, required=False,help='merge method, max or ave',default='ave')
    parser.add_argument('--num_conv', '-nc', type=int, required=False,help='num_conv',default=1)
    parser.add_argument('--num_lstm', '-nl', type=int, required=False,help='num_lstm',default=1)
    parser.add_argument('--num_denselayer', '-dl', type=int, required=False,help='num_denselayer',default=1)
    parser.add_argument('--ratio_negative', '-rn', type=int, required=False,help='ratio_negative',default=1)
    
    parser.add_argument('--rnaseq', '-r', action='store_true',help='rnaseq')
    parser.add_argument('--gencode', '-g', action='store_true',help='gencode')
    parser.add_argument('--unique35', '-u', action='store_true',help='unique35')
    parser.add_argument('--use_peak', '-a', action='store_true',help='use_peak')
    parser.add_argument('--use_cudnn', '-c', action='store_true',help='use_cudnn')
    parser.add_argument('--single_attention_vector', '-sa', action='store_true',help='single_attention_vector')

    return parser

def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    
    tf_name = args.tf_name
    model_type = args.model_type
    flanking = args.flanking
    output_dir = args.output_dir
    
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    learningrate = args.learningrate
    kernel_size = args.kernel_size
    num_filters = args.num_filters
    num_recurrent = args.num_recurrent
    num_dense = args.num_dense
    dropout_rate = args.dropout_rate
    rnn_dropout = args.rnn_dropout
    merge = args.merge
    
    num_conv = args.num_conv
    num_lstm = args.num_lstm
    num_denselayer = args.num_denselayer
    ratio_negative = args.ratio_negative
    
    rnaseq = args.rnaseq
    gencode = args.gencode
    unique35 = args.unique35
    
    use_peak = args.use_peak
    use_cudnn = args.use_cudnn
    single_attention_vector = args.single_attention_vector
    np.random.seed(2018)
    
    print(tf_name,model_type,flanking,rnaseq,gencode,unique35,output_dir)
    print(epochs,patience,batch_size,learningrate,kernel_size,num_filters,num_recurrent,num_dense,dropout_rate,merge)
    print(num_conv,num_lstm,num_denselayer,ratio_negative,use_peak,use_cudnn)
        
    
    
    num_meta = 0
    if rnaseq:
        num_meta = num_meta + 8

    
    if gencode:
        print('loading Gencode features')
        num_meta = num_meta + 6

    #If model already exist, continue training
    n_channel=6
    if not unique35:
        n_channel = 5
    model = get_model.make_model_attention(model_type=model_type,L=200+2*flanking,n_channel=n_channel, num_conv = num_conv,
                                                   num_denselayer=num_denselayer,num_lstm=num_lstm,kernel_size=kernel_size,num_filters=num_filters, 
                                                   num_recurrent=num_recurrent,num_dense=num_dense,dropout_rate = dropout_rate,
                                                   rnn_dropout=rnn_dropout,num_meta=num_meta,merge=merge,cudnn=use_cudnn,single_attention_vector=single_attention_vector)
    plot_model(model,show_shapes=True,show_layer_names=True,to_file='/home/chen/Dropbox/MU/workspace/encode_dream/test/model.png')
        
main()