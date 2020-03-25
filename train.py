import argparse
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.models import load_model
from pathlib import Path
import os
import get_model, utils

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Train a model.",formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--data_dir', '-i', type=str, required=True,help='path to the input data')
    parser.add_argument('--tf_name', '-t', type=str, required=True,help='name of the transcription factor')
    parser.add_argument('--output_dir', '-o', type=str, required=True,help='output path')
    parser.add_argument('--genome_fasta_file', '-gf', type=str, required=True,help='genome fasta file')
    parser.add_argument('--val_chr', '-v', type=str, required=False,help='name for validation chromosome',default='chr11')
    
    parser.add_argument('--bigwig_file_unique35', '-r', type=str, required=False,help='35bp uniqueness file, will not use this feature if left empty',default='')
    parser.add_argument('--rnaseq_data_file', '-r', type=str, required=False,help='RNA-Seq PCA data file, will not use this feature if left empty',default='')
    parser.add_argument('--gencode_file', '-g', type=str, required=False,help='Genomic annotation file, will not use this feature if left empty',default='')

    parser.add_argument('--attention_position', '-ap', type=str, required=False,\
                        help='Position of attention layers, can be attention_after_lstm, attention_before_lstm,attention_after_lstm,attention1d_after_lstm',default='attention_after_lstm')
    parser.add_argument('--flanking', '-f', type=int, required=False,help='flanking length',default=401)
    
    parser.add_argument('--epochs', '-e', type=int, required=False,help='epochs',default=60)
    parser.add_argument('--patience', '-p', type=int, required=False,help='patience',default=5)
    parser.add_argument('--batch_size', '-s', type=int, required=False,help='batch size',default=64)
    parser.add_argument('--learningrate', '-l', type=float, required=False,help='learningrate',default=0.001)
    parser.add_argument('--kernel_size', '-k', type=int, required=False,help='kernel size for Conv1D',default=34)
    parser.add_argument('--num_filters', '-nf', type=int, required=False,help='number of filters for Conv1D',default=64)
    parser.add_argument('--num_recurrent', '-nr', type=int, required=False,help='Output dim for LSTM',default=32)
    parser.add_argument('--num_dense', '-nd', type=int, required=False,help='Output dim for dense layers',default=64)
    parser.add_argument('--dropout_rate', '-d', type=float, required=False,help='dropout_rate for all layers except LSTM',default=0.1)
    parser.add_argument('--rnn_dropout1', '-rd1', type=float, required=False,help='dropout rate for LSTM',default=0.5)
    parser.add_argument('--rnn_dropout2', '-rd2', type=float, required=False,help='rnn dropout rate for LSTM',default=0.1)

    parser.add_argument('--merge', '-me', type=str, required=False,help='merge method, max or ave',default='ave')
    parser.add_argument('--num_conv', '-nc', type=int, required=False,help='Number of Conv1D layers',default=1)
    parser.add_argument('--num_lstm', '-nl', type=int, required=False,help='Number of LSTM layers',default=1)
    parser.add_argument('--num_denselayer', '-dl', type=int, required=False,help='Number of additional dense layers',default=1)
    parser.add_argument('--ratio_negative', '-rn', type=int, required=False,help='Ratio of negative samples to positive samples in each epoch',default=1)
    
    # parser.add_argument('--rnaseq', '-r', action='store_true',help='Use gene expression profile as an additional feature')
    # parser.add_argument('--gencode', '-g', action='store_true',help='Use genomic annotations as an additional feature')
    # parser.add_argument('--unique35', '-u', action='store_true',help='Use sequence uniqueness as an additional feature')
    
    parser.add_argument('--use_peak', '-a', action='store_true',help='should the positive bins sampled from peak regions?')
    parser.add_argument('--use_cudnn', '-c', action='store_true',help='use cudnnLSTM instead of LSTM, faster but will disable LSTM dropouts')
    parser.add_argument('--single_attention_vector', '-sa', action='store_true',help='merge attention weights in each position by averaging')

    parser.add_argument('--positive_weight', '-pw', type=float, required=False,help='weight for positive samples',default=1.0)
    parser.add_argument('--plot_model', '-pl', action='store_true',help='if the model architecture should be plotted')
    parser.add_argument('--random_seed', '-rs', type=int, required=False,help='random seed',default=0)
    parser.add_argument('--val_negative_ratio', '-vn', type=int, required=False,help='ratio for negative samples in validation',default=19)

    return parser

def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    
    data_dir = args.data_dir
    tf_name = args.tf_name
    genome_fasta_file = args.genome_fasta_file
    
    bigwig_file_unique35 = args.bigwig_file_unique35
    rnaseq_data_file = args.rnaseq_data_file
    gencode_file = args.gencode_file

    attention_position = args.attention_position
    flanking = args.flanking
    output_dir = args.output_dir
    val_chr = [args.val_chr]
    
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    learningrate = args.learningrate
    kernel_size = args.kernel_size
    num_filters = args.num_filters
    num_recurrent = args.num_recurrent
    num_dense = args.num_dense
    dropout_rate = args.dropout_rate
    rnn_dropout1 = args.rnn_dropout1
    rnn_dropout2 = args.rnn_dropout2
    merge = args.merge
    
    num_conv = args.num_conv
    num_lstm = args.num_lstm
    num_denselayer = args.num_denselayer
    ratio_negative = args.ratio_negative
    
    rnaseq = rnaseq_data_file!=''
    gencode = gencode_file!=''
    unique35 = bigwig_file_unique35!=''
    positive_weight = args.positive_weight    
    
    use_peak = args.use_peak
    use_cudnn = args.use_cudnn
    single_attention_vector = args.single_attention_vector
    double_strand = True
    val_negative_ratio = args.val_negative_ratio
    plot_model = args.plot_model
    if args.random_seed != 0:
        np.random.seed(args.random_seed)
        
    DNase_path =data_dir+ '/DNase'
    train_label_path = data_dir + '/label/train/'
    label_data_file = train_label_path+tf_name+'.train.labels.tsv.gz'
    
    train_peaks_path = data_dir + '/label/train_positive/'
    positive_peak_file = train_peaks_path+tf_name+'.train.peak.tsv'
    
    output_model_best = output_dir + '/best_model.h5'
    output_model = output_dir + attention_position + '.' + tf_name + '.{epoch:02d}.{val_acc:03f}.h5'
    output_history = output_dir + attention_position + '.' + tf_name + '.'+'1'+ '.'+str(flanking)+'.unique35'+str(unique35)+ '.RNAseq'+str(rnaseq)+ '.Gencode'+str(gencode)+'.csv'
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('loading labels')
    all_chr,cell_list,train_region,val_region,train_label_data,val_label_data,train_label_unbind,val_label_unbind,train_idx,val_idx = utils.import_label(label_data_file)
    genome = utils.import_genome(genome_fasta_file,all_chr)
    window_size = train_region.stop[0]-train_region.start[0]
    L = window_size+flanking*2
    num_meta = 0
    if rnaseq:
        num_meta = num_meta + 8
    if gencode:
        num_meta = num_meta + 6
        
    module_outfile = Path(output_model_best)
    if module_outfile.is_file():
        model = load_model(output_model_best,custom_objects={'Attention1D': get_model.Attention1D})
    else:
        n_channel=6
        if not unique35:
            n_channel = 5
        model = get_model.make_model(attention_position=attention_position,L=L,n_channel=n_channel, num_conv = num_conv,
                             num_denselayer=num_denselayer,num_lstm=num_lstm,kernel_size=kernel_size,num_filters=num_filters, 
                             num_recurrent=num_recurrent,num_dense=num_dense,dropout_rate = dropout_rate,
                             rnn_dropout=[rnn_dropout1,rnn_dropout2],num_meta=num_meta,merge=merge,cudnn=use_cudnn,
                             single_attention_vector=single_attention_vector)
    model.compile(Adam(lr=learningrate),loss='binary_crossentropy',metrics=['accuracy'])
    
    rnaseq_train = rnaseq_val = 0
    if rnaseq:
        rnaseq_data = pd.read_csv(rnaseq_data_file)
        rnaseq_train = rnaseq_data[cell_list].values
        rnaseq_val = rnaseq_data[cell_list].values
    else:
        rnaseq_train = rnaseq_val = 0
    if gencode:
        print('loading Gencode features')
        
        gencode_train,gencode_val = utils.import_gencode(gencode_file,train_idx,val_idx)
    else:
        gencode_train = gencode_val = 0
    
    #If model already exist, continue training


    if plot_model:
        from keras.utils import plot_model # require graphviz
        plot_model(model,show_shapes=True,show_layer_names=False,to_file=output_dir+'model.png')
    
    if use_peak:
        label_bind = pd.read_csv(positive_peak_file,sep='\t')
        train_label_bind = label_bind[label_bind.chrom.isin(set(all_chr)-set(val_chr))]
        val_label_bind = label_bind[label_bind.chrom.isin(val_chr)]
        datagen_train = utils.TrainGeneratorSingle_augmented(genome,bigwig_file_unique35,DNase_path,train_region,train_label_bind,
                                                       train_label_unbind,cell_list,rnaseq_train,gencode_train,True,unique35,
                                                       rnaseq,gencode,flanking,batch_size,ratio_negative,double_strand)
        datagen_val = utils.TrainGeneratorSingle_augmented(genome,bigwig_file_unique35,DNase_path,val_region,val_label_bind,
                                                     val_label_unbind,cell_list,rnaseq_val,gencode_val,False,unique35,
                                                     rnaseq,gencode,flanking,batch_size,val_negative_ratio,double_strand)
    else:
        
        datagen_train = utils.TrainGeneratorSingle(genome=genome,bw_dict_unique35=bigwig_file_unique35,DNase_path=DNase_path,
                                                  label_region=train_region,label_data=train_label_data,
                                                  label_data_unbind=train_label_unbind,cell_list=cell_list,rnaseq_data = rnaseq_train,
                                                  gencode_data=gencode_train,unique35=unique35,rnaseq=rnaseq,gencode=gencode,
                                                  flanking=flanking,batch_size=batch_size,ratio_negative=ratio_negative,double_strand = double_strand)
        datagen_val = utils.TrainGeneratorSingle(genome,bigwig_file_unique35,DNase_path,val_region,val_label_data,val_label_unbind,
                                                cell_list,rnaseq_val,gencode_val,unique35,rnaseq,gencode,flanking,batch_size,
                                                val_negative_ratio,double_strand)
    
    checkpointer1 = ModelCheckpoint(filepath=output_model,verbose=1, save_best_only=False, monitor='val_acc')
    checkpointer2 = ModelCheckpoint(filepath=output_model_best,verbose=1, save_best_only=True, monitor='val_acc')
    earlystopper = EarlyStopping(monitor='val_acc', patience=patience, verbose=1)
    csv_logger = CSVLogger(output_history,append=True)
    

    class_weight = {0: 1.0,1: positive_weight}   
    model.fit_generator(datagen_train,epochs=epochs,validation_data=datagen_val,verbose=1,class_weight=class_weight,
                        callbacks=[checkpointer1,checkpointer2, earlystopper, csv_logger])
    
if __name__ == '__main__':
    main()

# python train.py -i /home/chen/data/deepGRN/raw/ -t CTCF -ap resnetv1_lstm -o /home/chen/data/deepGRN/results/res_ctcf/ --plot_model --use_cudnn --unique35 -k 20 -nr 32 -dl 0 --use_peak
# python predict.py -i /home/chen/data/deepGRN/raw/ -m /home/chen/data/deepGRN/results/res_ctcf/best_model.h5 -c PC-3 -p /home/chen/data/deepGRN/raw/label/predict_region.bed -o /home/chen/data/deepGRN/results/res_ctcf/F.CTCF.PC-3.tab.gz -l /home/chen/data/deepGRN/raw/blacklist.bed.gz
# python score.py /home/chen/data/deepGRN/results/res_ctcf/F.CTCF.PC-3.tab.gz /home/chen/data/deepGRN/raw/label/final/ /home/chen/data/deepGRN/raw/label/predict_region.bed -o /home/chen/data/deepGRN/results/res_ctcf/F.CTCF.PC-3.txt
    
    
