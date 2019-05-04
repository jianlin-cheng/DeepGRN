import argparse
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.models import load_model
from pathlib import Path
import os,re,subprocess
import get_model
import utils

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Train a model.",formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--data_dir', '-i', type=str, required=True,help='data_dir')
    parser.add_argument('--tf_name', '-t', type=str, required=True,help='tf_name')
    parser.add_argument('--output_dir', '-o', type=str, required=True,help='output_dir')
    
    parser.add_argument('--train_script', '-ts', type=str, required=True,help='train_script')
    parser.add_argument('--predict_script', '-ps', type=str, required=True,help='predict_script')

    parser.add_argument('--attention_position', '-ap', type=str, required=False,\
                        help='Position of attention layers, can be attention1d_after_lstm, attention_before_lstm,attention_after_lstm,attention1d_after_lstm',default='attention_after_lstm')
    parser.add_argument('--flanking', '-f', type=int, required=False,help='flanking',default=401)
    parser.add_argument('--epochs', '-e', type=int, required=False,help='epochs',default=10)
    parser.add_argument('--patience', '-p', type=int, required=False,help='patience',default=5)
    parser.add_argument('--batch_size', '-s', type=int, required=False,help='batch_size',default=64)
    parser.add_argument('--learningrate', '-l', type=float, required=False,help='learningrate',default=0.001)
    parser.add_argument('--kernel_size', '-k', type=int, required=False,help='kernel_size for Conv1D',default=34)
    parser.add_argument('--num_filters', '-nf', type=int, required=False,help='num_filters for Conv1D',default=64)
    parser.add_argument('--num_recurrent', '-nr', type=int, required=False,help='Output dim for LSTM',default=64)
    parser.add_argument('--num_dense', '-nd', type=int, required=False,help='Output dim for dense layers',default=64)
    parser.add_argument('--dropout_rate', '-d', type=float, required=False,help='dropout_rate for all layers except LSTM',default=0.1)
    parser.add_argument('--rnn_dropout1', '-rd1', type=float, required=False,help='dropout_rate for LSTM',default=0.5)
    parser.add_argument('--rnn_dropout2', '-rd2', type=float, required=False,help='rnn_dropout_rate for LSTM',default=0.1)

    parser.add_argument('--merge', '-me', type=str, required=False,help='merge method, max or ave',default='ave')
    parser.add_argument('--num_conv', '-nc', type=int, required=False,help='Number of Conv1D layers',default=1)
    parser.add_argument('--num_lstm', '-nl', type=int, required=False,help='Number of LSTM layers',default=1)
    parser.add_argument('--num_denselayer', '-dl', type=int, required=False,help='Number of additional dense layers',default=1)
    parser.add_argument('--ratio_negative', '-rn', type=int, required=False,help='Ratio of negative samples to positive samples in each epoch',default=1)
    
    parser.add_argument('--rnaseq', '-r', action='store_true',help='rnaseq')
    parser.add_argument('--gencode', '-g', action='store_true',help='gencode')
    parser.add_argument('--unique35', '-u', action='store_true',help='unique35')
    parser.add_argument('--use_peak', '-a', action='store_true',help='use_peak')
    parser.add_argument('--use_cudnn', '-c', action='store_true',help='use cudnnLSTM instead of LSTM, faster but will disable LSTM dropouts')
    parser.add_argument('--single_attention_vector', '-sa', action='store_true',help='merge attention weights in each position by averaging')
#    parser.add_argument('--single_strand', '-ss', action='store_true',help='single_strand')
    
    parser.add_argument('--positive_weight', '-pw', type=float, required=False,help='positive_weight',default=1.0)
    parser.add_argument('--val_chr', '-v', type=str, required=False,help='Chromosomes for validation, split by comma',default='chr11')
    parser.add_argument('--val_label_file', '-vl', type=str, required=False,help='val_label_file',default='')
    parser.add_argument('--val_gencode_file', '-vg', type=str, required=False,help='val_gencode_file',default='')
    parser.add_argument('--val_negative_ratio', '-vn', type=int, required=False,help='val_negative_ratio',default=19)
    
    
    parser.add_argument('--plot_model', '-pl', action='store_true',help='plot_model')
    parser.add_argument('--random_seed', '-rs', type=int, required=False,help='random seed',default=0)

    
    return parser

def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    
    data_dir = args.data_dir
    tf_name = args.tf_name
    attention_position = args.attention_position
    flanking = args.flanking
    output_dir = args.output_dir
    
    train_script = args.train_script
    predict_script = args.predict_script
    
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
    
    rnaseq = args.rnaseq
    gencode = args.gencode
    unique35 = args.unique35
    
    use_peak = args.use_peak
    use_cudnn = args.use_cudnn
    single_attention_vector = args.single_attention_vector
    
    plot_model = args.plot_model
    if args.random_seed != 0:
        random_seed = args.random_seed
        np.random.seed(random_seed)
    else:
        random_seed = 1
    
    positive_weight = args.positive_weight
    val_chr = args.val_chr.split(',')
    val_label_file = args.val_label_file
    val_gencode_file = args.val_gencode_file
    val_negative_ratio = args.val_negative_ratio

    double_strand = True
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    file = open(output_dir+'/train_configs.txt','w') 
    file.write(str(args))
    file.close() 
    
    genome_fasta_file = data_dir+'/hg19.genome.fa'
    DNase_path =data_dir+ '/DNase'
    bigwig_file_unique35 = data_dir + '/wgEncodeDukeMapabilityUniqueness35bp.bigWig'
    rnaseq_data_file = data_dir + '/rnaseq_data.csv'
    gencode_train_file = data_dir + '/gencode_feature_train.tsv'
    
    train_label_path = data_dir + '/label/train/'
    train_peaks_path = data_dir + '/label/train_positive/'
        
    label_data_file = train_label_path+tf_name+'.train.labels.tsv.gz'
    positive_peak_file = train_peaks_path+tf_name+'.train.peak.tsv'
    output_model = output_dir + attention_position + '.' + tf_name + '.'+'1'+ '.'+str(flanking)+'.unique35'+str(unique35)+ '.RNAseq'+str(rnaseq)+ '.Gencode'+str(gencode)+'.h5'
    output_history = output_dir + attention_position + '.' + tf_name + '.'+'1'+ '.'+str(flanking)+'.unique35'+str(unique35)+ '.RNAseq'+str(rnaseq)+ '.Gencode'+str(gencode)+'.csv'
    

        
    print('loading labels')
    if val_label_file == '':
        label_region = pd.read_csv(label_data_file, sep='\t', header=0,usecols=['chr','start','stop'])
        all_chr = label_region['chr'].unique().tolist()
        train_chr = set(all_chr)-set(val_chr)
        val_idx = label_region.chr.isin(val_chr)
        train_idx = label_region.chr.isin(train_chr)
        cell_list = pd.read_csv(label_data_file, sep='\t', header=None,nrows=1).values.tolist()[0][3:]
        train_region = label_region[train_idx]
        train_region.index = range(train_region.shape[0])
        val_region = label_region[val_idx]
        val_region.index = range(val_region.shape[0])
        train_label_data_bind = pd.read_csv(label_data_file, sep='\t', header=0,usecols=range(3,3+len(cell_list)))=='B'
        train_label_data_unbind = pd.read_csv(label_data_file, sep='\t', header=0,usecols=range(3,3+len(cell_list)))=='U'
        val_label_data = pd.read_csv(label_data_file, sep='\t', header=0,usecols=range(3,3+len(cell_list)))[val_idx]
        
        train_label_data_bind = train_label_data_bind[train_idx].values
        train_label_data_unbind = train_label_data_unbind[train_idx].values

        
    else:
        train_region = pd.read_csv(label_data_file, sep='\t', header=0,usecols=['chr','start','stop'])
        val_region = pd.read_csv(val_label_file, sep='\t', header=0)
        # working in progress
            
    
    genome = utils.import_genome(genome_fasta_file,all_chr)
    window_size = train_region.stop[0]-train_region.start[0]
    
    num_meta = 0
    if rnaseq:
        num_meta = num_meta + 8
    if gencode:
        num_meta = num_meta + 6
        
    module_outfile = Path(output_dir+'/best_model.h5')
    if module_outfile.is_file():
        h = pd.read_csv(output_history)    
        best_auprc = h['val_auPRC'].max()
        epochs_trained = h.shape[0]-1
        model = load_model(output_dir+'/best_model.h5',custom_objects={'Attention1D': get_model.Attention1D})
    else:
        best_auprc = 0
        n_channel=6
        if not unique35:
            n_channel = 5
        model = get_model.make_model(attention_position=attention_position,L=window_size+2*flanking,n_channel=n_channel, num_conv = num_conv,
                                     num_denselayer=num_denselayer,num_lstm=num_lstm,kernel_size=kernel_size,num_filters=num_filters, 
                                     num_recurrent=num_recurrent,num_dense=num_dense,dropout_rate = dropout_rate,
                                     rnn_dropout=[rnn_dropout1,rnn_dropout2],num_meta=num_meta,merge=merge,cudnn=use_cudnn,
                                     single_attention_vector=single_attention_vector)
        model.compile(Adam(lr=learningrate),loss='binary_crossentropy',metrics=['accuracy'])
        file = open(output_history,'w') 
        file.write('epochs,loss,acc,val_auROC,val_auPRC\n') 
        file.close() 
        epochs_trained = 0
    
    
    rnaseq_train = rnaseq_val = 0
    if rnaseq:
        rnaseq_data = pd.read_csv(rnaseq_data_file)
        rnaseq_train = rnaseq_val = rnaseq_data[cell_list].values
    else:
        rnaseq_train = rnaseq_val = 0
    if gencode:
        print('loading Gencode features')
        gencode_train,gencode_val = utils.import_gencode(gencode_train_file,train_idx,val_idx)
    else:
        gencode_train = gencode_val = 0
    
    #If model already exist, continue training


    if plot_model:
        from keras.utils import plot_model # require graphviz
        plot_model(model,show_shapes=True,show_layer_names=False,to_file=output_dir+'model.png')
    
    if use_peak:
        label_bind = pd.read_csv(positive_peak_file,sep='\t')
        train_label_bind = label_bind[label_bind.chrom.isin(set(all_chr)-set(val_chr))]
        datagen_train = utils.TrainGeneratorSingle_augmented(genome,bigwig_file_unique35,DNase_path,train_region,train_label_bind,
                                                       train_label_data_unbind,cell_list,rnaseq_train,gencode_train,True,unique35,
                                                       rnaseq,gencode,flanking,batch_size,ratio_negative,double_strand)

    else:
        
        datagen_train = utils.TrainGeneratorSingle(genome=genome,bw_dict_unique35=bigwig_file_unique35,DNase_path=DNase_path,
                                                  label_region=train_region,label_data=train_label_data_bind,
                                                  label_data_unbind=train_label_data_unbind,cell_list=cell_list,rnaseq_data = rnaseq_train,
                                                  gencode_data=gencode_train,unique35=unique35,rnaseq=rnaseq,gencode=gencode,
                                                  flanking=flanking,batch_size=batch_size,ratio_negative=ratio_negative,double_strand = double_strand)



    datagen_val_list = []
    label_val_list = []
    for cell_name in cell_list:
        DNase_file = DNase_path+'/'+cell_name+'.1x.bw'
        val_region_cell = val_region[(val_label_data[cell_name] == 'B').values]
        val_region_cell = val_region_cell.append(val_region[(val_label_data[cell_name] == 'U').values].sample(n=val_region_cell.shape[0]*val_negative_ratio, random_state=random_seed),ignore_index=True) 
        label_val_cell = np.zeros(val_region_cell.shape[0])
        label_val_cell[0:np.sum(val_label_data[cell_name] == 'B')] = 1
        label_val_list.append(label_val_cell)
        datagen_val_list.append(utils.PredictionGeneratorSingle(genome,bigwig_file_unique35,DNase_file,val_region_cell,
                                                      rnaseq_val,gencode_val,unique35,rnaseq,gencode,flanking,batch_size=64))
 
    class_weight = {0: 1.0,1: positive_weight}
    train_flag = False
    
    wait_iter = 0
    for i in range(epochs):
        print('=========================== epoch '+str(i)+' =================================')
        history = model.fit_generator(datagen_train,epochs=1,verbose=1,class_weight=class_weight)
        datagen_train.on_epoch_end()
        j=0
        auroc_list = np.zeros(len(cell_list))
        auprc_list = np.zeros(len(cell_list))
        for datagen_val in datagen_val_list:
            print('validation on '+cell_list[j])
            datagen_val.start_idx = 0
            y_score = model.predict_generator(datagen_val,verbose=1)
            y_true = label_val_list[j]
            auroc,auprc = utils.calc_metrics(y_true,y_score)
            auroc_list[j] = auroc
            auprc_list[j] = auprc
            j = j+1
        current_auroc = np.mean(auroc_list)        
        current_auprc = np.mean(auprc_list)
        file = open(output_history,'a') 
        file.write(str(i+epochs_trained)+','+str(history.history['loss'][0])+','+str(history.history['acc'][0])+','+str(current_auroc)+','+str(current_auprc)+'\n') 
        file.close() 
        print('Validation auROC: '+str(current_auroc)+' auPRC: '+str(current_auprc))
        if current_auprc > best_auprc:
            print('Validation auPRC improved from '+str(best_auprc)+' to '+str(current_auprc))
            best_auprc = current_auprc
            wait_iter=0
            model.save(re.sub('.h5','.epoch'+str(i+epochs_trained)+'.h5',output_model))
            model.save(output_dir+'/best_model.h5')
        else:
            print('Validation auPRC did not improve from '+str(best_auprc))
            wait_iter = wait_iter+1
            if wait_iter > patience:
                print('Early stopping')
                train_flag = True
                break
    
    if train_flag:
        # submit another job for training
        subprocess.run(["sbatch",predict_script])
    else:
        subprocess.run(["sbatch",train_script])
        
    
if __name__ == '__main__':
    main()

# python train2.py -i /home/chen/data/deepGRN/raw/ -t NANOG -o /home/chen/data/deepGRN/reproduce_test2/  -rn 2  -pw 2

#data_dir = '/home/chen/data/deepGRN/raw/'
#tf_name = 'NANOG'
#output_dir = '/home/chen/data/deepGRN/result_test/'
#val_chr = ['chr11']
#attention_position = 'attention1d_after_lstm'
#val_label_file =''
#flanking = 401
#unique35 = rnaseq = gencode = True
#batch_size = 32
#ratio_negative = 15
#learningrate = 0.001
#model = get_model.make_model(attention_position=attention_position,L=1002,num_meta=14,num_conv=0,num_lstm=0,num_dense=1)
#model.compile(Adam(lr=learningrate),loss='binary_crossentropy',metrics=['accuracy'])
#patience = 5
#epochs=10
#use_peak = False
#plot_model = False
#random_seed=1