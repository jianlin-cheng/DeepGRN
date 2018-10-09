import numpy as np
import pandas as pd
from keras.utils import Sequence

import pyBigWig
from pyfaidx import Fasta

#import itertools
#from pybedtools import BedTool

# slice sequence from seq_start+1 to seq_end with length=seq_end-seq_start , return ndarray of length by 4
def genome_to_1hot(genome,chr_id,seq_start,seq_end):
    chr_str = list(genome[chr_id][seq_start:seq_end].seq.upper())
    d = dict(zip(['N','A','C','G','T'], range(6)))
    v = [d[x] for x in chr_str]
    y = np.eye(5)[v]
    return y[:,1:5]

def import_genome(genome_fasta_file,all_chr):
    genome = dict()
    fasta = Fasta(genome_fasta_file)
    for chr_id in all_chr:
        print('Loading '+chr_id)
        genome[chr_id] = fasta[chr_id][0:len(fasta[chr_id])]
    return genome

def import_bw_single(bigwig_file,genome,all_chr):
    bw_dict = dict([(key, []) for key in all_chr])
    bigwig = pyBigWig.open(bigwig_file)
    for chr_id in all_chr:
        print('Processing '+chr_id)
        bw_dict[chr_id]= np.array(bigwig.values(chr_id, 0, len(genome[chr_id])))
        bw_dict[chr_id][np.isnan(bw_dict[chr_id])] = 0
    bigwig.close()
    return bw_dict

def import_gencode(gencode_train_file,train_idx,val_idx):
    gencode_data = pd.read_csv(gencode_train_file, sep='\t', header=None,dtype=np.bool_)
    gencode_train = gencode_data[train_idx].values
    gencode_val = gencode_data[val_idx].values
    return gencode_train,gencode_val

def import_label(label_data_file,val_chr=['chr11']):
    label_region = pd.read_csv(label_data_file, sep='\t', header=0,usecols=['chr','start','stop'])
    all_chr = label_region['chr'].unique().tolist()
    train_chr = set(all_chr)-set(val_chr)
    val_idx = label_region.chr.isin(val_chr)
    train_idx = label_region.chr.isin(train_chr)
    cell_list = pd.read_csv(label_data_file, sep='\t', header=None,nrows=1).values.tolist()[0][3:]
    train_region = label_region[train_idx]
    val_region = label_region[val_idx]
    
    label_data = pd.read_csv(label_data_file, sep='\t', header=0,usecols=range(3,3+len(cell_list)))=='B'
    label_data_unbind = pd.read_csv(label_data_file, sep='\t', header=0,usecols=range(3,3+len(cell_list)))=='U'
    train_label_data = label_data[train_idx].values
    val_label_data = label_data[val_idx].values
    train_label_data_unbind = label_data_unbind[train_idx].values
    val_label_data_unbind = label_data_unbind[val_idx].values
    return all_chr,cell_list,train_region,val_region,train_label_data,val_label_data,train_label_data_unbind,val_label_data_unbind,train_idx,val_idx

    
def make_rc_x(x):
    rc_x = np.copy(x)
    rc_x[:,:,0] = x[:,:,3]==1
    rc_x[:,:,3] = x[:,:,0]==1
    rc_x[:,:,2] = x[:,:,1]==1
    rc_x[:,:,1] = x[:,:,2]==1
    rc_x = np.flip(rc_x, 1)
    return rc_x


def make_x(genome,bw_dict_unique35,bigwig_file,seq_start,seq_end,flanking,chr_id):
    bigwig = pyBigWig.open(bigwig_file)
    dnase_bw = np.array(bigwig.values(chr_id, seq_start, seq_end))
    dnase_bw[np.isnan(dnase_bw)] = 0
    bigwig.close()
    
    bigwig = pyBigWig.open(bw_dict_unique35)
    unique35_bw = np.array(bigwig.values(chr_id, seq_start, seq_end))
    unique35_bw[np.isnan(unique35_bw)] = 0
    bigwig.close()
    return np.concatenate((genome_to_1hot(genome,chr_id,seq_start,seq_end),unique35_bw[:, None],dnase_bw[:, None]),axis=1)
    
def make_x_batch(genome,bw_dict_unique35,bigwig_file_list,label_data,bins_batch,flanking):
    x_batch = np.zeros((bins_batch.shape[1],((bins_batch[1,0]-bins_batch[0,0]+1)*50+150)+2*flanking,6))
    for i in range(bins_batch.shape[1]):
        bin_start = bins_batch[0,i]
        bin_stop = bins_batch[1,i]
        seq_start = int(label_data.iloc[bin_start,1]-flanking)
        seq_end = int(label_data.iloc[bin_stop,2]+flanking)
        # check if sample is invalid due to crossing blacklisted region or chromosome
        if seq_end-seq_start == 50*(bin_stop-bin_start+1)+150+2*flanking:
            bigwig_file = bigwig_file_list[bins_batch[2,i]]
            chr_id = label_data.iloc[bin_start,0]
            x_batch[i,:,:] = make_x(genome,bw_dict_unique35,bigwig_file,seq_start,seq_end,flanking,chr_id)        
    return x_batch

class PredictionGeneratorSingle(Sequence):
    def __init__(self,genome,bw_dict_unique35,DNase_file,predict_region,rnaseq_data=0,gencode_data=0,unique35=True,rnaseq=False,gencode=False,flanking=0,batch_size=32):
        self.batch_size = batch_size
        self.genome = genome
        self.unique35 = unique35
        self.bw_dict_unique35 = bw_dict_unique35
        self.rnaseq = rnaseq
        self.rnaseq_data = rnaseq_data
        self.gencode = gencode
        self.gencode_data = gencode_data
        self.flanking = flanking
        self.predict_region = predict_region
        self.bigwig_file_list = [DNase_file]
        self.start_idx = 0

    def __len__(self):
        return int(np.ceil(self.predict_region.shape[0] / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of bins of the batch
        end_idx = min([self.start_idx+self.batch_size-1,self.predict_region.shape[0]-1])
        start_bin_idx = np.array(range(self.start_idx,end_idx+1))
        bins_batch = np.vstack((start_bin_idx,start_bin_idx,np.zeros_like(start_bin_idx)))
        self.start_idx = end_idx + 1
        
        # Generate data
        X = self.__data_generation(bins_batch)
        return X

    def __data_generation(self,bins_batch):
        x_fw = make_x_batch(self.genome,self.bw_dict_unique35,self.bigwig_file_list,self.predict_region,bins_batch,self.flanking)
        x_rc = make_rc_x(x_fw)
        if not self.unique35:
            x_fw = np.delete(x_fw,4,2)
            x_rc = np.delete(x_rc,4,2)
        if not self.rnaseq and not self.gencode:
            X = [x_fw,x_rc]
        elif self.rnaseq and self.gencode:
            rnaseq_batch = np.transpose(self.rnaseq_data[:,bins_batch[2,:]])
            gencode_batch = self.gencode_data[bins_batch[0,:],:]+0
            X = [x_fw,x_rc,np.concatenate((rnaseq_batch,gencode_batch),axis=1)]
        elif self.rnaseq:
            X = [x_fw,x_rc,np.transpose(self.rnaseq_data[:,bins_batch[2,:]])]
        elif self.gencode:
            gencode_batch = self.gencode_data[bins_batch[0,:],:]+0
            X = [x_fw,x_rc,gencode_batch]
        return X
