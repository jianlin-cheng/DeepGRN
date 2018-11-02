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
    d = dict(zip(['N','A','T','G','C'], range(6)))
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
    rc_x[:,:,0] = x[:,:,1]==1
    rc_x[:,:,1] = x[:,:,0]==1
    rc_x[:,:,2] = x[:,:,3]==1
    rc_x[:,:,3] = x[:,:,2]==1
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

def make_y(label_data,bin_start,bin_stop,cell_idx):
    return label_data[bin_start:(bin_stop+1),cell_idx]+0

def make_y_batch(label_data,bins_batch):
    y_batch = np.zeros((bins_batch.shape[1],(bins_batch[1,0]-bins_batch[0,0]+1),1))
    for i in range(bins_batch.shape[1]):
        bin_start = bins_batch[0,i]
        bin_stop = bins_batch[1,i]
        y_batch[i,:,:] = make_y(label_data,bin_start,bin_stop,bins_batch[2,i])
    return y_batch

# extract "B" or "U" bin indices from label_data, return is ndarray of n by 2, columns are bin indices and cell column indices
# If num=0 keep all row index info, otherwise sample num indices from each column 
def make_ID_list_single(label_region,label_data,num=0):
    label_IDs = np.nonzero(label_data[:,0])[0]
    if not num == 0:
        label_IDs = np.random.choice(label_IDs,size=num)
    label_IDs = np.vstack((label_IDs,np.zeros_like(label_IDs)))
    if label_data.shape[1] > 1:
        for col_idx in range(1,label_data.shape[1]):
            label_IDs_idx = np.nonzero(label_data[:,col_idx])[0]
            if not num == 0:
                label_IDs_idx = np.random.choice(label_IDs_idx,size=num)
            label_IDs_idx = np.vstack((label_IDs_idx,np.zeros_like(label_IDs_idx)+col_idx))
            label_IDs = np.concatenate((label_IDs,label_IDs_idx),axis=1)
    label_IDs = label_IDs.T
    np.random.shuffle(label_IDs)
    return label_IDs

def make_x_batch_positive(genome,bigwig_file_unique35,bigwig_file_list,label_data_bind,use_shift,flanking,window_size=200):
    seq_len = (flanking)*2+window_size
    x_batch = np.zeros((label_data_bind.shape[0],seq_len,6))
    for i in range(label_data_bind.shape[0]):
        peak_stop = label_data_bind.iloc[i,2]
        peak_start = label_data_bind.iloc[i,1]
        med = (peak_stop + peak_start)//2
        if use_shift:
            shift = peak_stop - med + window_size//2 -76
            med = med + np.random.randint(-shift, shift+1)
        seq_start = med - seq_len//2
        seq_end = med + seq_len//2
        bigwig_file = bigwig_file_list[label_data_bind.iloc[i,3]]
        chr_id = label_data_bind.iloc[i,0]
        x_batch[i,:,:] = make_x(genome,bigwig_file_unique35,bigwig_file,seq_start,seq_end,flanking,chr_id) 
    return x_batch

class TrainGeneratorSingle(Sequence):
    def __init__(self,genome,bigwig_file_unique35,DNase_path,label_region,label_data_bind,label_data_unbind,cell_list,rnaseq_data = 0,gencode_data=0,use_shift=False,unique35=True,rnaseq=False,gencode=False,flanking=0,batch_size=32,ratio_negative=1):
        self.batch_size = batch_size
        self.label_region = label_region
        self.label_data_bind = label_data_bind
        self.label_data_unbind = label_data_unbind
        self.cell_list = cell_list
        self.genome = genome
        self.unique35 = unique35
        self.bigwig_file_unique35 = bigwig_file_unique35
        self.flanking = flanking
        self.use_shift = use_shift
        self.rnaseq = rnaseq
        self.rnaseq_data = rnaseq_data
        self.gencode = gencode
        self.gencode_data = gencode_data
        self.ratio_negative = ratio_negative
        self.label_IDs_negative = make_ID_list_single(label_region,label_data_unbind,(label_data_bind.shape[0]//len(cell_list)+1)*ratio_negative)
        self.bigwig_file_list = [DNase_path+'/' +s +'.1x.bw' for s in cell_list]
        self.start_idx_positive = 0
        self.start_idx_negative = 0
        self.positive_indices = np.array(range(label_data_bind.shape[0]))
        self.on_epoch_end()

    def __len__(self):
        #Denotes the number of batches per epoch
        return self.label_data_bind.shape[0] // ((self.batch_size)//2)-1

    def __getitem__(self, index):
        # Generate indexes of bins of the batch
        batch_start_idx = self.start_idx_positive
        batch_end_idx = min([batch_start_idx+self.batch_size//2-1,self.label_data_bind.shape[0]-1])
        data_indices = self.positive_indices[batch_start_idx:(batch_end_idx+1)]
        end_idx_negative = min(self.start_idx_negative+self.batch_size//2*self.ratio_negative-1,self.label_IDs_negative.shape[0]-1)
        negative_batch = self.label_IDs_negative[self.start_idx_negative:(end_idx_negative+1)]
        self.start_idx_negative = end_idx_negative+1
        self.start_idx_positive = batch_end_idx+1
        bins_batch = np.vstack((negative_batch[:,0],negative_batch[:,0],negative_batch[:,1]))        
        # Generate data
        X, y = self.__data_generation(bins_batch,data_indices)
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.positive_indices)
        self.label_IDs_negative = make_ID_list_single(self.label_region,self.label_data_unbind,(self.label_data_bind.shape[0]//len(self.cell_list)+1)*self.ratio_negative)
        self.start_idx_positive = 0
        self.start_idx_negative = 0
        
    def __data_generation(self,bins_batch,data_indices):
        x_fw_positive = make_x_batch_positive(self.genome,self.bigwig_file_unique35,self.bigwig_file_list,self.label_data_bind.iloc[data_indices],self.use_shift,self.flanking)
        x_fw_negative = make_x_batch(self.genome,self.bigwig_file_unique35,self.bigwig_file_list,self.label_region,bins_batch,self.flanking)
        x_fw = np.concatenate((x_fw_positive,x_fw_negative))
        x_rc = make_rc_x(x_fw)
        if self.rnaseq:
        # rnaseq
            positive_cell_idx = self.label_data_bind.iloc[data_indices,3].tolist()
            negative_cell_idx = bins_batch[2,:].tolist()
            cell_idx = positive_cell_idx + negative_cell_idx
            rnaseq_batch = np.transpose(self.rnaseq_data[:,cell_idx])
        if self.gencode:
        #gencode
            positive_gencode_batch = self.label_data_bind.iloc[data_indices,4:]
            negative_gencode_batch = self.gencode_data[bins_batch[0,:],:]+0
            gencode_batch = np.concatenate((np.array(positive_gencode_batch),negative_gencode_batch))
        if not self.unique35:
            x_fw = np.delete(x_fw,4,2)
            x_rc = np.delete(x_rc,4,2)
        if not self.rnaseq and not self.gencode:
            X = [x_fw,x_rc]
        elif self.rnaseq and self.gencode:
            X = [x_fw,x_rc,np.concatenate((rnaseq_batch,gencode_batch),axis=1)]
        elif self.rnaseq:
            X = [x_fw,x_rc,rnaseq_batch]
        elif self.gencode:
            X = [x_fw,x_rc,gencode_batch]
            
        y = np.concatenate((np.zeros(x_fw_positive.shape[0])+1,np.zeros(x_fw_negative.shape[0])))
        return X, y


class DataGeneratorSingle(Sequence):
    def __init__(self,genome,bw_dict_unique35,DNase_path,label_region,label_data,label_data_unbind,cell_list,rnaseq_data = 0,gencode_data=0,unique35=True,rnaseq=False,gencode=False,flanking=0,batch_size=32,ratio_negative=1):
        label_IDs_positive = make_ID_list_single(label_region,label_data)
        self.batch_size = batch_size
        self.label_region = label_region
        self.label_data = label_data
        self.label_data_unbind = label_data_unbind
        self.cell_list = cell_list
        self.genome = genome
        self.unique35 = unique35
        self.bw_dict_unique35 = bw_dict_unique35
        self.flanking = flanking
        self.rnaseq = rnaseq
        self.rnaseq_data = rnaseq_data
        self.gencode = gencode
        self.gencode_data = gencode_data
        self.ratio_negative = ratio_negative
        self.label_IDs_positive = label_IDs_positive
        self.label_IDs_negative = make_ID_list_single(label_region,label_data_unbind,(label_IDs_positive.shape[0]//len(cell_list)+1)*ratio_negative)
        self.bigwig_file_list = [DNase_path+'/' +s +'.1x.bw' for s in cell_list]
        self.start_idx_positive = 0
        self.start_idx_negative = 0
        self.on_epoch_end()

    def __len__(self):
        #Denotes the number of batches per epoch
        return self.label_IDs_positive.shape[0] // ((self.batch_size)//2)-1

    def __getitem__(self, index):
        # Generate indexes of bins of the batch
        end_idx = min([self.start_idx_positive+self.batch_size//2-1,self.label_IDs_positive.shape[0]-1])
        positive_batch = self.label_IDs_positive[self.start_idx_positive:(end_idx+1)]
        end_idx_negative = min(self.start_idx_negative+self.batch_size//2*self.ratio_negative-1,self.label_IDs_negative.shape[0]-1)
        negative_batch = self.label_IDs_negative[self.start_idx_negative:(end_idx_negative+1)]
        
        self.start_idx_negative = end_idx_negative+1
        self.start_idx_positive = end_idx+1
        
        # mix with equal number of negative bins
        indexes_batch = np.concatenate((positive_batch,negative_batch),axis=0)
        bins_batch = np.vstack((indexes_batch[:,0],indexes_batch[:,0],indexes_batch[:,1]))        
        # Generate data
        X, y = self.__data_generation(bins_batch,positive_batch.shape[0],negative_batch.shape[0])
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.label_IDs_positive)
        self.label_IDs_negative = make_ID_list_single(self.label_region,self.label_data_unbind,(self.label_IDs_positive.shape[0]//len(self.cell_list))*self.ratio_negative+1)
        self.start_idx_positive = 0
        self.start_idx_negative = 0
        
    def __data_generation(self,bins_batch,num_positive,num_negative):
        x_fw = make_x_batch(self.genome,self.bw_dict_unique35,self.bigwig_file_list,self.label_region,bins_batch,self.flanking)
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
            
        y = np.concatenate((np.zeros(num_positive)+1,np.zeros(num_negative)))
        return X, y


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

#utility functions for training/predictions of multiple bins:
def make_ID_list(label_data,bin_num,bind='B'):
    label_IDs = (label_data.iloc[:,3]==bind).values.astype(int).ravel().nonzero()[0]
    label_IDs = np.vstack((label_IDs,np.zeros_like(label_IDs)))
    if label_data.shape[1] > 4:
        for col_idx in range(4,label_data.shape[1]):
            label_IDs_idx = (label_data.iloc[:,col_idx]==bind).values.astype(int).ravel().nonzero()[0]
            label_IDs_idx = np.vstack((label_IDs_idx,np.zeros_like(label_IDs_idx)+col_idx-3))
            label_IDs = np.concatenate((label_IDs,label_IDs_idx),axis=1)
    label_IDs = label_IDs.T
    label_IDs = label_IDs[np.logical_and(label_IDs[:,0]>=bin_num//2, label_IDs[:,0]+bin_num//2<=label_data.shape[0]-1)]
    return label_IDs
    
class DataGenerator(Sequence):
    def __init__(self,genome,bw_dict_unique35,DNase_path,label_data,rnaseq_data = 0,gencode_data=0,unique35=True,rnaseq=False,gencode=False,bin_num=1,flanking=0,batch_size=32,multibin=True):
        self.batch_size = batch_size
        self.label_data = label_data
        self.bin_num = bin_num
        self.genome = genome
        self.unique35 = unique35
        self.bw_dict_unique35 = bw_dict_unique35
        self.flanking = flanking
        self.multibin = multibin
        self.rnaseq = rnaseq
        self.rnaseq_data = rnaseq_data
        self.gencode = gencode
        self.gencode_data = gencode_data
        self.label_IDs_positive = make_ID_list(label_data,bin_num)
        self.label_IDs_negative = make_ID_list(label_data,bin_num,'U')
        self.bigwig_file_list = [DNase_path+'/' +s +'.1x.bw' for s in label_data.columns.values.tolist()[3:len(label_data.columns.values.tolist())]]
        self.on_epoch_end()

    def __len__(self):
        #Denotes the number of batches per epoch
        return int(np.floor(self.label_IDs_positive.shape[0] / ((self.batch_size)//2)))

    def __getitem__(self, index):
        # Generate indexes of bins of the batch
        batch_end = min((index+1)*(self.batch_size//2),len(self.label_IDs_positive))
        positive_batch = self.label_IDs_positive[index*(self.batch_size//2):batch_end]
        # mix with equal number of negative bins
        negative_batch = self.label_IDs_negative[np.random.choice(self.label_IDs_negative.shape[0],size=len(positive_batch))]
        indexes_batch = np.concatenate((positive_batch,negative_batch),axis=0)
#        np.random.shuffle(indexes_batch)
        bins_batch = np.vstack((indexes_batch[:,0]-self.bin_num//2,indexes_batch[:,0]+self.bin_num//2,indexes_batch[:,1]))        

        # Generate data
        x_fw, y = self.__data_generation(bins_batch)
        x_rc = make_rc_x(x_fw)
        
        if not self.unique35:
            x_fw = np.delete(x_fw,4,2)
            x_rc = np.delete(x_rc,4,2)
        if not self.rnaseq and not self.gencode:
            X = [x_fw,x_rc]
        elif self.rnaseq and self.gencode:
            rnaseq_batch = np.transpose(self.rnaseq_data[:,bins_batch[2,:]])
            gencode_batch = self.gencode_data[bins_batch[0,:],:]+0
#            gencode_batch = make_xgencode_batch(self.gencode_path,bins_batch,self.flanking,self.label_data)
            X = [x_fw,x_rc,np.concatenate((rnaseq_batch,gencode_batch),axis=1)]
        elif self.rnaseq:
            X = [x_fw,x_rc,np.transpose(self.rnaseq_data[:,bins_batch[2,:]])]
        elif self.gencode:
            gencode_batch = self.gencode_data[bins_batch[0,:],:]+0
#            gencode_batch = make_xgencode_batch(self.gencode_path,bins_batch,self.flanking,self.label_data)
            X = [x_fw,x_rc,gencode_batch]
            
        if not self.multibin:
            y = y[:,:,0]
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.label_IDs_positive)
        
    def __data_generation(self,bins_batch):
        x_fw = make_x_batch(self.genome,self.bw_dict_unique35,self.bigwig_file_list,self.label_data,bins_batch,self.flanking)
        y = make_y_batch(self.label_data,bins_batch)
        return x_fw, y
    
def parse_bed_merge(bed_merge):
    bin_len = ((bed_merge.iloc[:,2]-bed_merge.iloc[:,1]-150)/50).astype(int)
    bin_table = np.empty((len(bin_len),2))
    start_idx = 0
    for i in range(len(bin_len)):
        bin_table[i,0] = start_idx
        bin_table[i,1] = start_idx+bin_len[i]-1
        start_idx = start_idx+bin_len[i]
    return bin_table.astype(int)

def get_batch_num_seg(seg_size,bin_num,batch_size):
    if seg_size == 0:
        return 0
    elif seg_size <= bin_num:
        return 1
    elif seg_size < bin_num * batch_size:
        if seg_size % bin_num == 0:
            return 1
        else:
            return 2
    else:
        l = (seg_size // (bin_num*batch_size)) + get_batch_num_seg((seg_size % (bin_num*batch_size)),bin_num,batch_size)
        return int(l)
    
# Since all samples have been padding to bin_num during prediction, final output should remove those labels from padding
def remove_pred_padding(pred,seg_array,bin_num):
    seg_array1 = seg_array
    id_remove = []
    for i in range(seg_array1.shape[0]):
        m = (seg_array1[i,1]-seg_array1[i,0]+1) % bin_num
        if m > 0 :
            id_remove = id_remove + list(range(seg_array1[i,1]+1,seg_array1[i,1]+1+bin_num-m))
            seg_array1 = seg_array1 + bin_num - m
    pred_final = pred[:,:,0].flatten()
    return np.delete(pred_final,id_remove)

class PredictionGenerator(Sequence):
    def __init__(self,genome,bw_dict_unique35,DNase_file,seg_array,label_data,rnaseq_data=0,gencode_data=0,unique35=True,rnaseq=False,gencode=False,bin_num=1,flanking=0,batch_size=32,multibin=True):
        self.batch_size = batch_size
        self.bin_num = bin_num
        self.genome = genome
        self.unique35 = unique35
        self.bw_dict_unique35 = bw_dict_unique35
        self.rnaseq = rnaseq
        self.rnaseq_data = rnaseq_data
        self.gencode = gencode
        self.gencode_data = gencode_data
        self.flanking = flanking
        self.multibin = multibin
        self.seg_array = seg_array
        self.label_data = label_data
        self.bigwig_file_list = [DNase_file]
        self.start_idx = 0

    def __len__(self):
        #Denotes the number of batches per epoch
        seg_len = self.seg_array[:,1]-self.seg_array[:,0]+1
        return sum([get_batch_num_seg(seg_size,self.bin_num,self.batch_size) for seg_size in seg_len])

    def __getitem__(self, index):
        # Generate indexes of bins of the batch
        seg_end = self.seg_array[np.logical_and(self.seg_array[:,0]<=self.start_idx, self.seg_array[:,1]>=self.start_idx),1][0]
        max_batch_size = (seg_end-self.start_idx+1)//self.bin_num
        if max_batch_size > 0:
            if max_batch_size >= self.batch_size:
                current_batch_size = self.batch_size
            else:
                current_batch_size = max_batch_size
            start_bin_idx = np.array([self.start_idx+i*self.bin_num for i in range(current_batch_size)])
            bins_batch = np.vstack((start_bin_idx,start_bin_idx+self.bin_num-1,np.zeros_like(start_bin_idx)))
            self.start_idx += self.bin_num*current_batch_size
        else:
            # this batch will only have size 1
            bins_batch = np.vstack((self.start_idx,seg_end,0))
            self.start_idx = seg_end + 1
            
        # Generate data
        x_fw = self.__data_generation(bins_batch)
        
        # Padding with 0s
        if (x_fw.shape[1] != 50*self.bin_num+150+self.flanking*2) & self.multibin:
            x_fw = np.concatenate((x_fw,np.zeros((1,50*self.bin_num+150+self.flanking*2-x_fw.shape[1],x_fw.shape[2]))),axis=1)
        x_rc = make_rc_x(x_fw)
        if not self.unique35:
            x_fw = np.delete(x_fw,4,2)
            x_rc = np.delete(x_rc,4,2)
        if not self.rnaseq and not self.gencode:
            X = [x_fw,x_rc]
        elif self.rnaseq and self.gencode:
            rnaseq_batch = np.transpose(self.rnaseq_data[:,bins_batch[2,:]])
            gencode_batch = self.gencode_data[bins_batch[0,:],:]+0
#            gencode_batch = make_xgencode_batch(self.gencode_path,bins_batch,self.flanking,self.label_data)
            X = [x_fw,x_rc,np.concatenate((rnaseq_batch,gencode_batch),axis=1)]
        elif self.rnaseq:
            X = [x_fw,x_rc,np.transpose(self.rnaseq_data[:,bins_batch[2,:]])]
        elif self.gencode:
            gencode_batch = self.gencode_data[bins_batch[0,:],:]+0
#            gencode_batch = make_xgencode_batch(self.gencode_path,bins_batch,self.flanking,self.label_data)
            X = [x_fw,x_rc,gencode_batch]
            
        return X
        
    def __data_generation(self,bins_batch):
        x_fw = make_x_batch(self.genome,self.bw_dict_unique35,self.bigwig_file_list,self.label_data,bins_batch,self.flanking)
        return x_fw
