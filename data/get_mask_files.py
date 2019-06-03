import sys
import re
import itertools
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from pybedtools import BedTool, Interval

data_path = sys.argv[1]
#data_path = '/storage/htc/bdm/ccm3x/deepGRN/raw/'

#label_path = data_path+'label/train'
#label_peak_path = data_path+'label/train_positive/'
#train_bed_file = data_path+'label/train_regions.blacklistfiltered.bed'
#pred_bed_file = data_path+'label/predict_region.bed'
#blacklist_file = data_path+'blacklist.bed.gz'
#genome_sizes_file = data_path+'hg19.autoX.chrom.sizes'
#gencode_path = data_path+'gencode/'

L =1002
genome_window_size = 200


def get_gencode_feature(bed_input,gencode_path):
    cpg_bed = BedTool(gencode_path+'cpgisland.bed.gz')
    cds_bed = BedTool(gencode_path+'wgEncodeGencodeBasicV19.cds.merged.bed.gz')
    intron_bed = BedTool(gencode_path+'wgEncodeGencodeBasicV19.intron.merged.bed.gz')
    promoter_bed = BedTool(gencode_path+'wgEncodeGencodeBasicV19.promoter.merged.bed.gz')
    utr5_bed = BedTool(gencode_path+'wgEncodeGencodeBasicV19.utr5.merged.bed.gz')
    utr3_bed = BedTool(gencode_path+'wgEncodeGencodeBasicV19.utr3.merged.bed.gz')
    peaks= bed_input
    peaks_cpg_bedgraph = peaks.intersect(cpg_bed, wa=True, c=True)
    peaks_cds_bedgraph = peaks.intersect(cds_bed, wa=True, c=True)
    peaks_intron_bedgraph = peaks.intersect(intron_bed, wa=True, c=True)
    peaks_promoter_bedgraph = peaks.intersect(promoter_bed, wa=True, c=True)
    peaks_utr5_bedgraph = peaks.intersect(utr5_bed, wa=True, c=True)
    peaks_utr3_bedgraph = peaks.intersect(utr3_bed, wa=True, c=True)
    data = []    
    for cpg, cds, intron, promoter, utr5, utr3 in itertools.izip(peaks_cpg_bedgraph,peaks_cds_bedgraph,peaks_intron_bedgraph,peaks_promoter_bedgraph,peaks_utr5_bedgraph,peaks_utr3_bedgraph):
        gencode = [cpg.count, cds.count, intron.count, promoter.count, utr5.count, utr3.count]
        data.append(gencode)
    return np.array(data, dtype=bool)

def get_genome_bed(genome_sizes_file):
    genome_sizes_info = np.loadtxt(genome_sizes_file, dtype=str)
    chroms = list(genome_sizes_info[:,0])
    chroms_sizes = list(genome_sizes_info[:,1].astype(int))
    genome_bed = []
    for chrom, chrom_size in zip(chroms, chroms_sizes):
        genome_bed.append(Interval(chrom, 0, chrom_size))
    genome_bed = BedTool(genome_bed)
    return genome_bed

def get_blacklist(blacklist_file):
    blacklist = BedTool(blacklist_file)
    blacklist = blacklist.slop(g=genome_sizes_file, b=L)
    genome_sizes_info = np.loadtxt(genome_sizes_file, dtype=str)
    chroms = list(genome_sizes_info[:,0])
    chroms_sizes = list(genome_sizes_info[:,1].astype(int))
    blacklist2 = []
    for chrom, size in zip(chroms, chroms_sizes):
        blacklist2.append(Interval(chrom, 0, L))
        blacklist2.append(Interval(chrom, size - L, size))
    blacklist2 = BedTool(blacklist2)
    blacklist = blacklist.cat(blacklist2)
    return blacklist

def chroms_filter(feature, chroms):
    if feature.chrom in chroms:
        return True
    return False

def subset_chroms(chroms, bed):
    result = bed.filter(chroms_filter, chroms).saveas()
    return BedTool(result.fn)

genome_bed = get_genome_bed(genome_sizes_file)
train_bed = BedTool(train_bed_file)
pred_bed = BedTool(pred_bed_file)
blacklist_bed = get_blacklist(blacklist_file)

pred_bed_intersect_blacklist_count = pred_bed.intersect(blacklist_bed, wa=True, c=True, sorted=True)
nonblacklist_bools = np.array([i.count==0 for i in pred_bed_intersect_blacklist_count])
np.savetxt(data_path+"nonblacklist_bools.csv", nonblacklist_bools.astype(int), fmt='%1.0f', delimiter=',')

gencode_feature_train = get_gencode_feature(train_bed,gencode_path)
np.savetxt(data_path+'gencode_feature_train.tsv',gencode_feature_train, fmt='%1.0f', delimiter='\t')
gencode_feature_test = get_gencode_feature(pred_bed,gencode_path)
np.savetxt(data_path+'gencode_feature_test.tsv',gencode_feature_test, fmt='%1.0f', delimiter='\t')


tf_files = [f for f in listdir(label_path) if isfile(join(label_path, f))]
for tf_file in tf_files:
    tf_name = re.sub("(\..+)$", "", tf_file)
    label_train = pd.read_csv(data_path+'label/train/'+tf_name+'.train.labels.tsv',sep='\t',nrows=3)
    train_cells = list(label_train.columns.values)[3:label_train.shape[1]]
    pos_labels = pd.DataFrame(None)
    cell_idx = 0
    for train_cell in train_cells:
        relax_bed = BedTool(data_path+'chipseq/ChIPseq.'+train_cell+'.'+tf_name+'.relaxed.narrowPeak.gz')
        relax_bed = relax_bed.sort()
        relax_bed_slop = relax_bed.slop(g=genome_sizes_file, b=genome_window_size)
        nonnegative_bed = blacklist_bed.cat(relax_bed_slop)
        nonnegative_bed = nonnegative_bed.sort()
#        negative_bed  = train_bed.intersect(nonnegative_bed, wa=True, v=True, sorted=True,c=True)
#        negative_bed_df = pd.read_table(negative_bed.fn, names=['chrom', 'start', 'stop'],usecols =[0,1,2])
#        neg_labels = pd.concat((negative_bed_df,pd.DataFrame(gencode_negative+0)))
        
        conserv_bed = BedTool(data_path+'chipseq/ChIPseq.'+train_cell+'.'+tf_name+'.conservative.train.narrowPeak.gz')
        conserv_bed = conserv_bed.sort()
        conserv_bed_filtered = conserv_bed.intersect(blacklist_bed, wa=True, v=True, sorted=True)
        gencode_positive = get_gencode_feature(conserv_bed_filtered,gencode_path)
        conserv_bed_filtered_df = pd.read_table(conserv_bed_filtered.fn, names=['chrom', 'start', 'stop'],usecols =[0,1,2])
        pos_labels_cell = pd.concat((conserv_bed_filtered_df,
                        pd.DataFrame(np.zeros((conserv_bed_filtered_df.shape[0],1),dtype =int)+cell_idx),
                        pd.DataFrame(gencode_positive+0)),axis=1)
        if pos_labels.shape[0] == 0:
            pos_labels = pos_labels_cell
        else:
            pos_labels = pd.concat((pos_labels,pos_labels_cell))
        cell_idx +=1
        pos_labels.to_csv(label_peak_path+tf_name+'.train.peak.tsv',sep='\t',header = True,index = False)
        
