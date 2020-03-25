import sys
import re
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from pybedtools import BedTool, Interval

gencode_path = sys.argv[1]
genome_sizes_file = sys.argv[2]
bed_file = sys.argv[3]
outfile = sys.argv[4]


def get_gencode_feature(bed_input,gencode_path):
    cpg_bed = BedTool(gencode_path+'cpgisland.bed.gz')
    cds_bed = BedTool(gencode_path+'cds.bed.gz')
    intron_bed = BedTool(gencode_path+'intron.bed.gz')
    promoter_bed = BedTool(gencode_path+'promoter.bed.gz')
    utr5_bed = BedTool(gencode_path+'utr5.bed.gz')
    utr3_bed = BedTool(gencode_path+'utr3.bed.gz')
    peaks= bed_input
    peaks_cpg_bedgraph = peaks.intersect(cpg_bed, wa=True, c=True)
    peaks_cds_bedgraph = peaks.intersect(cds_bed, wa=True, c=True)
    peaks_intron_bedgraph = peaks.intersect(intron_bed, wa=True, c=True)
    peaks_promoter_bedgraph = peaks.intersect(promoter_bed, wa=True, c=True)
    peaks_utr5_bedgraph = peaks.intersect(utr5_bed, wa=True, c=True)
    peaks_utr3_bedgraph = peaks.intersect(utr3_bed, wa=True, c=True)
    data = []    
    for cpg, cds, intron, promoter, utr5, utr3 in zip(peaks_cpg_bedgraph,peaks_cds_bedgraph,peaks_intron_bedgraph,peaks_promoter_bedgraph,peaks_utr5_bedgraph,peaks_utr3_bedgraph):
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



def chroms_filter(feature, chroms):
    if feature.chrom in chroms:
        return True
    return False

def subset_chroms(chroms, bed):
    result = bed.filter(chroms_filter, chroms).saveas()
    return BedTool(result.fn)

genome_bed = get_genome_bed(genome_sizes_file)
train_bed = BedTool(bed_file)


gencode_feature_train = get_gencode_feature(train_bed,gencode_path)
np.savetxt(outfile,gencode_feature_train, fmt='%1.0f', delimiter='\t')

