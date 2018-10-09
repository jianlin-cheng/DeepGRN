gencode_path = '/storage/htc/bdm/ccm3x/deepGRN/raw/gencode/'

def get_gencode_feature(bed_input,outfile,gencode_path):
    from pybedtools import BedTool
    import numpy as np
    import itertools
    
    cpg_bed = BedTool(gencode_path+'cpgisland.bed.gz')
    cds_bed = BedTool(gencode_path+'wgEncodeGencodeBasicV19.cds.merged.bed.gz')
    intron_bed = BedTool(gencode_path+'wgEncodeGencodeBasicV19.intron.merged.bed.gz')
    promoter_bed = BedTool(gencode_path+'wgEncodeGencodeBasicV19.promoter.merged.bed.gz')
    utr5_bed = BedTool(gencode_path+'wgEncodeGencodeBasicV19.utr5.merged.bed.gz')
    utr3_bed = BedTool(gencode_path+'wgEncodeGencodeBasicV19.utr3.merged.bed.gz')
    peaks= BedTool(bed_input)

    peaks_cpg_bedgraph = peaks.intersect(cpg_bed, wa=True, c=True)
    peaks_cds_bedgraph = peaks.intersect(cds_bed, wa=True, c=True)
    peaks_intron_bedgraph = peaks.intersect(intron_bed, wa=True, c=True)
    peaks_promoter_bedgraph = peaks.intersect(promoter_bed, wa=True, c=True)
    peaks_utr5_bedgraph = peaks.intersect(utr5_bed, wa=True, c=True)
    peaks_utr3_bedgraph = peaks.intersect(utr3_bed, wa=True, c=True)

    data = []    
    for cpg, cds, intron, promoter, utr5, utr3 in itertools.izip(peaks_cpg_bedgraph,peaks_cds_bedgraph,peaks_intron_bedgraph,peaks_promoter_bedgraph,peaks_utr5_bedgraph,peaks_utr3_bedgraph):
        gencode = np.array([cpg.count, cds.count, intron.count, promoter.count, utr5.count, utr3.count], dtype=bool)
        gencode = np.array(gencode,dtype=int).tolist()
        data.append(gencode)
    np.savetxt(outfile, np.array(data), fmt='%.2e',delimiter="\t")

bed_input = '/storage/htc/bdm/ccm3x/deepGRN/raw/label/train_regions.blacklistfiltered.bed'
outfile = "/storage/htc/bdm/ccm3x/deepGRN/raw/gencode_feature_train.tsv"
get_gencode_feature(bed_input,outfile,gencode_path)

bed_input = '/storage/htc/bdm/ccm3x/deepGRN/raw/label/test_regions.blacklistfiltered.bed'
outfile = "/storage/htc/bdm/ccm3x/deepGRN/raw/gencode_feature_val.tsv"
get_gencode_feature(bed_input,outfile,gencode_path)