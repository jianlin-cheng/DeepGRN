## Generate demo data 

### With a single script:

get_rawdata/get_data.sh provided a demo to generate all the required data for the Dream-Encode Challenge 2016

```
bash get_rawdata/get_data.sh /home/yourname/data/inputs/
```

### Step by step:
Assuming all feature data will be generated at $DATAPATH (e.g. /home/yourname/data/inputs/)

Create directory structures

```
mkdir -p $DATAPATHbams
mkdir -p $DATAPATHDNase
mkdir -p $DATAPATHchipseq
mkdir -p $DATAPATHgencode
mkdir -p $DATAPATHlabel/train
mkdir -p $DATAPATHlabel/leaderboard
mkdir -p $DATAPATHlabel/final
mkdir -p $DATAPATHlabel/train_positive
mkdir -p $DATAPATHlabel/train_negative
```


Download annotation files for genomic elements

```
cd $DATAPATHgencode
wget https://github.com/uci-cbcl/FactorNet/raw/master/resources/wgEncodeGencodeBasicV19.cds.merged.bed.gz
wget https://github.com/uci-cbcl/FactorNet/raw/master/resources/wgEncodeGencodeBasicV19.intron.merged.bed.gz
wget https://github.com/uci-cbcl/FactorNet/raw/master/resources/wgEncodeGencodeBasicV19.promoter.merged.bed.gz
wget https://github.com/uci-cbcl/FactorNet/raw/master/resources/wgEncodeGencodeBasicV19.txStart.bed.gz
wget https://github.com/uci-cbcl/FactorNet/raw/master/resources/wgEncodeGencodeBasicV19.utr3.merged.bed.gz
wget https://github.com/uci-cbcl/FactorNet/raw/master/resources/wgEncodeGencodeBasicV19.utr5.merged.bed.gz
```

Get data from synapse(ChIP-Seq files, DNase files)

```
python get_syn_data.py $DATAPATH
```

Generate region file for prediction

```
cd $DATAPATHlabel/
gzip -d *.gz
awk '$1 == "chr1" {print}' test_regions.blacklistfiltered.bed > predict_region.bed
awk '$1 == "chr21" {print}' test_regions.blacklistfiltered.bed >> predict_region.bed
awk '$1 == "chr8" {print}' test_regions.blacklistfiltered.bed >> predict_region.bed
```

Download blacklist file, genome file, and uniqueness file from FactorNet repository since we also need to process the data by the way Factornet did.

```
cd $DATAPATH
wget https://github.com/uci-cbcl/FactorNet/raw/master/resources/blacklist.bed.gz
wget https://raw.githubusercontent.com/uci-cbcl/FactorNet/master/resources/hg19.autoX.chrom.sizes
# get unique 35 data
wget http://hgdownload.cse.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeMapability/wgEncodeDukeMapabilityUniqueness35bp.bigWig
wget http://hgdownload.cse.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeMapability/wgEncodeDacMapabilityConsensusExcludable.bed.gz
```

Generate 2 missing bigwig files(A549 and IMR-90) from bams

```
cd $DATAPATHbams
samtools merge A549.bam DNASE.A549.*
samtools index A549.bam
bamCoverage --bam A549.bam -o ../DNase/A549.1x.bw --outFileFormat bigwig --normalizeTo1x 2478297382 --ignoreForNormalization chrX chrM --Offset 1 --binSize 1 --numberOfProcessors 12 --skipNonCoveredRegions --blackListFileName ../wgEncodeDacMapabilityConsensusExcludable.bed.gz

samtools merge IMR-90.bam DNASE.IMR90.*
samtools index IMR-90.bam 
bamCoverage --bam IMR-90.bam -o ../DNase/IMR-90.1x.bw --outFileFormat bigwig --normalizeTo1x 2478297382 --ignoreForNormalization chrX chrM --Offset 1 --binSize 1 --numberOfProcessors 12 --skipNonCoveredRegions --blackListFileName ../wgEncodeDacMapabilityConsensusExcludable.bed.gz
```


Generate masks for genomic elements and blacklists, these can be user defined as long as they are binary(0,1) vectors with the same length with the training/prediction bins. For genomic elements, 0 means not overlapping and 1 means overlapping for each bin. For blacklists zero means overlapping with black list. Those bins will not be trained and force to be 0 in prediction.

```
cd $DATAPATH../src
python get_mask_files.py
```
