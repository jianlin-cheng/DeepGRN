# DeepGRN
Deep learning for modeling gene regulatory network

## Required softwares:

```
Python 3.6.6
bedtools 2.26.0
```

Python modules:
```
pandas 0.24.2
numpy 1.16.2
tensorflow 1.11.0
pyfaidx 0.5.5.2
pyfasta 0.5.2
pybedtools 0.8.0
pyBigWig 0.3.14
Keras 2.2.4
h5py 2.9.0
deepTools 3.2.1
```

Optional(for training with GPUs):

```
cuda 10
tensorflow-gpu 1.11.0
```

## Training (train.py)
Train models for TF binding site prediction:


```
usage: train.py [-h] --data_dir DATA_DIR --tf_name TF_NAME --output_dir
                OUTPUT_DIR [--attention_position ATTENTION_POSITION]
                [--flanking FLANKING] [--val_chr VAL_CHR] [--epochs EPOCHS]
                [--patience PATIENCE] [--batch_size BATCH_SIZE]
                [--learningrate LEARNINGRATE] [--kernel_size KERNEL_SIZE]
                [--num_filters NUM_FILTERS] [--num_recurrent NUM_RECURRENT]
                [--num_dense NUM_DENSE] [--dropout_rate DROPOUT_RATE]
                [--rnn_dropout1 RNN_DROPOUT1] [--rnn_dropout2 RNN_DROPOUT2]
                [--merge MERGE] [--num_conv NUM_CONV] [--num_lstm NUM_LSTM]
                [--num_denselayer NUM_DENSELAYER]
                [--ratio_negative RATIO_NEGATIVE] [--rnaseq] [--gencode]
                [--unique35] [--use_peak] [--use_cudnn]
                [--single_attention_vector] [--plot_model]
                [--random_seed RANDOM_SEED]
```                

### Arguments

* `-h, --help`            show this help message and exit
* `--data_dir DATA_DIR, -i DATA_DIR`
                        data_dir
* `--tf_name TF_NAME, -t TF_NAME`
                        tf_name
* `--output_dir OUTPUT_DIR, -o OUTPUT_DIR`
                        output_dir
* `--attention_position ATTENTION_POSITION, -ap ATTENTION_POSITION`
                        Position of attention layers, can be attention1d_after_lstm, attention_before_lstm,attention_after_lstm,attention1d_after_lstm
* `--flanking FLANKING, -f FLANKING`
                        flanking
* `--val_chr VAL_CHR, -v VAL_CHR`
                        val_chr
* `--epochs EPOCHS, -e EPOCHS`
                        epochs
* `--patience PATIENCE, -p PATIENCE`
                        patience
* `--batch_size BATCH_SIZE, -s BATCH_SIZE`
                        batch_size
* `--learningrate LEARNINGRATE, -l LEARNINGRATE`
                        learningrate
* `--kernel_size KERNEL_SIZE, -k KERNEL_SIZE`
                        kernel_size for Conv1D
* `--num_filters NUM_FILTERS, -nf NUM_FILTERS`
                        num_filters for Conv1D
* `--num_recurrent NUM_RECURRENT, -nr NUM_RECURRENT`
                        Output dim for LSTM
* `--num_dense NUM_DENSE, -nd NUM_DENSE`
                        Output dim for dense layers
* `--dropout_rate DROPOUT_RATE, -d DROPOUT_RATE`
                        dropout_rate for all layers except LSTM
* `--rnn_dropout1 RNN_DROPOUT1, -rd1 RNN_DROPOUT1`
                        dropout_rate for LSTM
* `--rnn_dropout2 RNN_DROPOUT2, -rd2 RNN_DROPOUT2`
                        rnn_dropout_rate for LSTM
* `--merge MERGE, -me MERGE`
                        merge method, max or ave
* `--num_conv NUM_CONV, -nc NUM_CONV`
                        Number of Conv1D layers
* `--num_lstm NUM_LSTM, -nl NUM_LSTM`
                        Number of LSTM layers
* `--num_denselayer NUM_DENSELAYER, -dl NUM_DENSELAYER`
                        Number of additional dense layers
* `--ratio_negative RATIO_NEGATIVE, -rn RATIO_NEGATIVE`
                        Ratio of negative samples to positive samples in each epoch
* `--rnaseq, -r`          rnaseq
* `--gencode, -g`         gencode
* `--unique35, -u`        unique35
* `--use_peak, -a`        use_peak
* `--use_cudnn, -c`       use cudnnLSTM instead of LSTM, faster but will disable LSTM dropouts
* `--single_attention_vector, -sa`
                        merge attention weights in each position by averaging
* `--plot_model, -pl     plot model as png file`
* `--random_seed RANDOM_SEED, -rs RANDOM_SEED`
                        random seed

## Prediction (predict.py)

Use models trained from train.py for new data prediction:


```
usage: predict.py [-h] --data_dir DATA_DIR --model_file MODEL_FILE --cell_name
                  CELL_NAME --predict_region_file PREDICT_REGION_FILE
                  --output_predict_path OUTPUT_PREDICT_PATH
                  [--batch_size BATCH_SIZE] [--blacklist_file BLACKLIST_FILE]
```

Predict a model.

### Arguments

* `-h, --help`            show this help message and exit
* `--data_dir DATA_DIR, -i DATA_DIR`
                        data_dir
* `--model_file MODEL_FILE, -m MODEL_FILE`
                        model_file
* `--cell_name CELL_NAME, -c CELL_NAME`
                        cell_name
* `--predict_region_file PREDICT_REGION_FILE, -p PREDICT_REGION_FILE`
                        predict_region_file
* `--output_predict_path OUTPUT_PREDICT_PATH, -o OUTPUT_PREDICT_PATH`
                        output_predict_path
* `--batch_size BATCH_SIZE, -b BATCH_SIZE`
                        batch_size
* `--blacklist_file BLACKLIST_FILE, -l BLACKLIST_FILE`
                        blacklist_file to use, no fitering if not provided

## Prepare your input data for prediction:

### Target region

The region to predict should be a bed format file indicate the region you wish to predict along the chromosome. If you use our trained model, we recommend you consider use 200bp for each prediction since the models are trained using this format. Example:

```
chr1	600	800
chr1	650	850
chr1	700	900
chr1	750	950
chr1	800	1000
```

### Genomic sequence

Genomic sequence is provided as fasta format. You can download these files from [UCSC Genome Browser](https://hgdownload.soe.ucsc.edu/downloads.html)
Please notice that if your validation chromosome name should be one of the chromosome names in this file.

### Chromatin accessibility data

Chromatin accessibility data should be prepared as BigWig format from DNase-Seq experiment. You should name your files as {cell_type}.1x.bw and put them into your/data/folder/DNase/  That is to say, one file for each cell type. These files should be generated from the read alignment files in the standard BAM file format. If you have replicates for the same cell type, you should first merge them with samtools:

```
samtools merge {cell_type}.bam {cell_type}.rep1.bam {cell_type}.rep2.bam
samtools index {cell_type}.bam
```

Then you can run the bamCoverage in [deeptools](https://deeptools.readthedocs.io/en/develop/content/tools/bamCoverage.html) to generate the required .bw file for each cell type. Here we use human genome as example:

`bamCoverage --bam ${i}.bam -o ${i}.1x.bw --outFileFormat bigwig --normalizeTo1x 2478297382 --ignoreForNormalization chrX chrM --Offset 1 --binSize 1  --blackListFileName blacklist.bed.gz --skipNonCoveredRegions`

You can provide a BED or GTF file containing regions that should be excluded from bamCoverage analyses. For human, we use the [low mapability region provided by UCSC Genome Browser](http://hgdownload.cse.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeMapability/wgEncodeDacMapabilityConsensusExcludable.bed.gz)

### Sequence uniqueness data [optional]
According to [UCSC Genome Browser](http://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg18&g=wgEncodeMapability). We use the Duke uniqueness score as an additional input. The Duke excluded regions track displays genomic regions for which mapped sequence tags were filtered out before signal generation and peak calling for Duke/UNC/UTA's Open Chromatin tracks. This track contains problematic regions for short sequence tag signal detection (such as satellites and rRNA genes). The Duke excluded regions track was generated for the ENCODE project.

The Duke uniqueness score tracks display how unique is each sequence on the positive strand starting at a particular base and of a particular length. Thus, the 35 bp track reflects the uniqueness of all 35 base sequences with the score being assigned to the first base of the sequence. Scores are normalized to between 0 and 1 with 1 representing a completely unique sequence and 0 representing the sequence occurs >4 times in the genome (excluding chrN_random and alternative haplotypes). A score of 0.5 indicates the sequence occurs exactly twice, likewise 0.33 for three times and 0.25 for four times. For Human genome, you can use the [Duke uniqueness tracks](http://hgdownload.soe.ucsc.edu/goldenPath/hg18/encodeDCC/wgEncodeMapability/wgEncodeDukeUniqueness35bp.wig.gz)  generated for the ENCODE project as tools in the development of the Open Chromatin tracks. For other genomes, you can generate your own sequence uniqueness data following this rule and use [wigToBigWig](https://www.encodeproject.org/software/wigtobigwig/) to convert it to the BigWig format.

### Gene expression profile [optional]
Follwing the [FactorNet](https://www.sciencedirect.com/science/article/pii/S1046202318303293) framework, we use the first eight principal components from the RNA-Seq experiments for each cell type. These principal components should be generated using the TPM(Transcripts Per Kilobase Million) with replicates merged by averging. You should save this input in your/data/folder/rnaseq_data.csv as a Comma Separated Values file with the first line indicate the type of the cells.

We provided a simple R script data/generate_pca.R to generate this data from TPM data. The input of this script should be a csv file containing a gene by cell type matrix with first row indicate the types of cell and the first column indicate the gene names. Usage:

`Rscript generate_pca.R [path/to/input.csv] [path/to/rnaseq_data.csv]`

### Genomic annotations [optional]
The annotation feature for each bin is encoded as a binary vector of length 6, with each value representing if there is an overlap between the input bin and each of the six genomic features (coding regions, intron, promoter, 5'/3'-UTR, and CpG island). For Human genome Hg19, we use annotations from the [FactorNet GitHub repo](https://github.com/uci-cbcl/FactorNet/tree/master/resources) to generate the required input and save them in data/hg19. For preparing this input for your own genome, you can use our python script: data/generate_annotation.py Usage:

`Python generate_annotation.py [gencode_path] [genome_sizes_file] [bed_file] [outfile]`

You should prepare your own (gzipped) bed files indicting the genomic features (coding regions, intron, promoter, 5'/3'-UTR, and CpG island) and save them as [cpgisland|cds|intron|promoter|utr5|utr3].bed.gz under the [gencode_path]. 

genome_sizes_file is the file containing size of each chromosome. For Hg19 the content is:

```
chr1	249250621
chr10	135534747
chr11	135006516
chr12	133851895
chr13	115169878
chr14	107349540
chr15	102531392
chr16	90354753
chr17	81195210
chr18	78077248
chr19	59128983
chr2	243199373
chr20	63025520
chr21	48129895
chr22	51304566
chr3	198022430
chr4	191154276
chr5	180915260
chr6	171115067
chr7	159138663
chr8	146364022
chr9	141213431
chrX	155270560
```

bed_file is the file indicating the target region and should be the same as the one you use for the prediction script.


## Prepare label data for custom training:

### label data

If you need to train your own models with custom data, instead of preparing a file indicating the target region, you should prepare a label file indicating the true labels of the binding status along the chromosome. In the DREAM-ENCODE challange, each bin falls into one of the three types: bound(B), unbound(U), or ambiguous(A), which is determined from the ChIP-Seq results. Bins overlapping with peaks and passing the Irreproducible Discovery Rate (IDR) check with a threshold of 5%  are labeled as bound. Bins that overlap with peaks but fail to pass the reproducibility threshold are labeled as ambiguous. All other bins are labeled as unbound. We do not use any ambiguous bins during the training or validation process according to the common practice. Therefore, each bin in the genomic sequence will either be a positive site (bounded) or a negative site (unbounded).
Example:

```
chr     start   stop    A549    H1-hESC HeLa-S3 HepG2   IMR-90  K562    MCF-7
chr10   600     800     U       U       U       U       U       U       U
chr10   650     850     U       U       U       U       U       U       U
chr10   700     900     U       U       U       U       U       U       U
chr10   750     950     U       U       U       U       U       U       U
chr10   800     1000    U       U       U       U       U       U       U

```

For each TF, you will need one such label file. You should save these (gzipped) label files under data_dir/label/train/ and name them as [TF_name].train.labels.tsv.gz

### Narrowpeak files for data augmentation during training [optional]

For TFs that have abundant binding sites, training performance and speed could benefit from sampling positive sample regions from the ChIP-Seq peak regions during each epoch. To do so, you would need to prepare a peak annotation file. We have provided the peak annotations as data/hg19/peak_annotation.gz for the TF and cells we used. For your own customized input, you can use data/generate_peaks.py to generate those peak annotations. Usage:

`Python generate_peaks.py [gencode_path] [genome_sizes_file] [bed_file] [outfile] [narrowPeak_path] [blacklist_file] [label_path] [label_peak_path] [genome_window_size] [flanking]`

[gencode_path] [genome_sizes_file] [bed_file] [outfile] are the same as described in the "Genomic annotations" section. 

narrowPeak_path is the path you store the ChIP-Seq narrowPeak files. In our experiment, these peaks are the reproducible peaks across pseudo-replicates that pass the 10% IDR threshold. The peaks are provided in the [narrowPeak format](https://genome.ucsc.edu/FAQ/FAQformat.html#format12). For each cell type and TF pair, the file  should be named as  ChIPseq.[train_cell].[tf_name].conservative.train.narrowPeak.gz'

blacklist_file is the file you want to exclude during analysis and can be empty if you do not wish to miss any regions

label_path is where the label files are stored. They should be located in data_dir/label/train/

label_peak_path is where the output annotation files are stored. They should be located in data_dir/label/train_positive/

genome_window_size is the size of each bin in your label files. In our experiment we set it to 200.

flanking is the length of upstream and downstream region that around your training sample. For example, we use 401.

## Generate demo data with single script:

get_rawdata/get_data.sh provided a demo to generate all the required data for the Dream-Encode Challenge 2016

```
bash get_rawdata/get_data.sh /home/yourname/data/inputs/
```

## Generate step by step:
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







