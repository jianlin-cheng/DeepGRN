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

### Arguments

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

optional arguments:
* `-h, --help`            show this help message and exit
* `data_dir DATA_DIR, -i DATA_DIR`
                        data_dir
* `tf_name TF_NAME, -t TF_NAME`
                        tf_name
* `output_dir OUTPUT_DIR, -o OUTPUT_DIR`
                        output_dir
* `attention_position ATTENTION_POSITION, -ap ATTENTION_POSITION`
                        Position of attention layers, can be attention1d_after_lstm, attention_before_lstm,attention_after_lstm,attention1d_after_lstm
* `flanking FLANKING, -f FLANKING`
                        flanking
* `val_chr VAL_CHR, -v VAL_CHR`
                        val_chr
* `epochs EPOCHS, -e EPOCHS`
                        epochs
* `patience PATIENCE, -p PATIENCE`
                        patience
* `batch_size BATCH_SIZE, -s BATCH_SIZE`
                        batch_size
* `learningrate LEARNINGRATE, -l LEARNINGRATE`
                        learningrate
* `kernel_size KERNEL_SIZE, -k KERNEL_SIZE`
                        kernel_size for Conv1D
* `num_filters NUM_FILTERS, -nf NUM_FILTERS`
                        num_filters for Conv1D
* `num_recurrent NUM_RECURRENT, -nr NUM_RECURRENT`
                        Output dim for LSTM
* `num_dense NUM_DENSE, -nd NUM_DENSE`
                        Output dim for dense layers
* `dropout_rate DROPOUT_RATE, -d DROPOUT_RATE`
                        dropout_rate for all layers except LSTM
* `rnn_dropout1 RNN_DROPOUT1, -rd1 RNN_DROPOUT1`
                        dropout_rate for LSTM
* `rnn_dropout2 RNN_DROPOUT2, -rd2 RNN_DROPOUT2`
                        rnn_dropout_rate for LSTM
* `merge MERGE, -me MERGE`
                        merge method, max or ave
* `num_conv NUM_CONV, -nc NUM_CONV`
                        Number of Conv1D layers
* `num_lstm NUM_LSTM, -nl NUM_LSTM`
                        Number of LSTM layers
* `num_denselayer NUM_DENSELAYER, -dl NUM_DENSELAYER`
                        Number of additional dense layers
* `ratio_negative RATIO_NEGATIVE, -rn RATIO_NEGATIVE`
                        Ratio of negative samples to positive samples in each epoch
* `rnaseq, -r`          rnaseq
* `gencode, -g`         gencode
* `unique35, -u`        unique35
* `use_peak, -a`        use_peak
* `use_cudnn, -c`       use cudnnLSTM instead of LSTM, faster but will disable LSTM dropouts
* `single_attention_vector, -sa`
                        merge attention weights in each position by averaging
* `plot_model, -pl     plot model as png file`
* `random_seed RANDOM_SEED, -rs RANDOM_SEED`
                        random seed

## Prediction (predict.py)

Use models trained from train.py for new data prediction:

### Arguments

```
usage: predict.py [-h] --data_dir DATA_DIR --model_file MODEL_FILE --cell_name
                  CELL_NAME --predict_region_file PREDICT_REGION_FILE
                  --output_predict_path OUTPUT_PREDICT_PATH
                  [--batch_size BATCH_SIZE] [--blacklist_file BLACKLIST_FILE]
```

Predict a model.

optional arguments:
  -h, --help            show this help message and exit
* `data_dir DATA_DIR, -i DATA_DIR`
                        data_dir
* `model_file MODEL_FILE, -m MODEL_FILE`
                        model_file
* `cell_name CELL_NAME, -c CELL_NAME`
                        cell_name
* `predict_region_file PREDICT_REGION_FILE, -p PREDICT_REGION_FILE`
                        predict_region_file
* `output_predict_path OUTPUT_PREDICT_PATH, -o OUTPUT_PREDICT_PATH`
                        output_predict_path
* `batch_size BATCH_SIZE, -b BATCH_SIZE`
                        batch_size
* `blacklist_file BLACKLIST_FILE, -l BLACKLIST_FILE`
                        blacklist_file to use, no fitering if not provided


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







