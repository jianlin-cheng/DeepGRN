## Feature importance analysis

predict_feature_importance.py: Predict without using one specific feature

For the prediction of each target, we set the values of each sequential feature (DNase-Seq, sequence, or uniqueness) to zero, or randomly switch the order of the vector for a non-sequential feature (genomic elements or RNA-Seq). The decrease of auPRC from these new predictions is used as the importance score of each feature

### Arguments

  * `-h, --help`            show this help message and exit
  * `--data_dir DATA_DIR, -i DATA_DIR`
                        path to the input data (required)
  * `--model_file MODEL_FILE, -m MODEL_FILE`
                        path to model file (required)
  * `--cell_name CELL_NAME, -c CELL_NAME`
                        cell name (required)
  * `--predict_region_file PREDICT_REGION_FILE, -p PREDICT_REGION_FILE`
                        predict region file (required)
  * `--output_predict_path OUTPUT_PREDICT_PATH, -o OUTPUT_PREDICT_PATH`
                        output path of prediction (required)
  * `--bigwig_file_unique35 BIGWIG_FILE_UNIQUE35, -bf BIGWIG_FILE_UNIQUE35`
                        35bp uniqueness file  (default: '')
  * `--rnaseq_data_file RNASEQ_DATA_FILE, -rf RNASEQ_DATA_FILE`
                        RNA-Seq PCA data file  (default: '')
  * `--gencode_file GENCODE_FILE, -gc GENCODE_FILE`
                        Genomic annotation file (default: '')
  * `--batch_size BATCH_SIZE, -b BATCH_SIZE`
                        batch size  (default: 512)
  * `--blacklist_file BLACKLIST_FILE, -l BLACKLIST_FILE`
                        blacklist_file to use, no fitering if not provided  (default: '')

## Extracting attention weights and DNase scores and sequence nearby from single and pairwise modules

extract_single1d_att_w.py, extract_single_att_w.py, extract_pair_att_w.py can be applied to corresponding attention model types

python extract_*.py [tf_name] [model_type] [data_dir] [model_file] [label_file] [pred_dir] [out_fa_file] [att_avg_out] [dnase_out]

`tf_name`: name of TF

`model_type`: type of model, should be single or pair

`data_dir`: path to input data

`model_file`: path to model file

`label_file`: path to label file

`pred_dir`: path contains all predictions

`out_fa_file,att_avg_out,dnase_out`: path where the DNanse, attention score and high-score sequence are stored.

Both attention weights and saliency scores are compted from trained models of corresponding TFs. For
attention scores, since it has a shorter length than the fold change and DNase values due to the convolution
layers in the model, we downsampled those longer sequences to fit the length of the original input sequence
in order to calculate the correlation.

## Extracting attention weights, saliency scores and fold change values from single module

python generate_saliency.py [tf_name] [cell_name] [data_dir] [fc_data_dir] [model_file] [bigwig_file_unique35] [model_summary_file] [label_file_path] [out_dir]

`tf_name`: name of TF

`cell_name`: type of cell

`data_dir`: path to input data

`fc_data_dir`:  Fold-enrichment signal coverage tracks: Genome-wide signal coverage tracks are provided in Bigwig format and contain fold-enrichment of ChIP-seq signal with respect to control at each nucleotide in the training chromosomes. ENCODE-DREAM data can be downloaded [here](https://www.synapse.org/#!Synapse:syn6181334)

`model_file`: path of model file

`bigwig_file_unique35`: path of 35bp uniqueness file

`model_summary_file`: CSV format summary file contain the model recontruction information so that the script can rebuild part of the model to extract saliency scores. Example:

```
tf_name,batch_size,learningrate,kernel_size,num_filters,num_recurrent,num_dense,dropout_rate,rnn_dropout1,rnn_dropout2,merge,num_conv,num_lstm,num_denselayer,ratio_negative,rnaseq,gencode,unique35,use_peak,use_cudnn,attention_position,single_attention_vector
CTCF,64,0.0001,20,64,64,64,0.1,0.5,0.1,ave,2,2,0,1,TRUE,TRUE,TRUE,TRUE,TRUE,attention_after_lstm,FALSE
E2F1,64,1.00E-05,30,128,64,64,0.1,0.1,0.1,ave,0,1,1,1,TRUE,TRUE,TRUE,FALSE,FALSE,attention_before_lstm,TRUE
EGR1,64,0.001,30,64,32,64,0.5,0.5,0.5,ave,1,1,1,1,TRUE,TRUE,TRUE,FALSE,FALSE,attention_before_lstm,TRUE
FOXA1,64,0.001,34,128,64,128,0.5,0.5,0.1,max,1,1,0,10,TRUE,TRUE,TRUE,TRUE,FALSE,attention1d_after_lstm_original,TRUE
FOXA2,64,0.001,30,64,32,128,0.5,0,0,ave,1,1,1,1,TRUE,TRUE,TRUE,FALSE,FALSE,attention1d_after_lstm_original,TRUE
GABPA,64,0.001,20,64,64,64,0.1,0.5,0.1,ave,1,1,1,1,TRUE,TRUE,TRUE,FALSE,FALSE,attention_after_lstm,TRUE
HNF4A,32,0.001,34,64,64,128,0.5,0.5,0.1,ave,1,1,0,1,TRUE,TRUE,TRUE,TRUE,FALSE,attention1d_after_lstm_original,TRUE
JUND,32,0.01,20,64,64,128,0.1,0.1,0.1,ave,1,2,1,1,TRUE,TRUE,TRUE,TRUE,FALSE,attention_after_lstm,TRUE
MAX,32,0.001,34,64,64,64,0.1,0.1,0.1,max,1,1,0,1,TRUE,TRUE,TRUE,TRUE,FALSE,attention1d_after_lstm_original,TRUE
NANOG,128,0.0005,15,64,64,64,0.1,0,0,ave,1,1,1,15,FALSE,FALSE,FALSE,FALSE,FALSE,attention1d_after_lstm_original,TRUE
REST,32,0.001,34,128,32,128,0.1,0.1,0.1,max,1,1,0,10,TRUE,TRUE,TRUE,TRUE,FALSE,attention1d_after_lstm_original,TRUE
TAF1,64,0.001,34,64,64,64,0.1,0.5,0.1,ave,1,2,0,5,TRUE,TRUE,TRUE,FALSE,FALSE,attention_before_lstm,TRUE
```

`out_dir`: output path

For saliency scores, the non-sequential input features are considered as weights that are not trainable and only the
partial derivative of output values to the sequential features are computed using the tf.gradients() function in
Tensorflow. Since both attention weights and saliency scores are high dimensional vectors and generally consist
only noisy values in many of their channels. We only use the channel with the highest correlation with the
fold change value for both attention scores and saliency scores for a fair comparison of their correlation.


## Motif detection over high attention scores regions.

Rscript plot_heatmap_attweights.R [tf_name] [result_path] [fimo_path] [model_type]

`tf_name`: name of the transcription factor

`result_path`: path where the DNanse, attention score and high-score sequence are stored. These results are generated from the extraction script in the same folder

`fimo_path`: path of the FIMO program, can be downloaded at http://meme-suite.org/doc/download.html

`model_type`: module type, should be one of single/pair


This R script is designed to illustrate the ability of attention weights to recognize motifs that contribute to
binding events from the genomic sequences.

For those positive samples without distinct DNase-Seq peaks, the patterns of genomic sequences are critical information for successful prediction. To test the ability of attention weights to recognize motifs that contribute to binding events from the genomic sequences, we first acquire the coordinates on the relative positions of maximum column sum of the attention weights from all positive bins in test datasets and extract a subsequence with a length of 20bp around each coordinate. To exclude samples that can be easily classified from patterns of DNase-Seq signal, we only select positive bins that have no significant coverage peaks (ratio between the highest score and average scores < 15). Then we run FIMO  to detect known motifs relevant to the TF of the model in the JASPAR database. We plot the attention scores of the samples that contain these subsequencesand the relative location of the regions with detected motifs in FIMO. In addition, we show that these maximum attention weights do not come from the DNase-Seq peaks near the motif regions by coincidence since no similar pattern is detected from the normalized DNase scores in the same regions .
