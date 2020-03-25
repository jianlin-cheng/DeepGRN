## Feature importance analysis

predict_feature_importance.py: Predict without using one specific feature

For the prediction of each target, we set the values of each sequential feature (DNase-Seq, sequence, or uniqueness) to zero, or randomly switch the order of the vector for a non-sequential feature (genomic elements or RNA-Seq). The decrease of auPRC from these new predictions is used as the importance score of each feature

### Arguments

  * `-h, --help`            show this help message and exit
  * `--data_dir DATA_DIR, -i DATA_DIR`
                        path to the input data
  * `--model_file MODEL_FILE, -m MODEL_FILE`
                        path to model file
  * `--cell_name CELL_NAME, -c CELL_NAME`
                        cell name
  * `--predict_region_file PREDICT_REGION_FILE, -p PREDICT_REGION_FILE`
                        predict region file
  * `--bigwig_file_unique35 BIGWIG_FILE_UNIQUE35, -bf BIGWIG_FILE_UNIQUE35`
                        35bp uniqueness file
  * `--rnaseq_data_file RNASEQ_DATA_FILE, -rf RNASEQ_DATA_FILE`
                        RNA-Seq PCA data file
  * `--gencode_file GENCODE_FILE, -gc GENCODE_FILE`
                        Genomic annotation file
  * `--output_predict_path OUTPUT_PREDICT_PATH, -o OUTPUT_PREDICT_PATH`
                        output path of prediction
  * `--batch_size BATCH_SIZE, -b BATCH_SIZE`
                        batch size
  * `--blacklist_file BLACKLIST_FILE, -l BLACKLIST_FILE`
                        blacklist_file to use, no fitering if not provided
  * `--feature_importance FEATURE_IMPORTANCE, -f FEATURE_IMPORTANCE`
                        name of the feature you want to exclude


## Extracting attention weights and saliency scores from single and pairwise modules

extract_single1d_att_w.py, extract_single_att_w.py, extract_pair_att_w.py, generate_saliency.py

Both attention weights and saliency scores are compted from trained models of corresponding TFs. For
saliency scores, the non-sequential input features are considered as weights that are not trainable and only the
partial derivative of output values to the sequential features are computed using the tf.gradients() function in
Tensorflow. Since both attention weights and saliency scores are high dimensional vectors and generally consist
only noisy values in many of their channels. We only use the channel with the highest correlation with the
fold change value for both attention scores and saliency scores for a fair comparison of their correlation. For
attention scores, since it has a shorter length than the fold change and DNase values due to the convolution
layers in the model, we downsampled those longer sequences to fit the length of the original input sequence
in order to calculate the correlation.


## Motif detection over high attention scores regions.

plot_heatmap_attweights.R

This R script is designed to illustrate the ability of attention weights to recognize motifs that contribute to
binding events from the genomic sequences.

For those positive samples without distinct DNase-Seq peaks, the patterns of genomic sequences are critical information for successful prediction. To test the ability of attention weights to recognize motifs that contribute to binding events from the genomic sequences, we first acquire the coordinates on the relative positions of maximum column sum of the attention weights from all positive bins in test datasets and extract a subsequence with a length of 20bp around each coordinate. To exclude samples that can be easily classified from patterns of DNase-Seq signal, we only select positive bins that have no significant coverage peaks (ratio between the highest score and average scores < 15). Then we run FIMO  to detect known motifs relevant to the TF of the model in the JASPAR database. We plot the attention scores of the samples that contain these subsequencesand the relative location of the regions with detected motifs in FIMO. In addition, we show that these maximum attention weights do not come from the DNase-Seq peaks near the motif regions by coincidence since no similar pattern is detected from the normalized DNase scores in the same regions .
