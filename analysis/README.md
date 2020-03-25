## Predict without using one specific feature

(predict_feature_importance.py)

```

--feature_importance feature_name, -f feature_name the feature you want to exclude

```




## Extracting attention weights and saliency scores from single and pairwise modules

(extract_single1d_att_w.py, extract_single_att_w.py, extract_pair_att_w.py, generate_saliency.py)

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
(plot_heatmap_attweights.R)
This R script is designed to illustrate the ability of attention weights to recognize motifs that contribute to
binding events from the genomic sequences.