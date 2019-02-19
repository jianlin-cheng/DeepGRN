# DeepGRN
Deep learning for modeling gene regulatory network

train.py Train models for TF binding site prediction:

usage: train.py [-h] --data_dir DATA_DIR --tf_name TF_NAME --output_dir
                OUTPUT_DIR [--model_type MODEL_TYPE] [--flanking FLANKING]
                [--val_chr VAL_CHR] [--epochs EPOCHS] [--patience PATIENCE]
                [--batch_size BATCH_SIZE] [--learningrate LEARNINGRATE]
                [--kernel_size KERNEL_SIZE] [--num_filters NUM_FILTERS]
                [--num_recurrent NUM_RECURRENT] [--num_dense NUM_DENSE]
                [--dropout_rate DROPOUT_RATE] [--rnn_dropout RNN_DROPOUT]
                [--merge MERGE] [--num_conv NUM_CONV] [--num_lstm NUM_LSTM]
                [--num_denselayer NUM_DENSELAYER]
                [--ratio_negative RATIO_NEGATIVE] [--rnaseq] [--gencode]
                [--unique35] [--use_peak] [--use_cudnn]
                [--single_attention_vector]


predict.py Use models trained from train.py for new data prediction:

usage: predict.py [-h] --data_dir DATA_DIR --model_file MODEL_FILE --cell_name
                  CELL_NAME --predict_region_file PREDICT_REGION_FILE
                  --output_predict_path OUTPUT_PREDICT_PATH
                  [--batch_size BATCH_SIZE]
                  
