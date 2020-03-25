## Generate custom input data 

Suppose your data are stored at you/data/folder, it should have the following structure:

```bash
|-- label (If the task is training)
???   |-- train
???   ???   |-- *.train.labels.tsv.gz
???   |-- train_positive (if using positive bin sampling)
???       |-- *.train.peak.tsv
|-- DNase
???   |-- *.1x.bw
|-- rnaseq_data.csv (if using rna-seq expression feature)
|-- genomic_annotation.tsv (if using genomic annotations feature)
|-- sequence_uniqueness.bw (if using sequence uniqueness feature)
|-- hg19.genome.fa (or your custom genome)
```

You will also need to prepare a .bed format file specifying the region your target region if the task is prediction.

For detailed instructions for data preparation, please refer to [Prepare label data for custom training](../README.md#Prepare)