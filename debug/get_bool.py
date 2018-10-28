tf_name = re.sub("(\..+)$", "", tf_file)
label_train = pd.read_csv(data_path+'label/train/'+tf_name+'.train.labels.tsv',sep='\t',nrows=3)
train_cells = list(label_train.columns.values)[3:label_train.shape[1]]
pos_labels = pd.DataFrame(None)
cell_idx = 0
for train_cell in train_cells:
    relax_bed = BedTool(data_path+'chipseq/ChIPseq.'+train_cell+'.'+tf_name+'.relaxed.narrowPeak.gz')
    relax_bed = relax_bed.sort()
    relax_bed_slop = relax_bed.slop(g=genome_sizes_file, b=genome_window_size)
    nonnegative_bed = blacklist_bed.cat(relax_bed_slop)
    nonnegative_bed = nonnegative_bed.sort()
#        negative_bed  = train_bed.intersect(nonnegative_bed, wa=True, v=True, sorted=True,c=True)
#        negative_bed_df = pd.read_table(negative_bed.fn, names=['chrom', 'start', 'stop'],usecols =[0,1,2])
#        neg_labels = pd.concat((negative_bed_df,pd.DataFrame(gencode_negative+0)))
    conserv_bed = BedTool(data_path+'chipseq/ChIPseq.'+train_cell+'.'+tf_name+'.conservative.train.narrowPeak.gz')
    conserv_bed = conserv_bed.sort()
    conserv_bed_filtered = conserv_bed.intersect(blacklist_bed, wa=True, v=True, sorted=True)
    gencode_positive = get_gencode_feature(conserv_bed_filtered,gencode_path)
    conserv_bed_filtered_df = pd.read_table(conserv_bed_filtered.fn, names=['chrom', 'start', 'stop'],usecols =[0,1,2])
    pos_labels = pd.concat((conserv_bed_filtered_df,
                    pd.DataFrame(np.zeros((conserv_bed_filtered_df.shape[0],1),dtype =int)+cell_idx),
                    pd.DataFrame(gencode_positive+0)),axis=1)
    if pos_labels.shape[0] == 0:
        pos_labels = pos_labels_cell
    else:
        pos_labels = pd.concat((pos_labels,pos_labels_cell))
    cell_idx +=1