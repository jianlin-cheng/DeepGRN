tf = 'NANOG'
input_dirs = ['data/H1-hESC']

L=1002
batch_size = 100

genome_sizes_file = 'resources/hg19.autoX.chrom.sizes'
genome_fasta_file = 'resources/hg19.fa'
blacklist_file = 'resources/blacklist.bed.gz'

genome_window_size = 200
genome_window_step = 50
shift_size = 20

from utils import *

valid_chroms = ['chr11']
valid_input_dirs = None
test_chroms = ['chr1', 'chr8', 'chr21']
epochs = 60
patience = 5
learningrate = 0.001
seed = 1
set_seed(seed)
dropout_rate = 0.1
L = 1000
w = 26

negatives = 1
meta = False
gencode = False
motif = False

num_motifs = 32
num_recurrent = 32
num_dense = 32
 
features = ['bigwig']    
    

def load_bigwigs(input_dirs):
    bigwig_names = None
    bigwig_files_list = []
    for input_dir in input_dirs:
        input_bigwig_info_file = input_dir + '/bigwig.txt'
        if not os.path.isfile(input_bigwig_info_file):
            input_bigwig_names = []
            input_bigwig_files_list = []
        else:
            input_bigwig_info = np.loadtxt(input_bigwig_info_file, dtype=str)
            if len(input_bigwig_info.shape) == 1:
                input_bigwig_info = np.reshape(input_bigwig_info, (-1,2))
            input_bigwig_names = list(input_bigwig_info[:, 1])
            input_bigwig_files = [input_dir + '/' + i for i in input_bigwig_info[:,0]]
        if bigwig_names is None:
            bigwig_names = input_bigwig_names
        else:
            assert bigwig_names == input_bigwig_names
        bigwig_files_list.append(input_bigwig_files)
    return bigwig_names, bigwig_files_list

import utils
utils.L = L


blacklist = make_blacklist()
chip_bed_list, relaxed_bed_list = zip(*parmap.map(get_chip_bed, input_dirs, tf, blacklist.fn))
nonnegative_regions_bed_file_list = parmap.map(nonnegative_wrapper, relaxed_bed_list, blacklist.fn)

num_cells = len(chip_bed_list)
chroms, chroms_sizes, genome_bed = get_genome_bed()
train_chroms = chroms
for chrom in valid_chroms + test_chroms:
    train_chroms.remove(chrom)
genome_bed_train, genome_bed_valid, genome_bed_test = \
    [subset_chroms(chroms_set, genome_bed) for chroms_set in
     (train_chroms, valid_chroms, test_chroms)]

print 'Splitting ChIP peaks into training, validation, and testing BEDs'
chip_bed_split_list = parmap.map(valid_test_split_wrapper, chip_bed_list, valid_chroms, test_chroms)
chip_bed_train_list, chip_bed_valid_list, chip_bed_test_list = zip(*chip_bed_split_list)

valid_chip_bed_list = None
valid_nonnegative_regions_bed_list = None
valid_bigwig_files_list = None
valid_meta_list = None 

valid_nonnegative_regions_bed_list = nonnegative_regions_bed_list
valid_bigwig_files_list = bigwig_files_list
valid_meta_list = meta_list


positive_label = [True]
#Train
print 'Extracting data from positive training BEDs'
positive_data_train_list = parmap.map(extract_data_from_bed,
                                      zip(chip_bed_train_list, bigwig_files_list, meta_list),
                                      True, positive_label, gencode)
