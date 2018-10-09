DATAPATH=/storage/htc/bdm/ccm3x/deepGRN/raw/

mkdir -p $DATAPATHbams
mkdir -p $DATAPATHDNase
mkdir -p $DATAPATHlabel/train
mkdir -p $DATAPATHlabel/final

# get data from synapse
source ~/py2/bin/activate
python get_syn_data.py $DATAPATH
gzip -d $DATAPATHlabel/train/*.gz
gzip -d $DATAPATHlabel/final/*.gz

# get unique 35 data
cd $DATAPATH
wget http://hgdownload.cse.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeMapability/wgEncodeDukeMapabilityUniqueness35bp.bigWig

# generate 2 missing bigwig files(A549 and IMR-90) from bams
module load samtools

cd $DATAPATHbams
samtools merge A549.bam DNASE.A549.*
samtools index A549.bam
bamCoverage --bam A549.bam -o ../DNase/A549.1x.bw --outFileFormat bigwig --normalizeTo1x 2478297382 --ignoreForNormalization chrX chrM --Offset 1 --binSize 1 --numberOfProcessors 12 --skipNonCoveredRegions

samtools merge IMR-90.bam DNASE.IMR90.*
samtools index IMR-90.bam 
bamCoverage --bam IMR-90.bam -o ../DNase/IMR-90.1x.bw --outFileFormat bigwig --normalizeTo1x 2478297382 --ignoreForNormalization chrX chrM --Offset 1 --binSize 1 --numberOfProcessors 12 --skipNonCoveredRegions
