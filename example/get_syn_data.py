import synapseclient
import synapseutils
import sys

data_path = sys.argv[1]
syn = synapseclient.Synapse()
syn.login()

# bigwig files from factornet
syn_download = syn.get('syn8073652', downloadLocation=data_path+'/DNase/')
syn_download = syn.get('syn8073583', downloadLocation=data_path+'/DNase/')
syn_download = syn.get('syn8074109', downloadLocation=data_path+'/DNase/')
syn_download = syn.get('syn8073618', downloadLocation=data_path+'/DNase/')
syn_download = syn.get('syn8073517', downloadLocation=data_path+'/DNase/')
syn_download = syn.get('syn8073483', downloadLocation=data_path+'/DNase/')
syn_download = syn.get('syn8073537', downloadLocation=data_path+'/DNase/')
syn_download = syn.get('syn8074140', downloadLocation=data_path+'/DNase/')
syn_download = syn.get('syn8074156', downloadLocation=data_path+'/DNase/')
syn_download = syn.get('syn8074090', downloadLocation=data_path+'/DNase/')
syn_download = syn.get('syn8074181', downloadLocation=data_path+'/DNase/')

# A549 and IMR-90 bam file for generating bigwig
syn_download = syn.get('syn7113332', downloadLocation=data_path+'/bams/')
syn_download = syn.get('syn7113353', downloadLocation=data_path+'/bams/')
syn_download = syn.get('syn7113334', downloadLocation=data_path+'/bams/')
syn_download = syn.get('syn6178045', downloadLocation=data_path+'/bams/')

syn_download = syn.get('syn6176276', downloadLocation=data_path+'/bams/')
syn_download = syn.get('syn6176274', downloadLocation=data_path+'/bams/')

# genome and labels
syn_download = syn.get('syn6184309', downloadLocation=data_path+'/')
syn_download = syn.get('syn6184311', downloadLocation=data_path+'/label/')
syn_download = syn.get('syn6184310', downloadLocation=data_path+'/label/')
syn_download = syn.get('syn6184317', downloadLocation=data_path+'/label/')
syn_download = syn.get('syn6184308', downloadLocation=data_path+'/label/')

# get chipseq peaks
syn_download = synapseutils.syncFromSynapse(syn, entity='syn6181337', path=data_path+'/chipseq/')
syn_download = synapseutils.syncFromSynapse(syn, entity='syn6181338', path=data_path+'/chipseq/')

# get labels for evaluation
files = synapseutils.syncFromSynapse(syn, 'syn10164048',path=data_path+'/label/final/')
files = synapseutils.syncFromSynapse(syn, 'syn10164047',path=data_path+'/label/leaderboard/')
files = synapseutils.syncFromSynapse(syn, 'syn7413983',path=data_path+'/label/train/')


