from utils import *
label_region=train_region
label_data = train_label_data
label_IDs_positive = make_ID_list_single(label_region,label_data)
label_IDs_negative = make_ID_list_single(label_region,label_data,'U')
bigwig_file_list = [DNase_path+'/' +s +'.1x.bw' for s in cell_list]

bigwig_file_list = [DNase_path+'/' +s +'.1x.bw' for s in label_data.columns.values.tolist()[3:len(label_data.columns.values.tolist())]]

np.random.shuffle(label_IDs_positive)
index = 0


batch_end = min((index+1)*(batch_size//2),len(label_IDs_positive))
positive_batch = label_IDs_positive[index*(batch_size//2):batch_end]
# mix with equal number of negative bins
negative_batch = label_IDs_negative[np.random.choice(label_IDs_negative.shape[0],size=len(positive_batch))]
indexes_batch = np.concatenate((positive_batch,negative_batch),axis=0)

# Generate data
bins_batch = np.vstack((indexes_batch[:,0]-bin_num//2,indexes_batch[:,0]+bin_num//2,indexes_batch[:,1]))

x_fw = make_x_batch(genome,bw_dict_unique35,bigwig_file_list,label_data,bins_batch,flanking)
x_rc = make_rc_x(x_fw)

rnaseq_batch = np.transpose(rnaseq_train[:,bins_batch[2,:]])
gencode_batch = gencode_data[bins_batch[0,:],:]
            
            

y = make_y_batch(label_data,bins_batch)




# prediction
batch_size =32
bin_num=5

seg_array = parse_bed_merge(bed_merge)
bigwig_file_list = [DNase_file]
seg_len = seg_array[:,1]-seg_array[:,0]+1
num_batch = sum([get_batch_num_seg(seg_size,bin_num,batch_size) for seg_size in seg_len])

t = 0
start_idx = 0
while(start_idx < label_data.shape[0]):
    seg_end = seg_array[np.logical_and(seg_array[:,0]<=start_idx, seg_array[:,1]>=start_idx),1][0]
    max_batch_size = (seg_end-start_idx+1)//bin_num
    if max_batch_size > 0:
        if max_batch_size >= batch_size:
            current_batch_size = batch_size
        else:
            current_batch_size = max_batch_size
        start_bin_idx = np.array([start_idx+i*bin_num for i in range(current_batch_size)])
        bins_batch = np.vstack((start_bin_idx,start_bin_idx+bin_num-1,np.zeros_like(start_bin_idx)))
        start_idx += bin_num*current_batch_size
    else:
        # this batch will only have size 1
        bins_batch = np.vstack((start_idx,seg_end,0))
        start_idx = seg_end + 1
    t +=1
    



x_fw = make_x_batch(genome,bw_dict_unique35,bigwig_file_list,label_data,bins_batch,flanking)

aaa=x_fw[None,0,range(300),:]
bbb = np.concatenate((aaa,np.zeros((1,50*bin_num+150-aaa.shape[1],aaa.shape[2]))),axis=1)


seg_array = utils.parse_bed_merge(bed_merge)
seg_array1 = seg_array
id_remove = []
for i in range(seg_array.shape[0]):
    m = (seg_array[i,1]-seg_array[i,0]+1) % bin_num
    if m > 0 :
        id_remove = id_remove + list(range(seg_array[i,1]+1,seg_array[i,1]+1+bin_num-m))
        seg_array = seg_array + bin_num - m
        
#pred = np.empty((len(datagen_val),bin_num,1))
pred_final = pred[:,:,0].flatten()
aaa=np.delete(pred_final,id_remove)




start_idx=11433
seg_end = seg_array[np.logical_and(seg_array[:,0]<=start_idx, seg_array[:,1]>=start_idx),1][0]
max_batch_size = (seg_end-start_idx+1)//bin_num
if max_batch_size > 0:
    if max_batch_size >= batch_size:
        current_batch_size = batch_size
        
    else:
        current_batch_size = max_batch_size
    start_bin_idx = np.array([start_idx+i*bin_num for i in range(current_batch_size)])
    bins_batch = np.vstack((start_bin_idx,start_bin_idx+bin_num-1,np.zeros_like(start_bin_idx)))
    start_idx += bin_num*current_batch_size
    
else:
    # this batch will only have size 1
    bins_batch = np.vstack((start_idx,seg_end,0))
    start_idx = seg_end + 1
bigwig_file_list = [DNase_file]
x_fw = utils.make_x_batch(genome,bw_dict_unique35,bigwig_file_list,label_data,bins_batch,flanking)

if (x_fw.shape[1] != 50*bin_num+150+flanking*2) & multibin:
    x_fw = np.concatenate((x_fw,np.zeros((1,50*bin_num+150-x_fw.shape[1],x_fw.shape[2]))),axis=1)




















