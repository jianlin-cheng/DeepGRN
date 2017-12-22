optimizers = c('sgd','rmsprop','adagrad','adadelta','adam','adamax','nadam')

slurm_gpu_long = '/storage/htc/bdm/ccm3x/slurm_configs/gpu_sub_long.sbatch'
slurm_gpu_short = '/storage/htc/bdm/ccm3x/slurm_configs/gpu_sub_short.sbatch'
slurm_cpu = '/storage/htc/bdm/ccm3x/slurm_configs/cpu_sub.sbatch'

run_src = '/storage/htc/bdm/ccm3x/src/run_cnn_2dout.py'

# for(i in optimizers){
#   system(paste('sbatch',slurm_gpu_long,run_src,1,2,i))
# }

for(i in 1:10){
  for(j in 1:10){
    system(paste('sbatch',slurm_cpu,run_src,i,j,'rmsprop'))
  }
}