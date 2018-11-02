#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition gpu3
#SBATCH --nodes=1
#SBATCH --ntasks=1         # leave at '1' unless using a MPI code
#SBATCH --cpus-per-task=1  # cores per task
#SBATCH --mem-per-cpu=32G  # memory per core (default is 1GB/core)
#SBATCH --time 1-23:15     # days-hours:minutes
#SBATCH --qos=normal
#SBATCH --account=general-gpu  # investors will replace this (e.g. `rc-gpu`)
#SBATCH --gres gpu:1

## labels and outputs
#SBATCH --job-name=cudnn_m
#SBATCH --output=results-%j.out  # %j is the unique jobID

#-------------------------------------------------------------------------------


echo "### Starting at: $(date) ###"
module load R/R-3.3.3
module load icu/icu-54.1 icu4c/icu4c-57.1 bedtools
module load cuda/cuda-9.0.176 cudnn/cudnn-7.2.1;

echo $*

cd /storage/htc/bdm/ccm3x/deepGRN/src

echo Enter py3_tf_gpu env...
source /storage/htc/bdm/ccm3x/py_env/py3_tf_gpu/bin/activate

echo start training...
echo python /storage/htc/bdm/ccm3x/deepGRN/src/train.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${24} ${25} ${26} ${27} ${28} -c 1
python /storage/htc/bdm/ccm3x/deepGRN/src/train.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${24} ${25} ${26} ${27} ${28} -c 1

echo start prediction...
echo python /storage/htc/bdm/ccm3x/deepGRN/src/model_predict.py ${19} ${20} ${21}
python /storage/htc/bdm/ccm3x/deepGRN/src/model_predict.py ${19} ${20} ${21}

deactivate
echo Leave py3_tf_gpu env...

echo Enter py2_theano_cpu env...
source /storage/htc/bdm/ccm3x/py_env/py2_theano_cpu/bin/activate

echo start evaluation...
echo Evaluating without blacklist...
echo python /storage/htc/bdm/ccm3x/deepGRN/src/score.py ${22} /storage/htc/bdm/ccm3x/deepGRN/raw/label/final ${23}
python /storage/htc/bdm/ccm3x/deepGRN/src/score.py ${22} /storage/htc/bdm/ccm3x/deepGRN/raw/label/final ${23}

echo Evaluating with blacklist...
echo python /storage/htc/bdm/ccm3x/deepGRN/src/score.py ${22} /storage/htc/bdm/ccm3x/deepGRN/raw/label/final ${23}.blacklisted -b /storage/htc/bdm/ccm3x/deepGRN/raw/nonblacklist_bools.csv
python /storage/htc/bdm/ccm3x/deepGRN/src/score.py ${22} /storage/htc/bdm/ccm3x/deepGRN/raw/label/final ${23}.blacklisted -b /storage/htc/bdm/ccm3x/deepGRN/raw/nonblacklist_bools.csv

echo add evaluation result to summary
Rscript /storage/htc/bdm/ccm3x/deepGRN/src/final_eval.R ${29}
