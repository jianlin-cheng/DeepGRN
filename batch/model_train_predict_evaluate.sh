#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition Lewis
#SBATCH --nodes=1
#SBATCH --ntasks=1         # leave at '1' unless using a MPI code
#SBATCH --cpus-per-task=4  # cores per task
#SBATCH --mem-per-cpu=8G  # memory per core (default is 1GB/core)
#SBATCH --time 1-23:15     # days-hours:minutes
#SBATCH --qos=normal
#SBATCH --account=general  # investors will replace this (e.g. `rc-gpu`)

## labels and outputs
#SBATCH --job-name=py_cpu
#SBATCH --output=results-%j.out  # %j is the unique jobID

## notifications
#SBATCH --mail-user=username@missouri.edu  # email address for notifications
#SBATCH --mail-type=END,FAIL  # which type of notifications to send
#-------------------------------------------------------------------------------

echo "### Starting at: $(date) ###"

# load modules then list loaded modules
module load icu/icu-54.1 icu4c/icu4c-57.1 bedtools
echo $*

cd /storage/htc/bdm/ccm3x/deepGRN/src

echo Enter py3_tf_cpu env...
source /storage/htc/bdm/ccm3x/py_env/py3_tf_cpu/bin/activate

echo start training...
python /storage/htc/bdm/ccm3x/deepGRN/src/model_train.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${24} ${25} ${26} ${27} ${28}

echo start prediction...
python /storage/htc/bdm/ccm3x/deepGRN/src/model_predict.py ${19} ${20} ${21}

deactivate
echo Leave py3_tf_cpu env...

echo Enter py2_theano_cpu env...
source /storage/htc/bdm/ccm3x/py_env/py2_theano_cpu/bin/activate

echo start evaluation...
echo Evaluating without blacklist...
python /storage/htc/bdm/ccm3x/deepGRN/src/score.py ${22} /storage/htc/bdm/ccm3x/deepGRN/raw/label/final ${23}

echo Evaluating with blacklist...
python /storage/htc/bdm/ccm3x/deepGRN/src/score.py ${22} /storage/htc/bdm/ccm3x/deepGRN/raw/label/final ${23}.blacklisted -b /storage/htc/bdm/ccm3x/deepGRN/raw/nonblacklist_bools.csv

echo add evaluation result to summary
Rscript /storage/htc/bdm/ccm3x/deepGRN/src/final_eval.R ${29}
