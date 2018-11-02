#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition Lewis
#SBATCH --nodes=1
#SBATCH --ntasks=1         # leave at '1' unless using a MPI code
#SBATCH --cpus-per-task=4  # cores per task
#SBATCH --mem-per-cpu=32G  # memory per core (default is 1GB/core)
#SBATCH --time 0-11:15     # days-hours:minutes
#SBATCH --qos=normal
#SBATCH --account=general  # investors will replace this (e.g. `rc-gpu`)

## labels and outputs
#SBATCH --job-name=m_peval
#SBATCH --output=results-%j.out  # %j is the unique jobID

## notifications
#SBATCH --mail-user=username@missouri.edu  # email address for notifications
#SBATCH --mail-type=END,FAIL  # which type of notifications to send
#-------------------------------------------------------------------------------

echo "### Starting at: $(date) ###"
module load R/R-3.3.3
module load icu/icu-54.1 icu4c/icu4c-57.1 bedtools
# load modules then list loaded modules

echo $*

cd /storage/htc/bdm/ccm3x/deepGRN/src

echo Enter py3_tf_cpu env...
source /storage/htc/bdm/ccm3x/py_env/py3_tf_cpu/bin/activate

echo start prediction...
python /storage/htc/bdm/ccm3x/deepGRN/src/model_predict.py ${1} ${2} ${3}

echo Leave py3_tf_cpu env...
deactivate
