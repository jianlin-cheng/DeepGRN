#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition Interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1         # leave at '1' unless using a MPI code
#SBATCH --cpus-per-task=1  # cores per task
#SBATCH --mem-per-cpu=32G  # memory per core (default is 1GB/core)
#SBATCH --time 0-3:15     # days-hours:minutes
#SBATCH --qos=normal
#SBATCH --account=general  # investors will replace this (e.g. `rc-gpu`)

## labels and outputs
#SBATCH --job-name=m_eval
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

echo Enter py2_theano_cpu env...
source /storage/htc/bdm/ccm3x/py_env/py2_theano_cpu/bin/activate

echo Start evaluation...
echo Evaluating without blacklist...
echo python /storage/htc/bdm/ccm3x/deepGRN/src/score.py ${1} /storage/htc/bdm/ccm3x/deepGRN/raw/label/final ${2}
python /storage/htc/bdm/ccm3x/deepGRN/src/score.py ${1} /storage/htc/bdm/ccm3x/deepGRN/raw/label/final ${2}

echo Evaluating with blacklist...
echo python /storage/htc/bdm/ccm3x/deepGRN/src/score.py ${1} /storage/htc/bdm/ccm3x/deepGRN/raw/label/final ${2}.blacklisted -b /storage/htc/bdm/ccm3x/deepGRN/raw/nonblacklist_bools.csv
python /storage/htc/bdm/ccm3x/deepGRN/src/score.py ${1} /storage/htc/bdm/ccm3x/deepGRN/raw/label/final ${2}.blacklisted -b /storage/htc/bdm/ccm3x/deepGRN/raw/nonblacklist_bools.csv

echo Leave py2_theano_cpu env...
deactivate

echo add evaluation result to summary
echo Rscript /storage/htc/bdm/ccm3x/deepGRN/src/final_eval.R ${3}
Rscript /storage/htc/bdm/ccm3x/deepGRN/src/final_eval.R ${3}
