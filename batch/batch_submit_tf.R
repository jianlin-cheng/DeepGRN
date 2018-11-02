p1 = '/storage/htc/bdm/ccm3x/deepGRN/results_original/'
p2 =  '/storage/htc/bdm/ccm3x/deepGRN/results_new/'

all_tf = gsub('\\..+','',list.files('/storage/htc/bdm/ccm3x/deepGRN/raw/label/final'))
# for(i in all_tf[-10]){
#   system(paste('Rscript /storage/htc/bdm/ccm3x/deepGRN/src/batch/submit/batch_train_predict_evaluate.R',i,p1,0,0))
# }
# system(paste('Rscript /storage/htc/bdm/ccm3x/deepGRN/src/batch/submit/batch_train_predict_evaluate.R',all_tf[10],p1,0,1))

for(i in all_tf[-1]){
  system(paste('Rscript /storage/htc/bdm/ccm3x/deepGRN/src/batch/submit/batch_train_predict_evaluate.R',i,p2,1,0))
}
system(paste('Rscript /storage/htc/bdm/ccm3x/deepGRN/src/batch/submit/batch_train_predict_evaluate.R',all_tf[1],p2,1,1))