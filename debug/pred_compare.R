pred_original_file = '/storage/htc/bdm/ccm3x/FactorNet/final_predictions/CTCF.PC-3.metaGENCODE_RNAseq_Unique35_DGF.bed'
pred_test_file = '/storage/htc/bdm/ccm3x/deepGRN/results/evaluate_factornet/predictions/CTCF.PC-3.metaGENCODE_RNAseq_Unique35_DGF_batch1_test.unique35True.RNAseqTrue.GencodeTrue.csv'
region_file = '/storage/htc/bdm/ccm3x/FactorNet/resources/predict_region.bed'
nonblacklist_bool_file = '/storage/htc/bdm/ccm3x/deepGRN/raw/nonblacklist_bools.csv'

library(data.table)
pred_original <- fread(pred_original_file,header = F,data.table = F)$V1[1:300000]
pred_test <- fread(pred_test_file,header = F,data.table = F)$V1
region <- fread(region_file,header = F,data.table = F)
blacklist_bool <- fread(nonblacklist_bool_file,header = F,data.table = F)$V1[1:300000]==0
pred_test[blacklist_bool] <- 0

aaa=abs(pred_original-pred_test)
names(aaa) <- as.character(1:length(aaa))
bbb=sort(aaa,decreasing=T)

