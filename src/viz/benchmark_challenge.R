aaa = read.csv('~/Dropbox/MU/workspace/encode_dream/performance/score/dream_encode_benchmark.csv',header = T,as.is = T)
benchmark_data = apply(aaa[,3:ncol(aaa)], 2, as.numeric)

crad_bagging <- crad <- matrix(0,nrow = nrow(benchmark_data),ncol = 4)
for(i in 1:ncol(crad)){
  crad[,i] <- (benchmark_data[,5]-benchmark_data[,i])>0
}
apply(crad, 2, mean)

