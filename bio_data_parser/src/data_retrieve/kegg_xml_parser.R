source('~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/src/utils.R')
kegg_file = '~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/kegg_xmls/'
library(igraph)
library(KEGGREST)
library(biomaRt)
data_mapping <- read.delim('~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/kegg_tcga_mapping.txt',as.is = T,sep = ' ',header = F)
el_all <- NULL
el_list <- list()
for(i in paste(data_mapping$V1,'.xml',sep = '')){
  el1 <- kegg_parser(paste(kegg_file,i,sep = ''))
  el_list[[gsub('.xml','',i)]] <- el1
  el_all <- rbind(el_all,el1)
}
write.table(unique(el_all),'~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/kegg_el.txt',row.names = F,quote = F)

#mapping kegg to ensembl
all_genes <- unique(c(el_all[,1],el_all[,2]))
ens_all <- NULL
i <- 1
while(i<length(all_genes)){
  query <- keggGet(all_genes[i:min(i+9,length(all_genes))])
  ens_ids <- gsub('Ensembl: ','',unlist(lapply(query, get_ens)))
  ens_all <- rbind(ens_all,cbind.data.frame(all_genes[i:min(i+9,length(all_genes))],ens_ids,stringsAsFactors=F))
  Sys.sleep(1)
  i <- i+10
}
ens_all[ens_all[,2]=='2952',2] <- 'ENSG00000277656' #manual fix
colnames(ens_all) <- c('KEGG','ENSG')
ens_mapping <- ens_all[,2]
names(ens_mapping) <- ens_all[,1]
el_all_ens <- muti_mapping(el_all,ens_mapping,1)
el_all_ens <- muti_mapping(el_all_ens,ens_mapping,2)


el_all_uniq <- as.matrix(unique(el_all_ens[,1:2]))
g_all <- graph_from_edgelist(el_all_uniq)
all_interactions <- paste(el_all_uniq[,1],el_all_uniq[,2],sep = '_')


kegg_adj <- matrix(0,nrow = length(el_list),ncol = length(all_interactions))
rownames(kegg_adj) <- names(el_list)
colnames(kegg_adj) <- all_interactions

for(i in 1:nrow(kegg_adj)){
  el_i <- el_list[[i]]
  el_i_ens <- muti_mapping(el_i,ens_mapping,1)
  el_i_ens <- muti_mapping(el_i_ens,ens_mapping,2)
  kegg_adj[i,paste(el_i_ens[,1],el_i_ens[,2],sep = '_')] <- el_i_ens[,3]
}

#sort genes by loci
all_ens_genes <- unlist(strsplit(ens_all[,2],' '))
human = useMart("ENSEMBL_MART_ENSEMBL", dataset = "hsapiens_gene_ensembl")
gene_loci <- getBM(attributes = c('chromosome_name','start_position','ensembl_gene_id'),filters = c('ensembl_gene_id'),mart = human,values = all_ens_genes)
gene_loci$chromosome_name[gene_loci$chromosome_name=='X'] <- 24
gene_loci$chromosome_name[gene_loci$chromosome_name=='CHR_HSCHR22_1_CTG7'] <- 22 #manual fix
gene_loci_sort <- gene_loci[order(gene_loci$chromosome_name,gene_loci$start_position),]

write.csv(kegg_adj,'~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/kegg_labels.csv',quote = F)
write.csv(ens_all,'~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/ens_all.csv',quote = F)
write.csv(gene_loci_sort,'~/Dropbox/MU/workspace/DeepGRN/bio_data_parser/data/gene_loci_sort.csv',quote = F)



