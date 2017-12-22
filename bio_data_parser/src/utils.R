
kegg_parser <- function(kegg_file){
  kegg_data <- read.delim(kegg_file,row.names = NULL,header = F,sep = '\n')
  #parse nodes
  entry_row <- grep('entry id.+gene',kegg_data$V1)
  entry_id <- c()
  entry_name <- c()
  for(i in entry_row){
    entry_id <- c(entry_id,gsub('.+id=(.+) name.+','\\1',kegg_data$V1[i]))
    entry_name <- c(entry_name,gsub('.+name=(.+) type.+','\\1',kegg_data$V1[i]))
  }
  names(entry_name) <- entry_id
  #parse edges
  edge_row <- grep('.+relation entry1.+',kegg_data$V1)
  el <- NULL
  for(i in edge_row){
    entry1_id <- gsub('.+relation entry1=(.+) entry2=.+','\\1',kegg_data$V1[i])
    entry2_id <- gsub('.+entry2=(.+) type.+','\\1',kegg_data$V1[i])
    e_type <- gsub('.+subtype name=(.+) value.+','\\1',kegg_data$V1[i+1])
    el <- rbind(el,c(entry1_id,entry2_id,e_type))
  }
  el <- el[el[,3]!= 'missing interaction'&el[,1]%in%names(entry_name)&el[,2]%in%names(entry_name),]
  el[el[,3]=='inhibition',3] <- -1
  el[el[,3]!=-1,3] <- 1
  # name convertion
  el2 <- NULL
  for(i in 1:nrow(el)){
    entry1_name <- unlist(strsplit(entry_name[el[i,1]],' '))
    el2 <- rbind(el2,cbind(entry1_name,el[i,2],el[i,3]))
  }
  el3 <- NULL
  for(i in 1:nrow(el2)){
    entry2_name <- unlist(strsplit(entry_name[el2[i,2]],' '))
    el3 <- rbind(el3,cbind(el2[i,1],entry2_name,el2[i,3]))
  }
  colnames(el3) <- c('entry1','entry2','type')
  return(el3)
}

muti_mapping <- function(x,mapping,col_ind,spt=' '){
  res <- NULL
  for(i in 1:nrow(x)){
    data1 <- x[i,] 
    if(length(grep(' ',mapping[data1[col_ind]]))==0){
      data1[col_ind] <- mapping[data1[col_ind]]
      res <- rbind(res,data1)
    }else{
      sub_data <- unlist(strsplit(mapping[data1[col_ind]],split = spt)[[1]])
      res_new <- matrix(rep(data1,length(sub_data)), nrow = length(sub_data), byrow = T)
      res_new[,col_ind] <- sub_data
      res <- rbind(res,res_new)
    }
  }
  return(res)
}

get_ens <- function(x){
  ens_id <- grep('^Ensembl',x$DBLINKS,value = T)
  if(length(ens_id)>0){
    return(ens_id)
  }else{
    return(x$ENTRY)
  }
}

make_groups <- function(x,grp_size){
  grp_num <- floor(x/grp_size)
  ind_list <- list()
  ind <- 1
  for(i in 1:grp_num){
    ind_list[[i]] <- ind:(ind+grp_size-1)
    i<- i+1
    ind <- ind+grp_size
  }
  if(max(ind_list[[grp_num]]) < x){
    ind_list[[grp_num]] <- c(ind_list[[grp_num]],max(ind_list[[grp_num]]): x)
  }
  return(ind_list)
}
