if (!require("tidyverse")) install.packages("tidyverse")

library( tidyverse )
args <- commandArgs(T)
x_raw <- file.path(args[1]) %>%
    read_csv( col_types=cols()) %>% select(-1) %>% as.data.frame()
x_pc <- t(prcomp(t(x_raw), center = TRUE, scale = TRUE,retx = T)$x)

write.csv(x_pc,args[2],row.names = F,quote = F)
