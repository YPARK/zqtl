#!/usr/bin/env Rscript
argv = commandArgs(trailingOnly = TRUE)

if(length(argv) < 2) {
    q()
}

PGS.DIR = argv[1]
OUT.FILE = argv[2]

library(dplyr)
library(tidyr)
library(readr)

## Read polygenic scores

read.pgs.y <- function(.file, digits=4) {
    .tab = suppressMessages(read_tsv(.file)) %>%
        select(fid, iid, ld.idx, y, y0) %>%
        mutate(y = round(y, digits)) %>% 
        mutate(y0 = round(y0, digits))
}

read.pgs.annotation <- function(.file, digits=4) {
    .tab = suppressMessages(read_tsv(.file)) %>%
        select(-nsnp, -n0, -y, -y0) %>% 
        gather(key='annotation', value='y.a',
               -fid, -iid, -ld.idx) %>%
        mutate(y.a = round(y.a, digits))
}

pgs.files = list.files(PGS.DIR, pattern = '.pgs.gz', full.names=TRUE)

## combine them

pgs.y.tab = lapply(pgs.files, read.pgs.y) %>%
    bind_rows()

pgs.annot.tab = lapply(pgs.files, read.pgs.annotation) %>%
    bind_rows()

out.tab = pgs.annot.tab %>% left_join(pgs.y.tab)

## save 
write_tsv(out.tab, OUT.FILE)
