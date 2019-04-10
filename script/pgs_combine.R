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

read.pgs.y <- function(.file) {
    .tab = suppressMessages(read_tsv(.file)) %>%
        select(fid, iid, y, y0)
}

read.pgs.annotation <- function(.file) {
    .tab = suppressMessages(read_tsv(.file)) %>%
        select(-ld.idx, -nsnp, -n0, -y, -y0) %>% 
        gather(key='annotation', value='y.a',
               - fid, - iid)
}

pgs.files = list.files(PGS.DIR, pattern = '.pgs.gz', full.names=TRUE)

## combine them

pgs.y.tab = lapply(pgs.files, read.pgs.y) %>%
    bind_rows() %>% 
    group_by(fid, iid) %>%
    summarize(y = sum(y, na.rm=TRUE), y0 = sum(y0, na.rm=TRUE)) %>%
    ungroup()

pgs.annot.tab = lapply(pgs.files, read.pgs.annotation) %>%
    bind_rows() %>% 
    group_by(fid, iid, annotation) %>%
    summarize(y = sum(y.a, na.rm=TRUE)) %>%
    ungroup() %>%
    spread(key=annotation, value=y, fill=0)

out.tab = pgs.y.tab %>% left_join(pgs.annot.tab)

## save 
write_tsv(out.tab, OUT.FILE)
