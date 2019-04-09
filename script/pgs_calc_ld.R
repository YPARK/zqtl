#!/usr/bin/env Rscript
argv = commandArgs(trailingOnly = TRUE)

if(length(argv) != 8) {
    print(length(argv))
    quit(status=1)
}

print(paste(argv, collapse=' '))

LD.IDX = as.integer(argv[1]) # e.g., 1
GWAS.FILE = argv[2]          # e.g., 'temp/2/gwas.bed.gz'
ANNOT.DIR = argv[3]          # e.g., 'temp/2/stratified/'
PLINK.HDR = argv[4]          # e.g., 'temp/2/plink'
BETA.COL.NAME = argv[5]      # e.g., 'lodds'
SE.COL.NAME = argv[6]        # e.g., 'se'
K = as.integer(argv[7])      # e.g., 10
OUT.FILE = argv[8]           # e.g., 'temp/out.gz'

library(dplyr)
library(readr)

dir.create(dirname(OUT.FILE), recursive=TRUE, showWarnings=FALSE)

plink = tryCatch(suppressMessages(zqtl::read.plink(PLINK.HDR)),
                 error = function(e) { return(NULL) })

if(is.null(plink)) {
    quit(status=1)
}

.bim.cols = c('chr', 'rs', 'missing', 'snp.loc', 'plink.a1', 'plink.a2')

x.bim = (plink$BIM) %>%
    (function(bim) { names(bim) = .bim.cols; bim }) %>%
    mutate(chr=gsub(chr, pattern='chr', replacement='')) %>%
    mutate(chr=paste('chr', chr, sep='')) %>%
    mutate(plink.pos = 1:n()) %>%
    select(snp.loc, plink.a1, plink.a2, plink.pos)

match.plink <- function(.tab) {
    .tab %>%
        rename(beta = BETA.COL.NAME) %>%
        rename(se = SE.COL.NAME) %>%
        rename(snp.loc = stop) %>%
        mutate(snp.loc = as.integer(snp.loc)) %>%
        left_join(x.bim, by = 'snp.loc') %>%
        na.omit() %>%
        mutate(s = if_else(a1 == plink.a1, 1, -1)) %>%
        mutate(beta = s * beta) %>%
        select(-s)
}

####################
## Read GWAS file ##
####################

.read.tsv <- function(...) suppressMessages(read_tsv(...))

gwas.tab = .read.tsv(GWAS.FILE) %>%
    match.plink()

if(nrow(gwas.tab) < 1) {
    write_tsv(data.frame(), OUT.FILE)
    q()
}

x.tib = gwas.tab %>%
    mutate(snp.loc = as.integer(snp.loc)) %>%
    select(snp.loc, plink.pos) %>%
    unique() %>%
    arrange(snp.loc) %>%
    mutate(x.pos = 1:n())

##########################################
## Take full SVD on the genotype matrix ##
##########################################

X = plink$BED[, x.tib$plink.pos, drop = FALSE]
ld.svd = zqtl::take.ld.svd(X)

K = min(K, ncol(ld.svd$U))

## regularize in the number of components

UDinv = sweep(ld.svd$U, 2, ld.svd$D, `/`)[, 1:K, drop = FALSE]
UD = sweep(ld.svd$U, 2, ld.svd$D, `*`)[, 1:K, drop = FALSE]
Vt = ld.svd$V.t[1:K, , drop = FALSE]
DVt = sweep(ld.svd$V.t, 1, ld.svd$D, `*`)[1:K, , drop = FALSE]

z = left_join(x.tib, gwas.tab, by = c("snp.loc", "plink.pos")) %>%
    mutate(z = beta/se) %>%
    select(z) %>%
    as.matrix()

## Take the total PGS
y = UDinv %*% (Vt %*% z)

################################
## Read stratified GWAS files ##
################################

annot.tbi.files = list.files(ANNOT.DIR, pattern = '.bed.gz.tbi', full.names=TRUE)
annot.bed.files = sapply(annot.tbi.files, gsub, pattern='.tbi', replacement='')

take.adjusted.y <- function(annot.x) {

    n1 = length(annot.x)

    if(length(n1) < 1) {
        return(matrix(0, nrow(U), 1))
    }

    ntot = nrow(x.tib)
    n0 = ntot - n1

    z.1 = z[annot.x, , drop = FALSE]
    Vt.1 = Vt[, annot.x, drop = FALSE]
    y.1 = UDinv %*% (Vt.1 %*% z.1)

    ## Adjust the influence of annotation potentially leaking out
    if(n0 > 0) {

        ## rbar = t(D*V0*1) * (D*V0*1) / n0
        DVt.0 = DVt[, -annot.x, drop = FALSE] / sqrt(n0)
        eta.0 = apply(DVt.0, 1, sum)
        rbar = sum(eta.0^2)

        ## z1'*V1*t(V0)*1
        Vt.0 = Vt[, -annot.x, drop = FALSE]
        Vt.0.sum = apply(Vt.0 / n0, 1, sum)

        delta.a = (t(z.1) %*% t(Vt.1) %*% Vt.0.sum / rbar)
        delta.a = as.numeric(delta.a)

        y.1 = y.1 - UD %*% apply(Vt.1, 1, sum) * delta.a

    }

    ## normalize by the number of variants
    y.1 = y.1 * (ntot / n1)
    return(y.1)
}

#########################
## For each annotation ##
#########################

n.annot = length(annot.bed.files)

Y.annot = matrix(0, nrow = nrow(y), ncol = n.annot)
annot.pos = NULL
annot.names = NULL

for(j in 1:n.annot) {

    .annot.file = annot.bed.files[j]
    .annot.name = basename(.annot.file) %>%
        gsub(pattern='.bed.gz',replacement='')

    annot.x = .read.tsv(.annot.file)

    if(nrow(annot.x) > 0) {
        annot.x = annot.x %>%
            match.plink() %>%
            left_join(x.tib, by = c("snp.loc", "plink.pos")) %>%
            na.omit()

        if(nrow(annot.x) > 0) {
            annot.x = annot.x %>%
                select(x.pos) %>%
                unlist(use.names=FALSE)

            if(length(annot.x) > 0) {
                Y.annot[, j] = take.adjusted.y(annot.x)
                annot.pos = c(annot.pos, annot.x) %>% unique()
            }
        }
    }
    annot.names = c(annot.names, .annot.name)
}

Y.annot = as.data.frame(Y.annot)
colnames(Y.annot) = annot.names

###########################
## Find the un-annotated ##
###########################

nsnp = nrow(z)
unannotated = setdiff(x.tib$x.pos, annot.pos)
n0 = length(unannotated)
y0 =  take.adjusted.y(unannotated)

out.tab = data.frame(ld.idx = LD.IDX,
                     fid = plink$FAM[, 1],
                     iid = plink$FAM[, 2],
                     nsnp = nsnp,
                     n0 = n0,
                     y=as.numeric(y),
                     y0=as.numeric(y0))

out.tab = cbind(out.tab, Y.annot)

write_tsv(out.tab, OUT.FILE)
