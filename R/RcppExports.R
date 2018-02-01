fit.zqtl <- function(effect,              # marginal effect : y ~ x
                     effect.se,           # marginal se : y ~ x
                     X,                   # X matrix
                     n = 0,               # sample size
                     C = NULL,            # covariate matrix (before LD)
                     C.delta = NULL,      # covariate matrix (already multified by LD)
                     factored = FALSE,    # Factored multiple traits
                     options = list()) {

    stopifnot(is.matrix(effect))
    stopifnot(is.matrix(effect.se))
    stopifnot(all(dim(effect) == dim(effect.se)))
    stopifnot(is.matrix(X))

    ## SNP confounding factors
    if(is.null(C)) {
        p <- dim(effect)[1]
        C <- matrix(1/p, p, 1)
    }

    if(is.null(C.delta)) {
        p <- dim(effect)[1]
        C.delta <- matrix(1, p, 1)
    }

    stopifnot(is.matrix(C))
    stopifnot(dim(effect)[1] == dim(C)[1])

    default.options <- list(tau.lb = -10,
                            tau.ub = -4,
                            pi.lb = -4,
                            pi.ub = -1,
                            tol = 1e-4,
                            gammax = 1000,
                            decay = 0,
                            rate = 1e-2,
                            nsample = 10,
                            vbiter = 2000,
                            verbose = TRUE,
                            do.stdize = TRUE,
                            min.se = 1e-4,
                            k = 1)

    if(length(options) == 0) {
        options <- default.options
    }
    options <- c(options, list(sample.size = n))

    ## call R/C++ functions ##
    if(factored) {
        return(.Call('rcpp_fac_zqtl', effect, effect.se, X, C, C.delta, options, PACKAGE = 'zqtl'))
    } else {
        return(.Call('rcpp_zqtl', effect, effect.se, X, C, C.delta, options, PACKAGE = 'zqtl'))
    }
}

fit.med.zqtl <- function(effect,              # marginal effect : y ~ x
                         effect.se,           # marginal se : y ~ x
                         effect.m,            # marginal : m ~ x
                         effect.m.se,         # marginal se : m ~ x
                         X.gwas,              # X matrix for GWAS
                         X.med = NULL,        # X matrix for mediation
                         n = 0,               # sample size of effect
                         n.med = 0,           # sample size of effect.m
                         C = NULL,            # covariate matrix
                         options = list()) {

    stopifnot(is.matrix(effect))
    stopifnot(is.matrix(effect.se))
    stopifnot(all(dim(effect) == dim(effect.se)))
    stopifnot(is.matrix(X.gwas))

    ## SNP confounding factors
    if(is.null(C)) {
        p <- dim(effect)[1]
        C <- matrix(1/p, p, 1)
    }

    ## X.gwas == X.med
    if(is.null(X.med)) {
        X.gwas <- X.med
    }

    stopifnot(is.matrix(C))
    stopifnot(dim(effect)[1] == dim(C)[1])

    default.options <- list(tau.lb = -10,
                            tau.ub = -4,
                            pi.lb = -4,
                            pi.ub = -1,
                            tol = 1e-4,
                            gammax = 1000,
                            decay = 0,
                            rate = 1e-2,
                            nsample = 10,
                            vbiter = 2000,
                            verbose = TRUE,
                            do.stdize = TRUE,
                            min.se = 1e-4,
                            k = 1)

    if(length(options) == 0) {
        options <- default.options
    }
    options <- c(options, list(sample.size = n, med.sample.size = n.med))

    stopifnot(dim(effect.m)[1] == dim(effect.m.se)[1])
    stopifnot(dim(effect.m)[2] == dim(effect.m.se)[2])

    ## call R/C++ functions ##
    ret <- .Call('rcpp_med_zqtl', PACKAGE = 'zqtl',
                 effect,
                 effect.se,
                 effect.m,
                 effect.m.se,
                 X.gwas,
                 X.med,
                 C,
                 options)
}

read.plink <- function(bed.header) {

    glue <- function(...) paste(..., sep='')
    fam.file <- glue(bed.header, '.fam')
    bim.file <- glue(bed.header, '.bim')
    bed.file <- glue(bed.header, '.bed')
    stopifnot(file.exists(fam.file))
    stopifnot(file.exists(bim.file))
    stopifnot(file.exists(bed.file))

    ## 1. read .fam and .bim file
    fam <- read.table(fam.file)
    bim <- read.table(bim.file)
    n <- dim(fam)[1]
    n.snp <- dim(bim)[1]

    ## 2. read .bed
    bed <- .Call('read_plink_bed', PACKAGE = 'zqtl', bed.file, n, n.snp)

    return(list(FAM=fam, BIM=bim, BED=bed))
}

take.ld.pairs <- function(X, cutoff = 0.05, stdize = FALSE) {
    ret <- .Call('rcpp_take_ld_pairs', PACKAGE = 'zqtl', X, cutoff, stdize);
    ret <- do.call(cbind, ret)
    return(ret)
}

take.ld.svd <- function(X, options = list()){
    return(.Call('rcpp_take_svd_xtx', PACKAGE = 'zqtl', X, options));
}
