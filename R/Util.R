################################################################
#' Covariance calculation (speed is a bit improved)
#' @name fast.cov
#' @usage fast.cov(x, y)
#' @param x matrix x
#' @param y matrix y
#' @return covariance matrix
#'
#' @export
#'
fast.cov <- function(x, y) {
    n.obs = crossprod(!is.na(x), !is.na(y))
    ret = crossprod(replace(x, is.na(x), 0),
                     replace(y, is.na(y), 0)) / n.obs
    return(ret)
}

################################################################
#' Covariance z-score calculation (speed is a bit improved)
#' @name fast.z.cov
#' @usage fast.z.cov(x, y)
#' @param x matrix x
#' @param y matrix y
#' @return covariance matrix
#'
#' @export
#'
fast.z.cov <- function(x, y) {
    n.obs = crossprod(!is.na(x), !is.na(y))
    ret = crossprod(replace(x, is.na(x), 0),
                     replace(y, is.na(y), 0)) / sqrt(n.obs)
    return(ret)
}

################################################################
#' Safe X'Y calculation
#' @name safe.XtY
#' @usage safe.XtY(x, y)
#' @param x matrix
#' @param y matrix
#' @return x'y result
#'
#' @export
#' 
safe.XtY <- function(x, y) {

    ntarget = nrow(x)
    nobs = crossprod(is.finite(x), is.finite(y))
    ret = crossprod(replace(x, !is.finite(x), 0),
                    replace(y, !is.finite(y), 0))

    ret = ret / pmax(nobs, 1) * ntarget
    return(ret)
}

################################################################
#' convert z-score to p-values (two-sided test)
#' @name zscore.pvalue
#' @usage zscore.pvalue(z)
#' @param z z-score
#' @return p-value
#'
#' @export
zscore.pvalue <- function(z) {
    2 * pnorm(abs(z), lower.tail = FALSE)
}

################################################################
#' output log message (override `sprintf`)
#' @name log.msg
#' @usage log.msg(stuff)
#' @param stuff input for sprintf
#'
#' @examples
#'
#' log.msg("Test Message %d", 1)
#'
#' @export
log.msg <- function(...) {
    ss = as.character(date())
    cat(sprintf('[%s] ', ss), sprintf(...), '\n', file = stderr(), sep = '')
}

################################################################
#' calculate univariate effect sizes and p-values
#'
#' @name calc.qtl.stat
#' @usage calc.qtl.stat(xx, yy)
#' @param xx n x p genotype matrix
#' @param yy n x t phenotype matrix
#' @param se.min mininum standard error (default: 1e-8)
#' @param verbose (default: FALSE)
#'
#' @return summary statistics matrix
#'
#' @export
#'
calc.qtl.stat <- function(xx, yy, se.min = 1e-8, verbose = FALSE) {

    loadNamespace("dplyr")
    loadNamespace("tidyr")

    .xx = scale(xx)
    .yy = scale(yy)

    rm.na.zero <- function(xx) {
        return(replace(xx, is.na(xx), 0))
    }

    ## cross-product is much faster than covariance function
    n.obs = crossprod(!is.na(.xx), !is.na(.yy))
    beta.mat = crossprod(.xx %>% rm.na.zero(), .yy %>% rm.na.zero()) / n.obs

    if(verbose){
        log.msg('Computed cross-products')
    }

    ## residual standard deviation
    resid.se.mat = matrix(NA, ncol(.xx), ncol(.yy))

    for(k in 1:ncol(.yy)) {

        beta.k = beta.mat[, k]
        yy.k = .yy[, k]
        err.k = sweep(sweep(.xx, 2, beta.k, `*`), 1, yy.k, `-`)
        se.k = apply(err.k, 2, sd, na.rm = TRUE)

        if(verbose) {
            log.msg('Residual on the column %d', k)
        }
        resid.se.mat[, k] = se.k + se.min
    }

    ## organize as consolidated table
    y.cols = 1:ncol(yy)
    colnames(beta.mat) = y.cols
    colnames(n.obs) = y.cols
    colnames(resid.se.mat) = y.cols

    beta.tab = beta.mat %>%
        as.data.frame() %>%
        dplyr::mutate(x.col = 1:n()) %>%
        tidyr::gather(key = 'y.col', value = 'beta', y.cols)

    resid.se.tab = resid.se.mat %>%
        as.data.frame() %>%
        dplyr::mutate(x.col = 1:n()) %>%
        tidyr::gather(key = 'y.col', value = 'resid.se', y.cols)

    nobs.tab = n.obs %>%
        as.data.frame() %>%
        dplyr::mutate(x.col = 1:n()) %>%
        tidyr::gather(key = 'y.col', value = 'n', y.cols)

    out.tab = beta.tab %>%
        dplyr::left_join(nobs.tab, by = c('x.col', 'y.col')) %>%
        dplyr::left_join(resid.se.tab, by = c('x.col', 'y.col')) %>%
        dplyr::mutate(se = resid.se/sqrt(n)) %>%
        dplyr::mutate(p.val = zscore.pvalue(beta/se))

    out.tab = out.tab %>%
        dplyr::mutate(x.col = as.integer(x.col)) %>%
        dplyr::mutate(y.col = as.integer(y.col))

    return(out.tab)
}

################################################################
#' Read binary PLINK format
#' @name read.plink
#' @usage read.plink(bed.header)
#' @param bed.header header for plink fileset
#' @return a list of FAM, BIM, BED data.
#' @author Yongjin Park, \email{ypp@@stat.ubc.ca}, \email{ypp.ubc@@gmail.com}
#' @export
read.plink <- function(bed.header) {

    fam.file = paste0(bed.header, '.fam')
    bim.file = paste0(bed.header, '.bim')
    bed.file = paste0(bed.header, '.bed')
    stopifnot(file.exists(fam.file))
    stopifnot(file.exists(bim.file))
    stopifnot(file.exists(bed.file))

    ## 1. read .fam and .bim file
    fam = read.table(fam.file, header = FALSE, stringsAsFactors = FALSE)
    bim = read.table(bim.file, header = FALSE, stringsAsFactors = FALSE)
    n = nrow(fam)
    n.snp = nrow(bim)

    ## 2. read .bed
    bed = .Call('read_plink_bed', PACKAGE = 'zqtl', bed.file, n, n.snp)

    return(list(FAM=fam, BIM=bim, BED=bed))
}

################################################################
#' Calculate covariance matrix and transform into triangular form for
#' visualization
#'
#' @name take.ld.pairs
#' @usage take.ld.pairs(X, cutoff, stdize)
#'
#' @param X genotype matrix
#' @param cutoff LD covariance cutoff (default: 0.05)
#' @param stdize standardize genotype matrix (default : FALSE)
#' @return \code{data.frame(x, y, x.pos, y.pos, cov)}
#' @export
take.ld.pairs <- function(X, cutoff = 0.05, stdize = FALSE) {
    ret = .Call('rcpp_take_ld_pairs', PACKAGE = 'zqtl', X, cutoff, stdize);
    ret = do.call(cbind, ret)
    return(ret)
}

################################################################
#' Decompose the scaled genotype matrix.
#'
#' @name take.ld.svd
#' @usage take.ld.svd(X, options, eigen.tol, eigen.reg, do.stdize)
#'
#' @param X n x p matrix
#' @param options a list of options
#' @param eigen.tol Error tolerance in Eigen decomposition (default: 0.01)
#' @param eigen.reg Regularization of Eigen decomposition (default: 0.0)
#' @param do.stdize Standardize (default: TRUE)
#'
#' @details
#'
#' Decompose \eqn{n^{-1/2}X = U D V^{\top}}{X/sqrt(n) = U D V'}
#' such that the LD matrix can become
#' \deqn{R = V D^{2} V^{\top}}{LD = V D^2 V'} for subsequent analysis.
#'
#' @export
#'
take.ld.svd <- function(X, options = list(),
                        eigen.tol = 1e-2,
                        eigen.reg = 0,
                        do.stdize = TRUE){

    opt.vars = c('eigen.tol', 'eigen.reg', 'do.stdize')

    .eval <- function(txt) eval(parse(text = txt))
    for(v in opt.vars) {
        val = .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] = val
        }
    }

    return(.Call('rcpp_take_svd_xtx', PACKAGE = 'zqtl', X, options));
}

###############################################################
#' Impute z-scores
#'
#' @export
#' @name impute.zscore
#' @usage impute.zscore(Z, V.t, observed)
#' @param Z incomplete z-score matrix
#' @param V.t t(V) matrix of the SVD result
#' @param observed SNP positions observed in Z
#' @return new z-score matrix
#'
#' @details
#'
#' Assuming that \eqn{\mathbf{y} = X_{0} \boldsymbol{\theta}_{0} +
#' X_{1} \boldsymbol{\theta}_{1} \cdots} and \eqn{n^{-1/2}X = U D
#' V^{\top}}, we have incomplete z-score \eqn{\mathbf{z}_{0} \sim N(V_{0}
#' D^2 V_{0}^{\top} , V_{0} D^2 V_{0}^{\top})}.  By linear
#' transformation, we have \deqn{\mathbf{y}_{0} = U D V_{0}^{\top}
#' \mathbf{\theta}_{0} \sim \mathcal{N}\!\left(U D^{-1}
#' V_{0}^{\top}\mathbf{z}_{0}, I\right).}  Using this we can derive
#' imputed z-score: \deqn{\tilde{\mathbf{z}} \approx
#' VDU^{\top}\mathbf{y}_{0} \sim \mathcal{N}\!\left(V
#' V_{0}^{\top}\mathbf{z}_{0}, VD^{2}V^{\top}\right).}
#'
impute.zscore <- function(Z, V.t, observed) {
    Vt0 = V.t[, observed, drop = FALSE]
    ret = t(V.t) %*% (Vt0 %*% Z)
    return(ret)
}

################################################################
#' Project z-scores of one LD block to another LD block
#'
#' @export
#' @name project.zscore
#' @usage project.zscore(ld.svd.tgt, ld.svd.src, z.src)
#' @param ld.svd.tgt SVD result on the target LD block
#' @param ld.svd.src SVD result on the source LD block
#' @param z.src z-score matrix of the source LD block
#' @param stdize standardize if TRUE, otherwise just center the mean
#' @return new z-score matrix
#'
#' @details
#' We assume the estimated individual phenotype:
#' \deqn{\mathbf{y}_{src} \sim \mathcal{N}\!\left(U_{src}D_{src}^{-1}V_{src}^{\top}\mathbf{z}_{src}, I\right).}
#' Therefore, a new z-score projected onto the target LD block will be:
#' \deqn{\mathbf{z}_{tgt} \sim \mathcal{N}\!\left(V_{tgt}D_{tgt}U_{tgt}^{\top}\mathbf{y}_{src}, V_{tgt}D_{tgt}^{2}V_{tgt}^{\top}\right).}
#'
project.zscore <- function(ld.svd.tgt, ld.svd.src, z.src, ...) {

    U.1 = ld.svd.tgt$U
    D.1 = ld.svd.tgt$D
    Vt.1 = ld.svd.tgt$V.t

    U.0 = ld.svd.src$U
    D.0 = ld.svd.src$D
    Vt.0 = ld.svd.src$V.t

    ret = (Vt.0 %*% z.src)
    ret = sweep(U.0, 2, D.0, `/`) %*% ret
    ret = t(U.1) %*% ret
    ret = sweep(t(Vt.1), 2, D.1, `*`) %*% ret

    ret = scale.zscore(ret, Vt.1, D.1, ...)
    return(ret)
}

################################################################
#' Standardize z-scores.
#'
#' @export
#' @name scale.zscore
#' @usage scale.zscore(Z, V.t, D, stdize = TRUE)
#'
#' @param Z z-score matrix
#' @param V.t t(V) matrix of the SVD result on genotype matrix
#' @param D D diagonal elements of the SVD result on genotype matrix
#' @param stdize standardize if TRUE, otherwise just center the mean
#'
#' @return locally standardized z-score matrix
#'
#' @details
#' Estimate \eqn{\mu} and \eqn{\tau} parameters in the model:
#' \deqn{\mathbf{z} \sim \mathcal{N}\!\left(R (\mu I), \tau^{2}R \right)}
#' then standardize z-scores by
#' \deqn{\mathbf{z} \gets (\mathbf{z} - \mu I) / \tau.}
#'
scale.zscore <- function(Z, V.t, D, stdize = TRUE) {
    loadNamespace("dplyr")

    .p = ncol(V.t)
    .K = nrow(V.t)

    if(.K < 10) {
        return(Z)
    }

    DV.t = sweep(V.t, 1, D, `*`)
    .y = sweep(V.t, 1, D, `/`) %*% Z
    .x = DV.t %*% matrix(1, ncol(V.t), 1)

    .xx = .x / sqrt(.p) ## to scale down for numerical stability
    .num = t(.xx) %*% .y
    .denom = sum(.xx * .x)

    .mu = .num / .denom

    z.mean = t(DV.t) %*% (.x %*% .mu)
    if(stdize) {
        z.sd = (.y - .x %*% .mu) %>%
            (function(.c) apply(.c^2, 2, sum) / (.K - 1))
        z.sd = sqrt(z.sd)
        ret = sweep(Z - z.mean, 2, pmax(z.sd, 1e-8), `/`)
    } else {
        ret = Z - z.mean
    }
    return(ret)
}
