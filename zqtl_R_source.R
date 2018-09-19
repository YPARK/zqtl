################################################################
#' Variational inference of zQTL regression
#'
#' @export
#' @name fit.zqtl
#'
#' @usage fit.zqtl(effect, effect.se, X)
#'
#' @param effect Marginal effect size matrix (SNP x trait)
#' @param effect.se Marginal effect size standard error matrix (SNP x trait)
#' @param n sample size of actual data (will ignore if n = 0)
#' @param X Design matrix (reference Ind x SNP)
#' @param C SNP confounding factors (SNP x confounder; default: NULL)
#' @param C.delta SNP confounding factors (SNP x confounder; default: NULL)
#' @param factored Fit factored QTL model (default: FALSE)
#' @param options A list of inference/optimization options.
#' @param do.hyper Hyper parameter tuning (default: FALSE)
#' @param do.rescale Rescale z-scores by standard deviation (default: FALSE)
#' @param tau Fixed value of tau
#' @param pi Fixed value of pi
#' @param tau.lb Lower-bound of tau (default: -10)
#' @param tau.ub Upper-bound of tau (default: -4)
#' @param pi.lb Lower-bound of pi (default: -4)
#' @param pi.ub Upper-bound of pi (default: -1)
#' @param tol Convergence criterion (default: 1e-4)
#' @param gammax Maximum precision (default: 1000)
#' @param rate Update rate (default: 1e-2)
#' @param decay Update rate decay (default: 0)
#' @param jitter SD of random jitter for mediation & factorization (default: 0.1)
#' @param nsample Number of stochastic samples (default: 10)
#' @param vbiter Number of variational Bayes iterations (default: 2000)
#' @param verbose Verbosity (default: TRUE)
#' @param k Rank of the factored model (default: 1)
#' @param svd.init initialize by SVD (default: TRUE)
#' @param right.nn non-negativity in factored effect (default: TRUE)
#' @param mu.min mininum non-negativity weight (default: 0.01)
#' @param print.interv Printing interval (default: 10)
#' @param nthread Number of threads during calculation (default: 1)
#' @param eigen.tol Error tolerance in Eigen decomposition (default: 0.01)
#' @param do.stdize Standardize (default: TRUE)
#' @param min.se Minimum level of SE (default: 1e-4)
#' @param rseed Random seed
#'
#'
#' @return a list of variational inference results
#'
#' @author Yongjin Park, \email{ypp@@csail.mit.edu}, \email{yongjin.peter.park@@gmail.com}
#'
#' @details
#'
#' Estimate true effect matrix from marginal effect sizes and standard errors (Hormozdiari et al., 2015; Zhu and Stephens, 2016):
#' \deqn{\mathbf{Z}_{t} \sim \mathcal{N}\!\left(R E^{-1}
#' \boldsymbol{\theta}_{t}, R\right)}{z[,t] ~ N(R inv(E) Theta[,t], R)}
#' where R is \eqn{p \times p}{p x p} LD / covariance matrix;
#' E is expected squared effect size matrix
#' (\eqn{\textsf{se}[\boldsymbol{\theta}_{t}^{\textsf{marg}}] + n^{-1} \langle \boldsymbol{\theta}_{t}^{\textsf{marg}} \rangle^{2}}{standard error + effect^2/n} matrix, diagonal);
#' \eqn{\mathbf{z}_{t}}{z[,t]} is \eqn{p \times 1}{p x 1} z-score vector of trait \eqn{t}{t}, or \eqn{\mathbf{z}_{t} = \boldsymbol{\theta}_{t}^{\textsf{marg}}/ \textsf{se}[\boldsymbol{\theta}_{t}^{\textsf{marg}}]}{z = theta.marg / se[theta.marg]}.
#'
#' In basic zQTL model, spasrse parameter matrix, \eqn{\theta}{theta}
#' was modeled with spike-slab prior.  We carry out posterior
#' inference by variational inference with surrogate distribution
#' first introduced in Carbonetto and Stephens (2012):
#'
#' \deqn{q(\theta|\alpha,\beta,\gamma) = \alpha
#' \mathcal{N}\!\left(\beta,\gamma^{-1}\right)}{q(theta|.) = alpha *
#' N(beta, 1/gamma)}
#'
#' We reparameterized \eqn{\alpha = \boldsymbol{\sigma}\!\left(\pi +
#' \delta\right)}{alpha = sigmoid(pi + delta)}, and \eqn{\gamma =
#' \gamma_{\textsf{max}}\boldsymbol{\sigma}\!\left(- \tau + \lambda
#' \right)}{gamma = gammax * sigmoid(- tau + lambda)} for numerical
#' stability.
#'
#' In factored zQTL model, we decompose sparse effect:
#' \deqn{\boldsymbol{\theta}_{t} = \boldsymbol{\theta}^{\textsf{left}}
#' \boldsymbol{\theta}_{t}^{\textsf{right}}}{theta = theta_left *
#' theta_right}
#'
#' @examples
#'
#' library(zqtl)
#' library(Matrix)
#'
#' n <- 500
#' p <- 2000
#' m <- 50
#'
#' set.seed(1)
#'
#' .rnorm <- function(a, b) matrix(rnorm(a * b), a, b)
#'
#' X <- .rnorm(n, p)
#' X0 <- X[, -(1:(p/2)), drop = FALSE]
#' X1 <- X[, (1:(p/2)), drop = FALSE]
#'
#' Y1 <- matrix(0, n, m)
#' Y <- matrix(0, n, m)
#' h2 <- 0.4
#'
#' c.snps <- sample(p / 2, 3)
#'
#' ## shared genetic variants
#' theta.left <- .rnorm(3, 1)
#' theta.right <- .rnorm(1, 3)
#' theta.shared <- theta.left \%*\% theta.right
#'
#' Y1[, 1:3] <- Y1[, 1:3] + X[, c.snps, drop = FALSE] \%*\% theta.shared
#'
#' v0 <- var(as.numeric(Y1[, 1:3]))
#' Y1[, -(1:3)] <- .rnorm(n, m - 3) * sqrt(v0)
#' v1 <- apply(Y1, 2, var)
#' Y1 <- Y1 + sweep(.rnorm(n, m), 2, c(sqrt(v1 * (1/h2 - 1))), `*`)
#'
#' ## introduce confounding factors
#' uu <- .rnorm(n, 5)
#' vv <- .rnorm(m, 5)
#' Y0 <- uu \%*\% t(vv)
#' Y <- Y1 + Y0
#'
#' ## just caculate univariate z-scores
#' fast.cov <- function(x, y) {
#'     n.obs <- crossprod(!is.na(x), !is.na(y))
#'     ret <- crossprod(replace(x, is.na(x), 0),
#'                      replace(y, is.na(y), 0)) / n.obs
#'     return(ret)
#' }
#'
#' fast.z.cov <- function(x, y) {
#'     n.obs <- crossprod(!is.na(x), !is.na(y))
#'     ret <- crossprod(replace(x, is.na(x), 0),
#'                      replace(y, is.na(y), 0)) / sqrt(n.obs)
#'     return(ret)
#' }
#'
#' xy.beta <- fast.cov(X, scale(Y))
#' z.xy <- fast.z.cov(X, scale(Y))
#' xy.beta.se <- xy.beta / (z.xy + 1e-4) + 1e-4
#'
#' vb.opt <- list(tol = 0, vbiter = 3000, jitter = 0.1,
#'                pi = -1, tau = -8, rate = 0.01, gammax = 1e2,
#'                eigen.tol = 1e-2, k = m, svd.init = TRUE)
#'
#' out <- fit.zqtl(xy.beta, xy.beta.se, X, factored = TRUE, options = vb.opt)
#'
#' ## show associations
#' .rnd <- c(c.snps, setdiff(sample(p, 20), c.snps))
#' image(Matrix(out$param.left$lodds[.rnd, ]))
#' image(Matrix(out$param.right$lodds))
#'
#'
fit.zqtl <- function(effect,              # marginal effect : y ~ x
                     effect.se,           # marginal se : y ~ x
                     X,                   # X matrix
                     n = 0,               # sample size
                     C = NULL,            # covariate matrix (before LD)
                     C.delta = NULL,      # covariate matrix (already multified by LD)
                     factored = FALSE,    # Factored multiple traits
                     options = list(),
                     do.hyper = FALSE,
                     do.rescale = FALSE,
                     tau = NULL,
                     pi = NULL,
                     tau.lb = -10,
                     tau.ub = -4,
                     pi.lb = -4,
                     pi.ub = -1,
                     tol = 1e-4,
                     gammax = 1e3,
                     rate = 1e-2,
                     decay = 0,
                     jitter = 1e-1,
                     nsample = 10,
                     vbiter = 2000,
                     verbose = TRUE,
                     k = 1,
                     svd.init = TRUE,
                     right.nn = FALSE,
                     mu.min = 1e-2,
                     print.interv = 10,
                     nthread = 1,
                     eigen.tol = 1e-2,
                     do.stdize = TRUE,
                     min.se = 1e-4,
                     rseed = NULL) {

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
        C.delta <- matrix(1/p, p, 1)
    }

    stopifnot(is.matrix(C))
    stopifnot(dim(effect)[1] == dim(C)[1])

    ## Override options
    opt.vars <- c('do.hyper', 'do.rescale', 'tau', 'pi', 'tau.lb',
                  'tau.ub', 'pi.lb', 'pi.ub', 'tol', 'gammax', 'rate', 'decay',
                  'jitter', 'nsample', 'vbiter', 'verbose', 'k', 'svd.init', 'right.nn', 'mu.min',
                  'print.interv', 'nthread', 'eigen.tol', 'do.stdize', 'min.se',
                  'rseed')

    .eval <- function(txt) eval(parse(text = txt))
    for(v in opt.vars) {
        val <- .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] <- val
        }
    }
    options[['sample.size']] <- n

    ## call R/C++ functions ##
    if(factored) {
        return(.Call('rcpp_fac_zqtl', effect, effect.se, X, C, C.delta, options, PACKAGE = 'zqtl'))
    } else {
        return(.Call('rcpp_zqtl', effect, effect.se, X, C, C.delta, options, PACKAGE = 'zqtl'))
    }
}

################################################################
#' Variational inference of zQTL factorization to identify potential
#' confounders across GWAS statistids
#'
#' @name fit.zqtl.factorize
#'
#' @usage fit.zqtl.factorize(effect, effect.se, X)
#'
#' @param effect Marginal effect size matrix (SNP x trait)
#' @param effect.se Marginal effect size standard error matrix (SNP x trait)
#' @param X Design matrix (reference Ind x SNP)
#' @param n sample size of actual data (will ignore if n = 0)
#' @param options A list of inference/optimization options.
#' @param do.hyper Hyper parameter tuning (default: FALSE)
#' @param do.rescale Rescale z-scores by standard deviation (default: FALSE)
#' @param tau Fixed value of tau
#' @param pi Fixed value of pi
#' @param tau.lb Lower-bound of tau (default: -10)
#' @param tau.ub Upper-bound of tau (default: -4)
#' @param pi.lb Lower-bound of pi (default: -4)
#' @param pi.ub Upper-bound of pi (default: -1)
#' @param tol Convergence criterion (default: 1e-4)
#' @param gammax Maximum precision (default: 1000)
#' @param rate Update rate (default: 1e-2)
#' @param decay Update rate decay (default: 0)
#' @param jitter SD of random jitter for mediation & factorization (default: 0.01)
#' @param nsample Number of stochastic samples (default: 10)
#' @param vbiter Number of variational Bayes iterations (default: 2000)
#' @param verbose Verbosity (default: TRUE)
#' @param k Rank of the factored model (default: 1)
#' @param svd.init initialize by SVD (default: TRUE)
#' @param right.nn non-negativity in factored effect (default: TRUE)
#' @param mu.min mininum non-negativity weight (default: 0.01)
#' @param print.interv Printing interval (default: 10)
#' @param nthread Number of threads during calculation (default: 1)
#' @param eigen.tol Error tolerance in Eigen decomposition (default: 0.01)
#' @param do.stdize Standardize (default: TRUE)
#' @param min.se Minimum level of SE (default: 1e-4)
#' @param rseed Random seed
#' @param factorization.model Factorization model; 0 = ind x factor, 1 = eigen x factor (default: 0)
#'
#' @return a list of variational inference results
#'
#' @author Yongjin Park, \email{ypp@@csail.mit.edu}, \email{yongjin.peter.park@@gmail.com}
#'
#' @details
#'
#' Our goal is to identify factorization of phenotype matrix: \deqn{Y
#' = U V}{Y = U V} where \eqn{Y}{Y} was used in the calculation of the
#' observed GWAS statsitics matrix \eqn{Z \propto X^{\top}Y}{Z ~ X'Y}.
#'
#' @examples
#'
#' library(zqtl)
#' library(Matrix)
#'
#' n <- 500
#' p <- 2000
#' m <- 50
#'
#' set.seed(1)
#'
#' .rnorm <- function(a, b) matrix(rnorm(a * b), a, b)
#'
#' X <- .rnorm(n, p)
#' X0 <- X[, -(1:(p/2)), drop = FALSE]
#' X1 <- X[, (1:(p/2)), drop = FALSE]
#'
#' Y1 <- matrix(0, n, m)
#' Y <- matrix(0, n, m)
#' h2 <- 0.4
#'
#' c.snps <- sample(p / 2, 3)
#'
#' ## shared genetic variants
#' theta.left <- .rnorm(3, 1)
#' theta.right <- .rnorm(1, 3)
#' theta.shared <- theta.left \%*\% theta.right
#'
#' Y1[, 1:3] <- Y1[, 1:3] + X[, c.snps, drop = FALSE] \%*\% theta.shared
#'
#' v0 <- var(as.numeric(Y1[, 1:3]))
#' Y1[, -(1:3)] <- .rnorm(n, m - 3) * sqrt(v0)
#' v1 <- apply(Y1, 2, var)
#' Y1 <- Y1 + sweep(.rnorm(n, m), 2, c(sqrt(v1 * (1/h2 - 1))), `*`)
#'
#' ## introduce confounding factors
#' uu <- .rnorm(n, 5)
#' vv <- .rnorm(m, 5)
#' Y0 <- uu \%*\% t(vv)
#' Y <- Y1 + Y0
#' Y <- scale(Y)
#'
#' fast.cov <- function(x, y) {
#'     n.obs <- crossprod(!is.na(x), !is.na(y))
#'     ret <- crossprod(replace(x, is.na(x), 0),
#'                      replace(y, is.na(y), 0)) / n.obs
#'     return(ret)
#' }
#'
#' fast.z.cov <- function(x, y) {
#'     n.obs <- crossprod(!is.na(x), !is.na(y))
#'     ret <- crossprod(replace(x, is.na(x), 0),
#'                      replace(y, is.na(y), 0)) / sqrt(n.obs)
#'     return(ret)
#' }
#'
#' xy.beta <- fast.cov(X, Y)
#' z.xy <- fast.z.cov(X, Y)
#' xy.beta.se <- xy.beta / (z.xy + 1e-4) + 1e-4
#'
#' vb.opt <- list(tol = 0, vbiter = 2000, jitter = 1e-1,
#'                pi = -1, rate = 0.01, gammax = 1e3,
#'                eigen.tol = 1e-1, k = m, right.nn = FALSE)
#'
#' out <- fit.zqtl.factorize(xy.beta, xy.beta.se, X, options = vb.opt)
#'
#' image(Matrix(head(out$param.indiv$theta, 20)))
#' image(Matrix(out$param.trait$theta))
#'
#' y.hat <- out$param.indiv$theta \%*\% t(out$param.trait$theta)
#'
#' plot(as.numeric(scale(y.hat)), as.numeric(scale(Y0)), pch = 19, cex = .3)
#' plot(as.numeric(scale(y.hat)), as.numeric(scale(Y1)), pch = 19, cex = .3)
#' plot(as.numeric(scale(y.hat)), as.numeric(scale(Y)), pch = 19, cex = .3)
#'
#' ## pure null
#'
#' xy.beta <- fast.cov(X0, scale(Y0))
#' z.xy <- fast.z.cov(X0, scale(Y0))
#' xy.beta.se <- xy.beta / (z.xy + 1e-4) + 1e-4
#'
#' out <- fit.zqtl.factorize(xy.beta, xy.beta.se, X0, options = vb.opt)
#'
#' image(Matrix(head(out$param.indiv$theta, 20)))
#' image(Matrix(out$param.trait$theta))
#'
#' y.hat <- out$param.indiv$theta \%*\% t(out$param.trait$theta)
#'
#' plot(as.numeric(scale(y.hat)), as.numeric(scale(Y0)), pch = 19, cex = .3)
#' plot(as.numeric(scale(y.hat)), as.numeric(scale(Y1)), pch = 19, cex = .3)
#' plot(as.numeric(scale(y.hat)), as.numeric(scale(Y)), pch = 19, cex = .3)
#'
#' @export
#'
fit.zqtl.factorize <- function(effect,              # marginal effect : y ~ x
                               effect.se,           # marginal se : y ~ x
                               X,                   # X matrix
                               n = 0,               # sample size
                               options = list(),
                               do.hyper = FALSE,
                               do.rescale = FALSE,
                               tau = NULL,
                               pi = NULL,
                               tau.lb = -10,
                               tau.ub = -4,
                               pi.lb = -4,
                               pi.ub = -1,
                               tol = 1e-4,
                               gammax = 1e3,
                               rate = 1e-2,
                               decay = 0,
                               jitter = 1e-1,
                               nsample = 10,
                               vbiter = 2000,
                               verbose = TRUE,
                               k = 1,
                               svd.init = TRUE,
                               right.nn = FALSE,
                               mu.min = 1e-2,
                               print.interv = 10,
                               nthread = 1,
                               eigen.tol = 1e-2,
                               do.stdize = TRUE,
                               min.se = 1e-4,
                               factorization.model = 0,
                               rseed = NULL) {

    stopifnot(is.matrix(effect))
    stopifnot(is.matrix(effect.se))
    stopifnot(all(dim(effect) == dim(effect.se)))
    stopifnot(nrow(effect) == ncol(X))
    stopifnot(is.matrix(X))

    ## Override options
    opt.vars <- c('do.hyper', 'do.rescale', 'tau', 'pi', 'tau.lb',
                  'tau.ub', 'pi.lb', 'pi.ub', 'tol', 'gammax', 'rate', 'decay',
                  'jitter', 'nsample', 'vbiter', 'verbose', 'k', 'svd.init', 'right.nn', 'mu.min',
                  'print.interv', 'nthread', 'eigen.tol', 'do.stdize', 'min.se',
                  'rseed', 'factorization.model')

    .eval <- function(txt) eval(parse(text = txt))
    for(v in opt.vars) {
        val <- .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] <- val
        }
    }
    options[['sample.size']] <- n

    ## call R/C++ functions ##
    return(.Call('rcpp_factorize', effect, effect.se, X, options, PACKAGE = 'zqtl'))
}

################################################################
#' Variational inference of zQTL mediation
#'
#' @export
#'
#' @name fit.med.zqtl
#'
#' @usage fit.med.zqtl(effect, effect.se, effect.m, effect.m.se, X.gwas)
#'
#' @param effect Marginal effect size matrix (SNP x trait)
#' @param effect.se Marginal effect size standard error matrix (SNP x trait)
#' @param effect.m Marignal genetic effects of mediators (SNP x mediator)
#' @param effect.m.se SE of the marginal genetic effects (SNP x mediator)
#' @param n sample size of GWAS data (will ignore if n = 0)
#' @param n.med sample size of mediation data (will ignore if n.med = 0)
#' @param X.gwas Design matrix (reference Ind.GWAS x SNP)
#' @param X.med Design matrix (reference Ind.MED x SNP)
#' @param C SNP confounding factors (SNP x confounder; default: NULL)#'
#' @param factored Fit factored model (default: FALSE)
#' @param options A list of inference/optimization options.
#' @param multivar.mediator Multivariate mediator QTL effect (default: FALSE)
#'
#' @param do.direct.estimation Estimate direct effect (default: TRUE)
#' @param do.control.backfire Estimate direct effect (default: FALSE)
#' @param do.med.two.step Estimate mediation in two steps (default: FALSE)
#' @param de.factorization Estimate direct effect by joint factorization (default: FALSE)
#' @param factorization.model Factorization model; 0 = ind x factor, 1 = eigen x factor (default: 0)
#' @param de.propensity Propensity sampling to estimate direct effect (default: FALSE)
#'
#' @param do.finemap.direct Fine-map direct effect SNPs (default: FALSE)
#' @param nboot Number of bootstraps followed by finemapping (default: 0 )
#' @param num.conditional number of conditional models
#' @param submodel.size size of each conditional model
#'
#'
#' @param do.var.calc variance calculation (default: FALSE)
#' @param num.strat.size Size of stratified sampling (default: 2)
#' @param num.duplicate.sample Duplicate number of independent components (default: 1)
#'
#' @param do.hyper Hyper parameter tuning (default: FALSE)
#' @param do.rescale Rescale z-scores by standard deviation (default: FALSE)
#' @param tau Fixed value of tau
#' @param pi Fixed value of pi
#' @param tau.lb Lower-bound of tau (default: -10)
#' @param tau.ub Upper-bound of tau (default: -4)
#' @param pi.lb Lower-bound of pi (default: -4)
#' @param pi.ub Upper-bound of pi (default: -1)
#' @param tol Convergence criterion (default: 1e-4)
#' @param gammax Maximum precision (default: 1000)
#' @param rate Update rate (default: 1e-2)
#' @param decay Update rate decay (default: 0)
#' @param jitter SD of random jitter for mediation & factorization (default: 0.01)
#' @param nsample Number of stochastic samples (default: 10)
#' @param vbiter Number of variational Bayes iterations (default: 2000)
#' @param verbose Verbosity (default: TRUE)
#' @param k Rank of the factored model (default: 1)
#' @param svd.init initialize by SVD (default: TRUE)
#' @param print.interv Printing interval (default: 10)
#' @param nthread Number of threads during calculation (default: 1)
#' @param eigen.tol Error tolerance in Eigen decomposition (default: 0.01)
#' @param do.stdize Standardize (default: TRUE)
#' @param min.se Minimum level of SE (default: 1e-4)
#' @param rseed Random seed
#'
#'
#'
#' @return a list of variational inference results
#'
#' @author Yongjin Park, \email{ypp@@csail.mit.edu}, \email{yongjin.peter.park@@gmail.com}
#'
#' @details
#'
#' Mediation analysis
#'
#' @examples
#'
#' library(zqtl)
#' n <- 500
#' p <- 1000
#' n.genes <- 10
#' h2 <- 0.8
#' n.causal.direct <- 3
#' n.causal.qtl <- 3
#' n.causal.gene <- 2
#'
#' set.seed(13)
#'
#' .rnorm <- function(a, b) matrix(rnorm(a * b), a, b)
#'
#' X.raw <- sapply(1:p, function(j) {
#'     f <- runif(1, min = 0.1, max = 0.9)
#'     rbinom(n, 2, f)
#' })
#'
#' X <- scale(X.raw)
#'
#' c.qtls <- matrix(sapply(1:n.genes, function(j) matrix(sample(p, n.causal.qtl))), nrow = n.causal.qtl)
#'
#' theta.snp.gene <- .rnorm(n.causal.qtl, n.genes) / n.causal.qtl
#'
#' gene.expr <- lapply(1:dim(c.qtls)[2], function(j) X[, c.qtls[,j], drop = FALSE] \%*\% theta.snp.gene[, j, drop = FALSE] + 0.5 * rnorm(n) )
#'
#' gene.expr <- do.call(cbind, gene.expr)
#'
#' theta.med <- sample(c(-1,1), n.causal.gene, TRUE)
#'
#' causal.direct <- sample(p, n.causal.direct)
#' theta.dir <- matrix(rnorm(n.causal.direct), n.causal.direct, 1) / n.causal.direct
#'
#' y <- gene.expr[,1:n.causal.gene,drop = FALSE] \%*\% theta.med # mediation
#' y <- y + X[, causal.direct, drop = FALSE] \%*\% theta.dir # direct
#' y <- y + rnorm(n) * c(sqrt(var(y) * (1/h2 - 1)))
#'
#' ## just caculate univariate z-scores
#' fast.cov <- function(x, y) {
#'     n.obs <- crossprod(!is.na(x), !is.na(y))
#'     ret <- crossprod(replace(x, is.na(x), 0),
#'                      replace(y, is.na(y), 0)) / n.obs
#'     return(ret)
#' }
#'
#' fast.z.cov <- function(x, y) {
#'     n.obs <- crossprod(!is.na(x), !is.na(y))
#'     ret <- crossprod(replace(x, is.na(x), 0),
#'                      replace(y, is.na(y), 0)) / sqrt(n.obs)
#'     return(ret)
#' }
#'
#' xy.beta <- fast.cov(X, scale(y))
#' z.xy <- fast.z.cov(X, scale(y))
#' xy.beta.se <- xy.beta / (z.xy + 1e-4) + 1e-4
#'
#' ################################################################
#' ## Use multivariate effects directly
#' vb.opt <- list(nsample = 10, vbiter = 3000, rate = 1e-2,
#'                gammax = 1e4, do.stdize = TRUE,
#'                pi = -0, tau = -4,
#'                do.hyper = FALSE, eigen.tol = 1e-2, tol = 1e-8,
#'                verbose = TRUE, min.se = 1e-4,
#'                print.interv = 200,
#'                multivar.mediator = TRUE)
#'
#' eqtl.true <- matrix(nrow = p, ncol = n.genes)
#' eqtl.se.true <- matrix(1, nrow = p, ncol = n.genes)
#'
#' for(j in 1:n.genes) {
#'     eqtl.true[c.qtls[,j], j] <- theta.snp.gene[,j]
#' }
#'
#' med.out <- fit.med.zqtl(effect = xy.beta,
#'                         effect.se = xy.beta.se,
#'                         effect.m = eqtl.true,
#'                         effect.m.se = eqtl.se.true,
#'                         X.gwas = X,
#'                         options = vb.opt)
#'
#' plot(med.out$param.mediated$lodds, ylab = 'lodds', main = 'mediation')
#'
#' ################################################################
#' vb.opt <- list(nsample = 10, vbiter = 3000, rate = 1e-2,
#'                gammax = 1e4, do.stdize = TRUE,
#'                pi = -0, tau = -4, submodel.size = 3,
#'                do.hyper = FALSE, eigen.tol = 1e-2, tol = 1e-8,
#'                verbose = TRUE, min.se = 1e-4,
#'                print.interv = 200,
#'                multivar.mediator = TRUE)
#'
#' eqtl.true <- matrix(nrow = p, ncol = n.genes)
#' eqtl.se.true <- matrix(1, nrow = p, ncol = n.genes)
#'
#' for(j in 1:n.genes) {
#'     eqtl.true[c.qtls[,j], j] <- theta.snp.gene[,j]
#' }
#'
#' med.out <- fit.med.zqtl(effect = xy.beta,
#'                         effect.se = xy.beta.se,
#'                         effect.m = eqtl.true,
#'                         effect.m.se = eqtl.se.true,
#'                         X.gwas = X,
#'                         options = vb.opt)
#'
#' plot(med.out$param.mediated$lodds, ylab = 'lodds', main = 'mediation')
#'
#'
fit.med.zqtl <- function(effect,              # marginal effect : y ~ x
                         effect.se,           # marginal se : y ~ x
                         effect.m,            # marginal : m ~ x
                         effect.m.se,         # marginal se : m ~ x
                         X.gwas,              # X matrix for GWAS
                         X.med = NULL,        # X matrix for mediation
                         n = 0,               # sample size of effect
                         n.med = 0,           # sample size of effect.m
                         C = NULL,            # covariate matrix
                         factored = FALSE,    # Factored model
                         options = list(),
                         multivar.mediator = FALSE,
                         de.propensity = FALSE,
                         de.factorization = FALSE,
                         factorization.model = 0,
                         num.strat.size = 2,
                         do.direct.estimation = TRUE,
                         do.control.backfire = FALSE,
                         do.med.two.step = FALSE,
                         do.finemap.direct = FALSE,
                         nboot = 0,
                         do.var.calc = FALSE,
                         med.lodds.cutoff = 0,
                         num.duplicate.sample = 1,
                         num.conditional = 0,
                         submodel.size = 1,
                         do.hyper = FALSE,
                         do.rescale = FALSE,
                         tau = NULL,
                         pi = NULL,
                         tau.lb = -10,
                         tau.ub = -4,
                         pi.lb = -4,
                         pi.ub = -1,
                         tol = 1e-4,
                         gammax = 1e3,
                         rate = 1e-2,
                         decay = 0,
                         jitter = 1e-1,
                         nsample = 10,
                         vbiter = 2000,
                         verbose = TRUE,
                         k = 1,
                         svd.init = TRUE,
                         print.interv = 10,
                         nthread = 1,
                         eigen.tol = 1e-2,
                         do.stdize = TRUE,
                         min.se = 1e-4,
                         rseed = NULL) {

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
        X.med <- X.gwas
    }

    stopifnot(is.matrix(C))
    stopifnot(dim(effect)[1] == dim(C)[1])

################################################################
    ## Override options
    opt.vars <- c('do.hyper', 'do.rescale', 'tau', 'pi', 'tau.lb',
                  'tau.ub', 'pi.lb', 'pi.ub', 'tol', 'gammax', 'rate', 'decay',
                  'jitter', 'nsample', 'vbiter', 'verbose', 'k', 'svd.init',
                  'print.interv', 'nthread', 'eigen.tol', 'do.stdize', 'min.se',
                  'rseed', 'do.var.calc', 'num.strat.size', 'nboot',
                  'multivar.mediator', 'de.propensity', 'de.factorization', 'factorization.model',
                  'do.direct.estimation', 'do.control.backfire', 'do.med.two.step', 'do.finemap.direct',
                  'med.lodds.cutoff', 'num.duplicate.sample', 'num.conditional',
                  'submodel.size')

    .eval <- function(txt) eval(parse(text = txt))
    for(v in opt.vars) {
        val <- .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] <- val
        }
    }

    options <- c(options, list(sample.size = n, med.sample.size = n.med))

    stopifnot(dim(effect.m)[1] == dim(effect.m.se)[1])
    stopifnot(dim(effect.m)[2] == dim(effect.m.se)[2])

    ## call R/C++ functions ##
    if(factored) {
        ret <- .Call('rcpp_fac_med_zqtl', PACKAGE = 'zqtl',
                     effect,
                     effect.se,
                     effect.m,
                     effect.m.se,
                     X.gwas,
                     X.med,
                     C,
                     options)
    } else {
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
    return(ret)
}

################################################################
#' Read binary PLINK format
#' @name read.plink
#' @usage read.plink(bed.header)
#' @param bed.header header for plink fileset
#' @return a list of FAM, BIM, BED data.
#' @author Yongjin Park, \email{ypp@@csail.mit.edu}, \email{yongjin.peter.park@@gmail.com}
#' @export
read.plink <- function(bed.header) {

    fam.file <- paste0(bed.header, '.fam')
    bim.file <- paste0(bed.header, '.bim')
    bed.file <- paste0(bed.header, '.bed')
    stopifnot(file.exists(fam.file))
    stopifnot(file.exists(bim.file))
    stopifnot(file.exists(bed.file))

    ## 1. read .fam and .bim file
    fam <- read.table(fam.file, header = FALSE, stringsAsFactors = FALSE)
    bim <- read.table(bim.file, header = FALSE, stringsAsFactors = FALSE)
    n <- dim(fam)[1]
    n.snp <- dim(bim)[1]

    ## 2. read .bed
    bed <- .Call('read_plink_bed', PACKAGE = 'zqtl', bed.file, n, n.snp)

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
    ret <- .Call('rcpp_take_ld_pairs', PACKAGE = 'zqtl', X, cutoff, stdize);
    ret <- do.call(cbind, ret)
    return(ret)
}

################################################################
#' Decompose the scaled genotype matrix \eqn{n^{-1/2}X = U D
#' V^{\top}}{X/sqrt(n) = U D V'} such that the LD matrix can become
#' \deqn{R = V D^{2} V^{\top}}{LD = V D^2 V'} for subsequent analysis.
#'
#' @name take.ld.svd
#' @usage take.ld.svd(X, options, eigen.tol, do.stdize)
#'
#' @param X n x p matrix
#' @param options a list of options
#' @param eigen.tol Error tolerance in Eigen decomposition (default: 0.01)
#' @param do.stdize Standardize (default: TRUE)
#'
#' @export
#'
take.ld.svd <- function(X, options = list(),
                        eigen.tol = 1e-2,
                        do.stdize = TRUE){

    opt.vars <- c('eigen.tol', 'do.stdize')

    .eval <- function(txt) eval(parse(text = txt))
    for(v in opt.vars) {
        val <- .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] <- val
        }
    }

    return(.Call('rcpp_take_svd_xtx', PACKAGE = 'zqtl', X, options));
}
