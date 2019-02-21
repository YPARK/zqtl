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
#' Project z-scores on to different LD block
#'
#' @export
#' @name project.zscore
#' @usage project.zscore(ld.src.tgt, ld.svd.src, z.src)
#' @param ld.src.tgt SVD result on the target LD block
#' @param ld.svd.src SVD result on the source LD block
#' @param z.src z-score matrix of the source LD block
#' @return new z-score matrix
#'
#' @details
#' We assume the estimated individual phenotype:
#' \deqn{\mathbf{y}_{src} \sim \mathcal{N}\!\left(U_{src}D_{src}^{-1}V_{src}^{\top}\mathbf{z}_{src}, I\right).}
#' Therefore, a new z-score projected onto the target LD block will be:
#' \deqn{\mathbf{z}_{tgt} \sim \mathcal{N}\!\left(V_{tgt}D_{tgt}U_{tgt}^{\top}\mathbf{y}_{src}, V_{tgt}D_{tgt}^{2}V_{tgt}^{\top}\right).}
#' 
project.zscore <- function(ld.src.tgt, ld.svd.src, z.src) {

    U.1 = ld.src.tgt$U
    D.1 = ld.src.tgt$D
    Vt.1 = ld.src.tgt$V.t

    U.0 = ld.svd.src$U
    D.0 = ld.svd.src$D
    Vt.0 = ld.svd.src$V.t

    ret = (Vt.0 %*% z.src)
    ret = sweep(U.0, 2, D.0, `/`) %*% ret
    ret = t(U.1) %*% ret
    ret = sweep(t(Vt.1), 2, D.1, `*`) %*% ret

    ret = scale.zscore(ret, Vt.1, D.1)
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

    require(dplyr)

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
            (function(.c) apply(.c^2, 2, sum) / (.K - 1)) %>%
            sqrt()
        ret = sweep(Z - z.mean, 2, pmax(z.sd, 1e-8), `/`)
    } else {
        ret = Z - z.mean
    }
    return(ret)
}


################################################################
#' Estimate standard deviation by the delta method
#'
#' @export
#' @name take.var.power10
#' @usage take.var.power10(var.result)
#'
#' @param var.result which contains two matrix/data.frame:
#' \itemize{
#' \item{mean: }{for the mean of log10 values,}
#' \item{var: }{for the variance of log10 values.}
#' }
#'
#' @details
#'
#' Say \eqn{x \sim \mathcal{N}\!\left(\mu, \sigma^{2}\right)} in its log10 scale.
#' Let \eqn{g(x) = 10^x}. Since \eqn{g(x) = \exp(x \log(10))} and
#' \eqn{g'(x) = \exp(x \log(10)) \log(10)}, by the delta method, we can approximate
#' the distribution of \eqn{g(x)} around mean value \eqn{\mu}.
#' \deqn{g(x) \sim \mathcal{N}\!\left(g(\mu), (g'(\mu))^2 \sigma^{2}\right)}.
#'
#' @return a list of mean and variance estimation.
#' \itemize{
#' \item{mean: }{mean matrix}
#' \item{sd: }{standard deviation matrix}
#' }
#'
take.var.power10 <- function(var.result) {
    m = var.result$mean
    v = var.result$var
    stopifnot(all(v >= 0))
    ret = list(mean = 10^(m), sd = exp(m * log(10)) * log(10) * sqrt(v))
    return(ret)
}

################################################################
#' Inference of zQTL regression without multivariate effect size.
#'
#' @export
#' @name fit.zqtl.vanilla
#'
#' @usage fit.zqtl.vanilla(effect, effect.se, X, multi.C, univar.C)
#'
#' @param effect Marginal effect size matrix (SNP x trait)
#' @param effect.se Marginal effect size standard error matrix (SNP x trait)
#' @param n sample size of actual data (will ignore if n = 0)
#' @param X Design matrix (reference Ind x SNP)
#' @param multi.C multivariate SNP annotations (SNP x something; default: NULL)
#' @param univar.C univariate SNP annotations (SNP x something; default: NULL)
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
#' @param print.interv Printing interval (default: 10)
#' @param nthread Number of threads during calculation (default: 1)
#' @param eigen.tol Error tolerance in Eigen decomposition (default: 0.01)
#' @param do.stdize Standardize (default: TRUE)
#' @param out.residual estimate residual z-scores (default: FALSE)
#' @param do.var.calc variance calculation (default: FALSE)
#' @param nboot Number of bootstraps (default: 0)
#' @param nboot.var Number of bootstraps for variance estimation (default: 100)
#' @param scale.var Scaled variance calculation (default: FALSE)
#' @param min.se Minimum level of SE (default: 1e-4)
#' @param rseed Random seed
#'
#' @return a list of variational inference results.
#' \itemize{
#' \item{conf.multi: }{ association with multivariate confounding variables}
#' \item{conf.uni: }{ association with univariate confounding variables}
#' \item{resid: }{ residuals}
#' \item{gwas.clean: }{ cleansed version of univariate GWAS effects}
#' \item{var: }{ variance decomposition results}
#' \item{llik: }{ log-likelihood trace over the optimization}
#' }
#'
#' @author Yongjin Park, \email{ypp@@csail.mit.edu}, \email{yongjin.peter.park@@gmail.com}
#'
#' @examples
#'
#' n = 500
#' p = 2000
#' m = 1
#'
#' set.seed(1)
#' .rnorm <- function(a, b) matrix(rnorm(a * b), a, b)
#' X = .rnorm(n, p)
#' Y = matrix(0, n, m)
#' h2 = 0.25
#' y1 = X[, 1:500] %*% .rnorm(500, 1) * sqrt(h2/500)
#' y0 = .rnorm(n, 1) * sqrt(1 - h2)
#' Y = y1 + y0
#' qtl.tab = calc.qtl.stat(X, Y)
#' qtl.1.tab = calc.qtl.stat(X, y1)
#' z.1 = matrix(qtl.1.tab$beta/qtl.1.tab$se, ncol = 1)
#' qtl.0.tab = calc.qtl.stat(X[sample(n), , drop = FALSE], y1)
#' z.0 = matrix(qtl.0.tab$beta/qtl.0.tab$se, ncol = 1)
#' z.out = fit.zqtl.vanilla(matrix(qtl.tab$beta, ncol=1),
#'                                matrix(qtl.tab$se, ncol=1),
#'                                X = X,
#'                                univar.C = cbind(z.1, z.0),
#'                                do.var.calc = TRUE,
#'                                scale.var = TRUE,
#'                                nboot.var = 100)
#'
#' library(dplyr)
#' .make.df <- function(x) {
#'     data.frame(mean = x$mean, sd = x$sd)
#' }
#' .power10 <- function(x, component) {
#'     take.var.power10(x) %>%
#'         .make.df() %>%
#'         mutate(component)
#' }
#' var.tab = bind_rows(.power10(z.out$var$univar.tot, 'univar.tot'),
#'                     .power10(z.out$var$multivar.tot, 'multivar.tot'),
#'                     .power10(z.out$var$resid, 'resid'),
#'                     .power10(z.out$var$univar, 'univar'))
#' print(var.tab)
#'
#'
fit.zqtl.vanilla <- function(effect,           # marginal effect : y ~ x
                             effect.se,        # marginal se : y ~ x
                             X,                # X matrix
                             n = 0,            # sample size
                             multi.C = NULL,   # covariate matrix (before LD)
                             univar.C = NULL,  # covariate matrix (already multiplied by LD)
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
                             print.interv = 10,
                             nthread = 1,
                             eigen.tol = 1e-2,
                             do.stdize = TRUE,
                             out.residual = FALSE,
                             do.var.calc = FALSE,
                             scale.var = TRUE,
                             nboot = 0,
                             nboot.var = 100,
                             min.se = 1e-4,
                             rseed = NULL) {

    stopifnot(is.matrix(effect))
    stopifnot(is.matrix(effect.se))
    stopifnot(all(dim(effect) == dim(effect.se)))
    stopifnot(is.matrix(X))

    p = ncol(X)

    ## SNP confounding factors
    if(is.null(multi.C)) {
        p = nrow(effect)
        multi.C = matrix(1/p, p, 1)
    }

    if(is.null(univar.C)) {
        p = nrow(effect)
        univar.C = matrix(1/p, p, 1)
    }

    stopifnot(is.matrix(multi.C))
    stopifnot(nrow(effect) == nrow(multi.C))

    stopifnot(is.matrix(univar.C))
    stopifnot(nrow(effect) == nrow(univar.C))

    ## Override options
    opt.vars = c('do.hyper', 'do.rescale', 'tau', 'pi', 'tau.lb',
                 'tau.ub', 'pi.lb', 'pi.ub', 'tol', 'gammax', 'rate', 'decay',
                 'jitter', 'nsample', 'vbiter', 'verbose',
                 'print.interv', 'nthread', 'eigen.tol', 'do.stdize', 'out.residual', 'min.se',
                 'rseed', 'do.var.calc', 'scale.var', 'nboot', 'nboot.var')

    .eval <- function(txt) eval(parse(text = txt))
    for(v in opt.vars) {
        val = .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] = val
        }
    }
    options[['sample.size']] = n

    ## call R/C++ functions ##
    z.out = .Call('rcpp_zqtl_vanilla',
                  effect, effect.se, X,
                  multi.C, univar.C,
                  options, PACKAGE = 'zqtl')

    if(options[['do.var.calc']]) {
        log.msg('Parsing variance calculation')


    }

    return(z.out)
}


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
#' @param A Annotation matrix (SNP x annotations; default: NULL)
#' @param multi.C multivariate SNP confounding factors (SNP x confounder; default: NULL)
#' @param univar.C univariate SNP confounding factors (SNP x confounder; default: NULL)
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
#' @param right.nn non-negativity in factored effect (default: FALSE)
#' @param mu.min mininum non-negativity weight (default: 0.01)
#' @param print.interv Printing interval (default: 10)
#' @param nthread Number of threads during calculation (default: 1)
#' @param eigen.tol Error tolerance in Eigen decomposition (default: 0.01)
#' @param do.stdize Standardize (default: TRUE)
#' @param out.residual estimate residual z-scores (default: FALSE)
#' @param do.var.calc variance calculation (default: FALSE)
#' @param nboot Number of bootstraps (default: 0)
#' @param nboot.var Number of bootstraps for variance estimation (default: 100)
#' @param scale.var Scaled variance calculation (default: FALSE)
#' @param min.se Minimum level of SE (default: 1e-4)
#' @param rseed Random seed
#'
#'
#' @return a list of variational inference results.
#' \itemize{
#' \item{param: }{ sparse genetic effect size (theta, theta.var, lodds)}
#' \item{param.left: }{ the left factor for the factored effect}
#' \item{param.right: }{ the left factor for the factored effect}
#' \item{conf.multi: }{ association with multivariate confounding variables}
#' \item{conf.uni: }{ association with univariate confounding variables}
#' \item{resid: }{ residuals}
#' \item{gwas.clean: }{ cleansed version of univariate GWAS effects}
#' \item{var: }{ variance decomposition results}
#' \item{llik: }{ log-likelihood trace over the optimization}
#' }
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
#' ###########################
#' ## A simple zQTL example ##
#' ###########################
#'
#' n = 500
#' p = 2000
#' m = 1
#'
#' set.seed(1)
#'
#' .rnorm <- function(a, b) matrix(rnorm(a * b), a, b)
#'
#' X = .rnorm(n, p)
#' Y = matrix(0, n, m)
#' h2 = 0.25
#'
#' c.snps = sample(p, 3)
#'
#' theta = .rnorm(3, m) * sqrt(h2 / 3)
#'
#' Y = X[, c.snps, drop=FALSE] %*% theta + .rnorm(n, m) * sqrt(1 - h2)
#'
#' library(dplyr)
#' library(tidyr)
#'
#' qtl.tab = calc.qtl.stat(X, Y)
#'
#' xy.beta = qtl.tab %>%
#'     dplyr::select(x.col, y.col, beta) %>%
#'     tidyr::spread(key = y.col, value = beta) %>%
#'     (function(x) matrix(x[1:p, seq(2, m+1)], ncol = 1))
#'
#' xy.se = qtl.tab %>%
#'     dplyr::select(x.col, y.col, se) %>%
#'     tidyr::spread(key = y.col, value = se) %>%
#'     (function(x) matrix(x[1:p, seq(2, m + 1)], ncol = 1))
#'
#' out = fit.zqtl(xy.beta, xy.se, X,
#'                      vbiter = 3000,
#'                      gammax = 1e3,
#'                      pi = 0,
#'                      eigen.tol = 1e-2,
#'                      do.var.calc = TRUE,
#'                      scale.var = TRUE)
#'
#' plot(out$param$lodds, main = 'association', pch = 19, cex = .5, col = 'gray', ylab = 'log-odds')
#' points(c.snps, out$param$lodds[c.snps], col = 'red', pch = 19)
#' legend('topright', legend = c('causal', 'others'), pch = 19, col = c('red','gray'))
#'
#' .var = c(out$var$param$mean,
#'          out$var$conf.mult$mean,
#'          out$var$conf.uni$mean,
#'          out$var$resid$mean,
#'          out$var$tot)
#'
#' barplot(height = 10^(.var),
#'         col = 2:6,
#'         names.arg = c('gen', 'c1', 'c2', 'resid', 'tot'),
#'         horiz = TRUE,
#'         ylab = 'category',
#'         xlab = 'estimated variance')
#'
fit.zqtl <- function(effect,              # marginal effect : y ~ x
                     effect.se,           # marginal se : y ~ x
                     X,                   # X matrix
                     n = 0,               # sample size
                     A = NULL,            # annotation matrix
                     multi.C = NULL,      # covariate matrix (before LD)
                     univar.C = NULL,     # covariate matrix (already multiplied by LD)
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
                     out.residual = FALSE,
                     do.var.calc = FALSE,
                     nboot = 0,
                     nboot.var = 100,
                     scale.var = FALSE,
                     min.se = 1e-4,
                     rseed = NULL) {

    stopifnot(is.matrix(effect))
    stopifnot(is.matrix(effect.se))
    stopifnot(all(dim(effect) == dim(effect.se)))
    stopifnot(is.matrix(X))

    p = ncol(X)

    ## SNP confounding factors
    if(is.null(multi.C)) {
        p = nrow(effect)
        multi.C = matrix(1/p, p, 1)
    }

    if(is.null(univar.C)) {
        p = nrow(effect)
        univar.C = matrix(1/p, p, 1)
    }

    stopifnot(is.matrix(multi.C))
    stopifnot(nrow(effect) == nrow(multi.C))

    stopifnot(is.matrix(univar.C))
    stopifnot(nrow(effect) == nrow(univar.C))

    ## Override options
    opt.vars = c('do.hyper', 'do.rescale', 'tau', 'pi', 'tau.lb',
                 'tau.ub', 'pi.lb', 'pi.ub', 'tol', 'gammax', 'rate', 'decay',
                 'jitter', 'nsample', 'vbiter', 'verbose', 'k', 'svd.init', 'right.nn', 'mu.min',
                 'print.interv', 'nthread', 'eigen.tol', 'do.stdize', 'out.residual', 'min.se',
                 'rseed', 'do.var.calc', 'scale.var', 'nboot', 'nboot.var')

    .eval <- function(txt) eval(parse(text = txt))
    for(v in opt.vars) {
        val = .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] = val
        }
    }
    options[['sample.size']] = n

    annotated = FALSE
    if(!is.null(A)) {
        stopifnot(is.matrix(A))
        stopifnot(nrow(A) == nrow(effect))
        annotated = TRUE
    }

    ## call R/C++ functions ##
    if(annotated) {
        return(.Call('rcpp_annot_zqtl', effect, effect.se, X, A, multi.C, univar.C, options, PACKAGE = 'zqtl'))
    } else if(factored) {
        return(.Call('rcpp_fac_zqtl', effect, effect.se, X, multi.C, univar.C, options, PACKAGE = 'zqtl'))
    } else {
        return(.Call('rcpp_zqtl', effect, effect.se, X, multi.C, univar.C, options, PACKAGE = 'zqtl'))
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
#' @return a list of variational inference results.
#' \itemize{
#' \item{param.left: }{ parameters for the left factors}
#' \item{param.right: }{ parameters for the right factors}
#' \item{llik: }{ log-likelihood trace over the optimization}
#' }
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
#' library(Matrix)
#'
#' n = 500
#' p = 1000
#' m = 50
#'
#' set.seed(1)
#'
#' .rnorm <- function(a, b) matrix(rnorm(a * b), a, b)
#'
#' X = .rnorm(n, p)
#' X0 = X[, -(1:(p/2)), drop = FALSE]
#' X1 = X[, (1:(p/2)), drop = FALSE]
#'
#' Y1 = matrix(0, n, m)
#' Y = matrix(0, n, m)
#' h2 = 0.4
#'
#' c.snps = sample(p / 2, 3)
#'
#' ## shared genetic variants
#' theta.left = .rnorm(3, 1)
#' theta.right = .rnorm(1, 3)
#' theta.shared = theta.left %*% theta.right
#'
#' Y1[, 1:3] = Y1[, 1:3] + X[, c.snps, drop = FALSE] %*% theta.shared
#'
#' v0 = var(as.numeric(Y1[, 1:3]))
#' Y1[, -(1:3)] = .rnorm(n, m - 3) * sqrt(v0)
#' v1 = apply(Y1, 2, var)
#' Y1 = Y1 + sweep(.rnorm(n, m), 2, c(sqrt(v1 * (1/h2 - 1))), `*`)
#'
#' ## introduce confounding factors
#' uu = .rnorm(n, 5)
#' vv = .rnorm(m, 5)
#' Y0 = uu %*% t(vv)
#' Y = Y1 + Y0
#' Y = scale(Y)
#'
#' xy.beta = fast.cov(X, Y)
#' z.xy = fast.z.cov(X, Y)
#' xy.beta.se = xy.beta / (z.xy + 1e-4) + 1e-4
#'
#' vb.opt = list(tol = 0, vbiter = 2000, jitter = 1e-1,
#'                pi = -1, rate = 0.01, gammax = 1e3,
#'                eigen.tol = 1e-1, k = m, right.nn = FALSE)
#'
#' out = fit.zqtl.factorize(xy.beta, xy.beta.se, X, options = vb.opt)
#'
#' image(Matrix(head(out$param.left$theta, 20)))
#' image(Matrix(out$param.right$theta))
#'
#' y.hat = out$param.left$theta %*% t(out$param.right$theta)
#'
#' plot(as.numeric(scale(y.hat)), as.numeric(scale(Y0)), pch = 19, cex = .3)
#' plot(as.numeric(scale(y.hat)), as.numeric(scale(Y1)), pch = 19, cex = .3)
#' plot(as.numeric(scale(y.hat)), as.numeric(scale(Y)), pch = 19, cex = .3)
#'
#' ## pure null
#'
#' xy.beta = fast.cov(X0, scale(Y0))
#' z.xy = fast.z.cov(X0, scale(Y0))
#' xy.beta.se = xy.beta / (z.xy + 1e-4) + 1e-4
#'
#' out = fit.zqtl.factorize(xy.beta, xy.beta.se, X0, options = vb.opt)
#'
#' image(Matrix(head(out$param.left$theta, 20)))
#' image(Matrix(out$param.right$theta))
#'
#' y.hat = out$param.left$theta %*% t(out$param.right$theta)
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
    opt.vars = c('do.hyper', 'do.rescale', 'tau', 'pi', 'tau.lb',
                 'tau.ub', 'pi.lb', 'pi.ub', 'tol', 'gammax', 'rate', 'decay',
                 'jitter', 'nsample', 'vbiter', 'verbose', 'k', 'svd.init', 'right.nn', 'mu.min',
                 'print.interv', 'nthread', 'eigen.tol', 'do.stdize', 'min.se',
                 'rseed', 'factorization.model')

    .eval <- function(txt) eval(parse(text = txt))
    for(v in opt.vars) {
        val = .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] = val
        }
    }
    options[['sample.size']] = n

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
#' @param multi.C SNP confounding factors (SNP x confounder; default: NULL)#'
#' @param factored Fit factored model (default: FALSE)
#' @param options A list of inference/optimization options.
#' @param multivar.mediator Multivariate mediator QTL effect (default: FALSE)
#'
#' @param do.direct.estimation Estimate direct effect (default: TRUE)
#'
#' @param de.factorization Estimate direct effect by joint factorization (default: TRUE)
#' @param factorization.model Factorization model; 0 = ind x factor, 1 = eigen x factor (default: 0)
#' @param de.propensity Propensity sampling to estimate direct effect (default: FALSE)
#'
#' @param do.finemap.direct Fine-map direct effect SNPs (default: FALSE)
#' @param nboot Number of bootstraps followed by finemapping (default: 0)
#' @param nboot.var Number of bootstraps for variance estimation (default: 100)
#' @param scale.var Scaled variance calculation (default: FALSE)
#' @param num.conditional number of conditional models
#' @param submodel.size size of each conditional model
#'
#' @param out.residual estimate residual z-scores (default: FALSE)
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
#' @return a list of variational inference results.
#' \itemize{
#' \item{param.mediated: }{ parameters for the mediated effects}
#' \item{param.unmediated: }{ parameters for the unmediated effects}
#' \item{param.intercept: }{ parameters for the intercept effect}
#' \item{param.covariate: }{ parameters for the multivariate covariate}
#' \item{param.covariate.uni: }{ parameters for the univariate covariate}
#' \item{bootstrap: }{ bootstrapped mediated parameters (if nboot > 0)}
#' \item{llik: }{ log-likelihood trace over the optimization}
#' }
#'
#' @author Yongjin Park, \email{ypp@@csail.mit.edu}, \email{yongjin.peter.park@@gmail.com}
#'
#' @details
#'
#' Mediation analysis
#'
#' @examples
#'
#' n = 500
#' p = 1000
#' n.genes = 10
#' h2 = 0.8
#' n.causal.direct = 1
#' n.causal.qtl = 3
#' n.causal.gene = 2
#' set.seed(13)
#'
#' .rnorm <- function(a, b) matrix(rnorm(a * b), a, b)
#'
#' X.raw = sapply(1:p, function(j) {
#'     f = runif(1, min = 0.1, max = 0.9)
#'     rbinom(n, 2, f)
#' })
#'
#' X = scale(X.raw)
#'
#' c.qtls = matrix(sapply(1:n.genes, function(j) matrix(sample(p, n.causal.qtl))), nrow = n.causal.qtl)
#'
#' theta.snp.gene = .rnorm(n.causal.qtl, n.genes) / n.causal.qtl
#'
#' gene.expr = lapply(1:dim(c.qtls)[2], function(j) X[, c.qtls[,j], drop = FALSE] %*% theta.snp.gene[, j, drop = FALSE] + 0.5 * rnorm(n) )
#'
#' gene.expr = do.call(cbind, gene.expr)
#'
#' theta.med = sample(c(-1,1), n.causal.gene, TRUE)
#'
#' causal.direct = sample(p, n.causal.direct)
#' theta.dir = matrix(rnorm(n.causal.direct), n.causal.direct, 1) / n.causal.direct
#'
#' y = gene.expr[,1:n.causal.gene,drop = FALSE] %*% theta.med # mediation
#' y = y + X[, causal.direct, drop = FALSE] %*% theta.dir # direct
#' y = y + rnorm(n) * c(sqrt(var(y) * (1/h2 - 1)))
#'
#' eqtl.tab = calc.qtl.stat(X, gene.expr)
#' gwas.tab = calc.qtl.stat(X, y)
#'
#' xg.beta = eqtl.tab %>%
#'     dplyr::select(x.col, y.col, beta) %>%
#'     tidyr::spread(key = y.col, value = beta) %>%
#'     (function(x) as.matrix(x[1:p, -1]))
#'
#' xg.se = eqtl.tab %>%
#'     dplyr::select(x.col, y.col, se) %>%
#'     tidyr::spread(key = y.col, value = se) %>%
#'     (function(x) as.matrix(x[1:p, -1]))
#'
#' xy.beta = gwas.tab %>%
#'     dplyr::select(x.col, y.col, beta) %>%
#'     tidyr::spread(key = y.col, value = beta) %>%
#'     (function(x) as.matrix(x[1:p, -1]))
#'
#' xy.se = gwas.tab %>%
#'     dplyr::select(x.col, y.col, se) %>%
#'     tidyr::spread(key = y.col, value = se) %>%
#'     (function(x) as.matrix(x[1:p, -1]))
#'
#' vb.opt = list(nsample = 10,
#'               vbiter = 3000,
#'               rate = 1e-2,
#'               gammax = 1e3,
#'               do.stdize = TRUE,
#'               pi = -0, tau = -4,
#'               do.hyper = FALSE,
#'               eigen.tol = 1e-2,
#'               tol = 1e-8,
#'               verbose = TRUE,
#'               print.interv = 200,
#'               multivar.mediator = FALSE)
#'
#' med.out = fit.med.zqtl(effect = xy.beta,
#'                              effect.se = xy.se,
#'                              effect.m = xg.beta,
#'                              effect.m.se = xg.se,
#'                              X.gwas = X,
#'                              options = vb.opt)
#'
#' barplot(height = med.out$param.mediated$lodds[, 1],
#'         names.arg = paste('g',1:n.genes),
#'         horiz = TRUE, xlab = 'log-odds', ylab = 'mediators')
#'
fit.med.zqtl <- function(effect,              # marginal effect : y ~ x
                         effect.se,           # marginal se : y ~ x
                         effect.m,            # marginal : m ~ x
                         effect.m.se,         # marginal se : m ~ x
                         X.gwas,              # X matrix for GWAS
                         X.med = NULL,        # X matrix for mediation
                         n = 0,               # sample size of effect
                         n.med = 0,           # sample size of effect.m
                         multi.C = NULL,      # multivariate covariate matrix
                         univar.C = NULL,     # univariate covariate matrix
                         factored = FALSE,    # Factored model
                         options = list(),
                         multivar.mediator = FALSE,
                         de.propensity = FALSE,
                         de.factorization = TRUE,
                         factorization.model = 0,
                         num.strat.size = 2,
                         do.direct.estimation = TRUE,
                         do.finemap.direct = FALSE,
                         nboot = 0,
                         nboot.var = 100,
                         scale.var = FALSE,
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
                         out.residual = FALSE,
                         min.se = 1e-4,
                         rseed = NULL) {

    stopifnot(is.matrix(effect))
    stopifnot(is.matrix(effect.se))
    stopifnot(all(dim(effect) == dim(effect.se)))
    stopifnot(is.matrix(X.gwas))

    p = ncol(X.gwas)

    ## SNP confounding factors
    if(is.null(multi.C)) {
        p = nrow(effect)
        multi.C = matrix(1/p, p, 1)
    }

    ## SNP confounding factors
    if(is.null(univar.C)) {
        univar.C = matrix(0, p, 1)
    }

    ## X.gwas == X.med
    if(is.null(X.med)) {
        X.med = X.gwas
    }

    stopifnot(is.matrix(multi.C))
    stopifnot(nrow(effect) == nrow(multi.C))

    stopifnot(is.matrix(univar.C))
    stopifnot(nrow(effect) == nrow(univar.C))

    ## Override options
    opt.vars = c('do.hyper', 'do.rescale', 'tau', 'pi', 'tau.lb',
                 'tau.ub', 'pi.lb', 'pi.ub', 'tol', 'gammax', 'rate', 'decay',
                 'jitter', 'nsample', 'vbiter', 'verbose', 'k', 'svd.init',
                 'print.interv', 'nthread', 'eigen.tol', 'do.stdize', 'out.residual', 'min.se',
                 'rseed', 'do.var.calc', 'num.strat.size', 'nboot', 'nboot.var', 'scale.var',
                 'multivar.mediator', 'de.propensity', 'de.factorization', 'factorization.model',
                 'do.direct.estimation', 'do.finemap.direct',
                 'med.lodds.cutoff', 'num.duplicate.sample', 'num.conditional',
                 'submodel.size')

    .eval <- function(txt) eval(parse(text = txt))
    for(v in opt.vars) {
        val = .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] = val
        }
    }

    options = c(options, list(sample.size = n, med.sample.size = n.med))

    stopifnot(nrow(effect.m) == nrow(effect.m.se))
    stopifnot(ncol(effect.m) == ncol(effect.m.se))

    ## call R/C++ functions ##
    if(factored) {
        ret = .Call('rcpp_fac_med_zqtl', PACKAGE = 'zqtl',
                    effect,
                    effect.se,
                    effect.m,
                    effect.m.se,
                    X.gwas,
                    X.med,
                    multi.C,
                    univar.C,
                    options)
    } else {
        ret = .Call('rcpp_med_zqtl', PACKAGE = 'zqtl',
                    effect,
                    effect.se,
                    effect.m,
                    effect.m.se,
                    X.gwas,
                    X.med,
                    multi.C,
                    univar.C,
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
#' @usage take.ld.svd(X, options, eigen.tol, do.stdize)
#'
#' @param X n x p matrix
#' @param options a list of options
#' @param eigen.tol Error tolerance in Eigen decomposition (default: 0.01)
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
                        do.stdize = TRUE){

    opt.vars = c('eigen.tol', 'do.stdize')

    .eval <- function(txt) eval(parse(text = txt))
    for(v in opt.vars) {
        val = .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] = val
        }
    }

    return(.Call('rcpp_take_svd_xtx', PACKAGE = 'zqtl', X, options));
}

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
#' @usage log.msg(...)
#' @param ... input for sprintf
#'
#' @export
#'
#' @usage
#'
#' log.msg('Test Message %d', 1)
#'
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
#'
#' @return summary statistics matrix
#'
#' @export
#'
calc.qtl.stat <- function(xx, yy, se.min = 1e-8) {

    require(dplyr)
    require(tidyr)

    .xx = scale(xx)
    .yy = scale(yy)

    rm.na.zero <- function(xx) {
        return(replace(xx, is.na(xx), 0))
    }

    ## cross-product is much faster than covariance function
    n.obs = crossprod(!is.na(.xx), !is.na(.yy))
    beta.mat = crossprod(.xx %>% rm.na.zero(), .yy %>% rm.na.zero()) / n.obs

    log.msg('Computed cross-products')

    ## residual standard deviation
    resid.se.mat = matrix(NA, ncol(.xx), ncol(.yy))

    for(k in 1:ncol(.yy)) {

        beta.k = beta.mat[, k]
        yy.k = .yy[, k]
        err.k = sweep(sweep(.xx, 2, beta.k, `*`), 1, yy.k, `-`)
        se.k = apply(err.k, 2, sd, na.rm = TRUE)

        log.msg('Residual on the column %d', k)
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
