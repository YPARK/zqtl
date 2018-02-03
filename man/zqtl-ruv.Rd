\name{fit.zqtl.ruv}
\alias{fit.zqtl.ruv}
\title{Variational inference of zQTL models}
\description{
Identification of potential confounders across multiple GWAS statistics
}
\usage{
fit.zqtl.ruv(effect,              # marginal effect : y ~ x
            effect.se,           # marginal se : y ~ x
            X,                   # X matrix
            n = 0,               # sample size
            options = list())
}
\arguments{
  \item{effect}{Marginal effect size matrix (SNP x trait)}
  \item{effect.se}{Marginal effect size standard error matrix (SNP x trait)}
  \item{n}{sample size of actual data (will ignore if n = 0)}
  \item{X}{Design matrix (reference Ind x SNP)}
  \item{options}{A list of inference/optimization options.}

  \item{do.hyper}{Hyper parameter tunign (default: FALSE)}
  \item{tau}{Fixed value of tau}
  \item{pi}{Fixed value of pi}

  \item{tau.lb}{Lower-bound of tau (default: -10)}
  \item{tau.ub}{Upper-bound of tau (default: -4)}
  \item{pi.lb}{Lower-bound of pi (default: -4)}
  \item{pi.ub}{Upper-bound of pi (default: -1)}
  \item{tol}{Convergence criterion (default: 1e-4)}
  \item{gammax}{Maximum precision (default: 1000)}
  \item{rate}{Update rate (default: 1e-2)}
  \item{decay}{Update rate decay (default: 0)}
  \item{jitter}{SD of random jitter for mediation & factorization (default: 0.01)}
  \item{nsample}{Number of stochastic samples (default: 10)}
  \item{vbiter}{Number of variational Bayes iterations (default: 2000)}
  \item{verbose}{Verbosity (default: TRUE)}
  \item{k}{Rank of the factored model (default: 1)}

  \item{out.resid}{output residuals (default: TRUE)}
  \item{print.interv}{Printing interval (default: 10)}
  \item{nthread}{Number of threads during calculation (default: 1)}
  \item{eigen.tol}{Error tolerance in Eigen decomposition (default: 0.1)}
  \item{do.stdize}{Standardize (default: TRUE)}
  \item{min.se}{Minimum level of SE}
  \item{rseed}{Random seed}
}
\value{
  \code{fit.zqtl.ruv} returns a list of variational parameters
}
\examples{

library(zqtl)

n <- 1000
p <- 1000
m <- 50

set.seed(1)
X.raw <- sapply(1:p, function(j) {
    f <- runif(1, min = 0.1, max = 0.9)
    rbinom(n, 2, f)
})

.rnorm <- function(a, b) matrix(rnorm(a * b), a, b)

X <- as.matrix(scale(X.raw))
Y <- matrix(0, n, m)
h2 <- 0.4

c.snps <- sample(p, 3)

## shared genetic variants
theta.left <- .rnorm(3, 2)
theta.right <- .rnorm(2, 3)
theta.shared <- theta.left \%*\% theta.right

Y[, 1:3] <- Y[, 1:3] + X[, c.snps, drop = FALSE] \%*\% theta.shared

v0 <- var(as.numeric(Y[, 1:3]))
v1 <- apply(Y, 2, var)
Y[, -(1:3)] <- .rnorm(n, m - 3) * sqrt(v0)
Y <- Y + sweep(.rnorm(n, m), 2, c(sqrt(v1 * (1/h2 - 1))), `*`)

Y <- scale(Y)

## introduce confounding factors
uu <- .rnorm(n, 1)
vv <- .rnorm(m, 1)
Y <- Y + uu \%*\% t(vv)

fast.cov <- function(x, y) {
    n.obs <- crossprod(!is.na(x), !is.na(y))
    ret <- crossprod(replace(x, is.na(x), 0),
                     replace(y, is.na(y), 0)) / n.obs
    return(ret)
}

fast.z.cov <- function(x, y) {
    n.obs <- crossprod(!is.na(x), !is.na(y))
    ret <- crossprod(replace(x, is.na(x), 0),
                     replace(y, is.na(y), 0)) / sqrt(n.obs)
    return(ret)
}

xy.beta <- fast.cov(X, Y)
z.xy <- fast.z.cov(X, Y)
xy.beta.se <- xy.beta / z.xy

vb.opt <- list(tol = 0, vbiter = 1500, jitter = 1e-2,
               pi = 0, rate = 0.01, gammax = 1e4,
               out.resid = TRUE, 
               eigen.tol = 1e-2, k = 10)

library(Matrix)
out <- fit.zqtl.ruv(xy.beta, xy.beta.se, X, options = vb.opt)
Z.conf <- out$Z.confounder

image(Matrix(head(Z.conf, 20)), main = 'Z.conf')
image(Matrix(head(out$step2$param.indiv$theta, 20)), main = 'left')
image(Matrix(out$step2$param.trait$theta), main = 'right')

################################################################
## run factord QTL on residuals
vb.opt <- list(tol = 0, vbiter = 1500, jitter = 1e-2,
               pi = -1, rate = 0.01, gammax = 1e4,
               eigen.tol = 1e-2, k = 12)

out <- fit.zqtl(out$effect.corrected, xy.beta.se, X, factored = TRUE, options = vb.opt)

print(out$conf.delta)

par(mfrow = c(3,4))
for(k in 1:12) {
    pip <- 1/(1+exp(-out$param.left$lodds[, k]))
    plot(pip, main = k, ylim = c(0, 1))
    points(c.snps, pip[c.snps], col = 2, pch = 19)
}

par(mfrow = c(3,4))
for(k in 1:12) {
    plot(out$param.right$theta[, k], main = k)
}

## run factord QTL with covariates
vb.opt <- list(tol = 0, vbiter = 1500, jitter = 1e-2,
               pi = -1, rate = 0.01, gammax = 1e4,
               eigen.tol = 1e-2, k = 12)

out <- fit.zqtl(xy.beta, xy.beta.se, X, C.delta = Z.conf, factored = TRUE, options = vb.opt)

print(out$conf.delta)

par(mfrow = c(3,4))
for(k in 1:12) {
    pip <- 1/(1+exp(-out$param.left$lodds[, k]))
    plot(pip, main = k, ylim = c(0, 1))
    points(c.snps, pip[c.snps], col = 2, pch = 19)
}

par(mfrow = c(3,4))
for(k in 1:12) {
    plot(out$param.right$theta[, k], main = k)
}

}

\details{}
\seealso{}
\keyword{}
