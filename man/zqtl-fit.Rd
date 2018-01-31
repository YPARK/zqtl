\name{fit.zqtl}
\alias{fit.zqtl}
\title{Variational inference of zQTL models}
\description{

  Estimate true effect matrix from marginal effect sizes and standard
  errors (Hormozdiari et al., 2015; Zhu and Stephens, 2016):

  \deqn{\mathbf{Z}_{t} \sim \mathcal{N}\!\left(R E^{-1}
    \boldsymbol{\theta}_{t}, R\right)}{z[,t] ~ N(R inv(E) Theta[,t], R)}

  where R is \eqn{p \times p}{p x p} LD / covariance matrix; E is expected
  squared effect size matrix
  (\eqn{\textsf{se}[\boldsymbol{\theta}_{t}^{\textsf{marg}}] + n^{-1}
    \langle \boldsymbol{\theta}_{t}^{\textsf{marg}} \rangle^{2}}{standard error
    + effect^2/n} matrix, diagonal); \eqn{\mathbf{z}_{t}}{z[,t]} is \eqn{p
    \times 1}{p x 1} z-score vector of trait \eqn{t}{t}, or \eqn{\mathbf{z}_{t} =
    \boldsymbol{\theta}_{t}^{\textsf{marg}}/
    \textsf{se}[\boldsymbol{\theta}_{t}^{\textsf{marg}}]}{z = theta.marg /
    se[theta.marg]}.
}
\usage{
fit.zqtl(effect,              # marginal effect : y ~ x
         effect.se,           # marginal se : y ~ x
         X,                   # X matrix
         n = 0,               # sample size
         C = NULL,            # covariate matrix
         factored = FALSE,    # Factored multiple traits
         options = list())
}
\arguments{
  \item{effect}{Marginal effect size matrix (SNP x trait)}
  \item{effect.se}{Marginal effect size standard error matrix (SNP x trait)}
  \item{n}{sample size of actual data (will ignore if n = 0)}
  \item{X}{Design matrix (reference Ind x SNP)}
  \item{C}{SNP confounding factors (SNP x confounder; default: NULL)}
  \item{factored}{Fit factored QTL model (default: FALSE)}
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
  \item{right.nn}{non-negativity in factored effect (default: FALSE)}
  \item{mu.min}{mininum non-negativity weight (default: 0.01)}

  \item{print.interv}{Printing interval (default: 10)}
  \item{nthread}{Number of threads during calculation (default: 1)}
  \item{eigen.tol}{Error tolerance in Eigen decomposition (default: 0.1)}
  \item{do.stdize}{Standardize (default: TRUE)}
  \item{min.se}{Minimum level of SE}
  \item{rseed}{Random seed}
  \item{weight.y}{Use 1/effect.se as weight in the linear model (default: TRUE)}
}
\value{
  \code{fit.zqtl} returns a list of variational parameters
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

X <- as.matrix(apply(X.raw, 2, function(x) x - mean(x, na.rm = TRUE)))
Y <- matrix(0, n, m)
h2 <- 0.4

c.snps <- sample(p, 3)

## shared genetic variants
theta.left <- sign(.rnorm(3, 1))
theta.right <- abs(.rnorm(1, 3))
theta.shared <- theta.left \%*\% theta.right

Y[, 1:3] <- Y[, 1:3] + X[, c.snps, drop = FALSE] \%*\% theta.shared

v0 <- var(as.numeric(Y[, 1:3]))
v1 <- apply(Y, 2, var)
Y[, -(1:3)] <- .rnorm(n, m - 3) * sqrt(v0)
Y <- Y + sweep(.rnorm(n, m), 2, c(sqrt(v1 * (1/h2 - 1))), `*`)
Y <- scale(Y)

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

vb.opt <- list(tol = 0, vbiter = 3000, jitter = 1e-2,
               pi = -1, rate = 0.01, gammax = 1e3,
               eigen.tol = 1e-1, k = 12)

out <- fit.zqtl(xy.beta, xy.beta.se, X, factored = TRUE, options = vb.opt)

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

vb.opt <- list(tol = 0, vbiter = 3000, jitter = 1e-2,
               pi = -1, rate = 0.01, gammax = 1e3,
               eigen.tol = 1e-1, k = 12, right.nn = TRUE)

out <- fit.zqtl(xy.beta, xy.beta.se, X, factored = TRUE, options = vb.opt)

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
\details{

  In basic zQTL model, parameter matrix, theta, can be decomposed into two
  parts: sparse strong effect and background polygenic effect matrices:

  \deqn{\boldsymbol{\theta}_{t} =
    \boldsymbol{\theta}_{t}^{\textsf{sparse}}}{theta = theta_sparse}

  Not to overfit data, each element of sparse coefficient,
  \eqn{\theta_{jt}}{theta[j,t]}, follows spike-slab prior.  We carry out
  posterior inference by variational inference with surrogate distribution
  first introduced in Carbonetto and Stephens (2012):

  \deqn{q(\theta|\alpha,\beta,\gamma) = \alpha
    \mathcal{N}\!\left(\beta,\gamma^{-1}\right)}{q(theta|.) = alpha *
    N(beta, 1/gamma)}

  We reparameterized \eqn{\alpha = \boldsymbol{\sigma}\!\left(\pi +
    \delta\right)}{alpha = sigmoid(pi + delta)}, and \eqn{\gamma =
    \gamma_{\textsf{max}}\boldsymbol{\sigma}\!\left(- \tau + \lambda
    \right)}{gamma = gammax * sigmoid(- tau + lambda)} for numerical
  stability.

  In factored zQTL model, we decompose sparse effect:
  \deqn{\boldsymbol{\theta}_{t} = \boldsymbol{\theta}^{\textsf{left}}
    \boldsymbol{\theta}_{t}^{\textsf{right}}}{theta = theta_left *
    theta_right}

}
\seealso{}
\keyword{}
