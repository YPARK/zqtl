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
         C = NULL,            # covariate matrix of multivariate effect
         C.delta = NULL,      # covariate matrix of univariate effect
         factored = FALSE,    # Factored multiple traits
         options = list())
}
\arguments{
  \item{effect}{Marginal effect size matrix (SNP x trait)}
  \item{effect.se}{Marginal effect size standard error matrix (SNP x trait)}
  \item{n}{sample size of actual data (will ignore if n = 0)}
  \item{X}{Design matrix (reference Ind x SNP)}
  \item{C}{SNP confounding factors (SNP x confounder; default: NULL)}
  \item{C.delta}{SNP confounding factors (SNP x confounder; default: NULL)}
  \item{factored}{Fit factored QTL model (default: FALSE)}
  \item{options}{A list of inference/optimization options.}

  \item{do.hyper}{Hyper parameter tunign (default: FALSE)}
  \item{do.rescale}{Rescale z-scores by standard deviation (default: FALSE)}
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
  \item{re.k}{Rank of the random effect model (default: 1)}

  \item{right.nn}{non-negativity in factored effect (default: TRUE)}
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
library(Matrix)

n <- 500
p <- 2000
m <- 50

set.seed(1)

.rnorm <- function(a, b) matrix(rnorm(a * b), a, b)

X <- .rnorm(n, p)
X0 <- X[, -(1:(p/2)), drop = FALSE]
X1 <- X[, (1:(p/2)), drop = FALSE]

Y1 <- matrix(0, n, m)
Y <- matrix(0, n, m)
h2 <- 0.4

c.snps <- sample(p / 2, 3)

## shared genetic variants
theta.left <- .rnorm(3, 1)
theta.right <- .rnorm(1, 3)
theta.shared <- theta.left \%*\% theta.right

Y1[, 1:3] <- Y1[, 1:3] + X[, c.snps, drop = FALSE] \%*\% theta.shared

v0 <- var(as.numeric(Y1[, 1:3]))
Y1[, -(1:3)] <- .rnorm(n, m - 3) * sqrt(v0)
v1 <- apply(Y1, 2, var)
Y1 <- Y1 + sweep(.rnorm(n, m), 2, c(sqrt(v1 * (1/h2 - 1))), `*`)

## introduce confounding factors
uu <- .rnorm(n, 5)
vv <- .rnorm(m, 5)
Y0 <- uu \%*\% t(vv)
Y <- Y1 + Y0

## just caculate univariate z-scores
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

xy.beta <- fast.cov(X, scale(Y))
z.xy <- fast.z.cov(X, scale(Y))
xy.beta.se <- xy.beta / (z.xy + 1e-4) + 1e-4

vb.opt <- list(tol = 0, vbiter = 3000, jitter = 0.01,
               pi = -2, tau = -8, rate = 0.01, gammax = 1e3,
               eigen.tol = 1e-1, k = m, re.k = m, nboot = 0,
               svd.init = TRUE)

out <- fit.zqtl(xy.beta, xy.beta.se, X, factored = TRUE, options = vb.opt)

## show associations
.rnd <- c(c.snps, setdiff(sample(p, 20), c.snps))
image(Matrix(out$param.left$lodds[.rnd, ]))
image(Matrix(out$param.right$lodds))

## show confounders
image(Matrix(head(out$rand.indiv$theta, 20)))
image(Matrix(out$rand.trait$theta))

y.hat <- out$rand.indiv$theta \%*\% t(out$rand.trait$theta)

plot(as.numeric(scale(y.hat)), as.numeric(scale(Y0)), pch = 19, cex = .3)
plot(as.numeric(scale(y.hat)), as.numeric(scale(Y1)), pch = 19, cex = .3)
plot(as.numeric(scale(y.hat)), as.numeric(scale(Y)), pch = 19, cex = .3)

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
