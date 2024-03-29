% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Util.R
\name{take.ld.svd}
\alias{take.ld.svd}
\title{Decompose the scaled genotype matrix.}
\usage{
take.ld.svd(X, options, eigen.tol, eigen.reg, do.stdize)
}
\arguments{
\item{X}{n x p matrix}

\item{options}{a list of options}

\item{eigen.tol}{Error tolerance in Eigen decomposition (default: 0.01)}

\item{eigen.reg}{Regularization of Eigen decomposition (default: 0.0)}

\item{do.stdize}{Standardize (default: TRUE)}
}
\description{
Decompose the scaled genotype matrix.
}
\details{
Decompose \eqn{n^{-1/2}X = U D V^{\top}}{X/sqrt(n) = U D V'}
such that the LD matrix can become
\deqn{R = V D^{2} V^{\top}}{LD = V D^2 V'} for subsequent analysis.
}
