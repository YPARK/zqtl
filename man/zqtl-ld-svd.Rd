\name{SVD of LD matrix}
\alias{take.ld.svd}
\title{Calculate covariance matrix and perform SVD}
\description{
  Decompose X/sqrt(n) = U D V' so that
  LD = V D^2 V' for subsequent analysis.
}
\usage{
take.ld.pairs(X, options=list())
}
\arguments{
  \item{X}{n x p matrix}
  \item{options=list(eigen.tol = 1e-8)}{numerical tolerance of SVD}
  \item{options=list(do.stdize = TRUE)}{standardized LD matrix}
}
\value{
  returns list(D, V.t)
}
\details{
}
\references{
}
\examples{
}
\author{Yongjin Park}

