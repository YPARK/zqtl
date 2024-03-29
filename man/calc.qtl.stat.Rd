% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Util.R
\name{calc.qtl.stat}
\alias{calc.qtl.stat}
\title{calculate univariate effect sizes and p-values}
\usage{
calc.qtl.stat(xx, yy)
}
\arguments{
\item{xx}{n x p genotype matrix}

\item{yy}{n x t phenotype matrix}

\item{se.min}{mininum standard error (default: 1e-8)}

\item{verbose}{(default: FALSE)}
}
\value{
summary statistics matrix
}
\description{
calculate univariate effect sizes and p-values
}
