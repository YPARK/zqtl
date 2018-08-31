# Summary-based QTL mapping

## Notes on Mac OS X users

For some unknown reason, compiled Eigen library does not work as
expected with `clang` compiler.  We recommend using `gcc` after
version 5.

## Installation (a simple version)

Just do this in `R`
```
> install.packages('zqtl_x.x.x.tar.gz')
```

## Installation (optional)

Prerequisite: `Rcpp`, `RcppEigen`, `RcppProgress` packages

Make sure your R development environment support `C++14` by including
`-std=c++14` to `CFLAGS` and `CXXFLAGS` in `~/.R/Makevars` file.
For instance,
```
CXX = g++-6
CXXFLAGS = -O3 -std=c++14
CFLAGS = -O3 -std=c++14
```

Build package locally.
```
$ R CMD build .
```

You will have `zqtl_x.x.x.tar.gz` gzipped file in current directory
(`x.x.x` can be any version of the package).  Install package within
R:

```
> install.packages('zqtl_x.x.x.tar.gz')
```

To speed up matrix-vector multiplication, one can compile codes with
Intel MKL library.  Currently `RcppEigen` does not support `BLAS` only
options.

Enjoy!

## Release notes

### v.1.3.1

- Make propensity sampling as second option
- Estimate the unmediated effect as before
- Drop weight features (not so useful)
- Allow multiple mediators in the conditional analysis

### v.1.3.0

- Counterfactual estimation of average unmediated effects via sampling
- Factored mediation model

### v.1.2.3

- Take multivariate effect sizes for mediator QTLs
- Initialization by dot product

### v.1.2.2

- Restrict number of mediator variables during the estimation of unmediated effects

### v.1.2.1

- Estimation of average unmediated effects
- Minor bugfix in regression (twice rescaling)

### v.1.2.0

- Matrix factorization for confounder correction
- Confirmed usefulness of non-negative parameters
- Simplified pleiotropy model in mediation analysis
- Adjust scales by standard deviation
- Variance model in mediation analysis

### v.1.1.0

- Spiked Gamma for the factored zQTL methods
- Additional covariate component for confounder correction

### v.1.0.1

- Variance calculation. See `note/variance.md` for details.
- Random effect component to account for uncertainty of individuals
- Removing LD-structure bias between mediation QTL and GWAS cohorts

### v.1.0.0

- Initial version migrated from MIT github

# Bug reports

Yongjin Park `ypp@csail.mit.edu`

