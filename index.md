# Summary-based QTL mapping

## Notes on Mac OS

For some unknown reasons, compiled Eigen library does not work as
expected with the `clang` compiler.  The matrix operations keep
producing lots of `nan` results.  We recommend using `gcc` after
version 5 that fully support `std=c++14` flag.

## Installation (a simple version)

Just do this in `R`
```
> install.packages('zqtl_x.x.x.tar.gz')
```

## Installation (optional)

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
options (this may not be true).

Enjoy!

# Release notes

### v 1.4.2

- Add faster SVD routine for a large genotype matrix
- Bugfix in randomized SVD

### v 1.4.1

- Regularization of eigen values (the `eigen.reg` parameter)

### v 1.4.0

- Add useful utilities

### v.1.3.6

- Improve variance calculation
- Partitioned variance calculation with annotation information

### v.1.3.5

- Report both types of residual variance estimation
- Lower the precision limit in SGVB steps
- Add adjusting routines for convenience


### v.1.3.4

- Remove "backfire control" and "two step optimization"
- Revive variance calculation with residual estimation
- Make a room for univariate confounding factors
- Output "clean" version of GWAS effects

### v.1.3.3

- Minor fix on smoothness of the unmediated effect
- Add `do.control.backfire` to regress out `M0 -> M`.
- Add experimental factorization model `factorization.model = 1`

### v.1.3.2

- Estimate unmediated factors by factorization
- Boost sample size by resampling eigen vectors
- Allow two step optimization
- Parametric bootstrap for sensitivity analysis

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

Yongjin Park `ypp@stat.ubc.ca`

