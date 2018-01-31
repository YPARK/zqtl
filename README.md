# Summary-based QTL mapping

## Installation

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
R CMD build .
```

You will have `zqtl_x.x.x.tar.gz` gzipped file in current directory
(`x.x.x` can be any version of the package).  Install package within
R:

```
install.packages('zqtl_x.x.x.tar.gz')
```

To speed up matrix-vector multiplication, one can compile codes with
Intel MKL library.  Currently `RcppEigen` does not support `BLAS` only
options.

Enjoy!

## Release notes

### v.1.1.0

- Spiked Gamma for the factored zQTL methods

### v.1.0.1

- Variance calculation. See `note/variance.md` for details.
- Random effect component to account for uncertainty of individuals
- Removing LD-structure bias between mediation QTL and GWAS cohorts

### v.1.0.0

- Initial version migrated from MIT github

# Bug reports

Yongjin Park `ypp@csail.mit.edu`

