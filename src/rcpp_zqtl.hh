// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
// [[Rcpp::depends(RcppZiggurat)]]
#include <Ziggurat.h>

using namespace Rcpp;

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "convergence.hh"
#include "mediation.hh"
#include "options.hh"
#include "parameters.hh"
#include "rcpp_util.hh"
#include "regression.hh"
#include "regression_factored.hh"
#include "regression_mediated.hh"
#include "residual.hh"
#include "sgvb_mediation_inference.hh"
#include "sgvb_regression_inference.hh"
#include "tuple_util.hh"
#include "zqtl_model.hh"

#ifndef RCPP_ZQTL_HH_
#define RCPP_ZQTL_HH_

using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using Vec = Eigen::Matrix<float, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<float, Eigen::ColMajor>;
using Scalar = Mat::Scalar;
using Index = Mat::Index;

// "util" header must come first
#include "eigen_sampler.hh"
#include "rcpp_zqtl_util.hh"

// mediation and regression headers
#include "rcpp_zqtl_factorization.hh"
#include "rcpp_zqtl_mediation.hh"
#include "rcpp_zqtl_regression.hh"

#endif
