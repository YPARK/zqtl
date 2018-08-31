////////////////////////////////////////////////////////////////
// A wrapper for eta = X * ThetaL * ThetaR' in Y ~ f(eta)
//
// (1) Theta is only created once, and referenced by many eta's.
// (2) Many eta's can be created to accommodate different random
// selection of data points (i.e., rows).
//

#include <memory>
#include <type_traits>

#include "eigen_util.hh"
#include "gaus_repr.hh"
#include "parameters.hh"
#include "rcpp_util.hh"

#ifndef MEDIATED_REGRESSION_HH_
#define MEDIATED_REGRESSION_HH_

////////////////////////////////////////////////////////////////
// X -> Y: regression of Y on X
// X * ThetaL * ThetaR'
//
// statistics of ThetaL is provided:
// ThetaLMean and ThetaLVar
//
template <typename Repr, typename ParamLeftMatrix, typename ParamRight>
struct mediated_regression_t {
  using ParamRightMatrix = typename param_traits<ParamRight>::Matrix;
  using Scalar = typename param_traits<ParamRight>::Scalar;
  using Index = typename param_traits<ParamRight>::Index;
  using ReprMatrix = typename Repr::DataMatrix;

  explicit mediated_regression_t(const ReprMatrix& xx,               // n x p
                                 const ReprMatrix& yy,               // n x m
                                 const ParamLeftMatrix& thetaLMean,  // p x k
                                 const ParamLeftMatrix& thetaLVar,   // p x k
                                 ParamRight& thetaR)
      : n(xx.rows()),
        p(xx.cols()),
        m(yy.cols()),
        k(thetaLMean.cols()),
        NobsR(m, k),
        ThetaLMean(thetaLMean),
        ThetaLVar(thetaLVar),
        ThetaR(thetaR),
        ThetaLMeanSq(p, k),
        ThetaRMeanSq(m, k),
        X(n, p),
        Xsq(n, p),
        G1R(m, k),
        G2R(m, k),
        Eta(make_gaus_repr(yy)) {
#ifdef DEBUG
    check_dim(ThetaLMean, p, k, "ThetaLMean in mediated_regression_t");
    check_dim(ThetaLVar, p, k, "ThetaLVar in mediated_regression_t");
    check_dim(ThetaR, m, k, "ThetaR in mediated_regression_t");
    check_dim(Eta, n, m, "Eta in mediated_regression_t");
#endif

    copy_matrix(mean_param(ThetaR), NobsR);

    // 1. compute Nobs
    // NobsR = O[Y'] * O[X] * O[L] (m x k)
    XYZ_nobs(yy.transpose(), xx, ThetaLMean, NobsR);

    // 2. copy X and Xsq removing missing values
    remove_missing(xx, X);
    Xsq = X.cwiseProduct(X);

    // 3. create representation Eta
    copy_matrix(NobsR, G1R);
    copy_matrix(NobsR, G2R);

    ThetaLMeanSq = ThetaLMean.cwiseProduct(ThetaLMean);

    copy_matrix(NobsR, ThetaRMeanSq);
    setConstant(ThetaRMeanSq, 0.0);

    this->resolve();
  }

  const Index n;
  const Index p;
  const Index m;
  const Index k;

  ParamRightMatrix NobsR;  // m x k

  const ParamLeftMatrix& ThetaLMean;  // p x k
  const ParamLeftMatrix& ThetaLVar;   // p x k

  ParamRight& ThetaR;  // m x k

  ParamLeftMatrix ThetaLMeanSq;   // p x k
  ParamRightMatrix ThetaRMeanSq;  // m x k

  ReprMatrix X;          // n x p
  ReprMatrix Xsq;        // n x p
  ParamRightMatrix G1R;  // m x k
  ParamRightMatrix G2R;  // m x k
  Repr Eta;              // n x m

  template <typename RNG>
  const ReprMatrix& sample(const RNG& rng) {
    return sample_repr(Eta, rng);
  }
  const ReprMatrix& repr_mean() const { return Eta.get_mean(); }
  const ReprMatrix& repr_var() const { return Eta.get_var(); }

  void add_sgd(const ReprMatrix& llik) { update_repr(Eta, llik); }

  void jitter(const Scalar sd) {
    perturb_param(ThetaR, sd);
    resolve_param(ThetaR);
    this->resolve();
  }

  template <typename RNG>
  void jitter(const Scalar sd, RNG& rng) {
    perturb_param(ThetaR, sd, rng);
    resolve_param(ThetaR);
    this->resolve();
  }

  // mean = X * E[L] * E[R]'
  // var = X^2 * (Var[L] * Var[R]' + E[L]^2 * Var[R]' + Var[L] * E[R']^2)
  void resolve() {
    ThetaRMeanSq =
        mean_param(ThetaR).cwiseProduct(mean_param(ThetaR)); /* (m x k)  */

    update_mean(
        Eta, X * ThetaLMean * mean_param(ThetaR).transpose()); /* mean x mean */
    update_var(
        Eta,
        Xsq * (ThetaLVar * var_param(ThetaR).transpose() +    /* var x var */
               ThetaLMeanSq * var_param(ThetaR).transpose() + /* mean^2 x var */
               ThetaLVar * ThetaRMeanSq.transpose()));        /* var x mean^2 */
  }

  /////////////////////////////////////////////////////////////////////////////
  // gradient w.r.t. E[R]                                                    //
  //     G1' * X * E[L]                   (m x n) (n x p) (p x k)            //
  //     + 2 * G2' * X^2 * Var[L] .* E[R] (m x n) (n x p) (p x k) .* (m x k) //
  //                                                                         //
  // gradient w.r.t. V[R]                                                    //
  //     G2' * X^2 * (Var[L] + E[L]^2)    (m x n) (n x p) (p x k)            //
  /////////////////////////////////////////////////////////////////////////////

  void eval_sgd() {
    Eta.summarize();

    ThetaRMeanSq =
        mean_param(ThetaR).cwiseProduct(mean_param(ThetaR)); /* (m x k) */

    // update of G1R

    // times_set(Xsq, ThetaLVar, temp_nk); /* (n x p) (p x k) = (n x k) */
    // trans_times_set(Eta.get_grad_type2(), temp_nk,
    //                 G1R); /* (n x m)' (n x k) = (m x k) */
    // G1R = 2.0 * G1R.cwiseProduct(mean_param(ThetaR)); /* (m x k) */
    //
    // times_set(X, ThetaLMean, temp_nk); /* (n x p) (p x k) = (n x k) */
    // trans_times_add(Eta.get_grad_type1(), temp_nk,
    //                 G1R); /* (n x m)' (n x k) = (m x k) */

    G1R = 2.0 * (Eta.get_grad_type2().transpose() * Xsq * ThetaLVar)
                    .cwiseProduct(mean_param(ThetaR)) +
          Eta.get_grad_type1().transpose() * X * ThetaLMean;

    // update of G2R

    // times_set(Xsq, ThetaLVar, temp_nk);    /* (n x p) (p x k) = (n x k) */
    // times_add(Xsq, ThetaLMeanSq, temp_nk); /* (n x p) (p x k) = (n x k) */
    // trans_times_set(Eta.get_grad_type2(), temp_nk,
    //                 G2R); /* (n x m)' (n x k) = (m x k) */

    G2R = Eta.get_grad_type2().transpose() * Xsq * (ThetaLVar + ThetaLMeanSq);

    eval_param_sgd(ThetaR, G1R, G2R, NobsR);
  }

  void update_sgd(const Scalar rate) {
    update_param_sgd(ThetaR, rate);
    resolve_param(ThetaR);
    this->resolve();
  }

  void eval_hyper_sgd() {
    this->eval_sgd();
    eval_hyperparam_sgd(ThetaR, G1R, G2R, NobsR);
  }

  void update_hyper_sgd(const Scalar rate) {
    update_hyperparam_sgd(ThetaR, rate);
    resolve_param(ThetaR);
    this->resolve();
  }

  struct square_op_t {
    Scalar operator()(const Scalar& x) const { return x * x; }
  } square_op;
};

template <typename ParamLeft, typename ParamRight, typename Scalar,
          typename Matrix>
struct get_mediated_regression_type;

template <typename ParamLeft, typename ParamRight, typename Scalar>
struct get_mediated_regression_type<
    ParamLeft, ParamRight, Scalar,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > {
  using type =
      mediated_regression_t<DenseReprMat<Scalar>, ParamLeft, ParamRight>;
};

template <typename ParamLeft, typename ParamRight, typename Scalar>
struct get_mediated_regression_type<ParamLeft, ParamRight, Scalar,
                                    Eigen::SparseMatrix<Scalar> > {
  using type =
      mediated_regression_t<SparseReprMat<Scalar>, ParamLeft, ParamRight>;
};

template <typename xDerived, typename yDerived, typename lDerived,
          typename ParamRight>
auto make_mediated_regression_eta(const Eigen::MatrixBase<xDerived>& xx,
                                  const Eigen::MatrixBase<yDerived>& yy,
                                  const Eigen::MatrixBase<lDerived>& thetaLMean,
                                  const Eigen::MatrixBase<lDerived>& thetaLVar,
                                  ParamRight& thetaR) {
  using Scalar = typename yDerived::Scalar;
  using Reg = mediated_regression_t<
      DenseReprMat<Scalar>,
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, ParamRight>;
  return Reg(xx.derived(), yy.derived(), thetaLMean.derived(),
             thetaLVar.derived(), thetaR);
}

template <typename xDerived, typename yDerived, typename lDerived,
          typename ParamRight>
auto make_mediated_regression_eta(const Eigen::SparseMatrixBase<xDerived>& xx,
                                  const Eigen::SparseMatrixBase<yDerived>& yy,
                                  const Eigen::MatrixBase<lDerived>& thetaLMean,
                                  const Eigen::MatrixBase<lDerived>& thetaLVar,
                                  ParamRight& thetaR) {
  using Scalar = typename yDerived::Scalar;
  using Reg = mediated_regression_t<
      SparseReprMat<Scalar>,
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, ParamRight>;
  return Reg(xx.derived(), yy.derived(), thetaLMean.derived(),
             thetaLVar.derived(), thetaR);
}

template <typename xDerived, typename yDerived, typename lDerived,
          typename ParamRight>
auto make_mediated_regression_eta_ptr(
    const Eigen::MatrixBase<xDerived>& xx,
    const Eigen::MatrixBase<yDerived>& yy,
    const Eigen::MatrixBase<lDerived>& thetaLMean,
    const Eigen::MatrixBase<lDerived>& thetaLVar, ParamRight& thetaR) {
  using Scalar = typename yDerived::Scalar;
  using Reg = mediated_regression_t<
      DenseReprMat<Scalar>,
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, ParamRight>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), thetaLMean.derived(),
                               thetaLVar.derived(), thetaR);
}

template <typename xDerived, typename yDerived, typename lDerived,
          typename ParamRight>
auto make_mediated_regression_eta_ptr(
    const Eigen::SparseMatrixBase<xDerived>& xx,
    const Eigen::SparseMatrixBase<yDerived>& yy,
    const Eigen::MatrixBase<lDerived>& thetaLMean,
    const Eigen::MatrixBase<lDerived>& thetaLVar, ParamRight& thetaR) {
  using Scalar = typename yDerived::Scalar;
  using Reg = mediated_regression_t<
      SparseReprMat<Scalar>,
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, ParamRight>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), thetaLMean.derived(),
                               thetaLVar.derived(), thetaR);
}

#endif
