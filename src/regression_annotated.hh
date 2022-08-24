////////////////////////////////////////////////////////////////
// A wrapper for eta = X * (A .* ThetaL) * ThetaR'
//
// X (n x p)
// A (p x K)
// ThetaL (p x K)
// ThetaR (m x K)
//

#include <memory>
#include <type_traits>

#include "eigen_util.hh"
#include "gaus_repr.hh"
#include "parameters.hh"
#include "rcpp_util.hh"

#ifndef ANNOTATED_REGRESSION_HH_
#define ANNOTATED_REGRESSION_HH_

template <typename Repr, typename ParamLeft, typename ParamRight>
struct annotated_regression_t {
  using ParamLeftMatrix = typename param_traits<ParamLeft>::Matrix;
  using ParamRightMatrix = typename param_traits<ParamRight>::Matrix;
  using Scalar = typename param_traits<ParamLeft>::Scalar;
  using Index = typename param_traits<ParamLeft>::Index;
  using ReprMatrix = typename Repr::DataMatrix;

  explicit annotated_regression_t(const ReprMatrix &xx, const ReprMatrix &yy,
                                  ParamLeft &thetaL, ParamRight &thetaR)
      : n(xx.rows()),
        p(xx.cols()),
        m(yy.cols()),
        k(thetaL.cols()),
        NobsL(p, k),
        NobsR(m, k),
        ThetaL(thetaL),
        ThetaR(thetaR),
        thetaLsq(p, k),
        thetaRsq(m, k),
        X(n, p),
        Xsq(n, p),
        G1L(p, k),
        G2L(p, k),
        G1R(m, k),
        G2R(m, k),
        temp_nk(n, k),
        temp_nk2(n, k),
        temp_pk(p, k),
        weight_pk(p, k),
        weight_pk_sq(p, k),
        max_weight(1.0),
        Eta(make_gaus_repr(yy)) {
#ifdef DEBUG
    check_dim(ThetaL, p, k, "ThetaL in annotated_regression_t");
    check_dim(ThetaR, m, k, "ThetaR in annotated_regression_t");
    check_dim(Eta, n, m, "Eta in annotated_regression_t");
#endif
    copy_matrix(ThetaL.theta, NobsL);
    copy_matrix(ThetaR.theta, NobsR);
    weight_pk.setConstant(max_weight);

    // 1. compute Nobs
    // Need to take into account of weights
    XYZ_nobs(xx.transpose(), yy, ThetaR.theta, NobsL);
    XYZ_nobs(yy.transpose(), xx, ThetaL.theta, NobsR);

    // 2. copy X and Xsq removing missing values
    remove_missing(xx, X);
    remove_missing(xx.unaryExpr([](const auto &x) { return x * x; }), Xsq);

    // 3. create representation Eta
    copy_matrix(NobsL, G1L);
    copy_matrix(NobsL, G2L);
    copy_matrix(NobsR, G1R);
    copy_matrix(NobsR, G2R);

    copy_matrix(NobsL, thetaLsq);
    copy_matrix(NobsR, thetaRsq);

    setConstant(thetaLsq, 0.0);
    setConstant(thetaRsq, 0.0);

    this->resolve();
  }

  const Index n;
  const Index p;
  const Index m;
  const Index k;

  ParamLeftMatrix NobsL;   // p x k
  ParamRightMatrix NobsR;  // m x k
  ParamLeft &ThetaL;       // p x k
  ParamRight &ThetaR;      // m x k

  ParamLeftMatrix thetaLsq;   // p x k
  ParamRightMatrix thetaRsq;  // m x k

  ReprMatrix X;             // n x p
  ReprMatrix Xsq;           // n x p
  ParamLeftMatrix G1L;      // p x k
  ParamLeftMatrix G2L;      // p x k
  ParamRightMatrix G1R;     // m x k
  ParamRightMatrix G2R;     // m x k
  ReprMatrix temp_nk;       // n x k
  ReprMatrix temp_nk2;      // n x k
  ReprMatrix temp_pk;       // p x k
  ReprMatrix weight_pk;     // p x k
  ReprMatrix weight_pk_sq;  // p x k
  Scalar max_weight;

  Repr Eta;  // n x m

  template <typename RNG>
  inline Eigen::Ref<const ReprMatrix> sample(RNG &rng) {
    return sample_repr(Eta, rng);
  }

  inline Eigen::Ref<const ReprMatrix> repr_mean() { return Eta.get_mean(); }
  inline Eigen::Ref<const ReprMatrix> repr_var() { return Eta.get_var(); }

  inline void add_sgd(const ReprMatrix &llik) { update_repr(Eta, llik); }

  template <typename Derived1, typename Derived2, typename Derived3>
  void set_weight_pk(const Eigen::MatrixBase<Derived1> &_weight,
                     const Eigen::MatrixBase<Derived2> &xx,
                     const Eigen::MatrixBase<Derived3> &yy) {
    ASSERT(_weight.rows() == p && _weight.cols() == k, "invalid weight matrix");
    weight_pk = _weight.derived();

    const Scalar max_val =
        weight_pk.cwiseAbs().maxCoeff() + static_cast<Scalar>(1e-4);

    if (max_weight < max_val) max_weight = max_val;

    weight_pk_sq = weight_pk.cwiseProduct(weight_pk);

    // NobsL = weight .* (O[X'] * O[Y] * O[R]) -> (p x k)
    // NobsR = O[Y'] * O[X] * (O[L] .* weight) -> (m x k)
    XYZ_nobs(xx.transpose(), yy, ThetaR.theta, NobsL);
    NobsL = NobsL.cwiseProduct(weight_pk);
    XYZ_nobs(yy.transpose(), xx, weight_pk, NobsR);
  }

  /////////////////////////////////////////////////////////////////////////
  // mean = X * (W .* E[L]) * E[R]'					 //
  // var = X^2 * (W2 .* Var[L] * Var[R]' + W2 .* E[L]^2 * Var[R]'	 //
  // 	       + W2 .* Var[L] * E[R']^2)				 //
  /////////////////////////////////////////////////////////////////////////

  inline void resolve() {
    thetaRsq = ThetaR.theta.cwiseProduct(ThetaR.theta); /* (m x k)  */
    thetaLsq = ThetaL.theta.cwiseProduct(ThetaL.theta); /* (p x k)  */

    update_mean(Eta, X * (ThetaL.theta.cwiseProduct(weight_pk)) *
                         ThetaR.theta.transpose());

    temp_nk = Xsq * (weight_pk_sq.cwiseProduct(ThetaL.theta_var + thetaLsq));

    temp_nk2 = Xsq * (weight_pk_sq.cwiseProduct(ThetaL.theta_var));

    update_var(Eta,
               temp_nk * ThetaR.theta_var.transpose() + temp_nk2 * thetaRsq);
  }

  /////////////////////////////////////////////////////////////////////////////
  // (1) gradient w.r.t. E[L]                                                //
  //     X' * G1 * E[R]                   (p x n) (n x m) (m x k)            //
  //     + 2 * X^2' * G2 * Var[R] .* E[L] (p x n) (n x m) (m x k) .* (p x k) //
  //                                                                         //
  // (2) gradient w.r.t. V[L]                                                //
  //     X^2' * G2 * (Var[R] + E[R]^2)    (p x n) (n x m) (m x k)            //
  //                                                                         //
  // (3) gradient w.r.t. E[R]                                                //
  //     G1' * X * E[L]                   (m x n) (n x p) (p x k)            //
  //     + 2 * G2' * X^2 * Var[L] .* E[R] (m x n) (n x p) (p x k) .* (m x k) //
  //                                                                         //
  // (4) gradient w.r.t. V[R]                                                //
  //     G2' * X^2 * (Var[L] + E[L]^2)    (m x n) (n x p) (p x k)            //
  /////////////////////////////////////////////////////////////////////////////

  inline void eval_sgd() {
    Eta.summarize();

    // (1) update of G1L -- reducing to [n x k] helps performance
    G1L = X.transpose() * Eta.get_grad_type1() * mean_param(ThetaR) +
          (Xsq.transpose() * Eta.get_grad_type2() * var_param(ThetaR))
                  .cwiseProduct(mean_param(ThetaL)) *
              static_cast<Scalar>(2.0);

    G1L = G1L.cwiseProduct(weight_pk);

    // (2) update of G2L
    G2L = Xsq.transpose() * Eta.get_grad_type2() *
          (var_param(ThetaR) +
           mean_param(ThetaR).cwiseProduct(mean_param(ThetaR)));

    G2L = G2L.cwiseProduct(weight_pk_sq);

    eval_param_sgd(ThetaL, G1L, G2L, NobsL);

    // (3) update of G1R
    G1R = Eta.get_grad_type1().transpose() * X *
              (weight_pk.cwiseProduct(mean_param(ThetaL))) +
          (Eta.get_grad_type2().transpose() * Xsq *
           weight_pk_sq.cwiseProduct(var_param(ThetaL)))
                  .cwiseProduct(mean_param(ThetaR)) *
              static_cast<Scalar>(2.0);

    // (4) update of G2R
    thetaLsq = ThetaL.theta.cwiseProduct(ThetaL.theta); /* (p x k)  */

    G2R = Eta.get_grad_type2().transpose() * Xsq *
          (weight_pk_sq.cwiseProduct(var_param(ThetaL)) +
           weight_pk.cwiseProduct(thetaLsq));

    eval_param_sgd(ThetaR, G1R, G2R, NobsR);
  }

  template <typename RNG>
  inline void jitter(const Scalar sd, RNG &rng) {
    perturb_param(ThetaL, sd, rng);
    perturb_param(ThetaR, sd, rng);
    resolve_param(ThetaL);
    resolve_param(ThetaR);
    this->resolve();
  }

  inline void init_by_svd(const ReprMatrix &yy, const Scalar sd) {
    ReprMatrix Yin;
    remove_missing(yy, Yin);

    ReprMatrix Ymean =
        Yin * ReprMatrix::Ones(Yin.cols(), 1) / static_cast<Scalar>(Yin.cols());

    ReprMatrix Y(Yin.rows(), k);
    for (Index j = 0; j < k; ++j) {
      Y.col(j) = Ymean.cwiseProduct(weight_pk.col(j));
    }

    ReprMatrix XtY = X.transpose() * Y / static_cast<Scalar>(n);
    XtY = XtY.cwiseProduct(weight_pk);

    Eigen::JacobiSVD<ReprMatrix> svd(XtY,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
    ParamLeftMatrix left = svd.matrixU() * sd;
    ThetaL.beta.setZero();
    ThetaL.beta.leftCols(k) = left.leftCols(k);
    ThetaR.beta.setConstant(sd);
    resolve_param(ThetaL);
    resolve_param(ThetaR);
    this->resolve();
  }

  inline void update_sgd(const Scalar rate) {
    update_param_sgd(ThetaL, rate);
    update_param_sgd(ThetaR, rate);
    resolve_param(ThetaL);
    resolve_param(ThetaR);
    this->resolve();
  }

  inline void eval_hyper_sgd() {
    this->eval_sgd();
    eval_hyperparam_sgd(ThetaL, G1L, G2L, NobsL);
    eval_hyperparam_sgd(ThetaR, G1R, G2R, NobsR);
  }

  inline void update_hyper_sgd(const Scalar rate) {
    update_hyperparam_sgd(ThetaL, rate);
    update_hyperparam_sgd(ThetaR, rate);
    resolve_param(ThetaL);
    resolve_param(ThetaR);
    this->resolve();
  }
};

template <typename ParamLeft, typename ParamRight, typename Scalar,
          typename Matrix>
struct get_annotated_regression_type;

template <typename ParamLeft, typename ParamRight, typename Scalar>
struct get_annotated_regression_type<
    ParamLeft, ParamRight, Scalar,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> {
  using type =
      annotated_regression_t<DenseReprMat<Scalar>, ParamLeft, ParamRight>;
};

template <typename xDerived, typename yDerived, typename ParamLeft,
          typename ParamRight>
auto make_annotated_regression_eta(const Eigen::MatrixBase<xDerived> &xx,
                                   const Eigen::MatrixBase<yDerived> &yy,
                                   ParamLeft &thetaL, ParamRight &thetaR) {
  using Scalar = typename yDerived::Scalar;
  using Reg =
      annotated_regression_t<DenseReprMat<Scalar>, ParamLeft, ParamRight>;
  return Reg(xx.derived(), yy.derived(), thetaL, thetaR);
}

template <typename xDerived, typename yDerived, typename ParamLeft,
          typename ParamRight>
auto make_annotated_regression_eta_ptr(const Eigen::MatrixBase<xDerived> &xx,
                                       const Eigen::MatrixBase<yDerived> &yy,
                                       ParamLeft &thetaL, ParamRight &thetaR) {
  using Scalar = typename yDerived::Scalar;
  using Reg =
      annotated_regression_t<DenseReprMat<Scalar>, ParamLeft, ParamRight>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), thetaL, thetaR);
}

#endif
