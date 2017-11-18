#include <memory>
#include <type_traits>

#include "eigen_util.hh"
#include "gaus_repr.hh"
#include "parameters.hh"
#include "rcpp_util.hh"

#ifndef MEDIATION_HH_
#define MEDIATION_HH_

////////////////////////////////////////////////////////////////
// Two types of output matrices:
//
// (1) M = n x k --> n individuals, k mediators
// (2) Y = r x t --> r individuals, t traits
//
// Parameter matrices
//
// (1) ThetaL linking genetics and mediators
// (2) ThetaR linking traits and mediators
//
// Two types of eta representations:
//
// (1) etaM ~ X * ThetaL           (n x k) = (n x p) (p x k)
// (2) etaY ~ W * ThetaL * ThetaR' (r x t) = (r x p) (p x k) (t x k)'
//
// Number of observations
//
// 1. (p x k) matrix
// O[ThetaL] = O[X'] * O[M] + O[W'] * O[Y] * O[ThetaR]
//  p x k      p x n  n x k   p x r  r x t    t x k
//
// 2. (t x k) matrix
// O[ThetaR] = O[Y'] * O[W] * O[ThetaL]
//  t x k      t x r   r x p   p x k
//
template <typename ReprM, typename ReprY, typename ParamLeft,
          typename ParamRight>
struct mediation_t;

template <typename D1, typename D2, typename D3, typename D4,
          typename ParamLeft, typename ParamRight>
auto make_mediation_eta(const Eigen::MatrixBase<D1>& x,
                        const Eigen::MatrixBase<D2>& m,
                        const Eigen::MatrixBase<D3>& w,
                        const Eigen::MatrixBase<D4>& y, ParamLeft& thetaL,
                        ParamRight& thetaR) {
  using Scalar = typename D1::Scalar;
  using Med = mediation_t<DenseReprMat<Scalar>, DenseReprMat<Scalar>, ParamLeft,
                          ParamRight>;
  return Med(x.derived(), m.derived(), w.derived(), y.derived(), thetaL,
             thetaR);
}

template <typename D1, typename D2, typename D3, typename D4,
          typename ParamLeft, typename ParamRight>
auto make_mediation_eta_ptr(const Eigen::MatrixBase<D1>& x,
                            const Eigen::MatrixBase<D2>& m,
                            const Eigen::MatrixBase<D3>& w,
                            const Eigen::MatrixBase<D4>& y, ParamLeft& thetaL,
                            ParamRight& thetaR) {
  using Scalar = typename D1::Scalar;
  using Med = mediation_t<DenseReprMat<Scalar>, DenseReprMat<Scalar>, ParamLeft,
                          ParamRight>;
  return std::make_shared<Med>(x.derived(), m.derived(), w.derived(),
                               y.derived(), thetaL, thetaR);
}

template <typename ParamLeft, typename ParamRight, typename Scalar,
          typename Matrix>
struct get_mediation_type;

template <typename ParamLeft, typename ParamRight, typename Scalar>
struct get_mediation_type<
    ParamLeft, ParamRight, Scalar,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > {
  using type = mediation_t<DenseReprMat<Scalar>, DenseReprMat<Scalar>,
                           ParamLeft, ParamRight>;
};

template <typename ParamLeft, typename ParamRight, typename Scalar>
struct get_mediation_type<ParamLeft, ParamRight, Scalar,
                          Eigen::SparseMatrix<Scalar> > {
  using type = mediation_t<SparseReprMat<Scalar>, SparseReprMat<Scalar>,
                           ParamLeft, ParamRight>;
};

/////////////////////
// implementations //
/////////////////////

template <typename ReprM, typename ReprY, typename ParamLeft,
          typename ParamRight>
struct mediation_t {
  using ParamLeftMatrix = typename param_traits<ParamLeft>::Matrix;
  using ParamRightMatrix = typename param_traits<ParamRight>::Matrix;
  using Scalar = typename param_traits<ParamLeft>::Scalar;
  using Index = typename param_traits<ParamLeft>::Index;
  using MMatrix = typename ReprM::DataMatrix;
  using YMatrix = typename ReprY::DataMatrix;

  explicit mediation_t(const MMatrix& xx, const MMatrix& mm, const YMatrix& ww,
                       const YMatrix& yy, ParamLeft& thetaL, ParamRight& thetaR)
      : n(xx.rows()),
        p(xx.cols()),
        k(mm.cols()),
        r(ww.rows()),
        t(yy.cols()),
        NobsL(p, k),
        NobsR(t, k),
        ThetaL(thetaL),
        ThetaR(thetaR),
        X(n, p),
        Xsq(n, p),
        W(r, p),
        Wsq(r, p),
        G1L(p, k),
        G2L(p, k),
        weight_pk(p, k),
        G1R(t, k),
        G2R(t, k),
        weight_pt(p, t),
        EtaM(make_gaus_repr(mm)),
        EtaY(make_gaus_repr(yy)),
        max_weight_pk(1.0),
        max_weight_pt(1.0) {
    // check dimensions : Y ~ W * ThetaL * ThetaR'
    //                    M ~ X * ThetaL
    check_dim(ThetaL, p, k,
              "ThetaL in mediation_t");  // (p x k) (t x k)' = (p x k)
    check_dim(ThetaR, t, k, "ThetaR in mediation_t");  //
    check_dim(EtaM, n, k, "EtaM in mediation_t");  // (n x p) (p x k) = (n x k)
    check_dim(EtaY, r, t, "EtaY in mediation_t");  // (r x k) (t x k) = (r x t)
    check_dim(mm, n, k, "M in mediation_t");
    check_dim(ww, r, p, "W in mediation_t");

    copy_matrix(ThetaL.theta, NobsL);
    copy_matrix(mean_param(ThetaR), NobsR);

    // 1. compute Nobs
    // O[ThetaL] = O[W'] * O[Y] * O[ThetaR]
    //           + O[X'] * O[M]
    XYZ_nobs(ww.transpose(), yy, mean_param(ThetaR), NobsL);
    ParamLeftMatrix temp_pk(p, k);
    XtY_nobs(xx, mm, temp_pk);
    NobsL += temp_pk;

    // O[ThetaR] = O[Y'] * O[W] * O[ThetaL]
    XYZ_nobs(yy.transpose(), ww, ThetaL.theta, NobsR);

    // 2. copy X and Xsq removing missing values
    remove_missing(xx, X);
    Xsq = X.cwiseProduct(X);

    remove_missing(ww, W);
    Wsq = W.cwiseProduct(W);

    weight_pk.setOnes();
    weight_pt.setOnes();

    // 3. create representation Eta
    copy_matrix(NobsL, G1L);
    copy_matrix(NobsL, G2L);
    copy_matrix(NobsR, G1R);
    copy_matrix(NobsR, G2R);

    this->resolve();
  }

  const Index n;  // number of individuals
  const Index p;  // number of predictors
  const Index k;  // number of mediators
  const Index r;  // rank of Y
  const Index t;  // number of traits

  ParamLeftMatrix NobsL;   // p x k
  ParamRightMatrix NobsR;  // t x k
  ParamLeft& ThetaL;       // p x k
  ParamRight& ThetaR;      // t x k

  MMatrix X;    // n x p
  MMatrix Xsq;  // n x p

  YMatrix W;    // r x p
  YMatrix Wsq;  // r x p

  ParamLeftMatrix G1L;  // p x k
  ParamLeftMatrix G2L;  // p x k

  ParamLeftMatrix weight_pk;  // p x t

  ParamRightMatrix G1R;  // t x k
  ParamRightMatrix G2R;  // t x k

  ParamRightMatrix weight_pt;  // p x t

  ReprM EtaM;  // n x m
  ReprY EtaY;  // k x t

  Scalar max_weight_pk;
  Scalar max_weight_pt;

  void jitter(const Scalar sd) {
    perturb_param(ThetaL, sd);
    perturb_param(ThetaR, sd);
    resolve_param(ThetaL);
    resolve_param(ThetaR);
    resolve();
  }

  template <typename RNG>
  void jitter(const Scalar sd, RNG& rng) {
    perturb_param(ThetaL, sd, rng);
    perturb_param(ThetaR, sd, rng);
    resolve_param(ThetaL);
    resolve_param(ThetaR);
    resolve();
  }

  template <typename Derived>
  void set_weight_pt(Eigen::MatrixBase<Derived>& _weight_pt) {
    ASSERT(_weight_pt.rows() == p && _weight_pt.cols() == t,
           "invalid weight_pt matrix");
    weight_pt = _weight_pt.derived();

    max_weight_pt = weight_pt.cwiseAbs().maxCoeff() + static_cast<Scalar>(1e-4);
  }

  template <typename Derived>
  void set_weight_pk(Eigen::MatrixBase<Derived>& _weight_pk) {
    ASSERT(_weight_pk.rows() == p && _weight_pk.cols() == k,
           "invalid weight_pk matrix");
    weight_pk = _weight_pk.derived();

    max_weight_pk = weight_pk.cwiseAbs().maxCoeff() + static_cast<Scalar>(1e-4);
  }

  // Initialize by solving linear systems
  // [p x t] -> [p x k] [t x k]'
  // mm [n x k], xx [n x p] -> xx' * mm [p x k]
  // yy [r x t], ww [r x p] -> ww' * yy [p x t]

  inline void init_by_ls(const MMatrix& mm, const YMatrix& yy,
                         const Scalar sd) {
    MMatrix M;
    YMatrix Y;

    remove_missing(mm, M);
    remove_missing(yy, Y);

    MMatrix XtM = X.transpose() * M / static_cast<Scalar>(n);
    YMatrix WtY = W.transpose() * Y / static_cast<Scalar>(r);

    ThetaL.beta.setZero();
    ThetaR.beta.setZero();

    ThetaL.beta = XtM * sd;
    for (Index j = 0; j < t; ++j) {
      ThetaR.beta.row(j) +=
          sd *
          XtM.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
              .solve(WtY.col(j))
              .transpose();
    }

    resolve_param(ThetaL);
    resolve_param(ThetaR);
    this->resolve();
  }

  // mean_m = X * E[L]
  // var_m  = X^2 * V[L]
  // mean_y = W * E[L] * E[R]'
  // var_y  = W^2 * (Var[L] * Var[R]' + E[L]^2 * Var[R]' + Var[L] * E[R']^2)
  void resolve() {
    // on M model
    update_mean(EtaM, X * ThetaL.theta.cwiseProduct(weight_pk)); /* X x mean */
    update_var(EtaM, Xsq *
                         ThetaL.theta_var.cwiseProduct(weight_pk).cwiseProduct(
                             weight_pk)); /* X.*X x var */

    // on Y model
    update_mean(EtaY, W * ((ThetaL.theta.cwiseProduct(weight_pk) *
                            mean_param(ThetaR).transpose())
                               .cwiseProduct(weight_pt))); /* mean x mean */

    update_var(
        EtaY, /* hopefully optimized away */
        Wsq *
            ((ThetaL.theta_var.cwiseProduct(weight_pk).cwiseProduct(weight_pk) *
                  var_param(ThetaR).transpose() + /* var x var */
              ThetaL.theta.cwiseProduct(ThetaL.theta).cwiseProduct(weight_pk) *
                  var_param(ThetaR).transpose() + /* mu^2 x var */
              ThetaL.theta_var.cwiseProduct(weight_pk).cwiseProduct(weight_pk) *
                  (mean_param(ThetaR).cwiseProduct(mean_param(ThetaR)))
                      .transpose()) /* var x mu^2 */
                 .cwiseProduct(weight_pt)
                 .cwiseProduct(weight_pt)));
  }

  void add_sgd(const MMatrix& llik_m, const YMatrix& llik_y) {
    update_repr(EtaM, llik_m);
    update_repr(EtaY, llik_y);
  }

  void add_sgd_m(const MMatrix& llik_m) { update_repr(EtaM, llik_m); }

  void add_sgd_y(const YMatrix& llik_y) { update_repr(EtaY, llik_y); }

  template <typename RNG>
  const YMatrix& sample_y(const RNG& rng) {
    return sample_repr(EtaY, rng);
  }

  template <typename RNG>
  const MMatrix& sample_m(const RNG& rng) {
    return sample_repr(EtaM, rng);
  }

  const YMatrix& get_sampled_y() const { return get_sampled_repr(EtaY); }

  const MMatrix& get_sampled_m() const { return get_sampled_repr(EtaM); }

  const YMatrix& repr_mean_y() const { return EtaY.get_mean(); }

  const MMatrix& repr_mean_m() const { return EtaM.get_mean(); }

  ///////////////////////////////////////////////////////////////////////////////
  // (1) gradient w.r.t. E[L]
  //     X' * GM1                            (p x n) (n x k)
  //     + (W' * GY1) * E[R]                 (p x r) (r x t) (t x k)
  //     + (2 * W^2' * GY2) * Var[R] .* E[L] (p x r) (r x t) (t x k) .* (p x k)
  //
  // (2) gradient w.r.t. V[L]
  //     X^2' * GM2                        (p x n) (n x k)
  //     + W^2' * GY2 * (Var[R] + E[R]^2)  (p x r) (r x t) (t x k)
  //
  // (3) gradient w.r.t. E[R]
  //     GY1' * W * E[L]                   (t x r) (r x p) (p x k)
  //     + 2 * GY2' * W^2 * Var[L] .* E[R] (t x r) (r x p) (p x k) .* (t x k)
  //
  // (4) gradient w.r.t. V[R]
  //     GY2' * W^2 * (Var[L] + E[L]^2)    (t x r) (r x p) (p x k)
  ///////////////////////////////////////////////////////////////////////////////

  void _compute_sgd_left() {
    // (1) update of G1L
    G1L = X.transpose() * EtaM.get_grad_type1() +
          (W.transpose() * EtaY.get_grad_type1()).cwiseProduct(weight_pt) *
              mean_param(ThetaR) +
          ((Wsq.transpose() * EtaY.get_grad_type2())
               .cwiseProduct(weight_pt)
               .cwiseProduct(weight_pt) *
           var_param(ThetaR))
                  .cwiseProduct(mean_param(ThetaL)) *
              static_cast<Scalar>(2.0);

    G1L = G1L.cwiseProduct(weight_pk);

    // times_set(EtaY.get_grad_type2(), var_param(ThetaR), temp_rk);
    // trans_times_set(Wsq, temp_rk, G1L);
    // G1L = 2.0 * G1L.cwiseProduct(ThetaL.theta);
    // times_set(EtaY.get_grad_type1(), mean_param(ThetaR), temp_rk);
    // trans_times_add(W, temp_rk, G1L);
    // trans_times_add(X, EtaM.get_grad_type1(), G1L);

    // (2) update of G2L
    G2L = Xsq.transpose() * EtaM.get_grad_type2() +
          (Wsq.transpose() * EtaY.get_grad_type2())
                  .cwiseProduct(weight_pt)
                  .cwiseProduct(weight_pt) *
              (mean_param(ThetaR).cwiseProduct(mean_param(ThetaR)) +
               var_param(ThetaR));

    G2L = G2L.cwiseProduct(weight_pk).cwiseProduct(weight_pk);

    // times_set(EtaY.get_grad_type2(), var_param(ThetaR), temp_rk);
    // times_add(EtaY.get_grad_type2(), thetaRsq, temp_rk);
    // trans_times_set(Wsq, temp_rk, G2L);
    // trans_times_add(Xsq, EtaM.get_grad_type2(), G2L);
  }

  void _compute_sgd_right() {
    // (3) update of G1R
    G1R = (EtaY.get_grad_type1().transpose() *
           W).cwiseProduct(weight_pt.transpose()) *
              mean_param(ThetaL).cwiseProduct(weight_pk) +
          ((EtaY.get_grad_type2().transpose() * Wsq)
               .cwiseProduct(weight_pt.transpose())
               .cwiseProduct(weight_pt.transpose()) *
           var_param(ThetaL).cwiseProduct(weight_pk).cwiseProduct(weight_pk))
                  .cwiseProduct(mean_param(ThetaR)) *
              static_cast<Scalar>(2.0);

    // times_set(Wsq, ThetaL.theta_var, temp_rk);
    // trans_times_set(EtaY.get_grad_type2(), temp_rk, G1R);
    // G1R = 2.0 * G1R.cwiseProduct(mean_param(ThetaR));
    // times_set(W, ThetaL.theta, temp_rk);
    // trans_times_add(EtaY.get_grad_type1(), temp_rk, G1R);

    // (4) update of G2R
    G2R = (EtaY.get_grad_type2().transpose() * Wsq)
              .cwiseProduct(weight_pt.transpose())
              .cwiseProduct(weight_pt.transpose()) *
          (var_param(ThetaL) +
           mean_param(ThetaL).cwiseProduct(mean_param(ThetaL)))
              .cwiseProduct(weight_pk)
              .cwiseProduct(weight_pk);

    // times_set(Wsq, ThetaL.theta_var, temp_rk);
    // times_add(Wsq, thetaLsq, temp_rk);
    // trans_times_set(EtaY.get_grad_type2(), temp_rk, G2R);
  }

  void eval_sgd() {
    EtaM.summarize();
    EtaY.summarize();

    this->_compute_sgd_left();
    this->_compute_sgd_right();

    eval_param_sgd(ThetaL, G1L, G2L, NobsL);
    eval_param_sgd(ThetaR, G1R, G2R, NobsR);
  }

  void eval_hyper_sgd() {
    EtaM.summarize();
    EtaY.summarize();

    this->_compute_sgd_left();
    this->_compute_sgd_right();

    eval_hyperparam_sgd(ThetaL, G1L, G2L, NobsL);
    eval_hyperparam_sgd(ThetaR, G1R, G2R, NobsR);
  }

  void update_sgd(const Scalar rate) {
    update_param_sgd(ThetaL, rate / max_weight_pk);
    update_param_sgd(ThetaR, rate / max_weight_pt);
    resolve_param(ThetaL);
    resolve_param(ThetaR);
    this->resolve();
  }

  void update_hyper_sgd(const Scalar rate) {
    update_hyperparam_sgd(ThetaL, rate / max_weight_pk);
    update_hyperparam_sgd(ThetaR, rate / max_weight_pt);
    resolve_param(ThetaL);
    resolve_param(ThetaR);
    this->resolve();
  }

  ////////////////////////////////////////////////////////////////
  // Just update gradient of thetaL only (without dependency with
  // mediatiion effect thetaR)
  void eval_sgd_xm() {
    EtaM.summarize();
    // (1) update of G1L
    trans_times_set(X, EtaM.get_grad_type1(), G1L);
    // (2) update of G2L
    trans_times_set(Xsq, EtaM.get_grad_type2(), G2L);
    eval_param_sgd(ThetaL, G1L, G2L, NobsL);
  }

  void eval_hyper_sgd_xm() {
    EtaM.summarize();
    // (1) update of G1L
    trans_times_set(X, EtaM.get_grad_type1(), G1L);
    // (2) update of G2L
    trans_times_set(Xsq, EtaM.get_grad_type2(), G2L);
    eval_hyperparam_sgd(ThetaL, G1L, G2L, NobsL);
  }

  void update_sgd_xm(const Scalar rate) {
    update_param_sgd(ThetaL, rate);
    resolve_param(ThetaL);
    this->resolve();
  }

  void update_hyper_sgd_xm(const Scalar rate) {
    update_hyperparam_sgd(ThetaL, rate);
    resolve_param(ThetaL);
    this->resolve();
  }

  // Update gradient of thetaR conditioning on the variational
  // distribution of thetaL
  void eval_sgd_my() {
    EtaY.summarize();
    this->_compute_sgd_right();
    eval_param_sgd(ThetaR, G1R, G2R, NobsR);
  }

  void eval_hyper_sgd_my() {
    EtaY.summarize();
    this->_compute_sgd_right();
    eval_hyperparam_sgd(ThetaR, G1R, G2R, NobsR);
  }

  void update_sgd_my(const Scalar rate) {
    update_param_sgd(ThetaR, rate);
    resolve_param(ThetaR);
    this->resolve();
  }

  void update_hyper_sgd_my(const Scalar rate) {
    update_hyperparam_sgd(ThetaR, rate);
    resolve_param(ThetaR);
    this->resolve();
  }
};

#endif
