#ifndef PARAM_MIXTURE_HH_
#define PARAM_MIXTURE_HH_

////////////////////////////////////////////////////////////////
// Prior distribution:
//
// theta ~ pi * N(0, v0 + v1) + (1 - pi) * N(0, v0)
//
// Variational approximation:
//
// theta | z = 1 ~ N(beta, w0 + w1)
// theta | z = 0 ~ N(0, w0)
//
// E[theta] = alpha * beta
// V[theta] = w0 + alpha * w1 + alpha * (1-alpha) * beta^2
//
////////////////////////////////////////////////////////////////

template <typename T, typename S>
struct param_mixture_t {
  typedef T data_t;
  typedef typename T::Scalar scalar_t;
  typedef typename T::Index index_t;
  typedef adam_t<T, scalar_t> grad_adam_t;
  typedef tag_param_mixture sgd_tag;
  typedef S sparsity_tag;

  template <typename Opt>
  explicit param_mixture_t(const index_t n1, const index_t n2, const Opt& opt)
      : nrow(n1),
        ncol(n2),

        alpha(nrow, ncol),
        alpha_aux(nrow, ncol),
        beta(nrow, ncol),

        omega(nrow, ncol),
        gamma_aux(nrow, ncol),

        omega_null(nrow, ncol),
        gamma_null_aux(nrow, ncol),

        theta(nrow, ncol),
        theta_var(nrow, ncol),

        grad_alpha_aux(nrow, ncol),
        grad_beta(nrow, ncol),
        grad_gamma_aux(nrow, ncol),
        grad_gamma_null_aux(nrow, ncol),

        r_m(opt.rate_m()),
        r_v(opt.rate_v()),

        pi_lodds_lb(opt.pi_lodds_lb()),
        pi_lodds_ub(opt.pi_lodds_ub()),
        tau_lodds_lb(opt.tau_lodds_lb()),
        tau_lodds_ub(opt.tau_lodds_ub()),

        adam_alpha_aux(r_m, r_v, nrow, ncol),
        adam_beta(r_m, r_v, nrow, ncol),
        adam_gamma_aux(r_m, r_v, nrow, ncol),
        adam_gamma_null_aux(r_m, r_v, nrow, ncol),

        adam_pi_aux(r_m, r_v),
        adam_tau_aux(r_m, r_v),
        adam_tau_null_aux(r_m, r_v),

        omega_op(opt.gammax(), tau_aux),
        omega_null_op(opt.gammax(), tau_null_aux),
        nu_op(opt.gammax(), tau_aux),
        nu_null_op(opt.gammax(), tau_null_aux),

        resolve_spike_op(pi_aux),
        grad_alpha_lo_op(pi_aux),

        grad_ln_ratio_alpha_op(nu_val, nu_null_val),

        grad_kl_gamma_op(nu_val, nu_null_val),
        grad_kl_gamma_null_op(nu_null_val),

        grad_gamma_chain_op(opt.gammax(), tau_aux),
        grad_gamma_chain_null_op(opt.gammax(), tau_null_aux),

        grad_prior_pi_op(pi_val),

        grad_kl_tau_op(nu_val, nu_null_val),

        grad_kl_tau_null_op(nu_null_val),

        grad_tau_chain_op(opt.gammax()) {}

  const index_t rows() const { return nrow; }
  const index_t cols() const { return ncol; }

  const index_t nrow;
  const index_t ncol;

  ////////////////////////////
  // variational parameters //
  ////////////////////////////

  T alpha;
  T alpha_aux;
  T beta;

  T omega;
  T gamma_aux;

  T omega_null;
  T gamma_null_aux;

  T theta;
  T theta_var;

  T grad_alpha_aux;
  T grad_beta;
  T grad_gamma_aux;
  T grad_gamma_null_aux;

  ////////////
  // priors //
  ////////////

  scalar_t pi_val;
  scalar_t pi_aux;
  scalar_t grad_pi_aux;

  scalar_t nu_val;
  scalar_t tau_aux;
  scalar_t grad_tau_aux;

  scalar_t nu_null_val;
  scalar_t tau_null_aux;
  scalar_t grad_tau_null_aux;

  // fixed hyperparameters
  const scalar_t r_m;
  const scalar_t r_v;

  const scalar_t pi_lodds_lb;
  const scalar_t pi_lodds_ub;

  const scalar_t tau_lodds_lb;
  const scalar_t tau_lodds_ub;

  ////////////////////////////////////////////////////////////////
  // adaptive gradient

  grad_adam_t adam_alpha_aux;

  grad_adam_t adam_beta;

  grad_adam_t adam_gamma_aux;

  grad_adam_t adam_gamma_null_aux;

  adam_t<scalar_t, scalar_t> adam_pi_aux;

  adam_t<scalar_t, scalar_t> adam_tau_aux;

  adam_t<scalar_t, scalar_t> adam_tau_null_aux;

  ////////////////////////////////////////////////////////////////
  // helper functors

  // omega = [1/gammax] * (1 + exp(-gamma_aux + tau_aux))
  struct resolve_var_op_t {
    explicit resolve_var_op_t(const scalar_t _gammax, const scalar_t& _tau_aux)
        : vmin(1 / _gammax), tau_aux(_tau_aux) {}

    inline const scalar_t operator()(const scalar_t& gam_aux) const {
      return vmin * (one_val + std::exp(-gam_aux + tau_aux)) + small_val;
    }

    const scalar_t vmin;
    const scalar_t& tau_aux;
    constexpr static scalar_t small_val = 1e-10;
  };

  resolve_var_op_t omega_op;       // omega_op(gammax, tau_aux)
  resolve_var_op_t omega_null_op;  // omega_null_op(gammax, tau_null_aux)

  // nu = [1/gammax] * (1 + exp(-tau_aux))
  struct resolve_prior_var_op_t {
    explicit resolve_prior_var_op_t(const scalar_t _gammax,
                                    const scalar_t& _tau_aux)
        : vmin(1 / _gammax), tau_aux(_tau_aux) {}

    inline const scalar_t operator()() const {
      return vmin * (one_val + std::exp(-tau_aux)) + small_val;
    }

    const scalar_t vmin;
    const scalar_t& tau_aux;
    constexpr static scalar_t small_val = 1e-10;
  };

  resolve_prior_var_op_t nu_op;       // nu_op(gammax, tau_aux)
  resolve_prior_var_op_t nu_null_op;  // nu_null_op(gammax, tau_null_aux)

  struct resolve_spike_t {
    explicit resolve_spike_t(const scalar_t& _pi_aux) : pi_aux(_pi_aux) {}
    inline const scalar_t operator()(const scalar_t& alpha_aux) const {
      const scalar_t lo = alpha_aux + pi_aux;
      if (-lo > large_exp_value) {
        return std::exp(lo) / (one_val + std::exp(lo));
      }
      return one_val / (one_val + std::exp(-lo));
    }
    const scalar_t& pi_aux;

  } resolve_spike_op;

  ////////////////////////
  // gradient operators //
  ////////////////////////

  /////////////////////
  // regarding alpha //
  /////////////////////

  // d KL / d alpha
  // ln pi/(1 - pi) - ln alpha / (1 - alpha)
  struct grad_kl_lodds_alpha_t {
    explicit grad_kl_lodds_alpha_t(const scalar_t& _lodds) : lodds(_lodds) {}
    inline const scalar_t operator()(const scalar_t& x) const {
      return lodds - x;
    }
    const scalar_t& lodds;
  } grad_alpha_lo_op;

  // (1 - 2 * a) * b^2
  struct grad_alpha_g2_t {
    inline const scalar_t operator()(const scalar_t& a,
                                     const scalar_t& b) const {
      return (one_val - two_val * a) * b * b;
    }
  } grad_alpha_g2_op;

  // ln(w1 + w0) - ln(w0) - ln(v1 + v0) + ln(v0)
  // = ln(1 + w1/w0)      - ln(1 + v1/v0)
  struct grad_ln_ratio_alpha_t {
    explicit grad_ln_ratio_alpha_t(const scalar_t& _v1, const scalar_t& _v0)
        : v1(_v1), v0(_v0) {}
    inline const scalar_t operator()(const scalar_t& w1,
                                     const scalar_t& w0) const {
      const scalar_t ln_ratio_prior = std::log(one_val + v1 / v0);
      return std::log(one_val + w1 / w0) - ln_ratio_prior;
    }

    const scalar_t& v1;
    const scalar_t& v0;
  } grad_ln_ratio_alpha_op;

  struct grad_alpha_chain_rule_t {
    inline const scalar_t operator()(const scalar_t& x,
                                     const scalar_t& a) const {
      return x * a * (one_val - a);
    }
  } grad_alpha_chain_op;

  ////////////////////
  // regarding beta //
  ////////////////////

  struct grad_beta_g2_t {
    inline const scalar_t operator()(const scalar_t& a,
                                     const scalar_t& b) const {
      return two_val * a * (one_val - a) * b;
    }
  } grad_beta_g2_op;

  /////////////////////
  // regarding gamma //
  /////////////////////

  // 0.5/(w1 + w0) - 0.5/(v0 + v1)
  struct grad_kl_gamma_t {
    explicit grad_kl_gamma_t(const scalar_t& _v1, const scalar_t& _v0)
        : v1(_v1), v0(_v0) {}

    inline const scalar_t operator()(const scalar_t& w1,
                                     const scalar_t& w0) const {
      const scalar_t inv_var_prior = half_val / (v0 + v1);
      return half_val / (w1 + w0) - inv_var_prior;
    }

    const scalar_t& v1;
    const scalar_t& v0;
  } grad_kl_gamma_op;

  // 0.5 * (1 - alpha) * (1/w0 - 1/v0)
  struct grad_kl_gamma_null_t {
    explicit grad_kl_gamma_null_t(const scalar_t& _v0) : v0(_v0) {}
    inline const scalar_t operator()(const scalar_t& a,
                                     const scalar_t& w0) const {
      return half_val * (one_val - a) * (one_val / w0 - one_val / v0);
    }
    const scalar_t& v0;
  } grad_kl_gamma_null_op;

  // omega = 1 / gamma
  // gamma = gammax * sigmoid(gam_aux - tau_aux)
  //
  // [d omega / d gamma] * [d gamma / d gamma_aux]
  //

  struct grad_gamma_chain_rule_t {
    explicit grad_gamma_chain_rule_t(const scalar_t _gammax,
                                     const scalar_t& _tau_aux)
        : vmin(1 / _gammax), tau_aux(_tau_aux) {}

    inline const scalar_t operator()(const scalar_t& gam_aux) const {
      return -vmin * std::exp(-gam_aux + tau_aux);
    }

    const scalar_t vmin;
    const scalar_t& tau_aux;
  };

  // Note: using different tau_aux

  grad_gamma_chain_rule_t grad_gamma_chain_op;  // grad_gamma_op(gammax,tau_aux)
  grad_gamma_chain_rule_t
      grad_gamma_chain_null_op;  // grad_gamma_op(gammax,tau_null_aux)

  //////////////////////
  // hyper-parameters //
  //////////////////////

  // grad of prior wrt pi_aux
  // alpha - pi
  struct grad_prior_pi_aux_t {
    explicit grad_prior_pi_aux_t(const scalar_t& _pi_val) : pi_val(_pi_val) {}
    inline const scalar_t operator()(const scalar_t& a) const {
      return a - pi_val;
    }
    const scalar_t& pi_val;
  } grad_prior_pi_op;

  // 0.5 * alpha * (theta_sq/(v1 + v0)^2 - 1/(v1 + v0))
  struct grad_kl_tau_t {
    explicit grad_kl_tau_t(const scalar_t& _v1, const scalar_t& _v0)
        : v1(_v1), v0(_v0) {}

    inline const scalar_t operator()(const scalar_t& a,
                                     const scalar_t& ts) const {
      const scalar_t vtot = small_val + v1 + v0;
      return (ts / vtot - one_val) / vtot * a * half_val;
    }

    const scalar_t& v1;
    const scalar_t& v0;
    constexpr static scalar_t small_val = 1e-8;
  } grad_kl_tau_op;

  // 0.5 * (1 - alpha) * (w0 / v0^2 - 1/v0)
  struct grad_kl_tau_null_t {
    explicit grad_kl_tau_null_t(const scalar_t& _v0) : v0(_v0) {}

    inline const scalar_t operator()(const scalar_t& a,
                                     const scalar_t& w0) const {
      return half_val * (one_val - a) * (w0 / v0 - one_val) / v0;
    }

    const scalar_t& v0;
    constexpr static scalar_t small_val = 1e-8;
  } grad_kl_tau_null_op;

  // - vmin * exp( - tau)
  struct grad_tau_chain_rule_t {
    explicit grad_tau_chain_rule_t(const scalar_t _gammax)
        : vmin(1 / _gammax) {}

    inline const scalar_t operator()(const scalar_t& tau_aux) const {
      return -vmin * std::exp(-tau_aux);
    }

    const scalar_t vmin;
  };

  grad_tau_chain_rule_t grad_tau_chain_op;  // grad_tau_chain_op(gammax)

  static constexpr scalar_t half_val = 0.5;
  static constexpr scalar_t one_val = 1.0;
  static constexpr scalar_t two_val = 2.0;
  static constexpr scalar_t large_exp_value = 20.0;  // exp(20) is too big
};

////////////////////////////////////////////////////////////////
// clear contents
template <typename Parameter>
void impl_initialize_param(Parameter& P, const tag_param_mixture) {
  // zero gradients
  setConstant(P.grad_alpha_aux, 0.0);
  setConstant(P.grad_beta, 0.0);
  setConstant(P.grad_gamma_aux, 0.0);
  setConstant(P.grad_gamma_null_aux, 0.0);
  P.grad_pi_aux = 0.0;
  P.grad_tau_aux = 0.0;
  P.grad_tau_null_aux = 0.0;

  // zero direction
  setConstant(P.beta, 0.0);

  // start from most relaxed
  P.pi_aux = P.pi_lodds_ub;
  setConstant(P.alpha_aux, 0.0);

  // complementary to each other
  setConstant(P.gamma_aux, P.tau_lodds_lb);
  setConstant(P.gamma_null_aux, P.tau_lodds_lb);
  P.tau_aux = P.tau_lodds_lb;
  P.tau_null_aux = P.tau_lodds_lb;
}

////////////////////////////////////////////////////////////////
// factory functions
template <typename scalar_t, typename Index, typename Opt>
auto make_dense_mixture(const Index n1, const Index n2, const Opt& opt) {
  using Mat = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
  using Param = param_mixture_t<Mat, tag_param_dense>;

  Param ret(n1, n2, opt);
  impl_initialize_param(ret, tag_param_mixture());
  resolve_param(ret);
  resolve_hyperparam(ret);

  return ret;
}

////////////////////////////////////////////////////////////////
// factory functions
template <typename scalar_t, typename Index, typename Opt>
auto make_dense_mixture_ptr(const Index n1, const Index n2, const Opt& opt) {
  using Mat = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
  using Param = param_mixture_t<Mat, tag_param_dense>;

  auto ret_ptr = std::make_shared<Param>(n1, n2, opt);
  Param& ret = *ret_ptr.get();
  impl_initialize_param(ret, tag_param_mixture());
  resolve_param(ret);
  resolve_hyperparam(ret);

  return ret_ptr;
}

////////////////////////////////////////////////////////////////
// initialize non-zeroness by adjacency A
template <typename scalar_t, typename Derived, typename Opt>
auto make_sparse_mixture(const Eigen::SparseMatrixBase<Derived>& A,
                         const Opt& opt) {
  const auto n1 = A.rows();
  const auto n2 = A.cols();

  using Mat = Eigen::SparseMatrix<scalar_t, Eigen::ColMajor>;
  using Param = param_mixture_t<Mat, tag_param_sparse>;

  Param ret(n1, n2, opt);
  const scalar_t eps = 1e-4;

  // just add epsilon * A to reserve spots
  initialize(A, ret.alpha, eps);
  initialize(A, ret.alpha_aux, eps);
  initialize(A, ret.beta, eps);

  initialize(A, ret.omega, eps);
  initialize(A, ret.gamma_aux, eps);

  initialize(A, ret.omega_null, eps);
  initialize(A, ret.gamma_null_aux, eps);

  initialize(A, ret.theta, eps);
  initialize(A, ret.theta_var, eps);

  initialize(A, ret.grad_alpha_aux, eps);
  initialize(A, ret.grad_beta, eps);
  initialize(A, ret.grad_gamma_aux, eps);
  initialize(A, ret.grad_gamma_null_aux, eps);

  impl_initialize_param(ret, tag_param_mixture());
  resolve_param(ret);
  resolve_hyperparam(ret);
  return ret;
}

// update parameters by calculated stochastic gradient
template <typename Parameter, typename scalar_t>
void impl_update_param_sgd(Parameter& P, const scalar_t rate,
                           const tag_param_mixture) {
  P.alpha_aux += update_adam(P.adam_alpha_aux, P.grad_alpha_aux) * rate;
  P.beta += update_adam(P.adam_beta, P.grad_beta) * rate;
  P.gamma_aux += update_adam(P.adam_gamma_aux, P.grad_gamma_aux) * rate;
  P.gamma_null_aux +=
      update_adam(P.adam_gamma_null_aux, P.grad_gamma_null_aux) * rate;
  resolve_param(P);
}

template <typename Parameter, typename scalar_t>
void impl_update_hyperparam_sgd(Parameter& P, const scalar_t rate,
                                const tag_param_mixture) {
  P.pi_aux += update_adam(P.adam_pi_aux, P.grad_pi_aux) * rate;
  P.tau_aux += update_adam(P.adam_tau_aux, P.grad_tau_aux) * rate;
  P.tau_null_aux +=
      update_adam(P.adam_tau_null_aux, P.grad_tau_null_aux) * rate;
  resolve_hyperparam(P);
}

// mean and variance
template <typename Parameter>
void impl_resolve_param(Parameter& P, const tag_param_mixture) {
  using scalar_t = typename Parameter::scalar_t;

  P.alpha = P.alpha_aux.unaryExpr(P.resolve_spike_op);
  P.theta = P.alpha.cwiseProduct(P.beta);
  P.omega = P.gamma_aux.unaryExpr(P.omega_op);

  const scalar_t nrow = static_cast<scalar_t>(P.rows());
  P.omega_null = P.gamma_null_aux.unaryExpr(P.omega_null_op) / nrow;

  P.theta_var =
      P.omega_null +
      P.alpha.cwiseProduct(P.omega + P.beta.cwiseProduct(P.beta)) -
      P.alpha.cwiseProduct(P.beta).cwiseProduct(P.alpha.cwiseProduct(P.beta));
}

template <typename Parameter>
void impl_resolve_hyperparam(Parameter& P, const tag_param_mixture) {
  if (P.pi_aux > P.pi_lodds_ub) P.pi_aux = P.pi_lodds_ub;
  if (P.pi_aux < P.pi_lodds_lb) P.pi_aux = P.pi_lodds_lb;

  if (P.tau_aux > P.tau_lodds_ub) P.tau_aux = P.tau_lodds_ub;
  if (P.tau_aux < P.tau_lodds_lb) P.tau_aux = P.tau_lodds_lb;

  if (P.tau_null_aux > P.tau_lodds_ub) P.tau_null_aux = P.tau_lodds_ub;
  if (P.tau_null_aux < P.tau_lodds_lb) P.tau_null_aux = P.tau_lodds_lb;

  P.pi_val = P.resolve_spike_op(0.0);

  P.nu_val = P.nu_op();
  P.nu_null_val = P.nu_null_op();
}

template <typename Parameter, typename scalar_t, typename RNG>
void impl_perturb_param(Parameter& P, const scalar_t sd, RNG& rng,
                        const tag_param_mixture) {
  std::normal_distribution<scalar_t> Norm;
  auto rnorm = [&rng, &Norm, &sd](const auto& x) { return sd * Norm(rng); };
  P.beta = P.beta.unaryExpr(rnorm);
  resolve_param(P);
}

template <typename Parameter, typename scalar_t>
void impl_perturb_param(Parameter& P, const scalar_t sd,
                        const tag_param_mixture) {
  std::mt19937 rng;
  impl_perturb_param(P, sd, rng, tag_param_mixture());
}

template <typename Parameter>
void impl_check_nan_param(Parameter& P, const tag_param_mixture) {
  auto is_nan = [](const auto& x) { return !std::isfinite(x); };
  auto num_nan = [&is_nan](const auto& M) { return M.unaryExpr(is_nan).sum(); };
  ASSERT(num_nan(P.alpha) == 0, "found in alpha");
  ASSERT(num_nan(P.beta) == 0, "found in beta");
  ASSERT(num_nan(P.theta) == 0, "found in theta");
  ASSERT(num_nan(P.theta_var) == 0, "found in theta_var");
}

template <typename Parameter>
const auto& impl_mean_param(Parameter& P, const tag_param_mixture) {
  return P.theta;
}

template <typename Parameter>
auto impl_log_odds_param(Parameter& P, const tag_param_mixture) {
  return P.alpha_aux.unaryExpr([&P](const auto& x) { return P.pi_aux + x; });
}

template <typename Parameter>
const auto& impl_var_param(Parameter& P, const tag_param_mixture) {
  return P.theta_var;
}

////////////////////////////////////////////////////////////////
// evaluate stochastic gradient descent step
template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_param_sgd(Parameter& P, const M1& G1, const M2& G2,
                         const M3& Nobs, const tag_param_mixture) {
  using scalar_t = typename Parameter::scalar_t;

  // Note: sparse matrix does not support .array()

  ////////////////////////////////
  // gradient w.r.t. alpha
  //
  // G1 .* beta +
  // G2 .* (w1 + (1-2alpha) * beta^2)
  //
  // dKL 1: lodds(pi) - lodds(alpha)
  //
  // dKL 2: 0.5 * [ ln(w1 + w0) - ln(v1 + v0)
  //                - ln(w0)    + ln(v0)
  //                + w0 / v0
  //                - theta_sq / (v1 + v0)   ]
  // where
  // theta_sq = beta^2 + w0 + w1

  const scalar_t var_prior = P.nu_val + P.nu_null_val;

  P.grad_alpha_aux =
      G1.cwiseProduct(P.beta) +
      G2.cwiseProduct(P.alpha.binaryExpr(P.beta, P.grad_alpha_g2_op)) +
      P.alpha_aux.unaryExpr(P.grad_alpha_lo_op) +
      0.5 *
          (P.omega.binaryExpr(P.omega_null, P.grad_ln_ratio_alpha_op) +
           P.omega_null / P.nu_null_val -
           (P.beta.cwiseProduct(P.beta) + P.omega + P.omega_null) / var_prior);

  // chain rule on alpha_aux
  P.grad_alpha_aux =
      P.grad_alpha_aux.binaryExpr(P.alpha, P.grad_alpha_chain_op);

  // adjust number of observations
  P.grad_alpha_aux = P.grad_alpha_aux.cwiseQuotient(Nobs);

  ////////////////////////////////
  // gradient w.r.t. beta
  //
  // G1 .* alpha +
  // G2 .* alpha .* (1 - alpha) .* beta
  //
  // dKL = -alpha .* beta / (v0 + v1)
  //

  P.grad_beta = G1.cwiseProduct(P.alpha) +
                G2.cwiseProduct(P.alpha.binaryExpr(P.beta, P.grad_beta_g2_op)) -
                P.alpha.cwiseProduct(P.beta) / var_prior;

  // adjust number of observations
  P.grad_beta = P.grad_beta.cwiseQuotient(Nobs);

  ////////////////////////////////
  // gradient w.r.t. gamma
  //
  // G2 .* alpha
  //
  // dKL = alpha * 0.5 * (1/(w1 + w0) - 1/(v0 + v1))
  //
  // chain rule:  .* (- Vmin .* exp(- gamma_aux + tau_aux))

  P.grad_gamma_aux =
      P.alpha
          .cwiseProduct(G2 +
                        P.omega.binaryExpr(P.omega_null, P.grad_kl_gamma_op))
          .cwiseProduct(P.gamma_aux.unaryExpr(P.grad_gamma_chain_op));

  P.grad_gamma_aux = P.grad_gamma_aux.cwiseQuotient(Nobs);

  // Note: scaling by the number of elements is extremely important
  const scalar_t nrow = P.rows();

  ////////////////////////////////
  // gradient w.r.t. gamma_null
  //
  // G2
  //
  // dKL = alpha * 0.5 * (1/(w1 + w0) - 1/(v0 + v1))
  //       (1-alpha) * 0.5 * (1/w0 - 1/v0)
  //
  // chain rule:  .* (- Vmin .* exp(- gamma_aux + tau_aux))

  P.grad_gamma_null_aux =
      (G2 + P.alpha.binaryExpr(P.omega, P.grad_kl_gamma_null_op) +
       P.alpha.cwiseProduct(
           P.omega.binaryExpr(P.omega_null, P.grad_kl_gamma_op)))
          .cwiseProduct(P.gamma_null_aux.unaryExpr(P.grad_gamma_chain_null_op));

  P.grad_gamma_null_aux = P.grad_gamma_null_aux.cwiseQuotient(Nobs) / nrow;
}

template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_hyperparam_sgd(Parameter& P, const M1& G1, const M2& G2,
                              const M3& Nobs, const tag_param_mixture) {
  using scalar_t = typename Parameter::scalar_t;
  const scalar_t ntot = Nobs.sum() + P.rows() * P.cols();
  const scalar_t nrow = P.rows();

  // First evaluate gradient
  impl_eval_param_sgd(P, G1, G2, Nobs, tag_param_mixture());

  ////////////////////////////////
  // gradient w.r.t. pi_aux
  P.grad_pi_aux = P.grad_alpha_aux.sum();
  P.grad_pi_aux += P.alpha.unaryExpr(P.grad_prior_pi_op).sum();
  P.grad_pi_aux /= ntot;

  ////////////////////////////////
  // gradient w.r.t. tau_aux
  //
  // - d / dw1
  // dKL / dv1   = alpha .* 0.5 .* (theta_sq/(v1 + v0)^2 - 1/(v1 + v0))
  // dv1 / dtau1 = -Vmin .* exp(-tau1_aux)
  //
  const scalar_t stuff =
      P.alpha
          .binaryExpr(P.beta.cwiseProduct(P.beta) + P.omega + P.omega_null,
                      P.grad_kl_tau_op)
          .sum();

  P.grad_tau_aux =
      -P.grad_gamma_aux.sum() + stuff * P.grad_tau_chain_op(P.tau_aux);

  P.grad_tau_aux /= ntot;

  // dKL / dv0   = alpha .* 0.5 .* (theta_sq/(v1 + v0)^2 - 1/(v1 + v0))
  //              (1-alpha) .* 0.5 .* (w0/v0^2 - 1/v0)
  // dv0 / dtau0 = -Vmin .* exp(-tau0_aux)

  const scalar_t stuff0 =
      P.alpha.binaryExpr(P.omega_null, P.grad_kl_tau_null_op).sum();

  P.grad_tau_null_aux = -P.grad_gamma_null_aux.sum() +
                        (stuff + stuff0) * P.grad_tau_chain_op(P.tau_null_aux);

  P.grad_tau_null_aux /= ntot;
}

template <typename Parameter>
void impl_write_param(Parameter& P, const std::string hdr, const std::string gz,
                      const tag_param_mixture) {
  write_data_file((hdr + ".theta" + gz), P.theta);
  write_data_file((hdr + ".theta_var" + gz), P.theta_var);
  typename Parameter::data_t temp =
      P.alpha_aux.unaryExpr([&P](const auto& x) { return P.pi_aux + x; });
  write_data_file((hdr + ".lodds" + gz), temp);
  write_data_file((hdr + ".spike" + gz), P.alpha);
  write_data_file((hdr + ".slab" + gz), P.beta);
}

#endif
