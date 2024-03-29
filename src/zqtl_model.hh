#ifndef ZQTL_MODEL_HH_
#define ZQTL_MODEL_HH_

//////////////////////////////////////////////////////
// RSS Model (Zhu & Stephens, 2016) :               //
// theta_hat ~ N(E R inv(E) theta, E R E)           //
// where E = sqrt(theta_hat^2 / n + theta_hat_var)  //
//                                                  //
// Redefine:                                        //
// y = V'z                                          //
// X = V'inv(E)                                     //
// eta = X theta                                    //
//////////////////////////////////////////////////////

////////////////////////////////////////
// Modified variance model            //
//                                    //
// var = D^2 * (1 + sigmoid(eta_var)) //
////////////////////////////////////////

template <typename T>
struct zqtl_model_t {
  using Scalar = typename T::Scalar;
  using Index = typename T::Index;
  using Data = T;

  template <typename X>
  using Dense = Eigen::MatrixBase<X>;

  explicit zqtl_model_t(const T& yy, const T& sn)
      : k(yy.rows()),
        m(yy.cols()),
        Y(k, m),
        var0(k, 1),
        var_mat(k, m),
        llik_mat(k, m),
        resid_mat(k, m),
        evidence_mat(k, m),
        distrib(0, 1) {
    is_obs_op<T> obs_op;
    evidence_mat = yy.unaryExpr(obs_op);
    remove_missing(yy, Y);
    remove_missing(sn, var0);

    var0 = var0.unaryExpr(add_vmin_op);

    llik_mat.setZero();
    resid_mat.setZero();
  }

  ////////////////////////////////////////////////////////////////
  // llik[i] = -var[i] eta[i]^2 / 2 + y[i] eta[i]
  template <typename Derived>
  const T& eval_eta(const Dense<Derived>& eta) {
    const Scalar half_val = static_cast<Scalar>(0.5);

    llik_mat = (var0.asDiagonal() * eta.cwiseProduct(eta) * (-half_val) +
                Y.cwiseProduct(eta))
                   .cwiseProduct(evidence_mat);
    return llik_mat;
  }

  template <typename Derived>
  const T& sample(const Dense<Derived>& eta) {
    auto rnorm = [this](const Scalar& x) { return distrib(rng); };
    Y = var0.cwiseSqrt().asDiagonal() * Y.unaryExpr(rnorm) +
        var0.asDiagonal() * eta;
    return Y;
  }

  ////////////////////////////////////////////////////////////////
  // llik[i] = -var[i] eta[i]^2 / 2 + y[i] eta[i]
  template <typename Derived, typename Derived2>
  const T& eval_eta_zeta(const Dense<Derived>& eta,
                         const Dense<Derived2>& zeta) {
    const Scalar half_val = static_cast<Scalar>(0.5);

    var_mat = var0.asDiagonal() * zeta.unaryExpr(upweight_var_op);

    llik_mat = (var_mat.cwiseProduct(eta).cwiseProduct(eta) * (-half_val) +
                Y.cwiseProduct(eta))
                   .cwiseProduct(evidence_mat);

    return llik_mat;
  }

  ////////////////////////////////////////////////////////////////
  // llik[i] = y[i] eta[i] - var[i] eta[i]^2 / 2
  //           + y[i] delta[i] / var[i] - delta[i]^2 / 2 * var[i]
  //           - delta[i] eta[i]
  //
  // y = V'z ~ N(S eta + delta, S)
  //
  template <typename Derived, typename Derived2>
  const T& eval_eta_delta(const Dense<Derived>& eta,
                          const Dense<Derived2>& delta) {
    const Scalar neg_half_val = static_cast<Scalar>(-0.5);

    llik_mat = (var0.asDiagonal() * eta.cwiseProduct(eta) * neg_half_val +
                Y.cwiseProduct(eta) +
                var0.cwiseInverse().asDiagonal() *
                    (Y.cwiseProduct(delta) +
                     delta.cwiseProduct(delta) * neg_half_val) -
                delta.cwiseProduct(eta))
                   .cwiseProduct(evidence_mat);

    return llik_mat;
  }

  template <typename Derived, typename Derived2, typename Derived3>
  const T& eval_eta_delta_zeta(const Dense<Derived>& eta,
                               const Dense<Derived2>& delta,
                               const Dense<Derived3>& zeta) {
    const Scalar neg_half_val = static_cast<Scalar>(-0.5);

    var_mat = var0.asDiagonal() * zeta.unaryExpr(upweight_var_op);

    llik_mat =
        (var_mat.cwiseProduct(eta).cwiseProduct(eta) * neg_half_val +
         Y.cwiseProduct(eta) +
         (Y.cwiseProduct(delta) + delta.cwiseProduct(delta) * neg_half_val)
             .cwiseQuotient(var_mat) -
         delta.cwiseProduct(eta))
            .cwiseProduct(evidence_mat);

    return llik_mat;
  }

  template <typename Derived, typename Derived2>
  const T& sample(const Dense<Derived>& eta, const Dense<Derived2>& delta) {
    auto rnorm = [this](const Scalar& x) { return distrib(rng); };
    Y = var0.cwiseSqrt().asDiagonal() * Y.unaryExpr(rnorm) +
        var0.asDiagonal() * eta + delta;
    return Y;
  }

  const T& llik() const { return llik_mat; }
  const Index nobs() const { return k; }
  const Index ntraits() const { return m; }

  const Index k;  // rank of out-of-sample LD
  const Index m;  // number of traits

 private:
  T Y;             // k x m, V' * Z
  T var0;          // k x 1, eigen values of LD matrix
  T var_mat;       // k x m, var0[k] * (1 + sigmoid(zeta[k, m]))
  T llik_mat;      // k x m
  T resid_mat;     // k x m
  T evidence_mat;  // k x m

  struct add_vmin_op_t {
    inline const Scalar operator()(const Scalar& v) const { return v + vmin; }
    static constexpr Scalar vmin = 1e-8;
  } add_vmin_op;

  // 1 + sigmoid(x)
  struct upweight_var_op_t {
    inline const Scalar operator()(const Scalar& x) const {
      if (-(x + offset) < large_value)
        return one_val + one_val / (one_val + fasterexp(-(x + offset)));
      return one_val +
             fasterexp(x + offset) / (one_val + fasterexp(x + offset));
    }
    const Scalar one_val = 1.0;
    const Scalar offset = -5.0;
    const Scalar large_value = 20.0;  // exp(20) is too big
  } upweight_var_op;

  dqrng::xoshiro256plus rng;
  dqrng::normal_distribution distrib;
};

#endif
