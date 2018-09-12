#ifndef RCPP_ZQTL_MEDIATION_HH_
#define RCPP_ZQTL_MEDIATION_HH_

struct effect_y_mat_t {
  explicit effect_y_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct effect_y_se_mat_t {
  explicit effect_y_se_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct effect_m_mat_t {
  explicit effect_m_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct effect_m_se_mat_t {
  explicit effect_m_se_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct conf_mat_t {
  explicit conf_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct geno_y_mat_t {
  explicit geno_y_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct geno_m_mat_t {
  explicit geno_m_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

///////////////////////////////////////////////////////////
// estimate residual effects using MR-Egger type of idea //
//                                                       //
// eta1 = D^{-2} V' Z_qtl Theta_med'                     //
// eta2 = V' Theta_dir                                   //
// eta3 = V' Theta_conf                                  //
///////////////////////////////////////////////////////////

template <typename DIRECT, typename MEDIATED_D, typename MEDIATED_E,
          typename... DATA>
Rcpp::List _variance_calculation(DIRECT& eta_direct, MEDIATED_D& delta_med,
                                 MEDIATED_E theta_med, options_t& opt,
                                 std::tuple<DATA...>&& data_tup);

template <typename DIRECT, typename MEDIATED_D, typename... DATA>
Rcpp::List _variance_calculation(DIRECT& eta_direct, MEDIATED_D& delta_med,
                                 options_t& opt,
                                 std::tuple<DATA...>&& data_tup);

template <typename MODEL_Y, typename DIRECT, typename CONF, typename MEDIATED_E,
          typename... DATA>
Rcpp::List _fine_map(MODEL_Y& model_y, DIRECT& eta_direct, CONF& eta_conf_y,
                     MEDIATED_E theta_med_org, options_t& opt,
                     std::tuple<DATA...>&& data_tup);

template <typename RNG, typename... DATA>
Mat _direct_effect_propensity(RNG& rng, options_t& opt,
                              std::tuple<DATA...>&& data_tup);

template <typename RNG, typename... DATA>
Mat _direct_effect_conditional(RNG& rng, options_t& opt,
                               std::tuple<DATA...>&& data_up);

template <typename RNG, typename... DATA>
Mat _direct_effect_factorization(RNG& rng, const options_t& opt,
                                 std::tuple<DATA...>&& data_tup);

bool check_mediation_input(const effect_y_mat_t& yy,        // z_y
                           const effect_y_se_mat_t& yy_se,  // z_y_se
                           const effect_m_mat_t& mm,        // z_m
                           const effect_m_se_mat_t& mm_se,  // z_m_se
                           const geno_y_mat_t& geno_y,      // genotype_y
                           const geno_m_mat_t& geno_m,      // genotype_m
                           const conf_mat_t& conf,          // snp confounder
                           options_t& opt);

std::tuple<Mat, Mat, Mat, Mat, Mat, Mat, Mat> preprocess_mediation_input(
    const effect_y_mat_t& yy,        // z_y
    const effect_y_se_mat_t& yy_se,  // z_y_se
    const effect_m_mat_t& mm,        // z_m
    const effect_m_se_mat_t& mm_se,  // z_m_se
    const geno_y_mat_t& geno_y,      // genotype_y
    const geno_m_mat_t& geno_m,      // genotype_m
    const conf_mat_t& conf,          // snp confounder
    options_t& opt);

template <typename RNG, typename... DATA>
Mat estimate_direct_effect(RNG& rng, options_t& opt,
                           std::tuple<DATA...>&& data_tup);

///////////////////////////////
// fit basic mediation model //
///////////////////////////////

Rcpp::List impl_fit_med_zqtl(const effect_y_mat_t& yy,        // z_y
                             const effect_y_se_mat_t& yy_se,  // z_y_se
                             const effect_m_mat_t& mm,        // z_m
                             const effect_m_se_mat_t& mm_se,  // z_m_se
                             const geno_y_mat_t& geno_y,      // genotype_y
                             const geno_m_mat_t& geno_m,      // genotype_m
                             const conf_mat_t& conf,          // snp confounder
                             options_t& opt) {
  if (!check_mediation_input(yy, yy_se, mm, mm_se, geno_y, geno_m, conf, opt)) {
    return Rcpp::List::create();
  }

  Mat Y, M, U, Vt, D2, VtI, VtC;

  std::tie(Y, M, U, Vt, D2, VtI, VtC) = preprocess_mediation_input(
      yy, yy_se, mm, mm_se, geno_y, geno_m, conf, opt);

  if (opt.verbose()) TLOG("Finished preprocessing\n\n");

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  // omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  zqtl_model_t<Mat> model_y(Y, D2);

  // intercept    ~ R 1 theta
  // Vt intercept ~ D2 Vt 1 theta
  auto theta_intercept = make_dense_slab<Scalar>(VtI.cols(), Y.cols(), opt);
  auto eta_intercept = make_regression_eta(VtI, Y, theta_intercept);
  eta_intercept.init_by_dot(Y, opt.jitter());

  // confounder -- or bias
  auto theta_conf_y = make_dense_slab<Scalar>(VtC.cols(), Y.cols(), opt);
  auto eta_conf_y = make_regression_eta(VtC, Y, theta_conf_y);
  eta_conf_y.init_by_dot(Y, opt.jitter());

  Mat llik0, llik, llik_fm;
  Rcpp::List param_unmediated_finemap = Rcpp::List::create();
  Rcpp::List param_unmed = Rcpp::List::create();
  Rcpp::List param_med = Rcpp::List::create();
  Rcpp::List bootstrap(opt.nboot());
  Rcpp::List var_decomp = Rcpp::List::create();
  Rcpp::List backfire = Rcpp::List::create();

  Mat M0;
  dummy_eta_t dummy;

  // Fine-map residual unmediated effect
  // This is fitted med w/o bootstrapping
  auto do_finemap_unmediated = [&](auto& theta_med, auto& delta_med) {
    auto theta_direct = make_dense_spike_slab<Scalar>(Vt.cols(), Y.cols(), opt);
    auto eta_direct = make_regression_eta(Vt, Y, theta_direct);
    eta_direct.init_by_dot(Y, opt.jitter());
    eta_intercept.resolve();
    eta_conf_y.resolve();
    delta_med.resolve();
    llik_fm = impl_fit_eta_delta(
        model_y, opt, rng,
        std::make_tuple(eta_direct, eta_intercept, eta_conf_y),
        std::make_tuple(dummy), std::make_tuple(dummy),
        std::make_tuple(delta_med));

    param_unmediated_finemap = param_rcpp_list(theta_direct);

    // Perform parametric bootstrap to estimate CIs.
    // This time we clamp direct effects.
    for (Index b = 0; b < opt.nboot(); ++b) {
      eta_direct.resolve();
      eta_conf_y.init_by_dot(Y, opt.jitter());
      eta_intercept.init_by_dot(Y, opt.jitter());
      delta_med.init_by_dot(Y, opt.jitter());
      impl_fit_eta_delta(model_y, opt, rng,
                         std::make_tuple(eta_intercept, eta_conf_y),
                         std::make_tuple(delta_med),
                         std::make_tuple(eta_direct), std::make_tuple(dummy));

      bootstrap[b] = param_rcpp_list(theta_med);
      if (opt.verbose())
        TLOG("Finished bootstrap [" << std::setw(10) << (b + 1) << " / "
                                    << std::setw(10) << opt.nboot() << "]");
    }
  };

  if (opt.do_direct_effect()) {
    M0 = estimate_direct_effect(rng, opt,
                                std::make_tuple(Y, M, U, Vt, D2, VtI, VtC));
    if (opt.verbose()) TLOG("Finished direct model estimation\n\n");

    // Adjust M ~ M0 to avoid the backfire
    if (opt.do_control_backfire()) {
      auto theta_adj = make_dense_spike_slab<Scalar>(M0.cols(), M.cols(), opt);
      auto delta_adj = make_regression_eta(M0, M, theta_adj);
      delta_adj.init_by_dot(M, opt.jitter());

      zqtl_model_t<Mat> model_adj(M, D2);

      if (opt.verbose()) TLOG("Adjust the backfire : M0 -> M\n\n");

      Mat llik_adj =
          impl_fit_eta_delta(model_adj, opt, rng, std::make_tuple(dummy),
                             std::make_tuple(delta_adj));

      if (opt.verbose()) TLOG("Finished adjusting the backfire\n\n");

      backfire = Rcpp::List::create(
          Rcpp::_["param"] = param_rcpp_list(theta_adj), Rcpp::_["M0"] = M0);

      delta_adj.resolve();

      auto theta_resid = make_dense_slab<Scalar>(M.rows(), M.cols(), opt);
      auto delta_resid = make_residual_eta(M, theta_resid);

      Mat llik_adj_resid = impl_fit_eta_delta(
          model_adj, opt, rng, std::make_tuple(dummy),
          std::make_tuple(delta_resid), std::make_tuple(dummy),
          std::make_tuple(delta_adj));

      if (opt.verbose()) TLOG("Finished the reconstruction of unbiased M\n\n");

      delta_resid.resolve();
      Mat mzhat = Vt.transpose() * delta_resid.repr_mean();

      if (opt.do_rescale()) {
        Mat mzhat_std = standardize_zscore(mzhat, Vt, D2.cwiseSqrt());
        M = Vt * mzhat_std;
      } else {
        Mat mzhat_std = center_zscore(mzhat, Vt, D2.cwiseSqrt());
        M = Vt * mzhat_std;
      }
      M0.resize(M0.rows(), 1);
      M0 = Mat::Ones(M0.rows(), 1) / static_cast<Scalar>(M0.rows());
    }

    // mediated
    auto theta_med = make_dense_spike_slab<Scalar>(M.cols(), Y.cols(), opt);
    auto delta_med = make_regression_eta(M, Y, theta_med);
    delta_med.init_by_dot(Y, opt.jitter());

    // the unmediated component -- smooth
    auto theta_unmed = make_dense_slab<Scalar>(M0.cols(), Y.cols(), opt);
    auto delta_unmed = make_regression_eta(M0, Y, theta_unmed);
    delta_unmed.init_by_dot(Y, opt.jitter());

    if (opt.do_med_two_step()) {
      // Just include the unmediated compnent
      llik0 = impl_fit_eta_delta(model_y, opt, rng,
                                 std::make_tuple(eta_intercept, eta_conf_y),
                                 std::make_tuple(delta_unmed));

      if (opt.verbose()) TLOG("Finished the null model estimation\n\n");

      delta_unmed.resolve();
      eta_intercept.resolve();
      eta_conf_y.resolve();

      // Clamping the unmediated effect...
      llik = impl_fit_eta_delta(model_y, opt, rng, std::make_tuple(dummy),
                                std::make_tuple(delta_med),
                                std::make_tuple(eta_intercept, eta_conf_y),
                                std::make_tuple(delta_unmed));
      if (opt.verbose()) TLOG("Finished two-step estimation\n\n");
    } else {
      llik = impl_fit_eta_delta(model_y, opt, rng,
                                std::make_tuple(eta_intercept, eta_conf_y),
                                std::make_tuple(delta_med, delta_unmed));
      if (opt.verbose()) TLOG("Finished joint model estimation\n\n");
    }
    param_unmed = param_rcpp_list(theta_unmed);
    param_med = param_rcpp_list(theta_med);

    if (opt.do_finemap_unmediated()) {
      do_finemap_unmediated(theta_med, delta_med);
    }

    if (opt.do_var_calc()) {
      var_decomp = _variance_calculation(eta_intercept, delta_med, theta_med,
                                         opt, std::make_tuple(Y, M, U, D2));
      if (opt.verbose()) TLOG("Finished variance decomposition\n\n");
    }
  } else {
    /////////////////////////
    // this must be smooth //
    /////////////////////////

    auto theta_unmed = make_dense_slab<Scalar>(Vt.cols(), Y.cols(), opt);
    auto eta_unmed = make_regression_eta(Vt, Y, theta_unmed);
    eta_unmed.init_by_dot(Y, opt.jitter());

    // mediated
    auto theta_med = make_dense_spike_slab<Scalar>(M.cols(), Y.cols(), opt);
    auto delta_med = make_regression_eta(M, Y, theta_med);

    llik = impl_fit_eta_delta(
        model_y, opt, rng,
        std::make_tuple(eta_intercept, eta_conf_y, eta_unmed),
        std::make_tuple(delta_med));

    param_unmed = param_rcpp_list(theta_unmed);
    param_med = param_rcpp_list(theta_med);
    if (opt.verbose()) TLOG("Finished joint model estimation\n\n");

    if (opt.do_finemap_unmediated()) {
      do_finemap_unmediated(theta_med, delta_med);
    }

    if (opt.do_var_calc()) {
      var_decomp = _variance_calculation(eta_intercept, delta_med, theta_med,
                                         opt, std::make_tuple(Y, M, U, D2));
      if (opt.verbose()) TLOG("Finished variance decomposition\n\n");
    }
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["U"] = U, Rcpp::_["Vt"] = Vt,
      Rcpp::_["D2"] = D2, Rcpp::_["M"] = M, Rcpp::_["M0"] = M0,
      Rcpp::_["param.mediated"] = param_med,
      Rcpp::_["param.unmediated"] = param_unmed,
      Rcpp::_["param.finemap.direct"] = param_unmediated_finemap,
      Rcpp::_["param.intercept"] = param_rcpp_list(theta_intercept),
      Rcpp::_["param.covariate"] = param_rcpp_list(theta_conf_y),
      Rcpp::_["llik"] = llik, Rcpp::_["llik.fm"] = llik_fm,
      Rcpp::_["llik.null"] = llik0, Rcpp::_["var.decomp"] = var_decomp,
      Rcpp::_["bootstrap"] = bootstrap, Rcpp::_["backfire"] = backfire);
}

//////////////////////////////////
// fit factored mediation model //
//////////////////////////////////

Rcpp::List impl_fit_fac_med_zqtl(const effect_y_mat_t& yy,        // z_y
                                 const effect_y_se_mat_t& yy_se,  // z_y_se
                                 const effect_m_mat_t& mm,        // z_m
                                 const effect_m_se_mat_t& mm_se,  // z_m_se
                                 const geno_y_mat_t& geno_y,      // genotype_y
                                 const geno_m_mat_t& geno_m,      // genotype_m
                                 const conf_mat_t& conf,  // snp confounder
                                 options_t& opt) {
  if (!check_mediation_input(yy, yy_se, mm, mm_se, geno_y, geno_m, conf, opt)) {
    return Rcpp::List::create();
  }

  Mat Y, M, U, Vt, D2, VtI, VtC;

  std::tie(Y, M, U, Vt, D2, VtI, VtC) = preprocess_mediation_input(
      yy, yy_se, mm, mm_se, geno_y, geno_m, conf, opt);

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  // omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  zqtl_model_t<Mat> model_y(Y, D2);

  ////////////////////////////////////////////////////////////////
  // factored mediation effect
  const Index Kmax = static_cast<Index>(opt.k());
  const Index K = std::min(std::min(Kmax, Y.cols()), M.cols());

  auto theta_med_left = make_dense_spike_slab<Scalar>(M.cols(), K, opt);
  auto theta_med_right = make_dense_spike_slab<Scalar>(Y.cols(), K, opt);

  auto delta_med =
      make_factored_regression_eta(M, Y, theta_med_left, theta_med_right);

  if (opt.mf_svd_init()) {
    delta_med.init_by_svd(Y, opt.jitter());
  } else {
    std::mt19937 _rng(opt.rseed());
    delta_med.jitter(opt.jitter(), _rng);
  }

  // intercept    ~ R 1 theta
  // Vt intercept ~ D2 Vt 1 theta
  auto theta_intercept = make_dense_slab<Scalar>(VtI.cols(), Y.cols(), opt);
  auto eta_intercept = make_regression_eta(VtI, Y, theta_intercept);
  eta_intercept.init_by_dot(Y, opt.jitter());

  // confounder -- or bias
  auto theta_conf_y = make_dense_slab<Scalar>(VtC.cols(), Y.cols(), opt);
  auto eta_conf_y = make_regression_eta(VtC, Y, theta_conf_y);
  eta_conf_y.init_by_dot(Y, opt.jitter());

  Rcpp::List out_unmed_param = Rcpp::List::create();

  Mat llik;
  Mat M0;
  if (opt.do_direct_effect()) {
    M0 = estimate_direct_effect(rng, opt,
                                std::make_tuple(Y, M, U, Vt, D2, VtI, VtC));

    if (opt.verbose()) TLOG("Finished direct model estimation\n\n");

    auto theta_unmed = make_dense_spike_slab<Scalar>(M0.cols(), Y.cols(), opt);
    auto delta_unmed = make_regression_eta(M0, Y, theta_unmed);
    delta_unmed.init_by_dot(Y, opt.jitter());

    llik = impl_fit_eta_delta(model_y, opt, rng,
                              std::make_tuple(eta_intercept, eta_conf_y),
                              std::make_tuple(delta_med, delta_unmed));

    out_unmed_param = param_rcpp_list(theta_unmed);
  } else {
    auto theta_unmed = make_dense_slab<Scalar>(Vt.cols(), Y.cols(), opt);
    auto eta_unmed = make_regression_eta(Vt, Y, theta_unmed);
    eta_unmed.init_by_dot(Y, opt.jitter());

    llik = impl_fit_eta_delta(
        model_y, opt, rng,
        std::make_tuple(eta_intercept, eta_conf_y, eta_unmed),
        std::make_tuple(delta_med));
    out_unmed_param = param_rcpp_list(theta_unmed);
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  if (opt.verbose()) TLOG("Finished joint model estimation\n\n");

  Rcpp::List out_left_param = param_rcpp_list(theta_med_left);
  Rcpp::List out_right_param = param_rcpp_list(theta_med_right);

  Rcpp::List var_decomp = Rcpp::List::create();

  if (opt.do_var_calc()) {
    var_decomp = _variance_calculation(eta_intercept, delta_med, theta_med_left,
                                       opt, std::make_tuple(Y, M, U, D2));

    if (opt.verbose()) TLOG("Finished variance decomposition\n\n");
  }

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["U"] = U, Rcpp::_["Vt"] = Vt,
      Rcpp::_["D2"] = D2, Rcpp::_["M"] = M, Rcpp::_["M0"] = M0,
      Rcpp::_["param.mediated.left"] = out_left_param,
      Rcpp::_["param.mediated.right"] = out_right_param,
      Rcpp::_["param.unmediated"] = out_unmed_param,
      Rcpp::_["param.intercept"] = param_rcpp_list(theta_intercept),
      Rcpp::_["param.covariate"] = param_rcpp_list(theta_conf_y),
      Rcpp::_["llik"] = llik, Rcpp::_["var.decomp"] = var_decomp);
}

////////////////////////////
// variance decomposition //
////////////////////////////

template <typename DIRECT, typename MEDIATED_D, typename... DATA>
Rcpp::List _variance_calculation(DIRECT& eta_direct, MEDIATED_D& delta_med,
                                 options_t& opt,
                                 std::tuple<DATA...>&& data_tup) {
  Mat Y, M, U, D2;
  std::tie(Y, M, U, D2) = data_tup;

  const Index _n = U.rows();
  const Index _T = Y.cols();
  const Scalar n = static_cast<Scalar>(U.rows());
  const Index nboot = 500;

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  // omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  Mat snUD = std::sqrt(n) * U * D2.cwiseSqrt().asDiagonal();
  Mat snUinvD = std::sqrt(n) * U * D2.cwiseSqrt().cwiseInverse().asDiagonal();
  Mat UinvD = U * D2.cwiseSqrt().cwiseInverse().asDiagonal();

  eta_direct.resolve();
  delta_med.resolve();

  Mat direct_ind(_n, _T);
  Mat med_ind(_n, _T);

  Mat temp(1, _T);
  running_stat_t<Mat> direct_stat(1, _T);
  running_stat_t<Mat> med_stat(1, _T);

  // direct   : X * theta_direct
  //            sqrt(n) U D Vt * theta_direct
  //            sqrt(n) U D eta_direct
  //
  for (Index b = 0; b < nboot; ++b) {
    direct_ind = snUD * eta_direct.sample(rng);
    column_var(direct_ind, temp);
    direct_stat(temp / n);
  }

  // mediated : X * inv(R) * mm.val * theta_med
  //            X * V inv(D2) Vt * mm.val * theta_med
  //            sqrt(n) U inv(D) * Vt * mm.val * theta_med
  //            sqrt(n) U inv(D) * delta_med
  //
  for (Index b = 0; b < nboot; ++b) {
    med_ind = snUinvD * delta_med.sample(rng);
    column_var(med_ind, temp);
    med_stat(temp / n);
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  return Rcpp::List::create(Rcpp::_["var.direct.mean"] = direct_stat.mean(),
                            Rcpp::_["var.direct.var"] = direct_stat.var(),
                            Rcpp::_["var.med.mean"] = med_stat.mean(),
                            Rcpp::_["var.med.var"] = med_stat.var());
}

template <typename DIRECT, typename MEDIATED_D, typename MEDIATED_E,
          typename... DATA>
Rcpp::List _variance_calculation(DIRECT& eta_direct, MEDIATED_D& delta_med,
                                 MEDIATED_E theta_med, options_t& opt,
                                 std::tuple<DATA...>&& data_tup) {
  Mat Y, M, U, D2;
  std::tie(Y, M, U, D2) = data_tup;

  const Index _n = U.rows();
  const Index _T = Y.cols();
  const Scalar n = static_cast<Scalar>(U.rows());
  const Index nboot = 500;
  const Index _K = M.cols();

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  // omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  Mat snUD = std::sqrt(n) * U * D2.cwiseSqrt().asDiagonal();
  Mat snUinvD = std::sqrt(n) * U * D2.cwiseSqrt().cwiseInverse().asDiagonal();
  Mat UinvD = U * D2.cwiseSqrt().cwiseInverse().asDiagonal();

  eta_direct.resolve();
  delta_med.resolve();

  Mat direct_ind(_n, _T);
  Mat med_ind(_n, _T);

  Mat temp(1, _T);
  running_stat_t<Mat> direct_stat(1, _T);
  running_stat_t<Mat> med_stat(1, _T);

  // direct   : X * theta_direct
  //            sqrt(n) U D Vt * theta_direct
  //            sqrt(n) U D eta_direct
  //
  for (Index b = 0; b < nboot; ++b) {
    direct_ind = snUD * eta_direct.sample(rng);
    column_var(direct_ind, temp);
    direct_stat(temp / n);
  }

  // mediated : X * inv(R) * mm.val * theta_med
  //            X * V inv(D2) Vt * mm.val * theta_med
  //            sqrt(n) U inv(D) * Vt * mm.val * theta_med
  //            sqrt(n) U inv(D) * delta_med
  //
  for (Index b = 0; b < nboot; ++b) {
    med_ind = snUinvD * delta_med.sample(rng);
    column_var(med_ind, temp);
    med_stat(temp / n);
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  Mat theta_med_mean = mean_param(theta_med);
  Mat var_med_each(_K, _T);

  for (Index k = 0; k < _K; ++k) {
    column_var(snUinvD * (M.col(k) * theta_med_mean.row(k)), temp);
    var_med_each.row(k) = temp / n;
  }

  return Rcpp::List::create(Rcpp::_["var.direct.mean"] = direct_stat.mean(),
                            Rcpp::_["var.direct.var"] = direct_stat.var(),
                            Rcpp::_["var.med.each"] = var_med_each,
                            Rcpp::_["var.med.mean"] = med_stat.mean(),
                            Rcpp::_["var.med.var"] = med_stat.var());
}

template <typename RNG, typename... DATA>
Mat _direct_effect_propensity(RNG& rng, options_t& opt,
                              std::tuple<DATA...>&& data_tup) {
  Mat mm, yy, Vt, D2, VtC, U;
  std::tie(mm, yy, Vt, D2, VtC, U) = data_tup;

  const Index max_n_submodel = opt.n_submodel_model();
  const Index n_submodel =
      (max_n_submodel > 0) ? std::min(max_n_submodel, mm.cols()) : mm.cols();

  const Index n_traits = yy.cols();

  Mat Y_resid = Mat::Zero(yy.rows(), n_traits * n_submodel);

  std::vector<Index> rand_med(mm.cols());

  std::shuffle(rand_med.begin(), rand_med.end(),
               std::mt19937{std::random_device{}()});

  const Index n_strat = Y_resid.rows() * opt.n_strat_size();
  Mat Y_strat(n_strat, n_traits);
  Mat Vt_strat(n_strat, Vt.cols());
  Mat D2_strat(n_strat, D2.cols());
  Mat VtC_strat(n_strat, VtC.cols());

  std::mt19937 rng_n(opt.rseed());
  discrete_sampler_t<Vec> randN(yy.rows());
  const Scalar half = static_cast<Scalar>(0.5);

  for (Index k = 0; k < n_submodel; ++k) {
    // sample eigen components inversely proportional to the
    // divergence of this mediation effect

    Index k_rand = rand_med.at(k);
    Mat Mk = mm.col(k_rand);

    Vec logScore = -(Mk.cwiseProduct(Mk)).cwiseQuotient(D2) * half;

    for (Index ri = 0; ri < n_strat; ++ri) {
      Index r = randN(logScore, rng_n);
      Y_strat.row(ri) = yy.row(r);
      Vt_strat.row(ri) = Vt.row(r);
      D2_strat.row(ri) = D2.row(r);
      VtC_strat.row(ri) = VtC.row(r);
    }

    zqtl_model_t<Mat> y_strat_model(Y_strat, D2_strat);

    auto theta = make_dense_slab<Scalar>(Vt_strat.cols(), Y_strat.cols(), opt);
    auto eta = make_regression_eta(Vt_strat, Y_strat, theta);
    eta.init_by_dot(Y_strat, opt.jitter());

    auto inter =
        make_dense_spike_slab<Scalar>(VtC_strat.cols(), Y_strat.cols(), opt);
    auto eta_intercept = make_regression_eta(VtC_strat, Y_strat, inter);
    eta_intercept.init_by_dot(Y_strat, opt.jitter());

    auto _llik = impl_fit_eta(y_strat_model, opt, rng,
                              std::make_tuple(eta, eta_intercept));

    eta.resolve();
    Mat _y = Vt * mean_param(theta);
    for (Index j = 0; j < _y.cols(); ++j) {
      Index kj = n_traits * k + j;
      Y_resid.col(kj) = _y.col(j);
    }

    if (opt.verbose())
      TLOG("Propensity sampling model : " << (k + 1) << " / " << n_submodel);
  }

  Mat _resid_z = Vt.transpose() * D2.asDiagonal() * Y_resid;
  Mat resid_z = _resid_z;
  resid_z = standardize_zscore(_resid_z, Vt, D2.cwiseSqrt());

  const Scalar denom = static_cast<Scalar>(resid_z.cols());
  Mat resid_z_mean = resid_z * Mat::Ones(resid_z.cols(), 1) / denom;

  return Vt * resid_z_mean;
}

template <typename RNG, typename... DATA>
Mat _direct_effect_conditional(RNG& rng, options_t& opt,
                               std::tuple<DATA...>&& data_tup) {
  Mat mm, yy, Vt, D2, VtC, U;
  std::tie(mm, yy, Vt, D2, VtC, U) = data_tup;

  const Index n_submodel = mm.cols();
  const Index max_n_submodel =
      std::max(n_submodel, static_cast<Index>(opt.n_submodel_model()));
  const Index n_traits = yy.cols();

  Mat Y_resid = Mat::Zero(yy.rows(), n_traits * max_n_submodel);

  const Index submodel_size =
      std::min(n_submodel - 1, static_cast<Index>(opt.n_submodel_size()));

  const Index n_med = mm.cols();
  std::vector<Index> rand_med(n_med);

  for (Index k = 0; k < max_n_submodel; ++k) {
    zqtl_model_t<Mat> y_model(yy, D2);

    if (k % n_med == 0) {
      std::shuffle(rand_med.begin(), rand_med.end(),
                   std::mt19937{std::random_device{}()});
    }

    // construct mediators not to regress out
    Mat Mk;
    if (submodel_size < 1) {
      Mk.resize(mm.rows(), 1);
      Mk = Mat::Zero(mm.rows(), 1);
    } else {
      Mk.resize(mm.rows(), submodel_size);
      for (Index j = 0; j < submodel_size; ++j) {
        Index k_rand = rand_med.at((k + j) % n_med);
        Mk.col(j) = mm.col(k_rand);
      }
    }

    auto med = make_dense_slab<Scalar>(Mk.cols(), yy.cols(), opt);
    auto delta = make_regression_eta(Mk, yy, med);

    auto theta = make_dense_spike_slab<Scalar>(Vt.cols(), yy.cols(), opt);
    auto eta = make_regression_eta(Vt, yy, theta);

    auto inter = make_dense_spike_slab<Scalar>(VtC.cols(), yy.cols(), opt);
    auto eta_inter = make_regression_eta(VtC, yy, inter);

    auto _llik =
        impl_fit_eta_delta(y_model, opt, rng, std::make_tuple(eta, eta_inter),
                           std::make_tuple(delta));

    eta.resolve();
    Mat _y = Vt * mean_param(theta);
    for (Index j = 0; j < _y.cols(); ++j) {
      Index kj = n_traits * k + j;
      Y_resid.col(kj) = _y.col(j);
    }

    if (opt.verbose())
      TLOG("Submodel mediator model : " << (k + 1) << " / " << max_n_submodel);
  }

  Mat _resid_z = Vt.transpose() * D2.asDiagonal() * Y_resid;
  Mat resid_z = _resid_z;
  resid_z = standardize_zscore(_resid_z, Vt, D2.cwiseSqrt());

  const Scalar denom = static_cast<Scalar>(resid_z.cols());
  Mat resid_z_mean = resid_z * Mat::Ones(resid_z.cols(), 1) / denom;

  return Vt * resid_z_mean;
}

////////////////////////////////////////////////////////////////
template <typename RNG, typename Left, typename Right, typename... DATA>
inline void _impl_de_factorization(Left& epsilon_left, Right& epsilon_right,
                                   const options_t& opt, RNG& rng,
                                   std::tuple<DATA...>&& data_tup) {
  Mat Target, D2, Design, DesignC;
  std::tie(Target, D2, Design, DesignC) = data_tup;

  auto theta_c =
      make_dense_spike_slab<Scalar>(DesignC.cols(), Target.cols(), opt);
  auto eta_c = make_regression_eta(DesignC, Target, theta_c);
  eta_c.init_by_dot(Target, opt.jitter());

  auto delta_random =
      make_factored_regression_eta(Design, Target, epsilon_left, epsilon_right);

  if (opt.mf_svd_init()) {
    Mat Dinv = D2.cwiseSqrt().cwiseInverse();
    delta_random.init_by_svd(Target, opt.jitter());
  } else {
    std::mt19937 _rng(opt.rseed());
    delta_random.jitter(opt.jitter(), _rng);
  }

  zqtl_model_t<Mat> y_model(Target, D2);

  Mat llik = impl_fit_eta_delta(y_model, opt, rng, std::make_tuple(eta_c),
                                std::make_tuple(delta_random));

  delta_random.resolve();
}

template <typename RNG, typename... DATA>
Mat _direct_effect_factorization(RNG& rng, const options_t& opt,
                                 std::tuple<DATA...>&& data_tup) {
  Mat mm, yy, Vt, D2, VtC, U;
  std::tie(mm, yy, Vt, D2, VtC, U) = data_tup;

  const Index n_traits = yy.cols();
  Mat YM(mm.rows(), mm.cols() + yy.cols());
  YM << yy, mm;

  // standardize [Y, M] z-scores
  Mat z_ym = Vt.transpose() * YM;
  Mat z_ym_std = z_ym;
  Mat D = D2.cwiseSqrt();

  if (opt.do_rescale()) {
    z_ym_std = standardize_zscore(z_ym, Vt, D);
  } else {
    z_ym_std = center_zscore(z_ym, Vt, D);
  }

  YM = Vt * z_ym_std;

  if (opt.verbose()) TLOG("Fit the factorization model");

  Mat Z0, lodds_mat, loading_mat;

  if (opt.de_factorization_model() == 1) {
    //////////////////////////////////////////////
    // model 1: D x (SVD factor x Trait factor) //
    //////////////////////////////////////////////
    Mat Design = D.asDiagonal();
    const Index K = std::min(std::min(YM.cols(), YM.rows()), Design.cols());
    // This must be sparse in both ways to select svd components
    auto epsilon_svd = make_dense_spike_slab<Scalar>(Design.cols(), K, opt);
    auto epsilon_trait = make_dense_spike_slab<Scalar>(YM.cols(), K, opt);

    _impl_de_factorization(epsilon_svd, epsilon_trait, opt, rng,
                           std::make_tuple(YM, D2, Design, VtC));

    Z0 = Vt.transpose() * Design * mean_param(epsilon_svd);
    lodds_mat = log_odds_param(epsilon_trait);
    loading_mat = mean_param(epsilon_trait);

  } else {
    /////////////////////////////////////////////////
    // model 0 : DUt x (Ind factor x Trait factor) //
    /////////////////////////////////////////////////

    Mat Design = D.asDiagonal() * U.transpose();
    const Index K = std::min(std::min(YM.cols(), YM.rows()), Design.cols());
    auto epsilon_indiv = make_dense_col_slab<Scalar>(Design.cols(), K, opt);
    auto epsilon_trait = make_dense_spike_slab<Scalar>(YM.cols(), K, opt);

    _impl_de_factorization(epsilon_indiv, epsilon_trait, opt, rng,
                           std::make_tuple(YM, D2, Design, VtC));

    Z0 = Vt.transpose() * Design * mean_param(epsilon_indiv);
    lodds_mat = log_odds_param(epsilon_trait);
    loading_mat = mean_param(epsilon_trait);
  }

  if (opt.verbose()) TLOG("Finished the factorization\n\n");

  //////////////////////////////////////////////////
  // Figure out Y only or M only components

  const Scalar cutoff = opt.med_lodds_cutoff();

  Scalar denom = 0.0;

  for (Index k = 0; k < Z0.cols(); ++k) {
    bool trait_on = false;
    bool mediator_on = false;
    Vec lodds = lodds_mat.col(k);
    Vec loading = loading_mat.col(k);
    for (Index t = 0; t < n_traits; ++t) {
      if (lodds(t) > cutoff) {
        if (opt.verbose())
          TLOG("On trait    [" << std::setw(10) << (t + 1) << "]"
                               << " @ [" << std::setw(10) << (k + 1) << "]");
        trait_on = true;
      }
    }

    for (Index t = n_traits; t < YM.cols(); ++t) {
      if (lodds(t) > cutoff) {
        if (opt.verbose())
          TLOG("On mediator [" << std::setw(10) << (t - n_traits + 1) << "]"
                               << " @ [" << std::setw(10) << (k + 1) << "]");

        mediator_on = true;
      }
    }

    Mat z_k = Z0.col(k);
    Mat z_k_std = opt.do_rescale() ? standardize_zscore(z_k, Vt, D)
                                   : center_zscore(z_k, Vt, D);
    Z0.col(k) = z_k_std;

    if (trait_on && mediator_on) {
      if (opt.verbose())
        TLOG("CommonFactor[" << std::setw(10) << (k + 1) << "]");
      Z0.col(k) = Z0.col(k) * 0.0;
    }
  }

  return Vt * Z0;
}

template <typename RNG, typename... DATA>
Mat estimate_direct_effect(RNG& rng, options_t& opt,
                           std::tuple<DATA...>&& data_tup) {
  Mat Y, M, U, Vt, D2, VtI, VtC;
  std::tie(Y, M, U, Vt, D2, VtI, VtC) = data_tup;

  Mat VtIC(Vt.rows(), VtC.cols() + VtI.cols());
  VtIC << VtI, VtC;

  Index n_trait = Y.cols();
  Mat M0;
  if (opt.do_de_factorization()) {
    if (opt.verbose())
      TLOG("Estimation of direct effect from factorization effects");
    M0 = _direct_effect_factorization(rng, opt,
                                      std::make_tuple(M, Y, Vt, D2, VtIC, U));
  } else if (opt.do_de_propensity()) {
    if (opt.verbose())
      TLOG("Estimation of direct effect by propensity sampling");
    M0.resize(M.rows(), n_trait);
    for (Index tt = 0; tt < n_trait; ++tt) {
      M0.col(tt) = _direct_effect_propensity(
          rng, opt, std::make_tuple(M, Y.col(tt), Vt, D2, VtIC, U));
      if (opt.verbose())
        TLOG("Finished on the trait : " << (tt + 1) << " / " << n_trait);
    }
  } else {
    if (opt.verbose()) TLOG("Estimation of direct effect by invariance");
    M0.resize(M.rows(), n_trait);
    for (Index tt = 0; tt < n_trait; ++tt) {
      M0.col(tt) = _direct_effect_conditional(
          rng, opt, std::make_tuple(M, Y.col(tt), Vt, D2, VtIC, U));
      if (opt.verbose())
        TLOG("Finished on the trait : " << (tt + 1) << " / " << n_trait);
    }
  }

  return M0;
}

bool check_mediation_input(const effect_y_mat_t& yy,        // z_y
                           const effect_y_se_mat_t& yy_se,  // z_y_se
                           const effect_m_mat_t& mm,        // z_m
                           const effect_m_se_mat_t& mm_se,  // z_m_se
                           const geno_y_mat_t& geno_y,      // genotype_y
                           const geno_m_mat_t& geno_m,      // genotype_m
                           const conf_mat_t& conf,          // snp confounder
                           options_t& opt) {
  //////////////////////
  // check dimensions //
  //////////////////////

  if (opt.with_ld_matrix()) {
    ELOG("Deprecated: we no longer use full LD matrix.");
    return false;
  }

  if (yy.val.rows() != yy_se.val.rows()) {
    ELOG("Check dimensions of effect and se on y");
    return false;
  }

  if (yy.val.rows() != geno_y.val.cols()) {
    ELOG("Check dimensions of genotype matrix on y");
    return false;
  }

  if (mm.val.rows() != mm_se.val.rows()) {
    ELOG("Check dimensions of effect and se on m");
    return false;
  }

  if (mm.val.cols() != mm_se.val.cols()) {
    ELOG("Check dimensions of effect and se on m");
    return false;
  }

  if (mm.val.rows() != geno_m.val.cols()) {
    ELOG("Check dimensions of genotype matrix on m");
    return false;
  }

  if (yy.val.rows() != conf.val.rows()) {
    ELOG("Check dimensions of C");
    return false;
  }

  if (opt.verbose()) TLOG("GWAS sample size = " << opt.sample_size());
  if (opt.verbose()) TLOG("Mediator sample size = " << opt.m_sample_size());

  return true;
}

std::tuple<Mat, Mat, Mat, Mat, Mat, Mat, Mat> preprocess_mediation_input(
    const effect_y_mat_t& yy,        // z_y
    const effect_y_se_mat_t& yy_se,  // z_y_se
    const effect_m_mat_t& mm,        // z_m
    const effect_m_se_mat_t& mm_se,  // z_m_se
    const geno_y_mat_t& geno_y,      // genotype_y
    const geno_m_mat_t& geno_m,      // genotype_m
    const conf_mat_t& conf,          // snp confounder
    options_t& opt) {
  const Scalar n = static_cast<Scalar>(opt.sample_size());
  const Scalar n1 = static_cast<Scalar>(opt.m_sample_size());

  /////////////////////////////////
  // Pre-process genotype matrix //
  /////////////////////////////////

  Mat _effect_y_z, effect_sqrt_y, weight_y;
  std::tie(_effect_y_z, effect_sqrt_y, weight_y) =
      preprocess_effect(yy.val, yy_se.val, n);

  Mat U, D, Vt;
  std::tie(U, D, Vt) = do_svd(geno_y.val, opt);
  Mat D2 = D.cwiseProduct(D);

  Mat effect_y_z = _effect_y_z;

  if (opt.do_rescale()) {
    effect_y_z = standardize_zscore(_effect_y_z, Vt, D);
    if (opt.verbose()) TLOG("Standardized z-scores of GWAS QTLs");
  } else {
    effect_y_z = center_zscore(_effect_y_z, Vt, D);
    if (opt.verbose()) TLOG("Centered z-scores of GWAS QTLs");
  }

  Mat Y = Vt * effect_y_z;

  Mat _effect_m_z, effect_sqrt_m, weight_m;
  std::tie(_effect_m_z, effect_sqrt_m, weight_m) =
      preprocess_effect(mm.val, mm_se.val, n1);
  Mat effect_m_z = _effect_m_z;

  ///////////////////////////////////////
  // construct mediation design matrix //
  ///////////////////////////////////////

  Mat M;
  if (opt.multi_med_effect()) {
    Mat safe_mm_val = mm.val;
    remove_missing(mm.val, safe_mm_val);
    Mat M0 = Vt * safe_mm_val;
    M = D2.asDiagonal() * M0;
    if (opt.verbose()) TLOG("Use multivariate mediation QTL statistics");

  } else {
    Mat U_m, D_m, Vt_m;
    std::tie(U_m, D_m, Vt_m) = do_svd(geno_m.val, opt);
    Mat D2_m = D_m.cwiseProduct(D_m);

    if (opt.do_rescale()) {
      effect_m_z = standardize_zscore(_effect_m_z, Vt_m, D_m);
      if (opt.verbose()) TLOG("Standardized z-scores of mediation QTLs");
    } else {
      effect_m_z = center_zscore(_effect_m_z, Vt_m, D_m);
      if (opt.verbose()) TLOG("Centered z-scores of mediation QTLs");
    }

    // alpha.uni           = S1 R1 inv(S1) alpha
    //
    // alpha               = S1 inv(R1) inv(S1) alpha.uni
    //                     = S1 V1 inv(D1^2) t(V1) inv(S1) alpha.uni
    //
    // t(V) R inv(S) alpha = t(V) V D^2 t(V) inv(S) alpha
    //                     = D^2 t(V) (S1/S) V1 inv(D1^2) t(V1) inv(S1)
    //                     alpha.uni
    //                   M = D^2 t(V) inv(S) S1 (V1/D1) t(V1/D1) * Z_alpha

    Mat Vt_m_d = D_m.cwiseInverse().asDiagonal() * Vt_m;

    // un-normalized version is more stable
    effect_m_z = Vt.transpose() * D2.asDiagonal() * Vt * Vt_m_d.transpose() *
                 Vt_m_d * effect_m_z;

    // Mat M(Vt.rows(), effect_m_z.cols());
    // Mat VtZ = Vt_m_d * effect_m_z;
    // Mat invS_S1 = weight_y.asDiagonal() * effect_sqrt_m;  // p x K
    // Mat stuff(Vt.rows(), 1);
    // for (Index k = 0; k < effect_m_z.cols(); ++k) {
    //   stuff = Vt_m_d.transpose() * VtZ.col(k);
    //   M.col(k) = D2.asDiagonal() * Vt * stuff.cwiseProduct(invS_S1.col(k));
    // }

    M.resize(Vt.rows(), effect_m_z.cols());
    M = Vt * effect_m_z;
    if (opt.verbose()) TLOG("Use summary mediation QTL statistics");
  }

  ////////////////////////////////////////////////////////////////
  // Other covariates

  Mat VtI = Vt * Mat::Ones(Vt.cols(), static_cast<Index>(1)) /
            static_cast<Scalar>(Vt.cols());
  Mat VtC = Vt * conf.val;

  if (opt.n_duplicate_sample() >= 2) {
    // duplicate samples to improve optimization
    const Index dup = opt.n_duplicate_sample();
    const Index one_time = 1;
    Y = Y.replicate(dup, one_time);
    M = M.replicate(dup, one_time);
    U = U.replicate(dup, one_time);
    Vt = Vt.replicate(dup, one_time);
    D2 = D2.replicate(dup, one_time);
    VtI = VtI.replicate(dup, one_time);
    VtC = VtC.replicate(dup, one_time);
  }

  return std::make_tuple(Y, M, U, Vt, D2, VtI, VtC);
}

#endif
