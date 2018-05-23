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

template <typename DIRECT, typename... DATA>
Rcpp::List _bootstrap_direct(const Mat obs_lodds, DIRECT& eta_direct,
                             const options_t& opt,
                             std::tuple<DATA...>&& data_tup);

template <typename... DATA>
Rcpp::List _bootstrap_marginal(const Mat obs_lodds, const options_t& opt,
                               std::tuple<DATA...>&& data_tup);

template <typename DIRECT, typename MEDIATED_D, typename MEDIATED_E,
          typename... DATA>
Rcpp::List _variance_calculation(DIRECT& eta_direct, MEDIATED_D& delta_med,
                                 MEDIATED_E theta_med, const options_t& opt,
                                 std::tuple<DATA...>&& data_tup);

template <typename MODEL_Y, typename DIRECT, typename CONF, typename MEDIATED_E,
          typename... DATA>
Rcpp::List _fine_map(MODEL_Y& model_y, DIRECT& eta_direct, CONF& eta_conf_y,
                     MEDIATED_E theta_med_org, const options_t& opt,
                     std::tuple<DATA...>&& data_tup);

Rcpp::List impl_fit_med_zqtl(const effect_y_mat_t& yy,        // z_y
                             const effect_y_se_mat_t& yy_se,  // z_y_se
                             const effect_m_mat_t& mm,        // z_m
                             const effect_m_se_mat_t& mm_se,  // z_m_se
                             const geno_y_mat_t& geno_y,      // genotype_y
                             const geno_m_mat_t& geno_m,      // genotype_m
                             const conf_mat_t& conf,          // snp confounder
                             const options_t& opt) {
  //////////////////////
  // check dimensions //
  //////////////////////

  if (opt.with_ld_matrix()) {
    ELOG("Deprecated: longer use full LD matrix.");
    return Rcpp::List::create();
  }

  if (yy.val.rows() != yy_se.val.rows()) {
    ELOG("Check dimensions of effect and se on y");
    return Rcpp::List::create();
  }

  if (mm.val.rows() != mm_se.val.rows()) {
    ELOG("Check dimensions of effect and se on m");
    return Rcpp::List::create();
  }

  if (mm.val.cols() != mm_se.val.cols()) {
    ELOG("Check dimensions of effect and se on m");
    return Rcpp::List::create();
  }

  if (yy.val.rows() != conf.val.rows()) {
    ELOG("Check dimensions of C");
    return Rcpp::List::create();
  }

  const Scalar n = static_cast<Scalar>(opt.sample_size());
  const Scalar n1 = static_cast<Scalar>(opt.m_sample_size());

  TLOG("GWAS sample size = " << opt.sample_size());
  TLOG("Mediator sample size = " << opt.m_sample_size());

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
  effect_y_z = center_zscore(_effect_y_z, Vt, D);
  TLOG("Centered z-scores of GWAS QTLs");
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
    if (opt.do_rescale()) {
      rescale(M0);
    }
    M = D2.asDiagonal() * M0;
    TLOG("Use multivariate mediation QTL statistics");

  } else {
    Mat U_m, D_m, Vt_m;
    std::tie(U_m, D_m, Vt_m) = do_svd(geno_m.val, opt);
    Mat D2_m = D_m.cwiseProduct(D_m);

    if (opt.do_rescale()) {
      effect_m_z = standardize_zscore(_effect_m_z, Vt_m, D_m);
      TLOG("Standardized z-scores of mediation QTLs");
    } else {
      effect_m_z = center_zscore(_effect_m_z, Vt_m, D_m);
      TLOG("Centered z-scores of mediation QTLs");
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

    TLOG("Use summary mediation QTL statistics");
  }

  ////////////////////////////////////////////////////////////////
  // Other covariates

  Mat VtI = Vt * Mat::Ones(Vt.cols(), static_cast<Index>(1)) /
            static_cast<Scalar>(Vt.cols());
  Mat VtC = Vt * conf.val;

#ifdef EIGEN_USE_MKL_ALL
  // random seed initialization
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  Mat var_intercept = Mat::Ones(Vt.rows(), 1);
  auto theta_var = make_dense_slab<Scalar>(var_intercept.cols(), Y.cols(), opt);
  auto zeta_var = make_regression_eta(var_intercept, Y, theta_var);

  ////////////////////////////////////////////////////////////////
  // STEP 1. Estimate one-mediator model
  ////////////////////////////////////////////////////////////////

  Index n_single = M.cols();
  if (opt.n_single_model() > 0) {
    n_single = std::min(static_cast<Index>(opt.n_single_model()), n_single);
  }

  Index n_trait = Y.cols();
  Mat Y_resid = Mat::Zero(Y.rows(), Y.cols() * n_single);

  std::vector<Index> rand_med(M.cols());
  std::shuffle(rand_med.begin(), rand_med.end(),
               std::mt19937{std::random_device{}()});

  for (Index k = 0; k < n_single; ++k) {
    zqtl_model_t<Mat> yk(Y, D2);

    // estimate each Vt * theta and combine them all
    auto theta_k = make_dense_slab<Scalar>(Vt.cols(), Y.cols(), opt);
    auto eta_k = make_regression_eta(Vt, Y, theta_k);
    eta_k.init_by_dot(Y, opt.jitter());

    Index k_rand = rand_med.at(k);
    Mat Mk = M.col(k_rand);

    auto med_k = make_dense_spike_slab<Scalar>(Mk.cols(), Y.cols(), opt);
    auto delta_k = make_regression_eta(Mk, Y, med_k);
    delta_k.init_by_dot(Y, opt.jitter());

    auto _llik = impl_fit_eta_delta(yk, opt, rng, std::make_tuple(eta_k),
                                    std::make_tuple(delta_k));

    Mat _y = Vt * mean_param(theta_k);
    for (Index j = 0; j < _y.cols(); ++j) {
      Index kj = n_trait * k + j;
      Y_resid.col(kj) = _y.col(j);
    }
    TLOG("Residual on the mediator " << (k + 1) << " / " << n_single);
  }

  Mat _resid_z = Vt.transpose() * D2.asDiagonal() * Y_resid;
  Mat resid_z = standardize_zscore(_resid_z, Vt, D);
  Y_resid = Vt * resid_z;

  ////////////////////////////////////////////////
  // STEP 2. Estimate average unmediated effect //
  ////////////////////////////////////////////////

  const Scalar denom = static_cast<Scalar>(n_single);
  Mat resid_z_mean = resid_z * Mat::Ones(n_single, 1) / denom;
  Mat M0 = Vt * resid_z_mean;

  if (opt.direct_model() > 1) {
    Index reK = opt.direct_model() - 1;
    if (reK > n_single) {
      reK = n_single;
    }
    TLOG("Estimate additional " << reK << " effects ");
    Mat mm = M0;
    auto avg_unmed =
        make_dense_spike_slab<Scalar>(mm.cols(), Y_resid.cols(), opt);
    auto delta_unmed = make_regression_eta(mm, Y_resid, avg_unmed);

    // use regular spike-slab on both sides
    auto thetaL = make_dense_spike_slab<Scalar>(Vt.cols(), reK, opt);
    auto thetaR = make_dense_col_spike_slab<Scalar>(Y_resid.cols(), reK, opt);
    auto eta_f = make_factored_regression_eta(Vt, Y_resid, thetaL, thetaR);
    eta_f.init_by_svd(Y_resid, opt.jitter());

    zqtl_model_t<Mat> _model_r(Y_resid, D2);
    Mat _llik = impl_fit_eta_delta(_model_r, opt, rng, std::make_tuple(eta_f),
                                   std::make_tuple(delta_unmed));

    Mat zz = Vt.transpose() * D2.asDiagonal() * Vt * mean_param(thetaL);
    Mat LO = log_odds_param(thetaR);

    TLOG("Log-odds:\n" << LO.transpose());

    const Scalar cutoff = opt.med_lodds_cutoff();

    Index K = 0;
    for (Index k = 0; k < reK; ++k) {
      if (LO(0, k) > cutoff) ++K;
    }

    if (K > 1) {
      M0.resize(mm.rows(), 1 + K);
      Index pos = 0;
      M0.col(pos) = mm;
      for (Index k = 0; k < reK; ++k) {
        if (LO(0, k) >= cutoff) {
          M0.col(++pos) = zz.col(k);
        }
      }
    }

    TLOG("Finished factorization");
  }

  ////////////////////////////////////////////////////////////////
  // STEP 3. Estimate the full model
  ////////////////////////////////////////////////////////////////

  zqtl_model_t<Mat> model_y(Y, D2);

  ////////////////////////////////////////////////////////////////
  // construct delta_med to capture overall (potential) mediation effect
  auto theta_med = make_dense_spike_slab<Scalar>(M.cols(), Y.cols(), opt);
  auto delta_med = make_regression_eta(M, Y, theta_med);
  delta_med.init_by_dot(Y, opt.jitter());

  // intercept    ~ R 1 theta
  // Vt intercept ~ D2 Vt 1 theta
  auto theta_intercept = make_dense_slab<Scalar>(VtI.cols(), Y.cols(), opt);
  auto eta_intercept = make_regression_eta(VtI, Y, theta_intercept);

  // confounder -- or bias
  auto theta_conf_y = make_dense_slab<Scalar>(VtC.cols(), Y.cols(), opt);
  auto eta_conf_y = make_regression_eta(VtC, Y, theta_conf_y);

  auto theta_unmed = make_dense_spike_slab<Scalar>(M0.cols(), Y.cols(), opt);
  auto delta_unmed = make_regression_eta(M0, Y, theta_unmed);
  delta_unmed.init_by_dot(Y, opt.jitter());

  auto llik = impl_fit_eta_delta(model_y, opt, rng,
                                 std::make_tuple(eta_intercept, eta_conf_y),
                                 std::make_tuple(delta_med, delta_unmed));

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  TLOG("Finished joint model estimation\n\n");

  Rcpp::List var_decomp = _variance_calculation(
      eta_intercept, delta_med, theta_med, opt, std::make_tuple(Y, M, U, D2));

  TLOG("Finished variance decomposition\n\n");

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["resid.Z"] = resid_z, Rcpp::_["U"] = U,
      Rcpp::_["M"] = M, Rcpp::_["Vt"] = Vt, Rcpp::_["S.inv.y"] = weight_y,
      Rcpp::_["D2"] = D2,
      Rcpp::_["param.mediated"] = param_rcpp_list(theta_med),
      Rcpp::_["param.unmediated"] = param_rcpp_list(theta_unmed),
      Rcpp::_["param.intercept"] = param_rcpp_list(theta_intercept),
      Rcpp::_["param.covariate"] = param_rcpp_list(theta_conf_y),
      Rcpp::_["param.var"] = param_rcpp_list(theta_var), Rcpp::_["llik"] = llik,
      Rcpp::_["var.decomp"] = var_decomp);
}

template <typename DIRECT, typename MEDIATED_D, typename MEDIATED_E,
          typename... DATA>
Rcpp::List _variance_calculation(DIRECT& eta_direct, MEDIATED_D& delta_med,
                                 MEDIATED_E theta_med, const options_t& opt,
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
  omp_set_num_threads(opt.nthread());
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

#endif
