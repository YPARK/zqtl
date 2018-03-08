#ifndef RCPP_ZQTL_REGRESSION_HH_
#define RCPP_ZQTL_REGRESSION_HH_

Rcpp::List impl_fit_zqtl(const Mat& _effect, const Mat& _effect_se,
                         const Mat& X, const Mat& C, const Mat& Cdelta,
                         const options_t& opt) {
  if (opt.with_ld_matrix()) {
    ELOG("Deprecated");
    return Rcpp::List::create();
  }

  if (!opt.with_ld_matrix() && X.cols() != _effect.rows()) {
    ELOG("X and effect contain different number of variables");
    return Rcpp::List::create();
  }

  if (_effect.rows() != _effect_se.rows()) {
    ELOG("Check dimensions of effect and se");
    return Rcpp::List::create();
  }

  if (_effect.rows() != C.rows()) {
    ELOG("Check dimensions of C");
    return Rcpp::List::create();
  }

  if (_effect.rows() != Cdelta.rows()) {
    ELOG("Check dimensions of Cdelta");
    return Rcpp::List::create();
  }

  Mat U, D, D2, Vt;
  std::tie(U, D, Vt) = do_svd(X, opt);
  D2 = D.cwiseProduct(D);
  TLOG("Finished SVD of genotype matrix");

  const Scalar sample_size = static_cast<Scalar>(opt.sample_size());
  Mat _effect_z, effect_sqrt, weight;

  std::tie(_effect_z, effect_sqrt, weight) =
      preprocess_effect(_effect, _effect_se, sample_size);

  Mat effect_z = _effect_z;

  if (opt.do_rescale()) {
    effect_z = standardize_zscore(_effect_z, Vt, D);
    TLOG("Standardized z-scores");
  } else {
    effect_z = center_zscore(_effect_z, Vt, D);
    TLOG("Centered z-scores");
  }

  Mat Y = Vt * effect_z;   // GWAS
  Mat VtC = Vt * C;        // confounder
  Mat VtCd = Vt * Cdelta;  // To correct for phenotype corr

  zqtl_model_t<Mat> model(Y, D2);

  TLOG("Constructed zqtl model");

  // eta_conf = Vt * inv(effect_sq) * C * theta_conf
  auto theta_c = make_dense_spike_slab<Scalar>(VtC.cols(), Y.cols(), opt);
  auto eta_c = make_regression_eta(VtC, Y, theta_c);

  // delta_conf = Vt * Cdelta * theta_conf
  auto theta_c_delta =
      make_dense_spike_slab<Scalar>(VtCd.cols(), Y.cols(), opt);
  auto delta_c = make_regression_eta(VtCd, Y, theta_c_delta);

  // mean effect size --> can be sparse matrix
  auto theta = make_dense_spike_slab<Scalar>(Vt.cols(), Y.cols(), opt);
  auto eta = make_regression_eta(Vt, Y, theta);
  if (opt.weight_y()) eta.set_weight(weight);
  TLOG("Constructed effects");

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  omp_set_num_threads(opt.nthread());
#else
  // random seed initialization
  std::mt19937 rng(opt.rseed());
#endif

  auto llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(eta, eta_c),
                                 std::make_tuple(delta_c));

  // take residuals
  Rcpp::List resid = Rcpp::List::create();
  if (opt.out_resid()) {
    TLOG("Estimate the residuals");

    auto theta_resid = make_dense_slab<Scalar>(Y.rows(), Y.cols(), opt);
    auto delta_resid = make_residual_eta(Y, theta_resid);

    eta.resolve();
    eta_c.resolve();
    delta_c.resolve();

    dummy_eta_t dummy;
    Mat llik_resid = impl_fit_eta_delta(
        model, opt, rng, std::make_tuple(dummy), std::make_tuple(delta_resid),
        std::make_tuple(eta, eta_c), std::make_tuple(delta_c));

    delta_resid.resolve();
    Mat Zhat = Vt.transpose() * delta_resid.repr_mean();
    Mat effect_hat = Zhat.cwiseProduct(effect_sqrt);

    resid = Rcpp::List::create(Rcpp::_["llik"] = llik_resid,
                               Rcpp::_["param"] = param_rcpp_list(theta_resid),
                               Rcpp::_["Z.hat"] = Zhat,
                               Rcpp::_["effect.hat"] = effect_hat);
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  TLOG("Successfully finished regression!");

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["U"] = U, Rcpp::_["Vt"] = Vt,
      Rcpp::_["D2"] = D2, Rcpp::_["S.inv"] = weight,
      Rcpp::_["param"] = param_rcpp_list(theta),
      Rcpp::_["conf"] = param_rcpp_list(theta_c),
      Rcpp::_["conf.delta"] = param_rcpp_list(theta_c_delta),
      Rcpp::_["resid"] = resid, Rcpp::_["llik"] = llik);
}

////////////////////////////////////////////////////////////////
// Factored QTL modeling
Rcpp::List impl_fit_fac_zqtl(const Mat& _effect, const Mat& _effect_se,
                             const Mat& X, const Mat& C, const Mat& Cdelta,
                             const options_t& opt) {
  if (opt.with_ld_matrix()) {
    ELOG("Deprecated: longer use full LD matrix.");
    return Rcpp::List::create();
  }

  if (_effect.rows() != _effect_se.rows()) {
    ELOG("Check dimensions of effect and se");
    return Rcpp::List::create();
  }

  if (_effect.rows() != C.rows()) {
    ELOG("Check dimensions of C");
    return Rcpp::List::create();
  }

  if (_effect.rows() != Cdelta.rows()) {
    ELOG("Check dimensions of Cdelta");
    return Rcpp::List::create();
  }

  Mat U, D, D2, Vt;
  std::tie(U, D, Vt) = do_svd(X, opt);
  D2 = D.cwiseProduct(D);
  TLOG("Finished SVD of genotype matrix");

  const Scalar sample_size = static_cast<Scalar>(opt.sample_size());
  Mat _effect_z, effect_sqrt, weight;

  std::tie(_effect_z, effect_sqrt, weight) =
      preprocess_effect(_effect, _effect_se, sample_size);

  Mat effect_z = _effect_z;

  if (opt.do_rescale()) {
    effect_z = standardize_zscore(_effect_z, Vt, D);
    TLOG("Standardized z-scores");
  } else {
    effect_z = center_zscore(_effect_z, Vt, D);
    TLOG("Centered z-scores");
  }

  Mat Y = Vt * effect_z;
  zqtl_model_t<Mat> model(Y, D2);

  ////////////////////////////////////////////////////////////////
  // constrcut parameters
  const Index K = std::min(static_cast<Index>(opt.k()), Y.cols());

  // confounder
  Mat VtC = Vt * C;
  auto theta_c = make_dense_spike_slab<Scalar>(VtC.cols(), Y.cols(), opt);
  auto eta_c = make_regression_eta(VtC, Y, theta_c);

  ////////////////////////////////////////////////////////////////
  // This is useful to correct for phenotype correlations
  // delta_conf = Vt * Cdelta * theta_conf
  Mat VtCd = Vt * Cdelta;
  auto theta_c_delta =
      make_dense_spike_slab<Scalar>(VtCd.cols(), Y.cols(), opt);
  auto delta_c = make_regression_eta(VtCd, Y, theta_c_delta);

  ////////////////////////////////////////////////////////////////
  // Match scales -- just to help inference

  if (opt.do_rescale()) {
    rescale(Y);
    rescale(Vt);
    rescale(VtC);
    rescale(VtCd);
  }

    ////////////////////////////////////////////////////////////////
    // factored parameters
#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  omp_set_num_threads(opt.nthread());
#else
  // random seed initialization
  std::mt19937 rng(opt.rseed());
#endif

  Rcpp::List out_left_param = Rcpp::List::create();
  Rcpp::List out_right_param = Rcpp::List::create();
  Mat llik;

  if (opt.mf_right_nn()) {
    // use non-negative gamma
    auto theta_left = make_dense_spike_slab<Scalar>(Vt.cols(), K, opt);
    auto theta_right = make_dense_spike_gamma<Scalar>(Y.cols(), K, opt);

    auto eta_f = make_factored_regression_eta(Vt, Y, theta_left, theta_right);
    if (opt.weight_y()) eta_f.set_weight(weight);

    if (opt.mf_svd_init()) {
      eta_f.init_by_svd(Y, opt.jitter());
    } else {
      std::mt19937 _rng(opt.rseed());
      eta_f.jitter(opt.jitter(), _rng);
    }

    llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(eta_f, eta_c),
                              std::make_tuple(delta_c));

    out_left_param = param_rcpp_list(theta_left);
    out_right_param = param_rcpp_list(theta_right);
  } else {
    // use regular spike-slab on both sides
    auto theta_left = make_dense_spike_slab<Scalar>(Vt.cols(), K, opt);
    auto theta_right = make_dense_spike_slab<Scalar>(Y.cols(), K, opt);

    auto eta_f = make_factored_regression_eta(Vt, Y, theta_left, theta_right);
    if (opt.weight_y()) eta_f.set_weight(weight);

    if (opt.mf_svd_init()) {
      eta_f.init_by_svd(Y, opt.jitter());
    } else {
      std::mt19937 _rng(opt.rseed());
      eta_f.jitter(opt.jitter(), _rng);
    }

    llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(eta_f, eta_c),
                              std::make_tuple(delta_c));

    out_left_param = param_rcpp_list(theta_left);
    out_right_param = param_rcpp_list(theta_right);
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  TLOG("Successfully finished factored regression!");

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["U"] = U, Rcpp::_["Vt"] = Vt,
      Rcpp::_["VtCd"] = VtCd, Rcpp::_["VtC"] = VtC, Rcpp::_["D2"] = D2,
      Rcpp::_["S.inv"] = weight, Rcpp::_["param.left"] = out_left_param,
      Rcpp::_["param.right"] = out_right_param,
      Rcpp::_["conf"] = param_rcpp_list(theta_c),
      Rcpp::_["conf.delta"] = param_rcpp_list(theta_c_delta),
      Rcpp::_["llik"] = llik);
}

#endif
