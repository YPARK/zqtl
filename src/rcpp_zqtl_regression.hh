#ifndef RCPP_ZQTL_REGRESSION_HH_
#define RCPP_ZQTL_REGRESSION_HH_

Rcpp::List impl_fit_zqtl(const Mat& _effect, const Mat& _effect_se,
                         const Mat& X, const Mat& C, const options_t& opt) {
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

  Mat U, D2, Vt;
  std::tie(U, D2, Vt) = do_svd(X, opt);
  D2 = D2.cwiseProduct(D2);
  TLOG("Finished SVD of genotype matrix");

  const Scalar sample_size = static_cast<Scalar>(opt.sample_size());
  Mat effect_z, weight;

  std::tie(effect_z, weight) =
      preprocess_effect(_effect, _effect_se, sample_size);

  Mat Y = Vt * effect_z;
  TLOG("Finished eigen-decomposition");

  zqtl_model_t<Mat> model(Y, D2);

  TLOG("Constructed zqtl model");

  // confounder
  // eta_conf = Vt * inv(effect_sq) * C * theta_conf
  Mat VtC = Vt * C;
  auto theta_c = make_dense_spike_slab<Scalar>(VtC.cols(), Y.cols(), opt);
  auto eta_c = make_regression_eta(VtC, Y, theta_c);

  // mean effect size --> can be sparse matrix
  auto theta = make_dense_spike_slab<Scalar>(Vt.cols(), Y.cols(), opt);
  auto eta = make_regression_eta(Vt, Y, theta);
  if (opt.weight_y()) eta.set_weight(weight);

  // delta offset
  Mat Id = Mat::Ones(Y.rows(), 1) / static_cast<Scalar>(Vt.cols());
  auto theta_delta_rand =
      make_dense_spike_slab<Scalar>(Id.cols(), Y.cols(), opt);
  auto delta_rand = make_regression_eta(Id, Y, theta_delta_rand);

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
                                 std::make_tuple(delta_rand));

  auto resid = Rcpp::List::create();
  Mat llik_resid;

  // V' z ~ N(V' (w .* r) + ..., D^2)
  if (opt.out_resid()) {
    // delta_u = D t(U) epsilon
    auto epsilon_resid = make_dense_col_slab<Scalar>(U.rows(), Y.cols(), opt);

    Mat DUt = D2.cwiseSqrt().asDiagonal() * U.transpose();
    auto delta_resid = make_regression_eta(DUt, Y, epsilon_resid);

    dummy_eta_t dummy;
    eta.resolve();
    eta_c.resolve();
    delta_rand.resolve();

    llik_resid = impl_fit_eta_delta(
        model, opt, rng, std::make_tuple(dummy), std::make_tuple(delta_resid),
        std::make_tuple(eta, eta_c), std::make_tuple(delta_rand));

    resid = param_rcpp_list(epsilon_resid);
    TLOG("Calibrated residual effect");
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["Vt"] = Vt, Rcpp::_["D2"] = D2,
      Rcpp::_["S.inv"] = weight, Rcpp::_["param"] = param_rcpp_list(theta),
      Rcpp::_["delta"] = param_rcpp_list(theta_delta_rand),
      Rcpp::_["conf"] = param_rcpp_list(theta_c), Rcpp::_["resid"] = resid,
      Rcpp::_["llik"] = llik, Rcpp::_["llik.resid"] = llik_resid);
}

////////////////////////////////////////////////////////////////
// Factored QTL modeling
Rcpp::List impl_fit_fac_zqtl(const Mat& _effect, const Mat& _effect_se,
                             const Mat& X, const Mat& C, const options_t& opt) {
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

  Mat U, D2, Vt;
  std::tie(U, D2, Vt) = do_svd(X, opt);
  D2 = D2.cwiseProduct(D2);
  TLOG("Finished SVD of genotype matrix");

  TLOG("Finished eigen-decomposition");

  const Scalar sample_size = static_cast<Scalar>(opt.sample_size());
  Mat effect_z, weight;

  std::tie(effect_z, weight) =
      preprocess_effect(_effect, _effect_se, sample_size);

  Mat Y = Vt * effect_z;
  zqtl_model_t<Mat> model(Y, D2);

  // constrcut parameters
  const Index K = std::min(static_cast<Index>(opt.k()), Y.cols());

  // confounder
  Mat VtC = Vt * C;
  auto theta_c = make_dense_spike_slab<Scalar>(VtC.cols(), Y.cols(), opt);
  auto eta_c = make_regression_eta(VtC, Y, theta_c);

  // factored parameters
  auto theta_left = make_dense_spike_slab<Scalar>(Vt.cols(), K, opt);
  auto theta_right = make_dense_spike_slab<Scalar>(Y.cols(), K, opt);
  auto eta = make_factored_regression_eta(Vt, Y, theta_left, theta_right);
  if (opt.weight_y()) eta.set_weight(weight);

  if (opt.mf_svd_init()) {
    eta.init_by_svd(Y, opt.jitter());
  } else {
    std::mt19937 rng(opt.rseed());
    eta.jitter(opt.jitter(), rng);
  }

  // delta offset
  Mat Id = Mat::Ones(Y.rows(), 1) / static_cast<Scalar>(Vt.cols());
  auto theta_delta_rand =
      make_dense_spike_slab<Scalar>(Id.cols(), Y.cols(), opt);
  auto delta_rand = make_regression_eta(Id, Y, theta_delta_rand);

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  omp_set_num_threads(opt.nthread());
#else
  // random seed initialization
  std::mt19937 rng(opt.rseed());
#endif

  auto llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(eta, eta_c),
                                 std::make_tuple(delta_rand));

  // Residuals to capture discrepancy in the reference individuals
  auto resid = Rcpp::List::create();
  Mat llik_resid;

  // V' z ~ N(V' r + ..., D^2)
  if (opt.out_resid()) {
    // delta_u = D t(U) epsilon
    auto epsilon_resid = make_dense_col_slab<Scalar>(U.rows(), Y.cols(), opt);

    Mat DUt = D2.cwiseSqrt().asDiagonal() * U.transpose();
    auto delta_resid = make_regression_eta(DUt, Y, epsilon_resid);

    dummy_eta_t dummy;
    eta.resolve();
    eta_c.resolve();
    delta_rand.resolve();

    llik_resid = impl_fit_eta_delta(
        model, opt, rng, std::make_tuple(dummy), std::make_tuple(delta_resid),
        std::make_tuple(eta, eta_c), std::make_tuple(delta_rand));

    resid = param_rcpp_list(epsilon_resid);
    TLOG("Calibrated residual effect");
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["Vt"] = Vt, Rcpp::_["D2"] = D2,
      Rcpp::_["S.inv"] = weight,
      Rcpp::_["param.left"] = param_rcpp_list(theta_left),
      Rcpp::_["param.right"] = param_rcpp_list(theta_right),
      Rcpp::_["delta"] = param_rcpp_list(theta_delta_rand),
      Rcpp::_["conf"] = param_rcpp_list(theta_c), Rcpp::_["resid"] = resid,
      Rcpp::_["llik"] = llik, Rcpp::_["llik.resid"] = llik_resid);
}

#endif
