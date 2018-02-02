#ifndef RCPP_ZQTL_RUV_HH_
#define RCPP_ZQTL_RUV_HH_

Rcpp::List impl_fit_ruv(const Mat& _effect, const Mat& _effect_se, const Mat& X,
                        const options_t& opt) {
  if (opt.with_ld_matrix()) {
    ELOG("Deprecated: longer use full LD matrix.");
    return Rcpp::List::create();
  }

  if (_effect.rows() != _effect_se.rows()) {
    ELOG("Check dimensions of effect and se");
    return Rcpp::List::create();
  }

  Mat U, D2, Vt;
  std::tie(U, D2, Vt) = do_svd(X, opt);
  D2 = D2.cwiseProduct(D2);
  TLOG("Finished SVD of genotype matrix");

  const Scalar sample_size = static_cast<Scalar>(opt.sample_size());
  Mat effect_z, effect_sqrt, weight;

  std::tie(effect_z, effect_sqrt, weight) =
      preprocess_effect(_effect, _effect_se, sample_size);

  Mat Y = Vt * effect_z;
  Mat DUt = D2.cwiseSqrt().asDiagonal() * U.transpose();
  zqtl_model_t<Mat> model(Y, D2);
  dummy_eta_t dummy;

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  omp_set_num_threads(opt.nthread());
#else
  // random seed initialization
  std::mt19937 rng(opt.rseed());
#endif

  ////////////////////////////////////////////
  // 1. fit zQTL and estimate the residuals //
  ////////////////////////////////////////////
  Mat R = Y;
  Rcpp::List step1 = Rcpp::List::create();
  {
    TLOG("Perform RUV-r [1]: first take residuals");
    zqtl_model_t<Mat> model(Y, D2);

    auto theta = make_dense_spike_slab<Scalar>(Vt.cols(), Y.cols(), opt);
    auto eta = make_regression_eta(Vt, Y, theta);
    if (opt.weight_y()) eta.set_weight(weight);

    // Y ~ D2 * (Vt * theta) = D2 * eta
    Mat llik = impl_fit_eta(model, opt, rng, std::make_tuple(eta));

    TLOG("Estimate the residuals");
    eta.resolve();
    Mat Yhat = eta.repr_mean();
    R = Y - D2.asDiagonal() * Yhat;

    TLOG("Residual : " << R.minCoeff() << " ~ " << R.maxCoeff());

    step1 = Rcpp::List::create(Rcpp::_["R"] = R,
                               Rcpp::_["theta"] = param_rcpp_list(theta),
                               Rcpp::_["llik"] = llik);
  }

  ////////////////////////////////////////
  // 2. factorization on the residuals  //
  ////////////////////////////////////////
  const Index K = std::min(static_cast<Index>(opt.k()), R.cols());
  Mat W = Mat::Zero(Y.rows(), K);
  Mat Zconf = Mat::Zero(_effect.rows(), K);
  Mat effect_hat = _effect;
  Mat Zhat = _effect.cwiseQuotient(_effect_se);

  Rcpp::List step2 = Rcpp::List::create();
  {
    TLOG("Perform RUV-r [2]: factorization of the residuals");

    // random effect to remove non-genetic bias
    auto epsilon_indiv = make_dense_col_slab<Scalar>(DUt.cols(), K, opt);
    auto epsilon_trait = make_dense_col_spike_slab<Scalar>(R.cols(), K, opt);

    auto delta_random =
        make_factored_regression_eta(DUt, R, epsilon_indiv, epsilon_trait);

    Mat Rd = D2.cwiseSqrt().cwiseInverse().asDiagonal() * R;
    delta_random.init_by_svd(R, opt.jitter());

    zqtl_model_t<Mat> model(R, D2);
    Mat llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(dummy),
                                  std::make_tuple(delta_random));

    const Scalar n = DUt.cols();
    W = DUt * mean_param(epsilon_indiv);
    Zconf = Vt.transpose() * mean_param(epsilon_indiv);

    TLOG("W : " << W.minCoeff() << " ~ " << W.maxCoeff());

    step2 = Rcpp::List::create(
        Rcpp::_["param.indiv"] = param_rcpp_list(epsilon_indiv),
        Rcpp::_["param.trait"] = param_rcpp_list(epsilon_trait),
        Rcpp::_["llik"] = llik);
  }

  ////////////////////////////////////////////////////////
  // 3. fit the full model with the inferred covariates //
  ////////////////////////////////////////////////////////
  Rcpp::List step3 = Rcpp::List::create();
  if (opt.out_resid()) {
    TLOG("Perform RUV-r [3]: regression on the confounders");
    auto theta_conf = make_dense_spike_slab<Scalar>(W.cols(), Y.cols(), opt);
    auto delta_conf = make_regression_eta(W, Y, theta_conf);

    zqtl_model_t<Mat> model(Y, D2);
    Mat llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(dummy),
                                  std::make_tuple(delta_conf));

    TLOG("Estimate the clean effect sizes");

    auto theta_resid = make_dense_slab<Scalar>(Y.rows(), Y.cols(), opt);
    auto delta_resid = make_residual_eta(Y, theta_resid);

    delta_conf.resolve();
    Mat llik_resid = impl_fit_eta_delta(
        model, opt, rng, std::make_tuple(dummy), std::make_tuple(delta_resid),
        std::make_tuple(dummy), std::make_tuple(delta_conf));

    delta_resid.resolve();
    Zhat = Vt.transpose() * delta_resid.repr_mean();
    effect_hat = Zhat.cwiseProduct(effect_sqrt);

    step3 = Rcpp::List::create(
        Rcpp::_["llik"] = llik, Rcpp::_["llik.resid"] = llik_resid,
        Rcpp::_["multivariate"] = param_rcpp_list(theta_resid),
        Rcpp::_["theta.conf"] = param_rcpp_list(theta_conf));
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif
  return Rcpp::List::create(Rcpp::_["effect.corrected"] = effect_hat,
                            Rcpp::_["Z.hat.corrected"] = Zhat,
                            Rcpp::_["Z.confounder"] = Zconf,
                            Rcpp::_["step1"] = step1, Rcpp::_["step2"] = step2,
                            Rcpp::_["step3"] = step3);
}

#endif
