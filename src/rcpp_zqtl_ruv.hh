#ifndef RCPP_ZQTL_RUV_HH_
#define RCPP_ZQTL_RUV_HH_

// _effect        = effect size
// _effect_se     = effect standard error
// X              = reference panel genotype matrix
//
// ctrl_effect    = effect size of the control LD block
// ctrl_effect_se = standard error of the control LD block
// X0             = reference panel for the control LD block
//
//
// 1. Fit confounder factrization
//
//        Z0 ~ 1/sqrt(n) X0' * C * right'
//        Y0 ~ D U0' * C * right'
//
// 2. Run factored multi-trait zqtl
//
//        Z ~ R * thetaL * theraR' + D U' * C * theta_covar
//        Y ~ D^2 * V' * thetaL * thetaR' +
//                  DU' * C * theta_covar
//
Rcpp::List impl_fit_ruv_twostep(const Mat& _effect,          //
                                const Mat& _effect_se,       //
                                const Mat& X,                //
                                const Mat& _ctrl_effect,     //
                                const Mat& _ctrl_effect_se,  //
                                const Mat& X0,               //
                                const options_t& opt) {
  // if (opt.with_ld_matrix()) {
  //   ELOG("Deprecated: longer use full LD matrix.");
  //   return Rcpp::List::create();
  // }

  // if (_effect.rows() != _effect_se.rows()) {
  //   ELOG("Check dimensions of effect and se");
  //   return Rcpp::List::create();
  // }

  // if (_ctrl_effect.rows() != _ctrl_effect_se.rows()) {
  //   ELOG("Check dimensions of effect and se on the control LD block");
  //   return Rcpp::List::create();
  // }

  // Mat U, D, Dsq, Vt;
  // std::tie(U, D, Vt) = do_svd(X, opt);
  // Dsq = D.cwiseProduct(D);
  // TLOG("Finished SVD of genotype matrix : X");

  // Mat U0, D0, D0sq, V0t;
  // std::tie(U0, D0, V0t) = do_svd(X0, opt);
  // D0sq = D0.cwiseProduct(D0);
  // TLOG("Finished SVD of genotype matrix : X0");

  // const Scalar sample_size = static_cast<Scalar>(opt.sample_size());
  // Mat Z, S, W;
  // std::tie(Z, S, W) = preprocess_effect(_effect, _effect_se, sample_size);

  // Mat Z0, S0, W0;
  // std::tie(Z0, S0, W0) =
  //     preprocess_effect(_ctrl_effect, _ctrl_effect_se, sample_size);

  // dummy_eta_t dummy;

  // // step 1. factorization
  // const Index K = std::min(static_cast<Index>(opt.k()), Z0.cols());
  //   Mat Y0 = V0t * Z0;
  //   Mat DU0t = D0.asDiagonal() * U0.transpose();
  // auto theta_covar = make_dense_col_slab<Scalar>(DU0t.cols(), K, opt);
  // auto theta_trait = make_dense_col_spike_slab<Scalar>(Z0.cols(), K, opt);
  // Rcpp::List step1 = Rcpp::List::create();
  // {
  //   auto delta_mf =
  //       make_factored_regression_eta(DU0t, Y0, theta_covar, theta_trait);

  //   zqtl_model_t<Mat> model_step1(Y0, D0sq);
  //   Mat llik_mf =
  //       impl_fit_eta_delta(model_step1, opt, rng, std::make_tuple(dummy),
  //                          std::make_tuple(delta_mf));

  //   step1 = Rcpp::List::create(
  //       Rcpp::_["Y"] = Y0, Rcpp::_["DUt"] = DU0t,
  //       Rcpp::_["theta.covar"] = param_rcpp_list(theta_covar),
  //       Rcpp::_["theta.trait"] = param_rcpp_list(theta_trait),
  //       Rcpp::_["llik"] = llik_mf);
  // }

  // // step2. model estimation and report 
  // Mat Y = Vt * Z;
  // Mat DUt = D.cwiseSqrt().asDiagonal() * U.transpose();

  // Mat W = DUt * mean_param(theta_covar); // L x 

}

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
    zqtl_model_t<Mat> model_step1(Y, D2);

    auto theta = make_dense_slab<Scalar>(Vt.cols(), Y.cols(), opt);
    auto eta = make_regression_eta(Vt, Y, theta);
    if (opt.weight_y()) eta.set_weight(weight);

    // Y ~ D2 * (Vt * theta) = D2 * eta
    Mat llik = impl_fit_eta(model_step1, opt, rng, std::make_tuple(eta));

    TLOG("Estimate the residuals");
    eta.resolve();
    Mat Yhat = eta.repr_mean();
    R = Y - D2.asDiagonal() * Yhat;

    TLOG("Y : " << Y.minCoeff() << " ~ " << Y.maxCoeff());
    TLOG("Yhat : " << Yhat.minCoeff() << " ~ " << Yhat.maxCoeff());
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

    delta_random.init_by_svd(R, opt.jitter());

    zqtl_model_t<Mat> model_step2(R, D2);
    Mat llik = impl_fit_eta_delta(model_step2, opt, rng, std::make_tuple(dummy),
                                  std::make_tuple(delta_random));

    W = DUt * mean_param(epsilon_indiv);
    Zconf = Vt.transpose() * mean_param(epsilon_indiv);

    TLOG("W : " << W.minCoeff() << " ~ " << W.maxCoeff());
    TLOG("Zconf : " << Zconf.minCoeff() << " ~ " << Zconf.maxCoeff());

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

    zqtl_model_t<Mat> model_step3(Y, D2);
    Mat llik = impl_fit_eta_delta(model_step3, opt, rng, std::make_tuple(dummy),
                                  std::make_tuple(delta_conf));

    TLOG("Estimate the clean effect sizes");

    delta_conf.resolve();
    Zhat = Vt.transpose() * (Y - delta_conf.repr_mean());
    effect_hat = Zhat.cwiseProduct(effect_sqrt);

    step3 =
        Rcpp::List::create(Rcpp::_["llik"] = llik,
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
