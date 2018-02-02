#ifndef RCPP_ZQTL_FACTORIZATION_HH_
#define RCPP_ZQTL_FACTORIZATION_HH_

////////////////////////////////////////////////////////////////
// Just Run matrix factorization and output learned factors and
// residuals for the follow-up analysis
//

Rcpp::List impl_fit_factorization(const Mat& _effect, const Mat& _effect_se,
                                  const Mat& X, const options_t& opt) {
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
  zqtl_model_t<Mat> model(Y, D2);
  dummy_eta_t dummy;

  // Factorization
  const Index K = std::min(static_cast<Index>(opt.k()), Y.cols());

  // random effect to remove non-genetic bias
  Mat DUt = D2.cwiseSqrt().asDiagonal() * U.transpose();
  auto epsilon_indiv = make_dense_col_slab<Scalar>(DUt.cols(), K, opt);
  auto epsilon_trait = make_dense_col_spike_slab<Scalar>(Y.cols(), K, opt);

  auto delta_random =
      make_factored_regression_eta(DUt, Y, epsilon_indiv, epsilon_trait);

  delta_random.init_by_svd(Y, opt.jitter());

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  TLOG("Fit the factorization model");

  Mat llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(dummy),
                                std::make_tuple(delta_random));

  Rcpp::List resid = Rcpp::List::create();

  if (opt.out_resid()) {
    TLOG("Estimate the residuals");

    auto theta_resid = make_dense_slab<Scalar>(Y.rows(), Y.cols(), opt);
    auto delta_resid = make_residual_eta(Y, theta_resid);

    delta_random.resolve();
    Mat llik_resid = impl_fit_eta_delta(
        model, opt, rng, std::make_tuple(dummy), std::make_tuple(delta_resid),
        std::make_tuple(dummy), std::make_tuple(delta_random));

    delta_resid.resolve();
    Mat Zhat = Vt.transpose() * delta_resid.repr_mean();
    Mat effect_hat = Zhat.cwiseProduct(effect_sqrt);

    resid =
        Rcpp::List::create(Rcpp::_["llik"] = llik_resid,
                           Rcpp::_["param"] = param_rcpp_list(theta_resid),
                           Rcpp::_["Z.hat"] = Zhat, Rcpp::_["effect.hat"] = effect_hat);
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  TLOG("Successfully finished!");

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["U"] = U, Rcpp::_["Vt"] = Vt,
      Rcpp::_["D2"] = D2, Rcpp::_["S.inv"] = weight,
      Rcpp::_["param.indiv"] = param_rcpp_list(epsilon_indiv),
      Rcpp::_["param.trait"] = param_rcpp_list(epsilon_trait),
      Rcpp::_["llik"] = llik, Rcpp::_["resid"] = resid);
}

#endif
