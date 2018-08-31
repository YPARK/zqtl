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

  Mat U, D, Vt;
  std::tie(U, D, Vt) = do_svd(X, opt);
  Mat D2 = D.cwiseProduct(D);
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

  // Intercept term
  Mat C = Mat::Ones(Vt.cols(), 1);
  Mat VtC = Vt * (C / static_cast<Scalar>(Vt.cols()));
  // random effect to remove non-genetic bias
  Mat DUt = D2.cwiseSqrt().asDiagonal() * U.transpose();

  zqtl_model_t<Mat> model(Y, D2);
  // dummy_eta_t dummy;

  // eta_conf = Vt * inv(effect_sq) * C * theta_conf
  auto theta_c = make_dense_spike_slab<Scalar>(VtC.cols(), Y.cols(), opt);
  auto eta_c = make_regression_eta(VtC, Y, theta_c);

  // Factorization
  const Index K = std::min(static_cast<Index>(opt.k()), Y.cols());

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  // omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  TLOG("Fit the factorization model");
  Mat llik;
  Rcpp::List indiv_out = Rcpp::List::create();
  Rcpp::List trait_out = Rcpp::List::create();

  if (opt.mf_right_nn()) {
    // use non-negative gamma
    auto epsilon_indiv = make_dense_col_spike_slab<Scalar>(DUt.cols(), K, opt);
    auto epsilon_trait = make_dense_spike_gamma<Scalar>(Y.cols(), K, opt);

    auto delta_random =
        make_factored_regression_eta(DUt, Y, epsilon_indiv, epsilon_trait);

    delta_random.init_by_svd(D.cwiseInverse().asDiagonal() * Y, opt.jitter());

    llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(eta_c),
                              std::make_tuple(delta_random));
    indiv_out = param_rcpp_list(epsilon_indiv);
    trait_out = param_rcpp_list(epsilon_trait);

  } else {
    auto epsilon_indiv = make_dense_col_slab<Scalar>(DUt.cols(), K, opt);
    auto epsilon_trait = make_dense_col_spike_slab<Scalar>(Y.cols(), K, opt);

    auto delta_random =
        make_factored_regression_eta(DUt, Y, epsilon_indiv, epsilon_trait);

    delta_random.init_by_svd(D.cwiseInverse().asDiagonal() * Y, opt.jitter());

    llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(eta_c),
                              std::make_tuple(delta_random));
    indiv_out = param_rcpp_list(epsilon_indiv);
    trait_out = param_rcpp_list(epsilon_trait);
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  TLOG("Successfully finished factorization!");

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["U"] = U, Rcpp::_["Vt"] = Vt,
      Rcpp::_["D2"] = D2, Rcpp::_["S.inv"] = weight,
      Rcpp::_["conf"] = param_rcpp_list(theta_c),
      Rcpp::_["param.indiv"] = indiv_out, Rcpp::_["param.trait"] = trait_out,
      Rcpp::_["llik"] = llik);
}

#endif
