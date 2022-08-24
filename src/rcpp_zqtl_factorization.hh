#ifndef RCPP_ZQTL_FACTORIZATION_HH_
#define RCPP_ZQTL_FACTORIZATION_HH_

////////////////////////////////////////////////////////////////
template <typename RNG, typename Left, typename Right, typename... DATA>
auto _impl_factorization(Left& epsilon_left, Right& epsilon_right,
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

  Rcpp::List left_out = param_rcpp_list(epsilon_left);
  Rcpp::List right_out = param_rcpp_list(epsilon_right);
  return std::make_tuple(llik, left_out, right_out);
}

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

  dqrng::xoshiro256plus rng(opt.rseed());

  Mat Y = Vt * effect_z;

  TLOG("Fit the factorization model");
  Mat llik;
  Rcpp::List left_out = Rcpp::List::create();
  Rcpp::List trait_out = Rcpp::List::create();

  // Intercept term
  Mat C = Mat::Ones(Vt.cols(), 1);
  Mat VtC = Vt * (C / static_cast<Scalar>(Vt.cols()));

  if (opt.de_factorization_model() == 1) {
    Mat Design = D2.cwiseSqrt().asDiagonal();
    const Index K = std::min(static_cast<Index>(opt.k()),
                             std::min(Y.cols(), Design.cols()));

    if (opt.mf_right_nn()) {
      // use non-negative gamma
      auto epsilon_svd = make_dense_spike_slab<Scalar>(Design.cols(), K, opt);
      auto epsilon_trait = make_dense_spike_gamma<Scalar>(Y.cols(), K, opt);

      std::tie(llik, left_out, trait_out) =
          _impl_factorization(epsilon_svd, epsilon_trait, opt, rng,
                              std::make_tuple(Y, D2, Design, VtC));

    } else {
      auto epsilon_svd = make_dense_spike_slab<Scalar>(Design.cols(), K, opt);
      auto epsilon_trait = make_dense_col_spike_slab<Scalar>(Y.cols(), K, opt);

      std::tie(llik, left_out, trait_out) =
          _impl_factorization(epsilon_svd, epsilon_trait, opt, rng,
                              std::make_tuple(Y, D2, Design, VtC));
    }

  } else {
    // random effect to remove non-genetic bias
    Mat Design = D2.cwiseSqrt().asDiagonal() * U.transpose();
    const Index K = std::min(static_cast<Index>(opt.k()), Y.cols());

    if (opt.mf_right_nn()) {
      // use non-negative gamma
      auto epsilon_indiv =
          make_dense_col_spike_slab<Scalar>(Design.cols(), K, opt);
      auto epsilon_trait = make_dense_spike_gamma<Scalar>(Y.cols(), K, opt);

      std::tie(llik, left_out, trait_out) =
          _impl_factorization(epsilon_indiv, epsilon_trait, opt, rng,
                              std::make_tuple(Y, D2, Design, VtC));

    } else {
      auto epsilon_indiv = make_dense_col_slab<Scalar>(Design.cols(), K, opt);
      auto epsilon_trait = make_dense_col_spike_slab<Scalar>(Y.cols(), K, opt);

      std::tie(llik, left_out, trait_out) =
          _impl_factorization(epsilon_indiv, epsilon_trait, opt, rng,
                              std::make_tuple(Y, D2, Design, VtC));
    }
  }

  TLOG("Successfully finished factorization!");

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["U"] = U, Rcpp::_["Vt"] = Vt,
      Rcpp::_["D2"] = D2, Rcpp::_["S.inv"] = weight,
      Rcpp::_["param.left"] = left_out, Rcpp::_["param.right"] = trait_out,
      Rcpp::_["llik"] = llik);
}

#endif
