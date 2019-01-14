#ifndef RCPP_ZQTL_ADJUST_HH_
#define RCPP_ZQTL_ADJUST_HH_

//////////////////////////////////////////////////////
// simply adjust Cdelta without multivariate effect //
//////////////////////////////////////////////////////

Rcpp::List impl_adjust_zqtl(const Mat &_effect, const Mat &_effect_se,
			    const Mat &X, const Mat &C, const Mat &Cdelta,
			    const options_t &opt) {
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
  eta_c.init_by_dot(Y, opt.jitter());

  // delta_conf = Vt * Cdelta * theta_conf
  auto theta_c_delta =
      make_dense_spike_slab<Scalar>(VtCd.cols(), Y.cols(), opt);
  auto delta_c = make_regression_eta(VtCd, Y, theta_c_delta);
  delta_c.init_by_dot(Y, opt.jitter());

  TLOG("Constructed effects");

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  // omp_set_num_threads(opt.nthread());
#else
  // random seed initialization
  std::mt19937 rng(opt.rseed());
#endif


  ///////////////////////////
  // report clean z-scores //
  ///////////////////////////

  // Just remove contributions from eta_c and delta_c
  // z_c = V * D^2 * (Vt * theta_c)
  // z_d = V * (Vt * d * theta_d)

  Mat z_clean = effect_z;
  Rcpp::List clean(opt.nboot());

  auto remove_confounders = [&]() {
    for (Index b = 0; b < opt.nboot(); ++b) {
      eta_c.resolve();
      delta_c.resolve();

      z_clean = effect_z;
      z_clean -= Vt.transpose() * D2.asDiagonal() * eta_c.sample(rng);
      z_clean -= Vt.transpose() * delta_c.sample(rng);

      clean[b] = z_clean.cwiseProduct(effect_sqrt);
    }
  };

  //////////////////////////
  // Fit regression model //
  //////////////////////////

  auto llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(eta_c),
                                 std::make_tuple(delta_c));

  if (opt.nboot() > 0) {
    remove_confounders();
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  TLOG("Successfully finished regression!");

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["U"] = U, Rcpp::_["Vt"] = Vt,
      Rcpp::_["D2"] = D2, Rcpp::_["S.inv"] = weight,
      Rcpp::_["conf.multi"] = param_rcpp_list(theta_c),
      Rcpp::_["conf.uni"] = param_rcpp_list(theta_c_delta),
      Rcpp::_["llik"] = llik,
      Rcpp::_["gwas.clean"] = clean);
}


#endif
