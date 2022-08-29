#ifndef RCPP_ZQTL_REGRESSION_HH_
#define RCPP_ZQTL_REGRESSION_HH_

Rcpp::List impl_fit_zqtl(const Mat &_effect, const Mat &_effect_se,
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

  // mean effect size --> can be sparse matrix
  auto theta = make_dense_spike_slab<Scalar>(Vt.cols(), Y.cols(), opt);
  auto eta = make_regression_eta(Vt, Y, theta);
  eta.init_by_dot(Y, opt.jitter());

  TLOG("Constructed effects");

  // random seed initialization
  dqrng::xoshiro256plus rng(opt.rseed());

  Mat xi(Vt.rows(), Y.cols());
  Mat Dinv = D.cwiseInverse();

  ////////////////////////////////
  // Note : eta ~ Vt * theta    //
  // z = V * D^2 * (Vt * theta) //
  // xi = D^-1 * Vt * (z * se)  //
  // var = sum(xi * xi)         //
  ////////////////////////////////

  log10_trunc_op_t<Scalar> log10_op(1e-10);

  auto take_eta_var = [&](auto &_eta) {
    const Index K = Vt.rows();
    const Index m = effect_sqrt.cols();
    const Index p = Vt.cols();
    const Scalar denom = static_cast<Scalar>(K);

    _eta.resolve();

    Mat temp(1, m);
    Mat temp_Km(K, m);
    running_stat_t<Mat> _stat(1, m);
    // running_stat_t<Mat> _stat_null(1, m);
    Mat onesK = Mat::Ones(1, K);
    Mat z(p, m);  // projected GWAS
    Mat obs;

    temp_Km = _eta.repr_mean();
    z = Vt.transpose() * D2.asDiagonal() * temp_Km;
    if (opt.scale_var_calc()) {
      z = z.cwiseProduct(effect_sqrt);
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesK * (xi.cwiseProduct(xi));
    } else {
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesK * (xi.cwiseProduct(xi)) / denom;
    }
    obs = obs.unaryExpr(log10_op);

    for (Index b = 0; b < opt.nboot_var(); ++b) {
      temp_Km = _eta.sample(rng);
      z = Vt.transpose() * D2.asDiagonal() * temp_Km;
      if (opt.scale_var_calc()) {
        z = z.cwiseProduct(effect_sqrt);
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesK * (xi.cwiseProduct(xi));
      } else {
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesK * (xi.cwiseProduct(xi)) / denom;
      }
      _stat(temp.unaryExpr(log10_op));
    }

    return Rcpp::List::create(Rcpp::_["mean"] = _stat.mean(),
                              Rcpp::_["var"] = _stat.var(),
                              Rcpp::_["obs"] = obs);
  };

  ////////////////////////////////
  // delta ~ D^2 * Vt * theta   //
  // z = V * (D^2 * Vt * theta) //
  // xi = D^-1 * Vt * (z .* se) //
  // var = sum(xi * xi)         //
  ////////////////////////////////

  auto take_delta_var = [&](auto &_delta) {
    const Index K = Vt.rows();
    const Index m = effect_sqrt.cols();
    const Index p = Vt.cols();
    const Scalar denom = static_cast<Scalar>(K);

    _delta.resolve();

    Mat temp(1, m);
    Mat temp_Km(K, m);
    running_stat_t<Mat> _stat(1, m);
    // running_stat_t<Mat> _stat_null(1, m);
    Mat onesK = Mat::Ones(1, K);
    Mat z(p, m);  // projected GWAS

    temp_Km = _delta.repr_mean();
    z = Vt.transpose() * temp_Km;
    Mat obs;
    if (opt.scale_var_calc()) {
      z = z.cwiseProduct(effect_sqrt);
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesK * (xi.cwiseProduct(xi));
    } else {
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesK * (xi.cwiseProduct(xi)) / denom;
    }
    obs = obs.unaryExpr(log10_op);

    for (Index b = 0; b < opt.nboot_var(); ++b) {
      temp_Km = _delta.sample(rng);
      z = Vt.transpose() * temp_Km;
      if (opt.scale_var_calc()) {
        z = z.cwiseProduct(effect_sqrt);
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesK * (xi.cwiseProduct(xi));
      } else {
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesK * (xi.cwiseProduct(xi)) / denom;
      }
      _stat(temp.unaryExpr(log10_op));
    }

    return Rcpp::List::create(Rcpp::_["mean"] = _stat.mean(),
                              Rcpp::_["var"] = _stat.var(),
                              Rcpp::_["obs"] = obs);
  };

  /////////////////////////////////
  // calculate residual z-scores //
  /////////////////////////////////

  Rcpp::List resid = Rcpp::List::create();

  auto theta_resid = make_dense_slab<Scalar>(Y.rows(), Y.cols(), opt);
  auto delta_resid = make_residual_eta(Y, theta_resid);
  delta_resid.init_by_y(Y, opt.jitter());

  auto take_residual = [&]() {
    TLOG("Estimate the residuals");

    eta.resolve();
    eta_c.resolve();
    delta_c.resolve();

    dummy_eta_t dummy;

    // Take residual variance
    Mat llik_resid = impl_fit_eta_delta(
        model, opt, rng, std::make_tuple(dummy), std::make_tuple(delta_resid),
        std::make_tuple(eta, eta_c), std::make_tuple(delta_c));

    resid = Rcpp::List::create(Rcpp::_["llik"] = llik_resid,
                               Rcpp::_["param"] = param_rcpp_list(theta_resid));
  };

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

  auto llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(eta, eta_c),
                                 std::make_tuple(delta_c));

  if (opt.out_resid()) {
    take_residual();
  }

  auto take_tot_var = [&]() {
    const Index K = Vt.rows();
    const Scalar sK = static_cast<Scalar>(Vt.rows());
    Mat temp = Dinv.asDiagonal() * Y;
    Mat ret =
        (Mat::Ones(1, K) * temp.cwiseProduct(temp) / sK).unaryExpr(log10_op);
    return Rcpp::wrap(ret);
  };

  Rcpp::List var_decomp = Rcpp::List::create();

  if (opt.do_var_calc()) {
    if (!opt.out_resid()) {  // we need residuals
      TLOG("We need to calibrate the residuals");
      take_residual();
    }

    auto _var_mult = take_eta_var(eta);
    auto _var_conf_mult = take_eta_var(eta_c);
    auto _var_conf_uni = take_delta_var(delta_c);
    auto _var_resid = take_delta_var(delta_resid);
    auto _var_tot = take_tot_var();

    var_decomp = Rcpp::List::create(
        Rcpp::_["param"] = _var_mult,            // multivariate
        Rcpp::_["conf.multi"] = _var_conf_mult,  // confounder multi
        Rcpp::_["conf.uni"] = _var_conf_uni,     // confounder uni
        Rcpp::_["resid"] = _var_resid,           // resid
        Rcpp::_["tot"] = _var_tot);
  }

  if (opt.nboot() > 0) {
    remove_confounders();
  }

  TLOG("Successfully finished regression!");

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["U"] = U, Rcpp::_["Vt"] = Vt,
      Rcpp::_["D2"] = D2, Rcpp::_["S.inv"] = weight,
      Rcpp::_["param"] = param_rcpp_list(theta),
      Rcpp::_["conf.multi"] = param_rcpp_list(theta_c),
      Rcpp::_["conf.uni"] = param_rcpp_list(theta_c_delta),
      Rcpp::_["resid"] = resid, Rcpp::_["llik"] = llik,
      Rcpp::_["gwas.clean"] = clean, Rcpp::_["var"] = var_decomp);
}

////////////////////////////////////////////////////////////////
// Factored QTL modeling
Rcpp::List impl_fit_fac_zqtl(const Mat &_effect, const Mat &_effect_se,
                             const Mat &X, const Mat &C, const Mat &Cdelta,
                             const options_t &opt) {
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
  eta_c.init_by_dot(Y, opt.jitter());

  ////////////////////////////////////////////////////////////////
  // This is useful to correct for phenotype correlations
  // delta_conf = Vt * Cdelta * theta_conf
  Mat VtCd = Vt * Cdelta;
  auto theta_c_delta =
      make_dense_spike_slab<Scalar>(VtCd.cols(), Y.cols(), opt);
  auto delta_c = make_regression_eta(VtCd, Y, theta_c_delta);
  delta_c.init_by_dot(Y, opt.jitter());

  ////////////////////////////////////////////////////////////////
  // factored parameters
  dqrng::xoshiro256plus rng(opt.rseed());

  Mat xi(Vt.rows(), Y.cols());
  Mat Dinv = D.cwiseInverse();

  ////////////////////////////////
  // Note : eta ~ Vt * theta    //
  // z = V * D^2 * (Vt * theta) //
  // xi = D^-1 * Vt * (z * se)  //
  // var = sum(xi * xi)         //
  ////////////////////////////////

  log10_trunc_op_t<Scalar> log10_op(1e-10);

  auto take_eta_var = [&](auto &_eta) {
    const Index K = Vt.rows();
    const Index m = effect_sqrt.cols();
    const Index p = Vt.cols();
    const Scalar denom = static_cast<Scalar>(K);

    _eta.resolve();

    Mat temp(1, m);
    Mat temp_Km(K, m);
    running_stat_t<Mat> _stat(1, m);
    // running_stat_t<Mat> _stat_null(1, m);
    Mat onesK = Mat::Ones(1, K);
    Mat z(p, m);  // projected GWAS
    Mat obs;

    temp_Km = _eta.repr_mean();
    z = Vt.transpose() * D2.asDiagonal() * temp_Km;
    if (opt.scale_var_calc()) {
      z = z.cwiseProduct(effect_sqrt);
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesK * (xi.cwiseProduct(xi));
    } else {
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesK * (xi.cwiseProduct(xi)) / denom;
    }
    obs = obs.unaryExpr(log10_op);

    for (Index b = 0; b < opt.nboot_var(); ++b) {
      temp_Km = _eta.sample(rng);
      z = Vt.transpose() * D2.asDiagonal() * temp_Km;
      if (opt.scale_var_calc()) {
        z = z.cwiseProduct(effect_sqrt);
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesK * (xi.cwiseProduct(xi));
      } else {
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesK * (xi.cwiseProduct(xi)) / denom;
      }
      _stat(temp.unaryExpr(log10_op));
    }

    return Rcpp::List::create(Rcpp::_["mean"] = _stat.mean(),
                              Rcpp::_["var"] = _stat.var(),
                              Rcpp::_["obs"] = obs);
  };

  ////////////////////////////////
  // delta ~ D^2 * Vt * theta   //
  // z = V * (D^2 * Vt * theta) //
  // xi = D^-1 * Vt * (z .* se) //
  // var = sum(xi * xi)         //
  ////////////////////////////////

  auto take_delta_var = [&](auto &_delta) {
    const Index K = Vt.rows();
    const Index m = effect_sqrt.cols();
    const Index p = Vt.cols();
    const Scalar denom = static_cast<Scalar>(K);

    _delta.resolve();

    Mat temp(1, m);
    Mat temp_Km(K, m);
    running_stat_t<Mat> _stat(1, m);
    // running_stat_t<Mat> _stat_null(1, m);
    Mat onesK = Mat::Ones(1, K);
    Mat z(p, m);  // projected GWAS

    temp_Km = _delta.repr_mean();
    z = Vt.transpose() * temp_Km;
    Mat obs;
    if (opt.scale_var_calc()) {
      z = z.cwiseProduct(effect_sqrt);
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesK * (xi.cwiseProduct(xi));
    } else {
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesK * (xi.cwiseProduct(xi)) / denom;
    }
    obs = obs.unaryExpr(log10_op);

    for (Index b = 0; b < opt.nboot_var(); ++b) {
      temp_Km = _delta.sample(rng);
      z = Vt.transpose() * temp_Km;
      if (opt.scale_var_calc()) {
        z = z.cwiseProduct(effect_sqrt);
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesK * (xi.cwiseProduct(xi));
      } else {
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesK * (xi.cwiseProduct(xi)) / denom;
      }
      _stat(temp.unaryExpr(log10_op));
    }

    return Rcpp::List::create(Rcpp::_["mean"] = _stat.mean(),
                              Rcpp::_["var"] = _stat.var(),
                              Rcpp::_["obs"] = obs);
  };

  /////////////////////////////////
  // calculate residual z-scores //
  /////////////////////////////////

  Rcpp::List resid = Rcpp::List::create();

  auto theta_resid = make_dense_slab<Scalar>(Y.rows(), Y.cols(), opt);
  auto delta_resid = make_residual_eta(Y, theta_resid);
  auto theta_tot = make_dense_slab<Scalar>(Y.rows(), Y.cols(), opt);

  auto take_residual = [&](auto &_eta_f) {
    TLOG("Estimate the residuals");

    _eta_f.resolve();
    eta_c.resolve();
    delta_c.resolve();

    dummy_eta_t dummy;

    // Take residual variance
    Mat llik_resid = impl_fit_eta_delta(
        model, opt, rng, std::make_tuple(dummy), std::make_tuple(delta_resid),
        std::make_tuple(_eta_f, eta_c), std::make_tuple(delta_c));

    resid = Rcpp::List::create(Rcpp::_["llik"] = llik_resid,
                               Rcpp::_["param"] = param_rcpp_list(theta_resid));
  };

  ///////////////////////////
  // report clean z-scores //
  ///////////////////////////

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

  ///////////////////
  // model fitting //
  ///////////////////

  Rcpp::List out_left_param = Rcpp::List::create();
  Rcpp::List out_right_param = Rcpp::List::create();
  Mat llik;
  Rcpp::List var_factored = Rcpp::List::create();

  if (opt.mf_right_nn()) {
    // use non-negative gamma
    auto theta_left = make_dense_spike_slab<Scalar>(Vt.cols(), K, opt);
    auto theta_right = make_dense_spike_gamma<Scalar>(Y.cols(), K, opt);

    auto eta_f = make_factored_regression_eta(Vt, Y, theta_left, theta_right);

    if (opt.mf_svd_init()) {
      eta_f.init_by_svd(Y, opt.jitter());
    } else {
      dqrng::xoshiro256plus _rng(opt.rseed());
      eta_f.jitter(opt.jitter(), _rng);
    }

    llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(eta_f, eta_c),
                              std::make_tuple(delta_c));

    out_left_param = param_rcpp_list(theta_left);
    out_right_param = param_rcpp_list(theta_right);

    if (opt.out_resid()) take_residual(eta_f);

    if (opt.do_var_calc()) {
      if (!opt.out_resid()) take_residual(eta_f);
      var_factored = take_eta_var(eta_f);
    }

  } else {
    // use regular spike-slab on both sides
    auto theta_left = make_dense_spike_slab<Scalar>(Vt.cols(), K, opt);
    auto theta_right = make_dense_spike_slab<Scalar>(Y.cols(), K, opt);

    auto eta_f = make_factored_regression_eta(Vt, Y, theta_left, theta_right);

    if (opt.mf_svd_init()) {
      eta_f.init_by_svd(Y, opt.jitter());
    } else {
      dqrng::xoshiro256plus _rng(opt.rseed());
      eta_f.jitter(opt.jitter(), _rng);
    }

    llik = impl_fit_eta_delta(model, opt, rng, std::make_tuple(eta_f, eta_c),
                              std::make_tuple(delta_c));

    out_left_param = param_rcpp_list(theta_left);
    out_right_param = param_rcpp_list(theta_right);

    if (opt.out_resid()) take_residual(eta_f);

    if (opt.do_var_calc()) {
      if (!opt.out_resid()) take_residual(eta_f);
      var_factored = take_eta_var(eta_f);
    }
  }

  auto take_tot_var = [&]() {
    const Index K = Vt.rows();
    const Scalar sK = static_cast<Scalar>(Vt.rows());
    Mat temp = Dinv.asDiagonal() * Y;
    Mat ret =
        (Mat::Ones(1, K) * temp.cwiseProduct(temp) / sK).unaryExpr(log10_op);
    return Rcpp::wrap(ret);
  };

  Rcpp::List var_decomp = Rcpp::List::create();

  if (opt.do_var_calc()) {
    auto _var_conf_mult = take_eta_var(eta_c);
    auto _var_conf_uni = take_delta_var(delta_c);
    auto _var_resid = take_delta_var(delta_resid);
    auto _var_tot = take_tot_var();

    var_decomp = Rcpp::List::create(
        Rcpp::_["factored"] = var_factored,
        Rcpp::_["conf.multi"] = _var_conf_mult,  // confounder multi
        Rcpp::_["conf.uni"] = _var_conf_uni,     // confounder uni
        Rcpp::_["resid"] = _var_resid,           // resid
        Rcpp::_["tot"] = _var_tot);
  }

  if (opt.nboot() > 0) {
    remove_confounders();
  }

  TLOG("Successfully finished factored regression!");

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["U"] = U, Rcpp::_["Vt"] = Vt,
      Rcpp::_["VtCd"] = VtCd, Rcpp::_["VtC"] = VtC, Rcpp::_["D2"] = D2,
      Rcpp::_["S.inv"] = weight, Rcpp::_["param.left"] = out_left_param,
      Rcpp::_["param.right"] = out_right_param,
      Rcpp::_["conf.multi"] = param_rcpp_list(theta_c),
      Rcpp::_["conf.uni"] = param_rcpp_list(theta_c_delta),
      Rcpp::_["llik"] = llik, Rcpp::_["gwas.clean"] = clean,
      Rcpp::_["var"] = var_decomp);
}

#endif
