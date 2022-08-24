#ifndef RCPP_ZQTL_ANNOT_HH_
#define RCPP_ZQTL_ANNOT_HH_

Rcpp::List impl_zqtl_annot(const Mat &_effect,     // p x m
                           const Mat &_effect_se,  // p x m
                           const Mat &X,           // n x p
                           const Mat &A,           // p x k
                           const Mat &C,           // p x r
                           const Mat &Cdelta,      // p x s
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

  if (_effect.rows() != A.rows()) {
    ELOG("Check dimensions of A");
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
  Mat VtC = Vt * C;        // SNP-level annotations
  Mat VtCd = Vt * Cdelta;  // z-score annotations

  zqtl_model_t<Mat> model(Y, D2);

  TLOG("Constructed zqtl model");

  ////////////////////////////////////////////////////////////////
  // create annotated regression effect
  const Index K = A.cols();

  ////////////////////////////////////////////////////////////////
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

  auto take_eta_var_each = [&](auto &_x, auto &_annot, auto &_theta_left,
                               auto &_theta_right) {
    const Index SV = Vt.rows();
    const Index m = effect_sqrt.cols();
    const Index p = Vt.cols();
    const Scalar denom = static_cast<Scalar>(SV);

    const Index K = _theta_left.cols();

    Mat eta_k(SV, m);     // stochastic realization
    Mat theta_s(p, K);    // stochastic realization
    Mat loading_s(m, K);  // stochastic realization
    Mat z(p, m);          // projected GWAS
    Mat temp(K, m);       // temporary stat
    Mat onesSV = Mat::Ones(1, SV);
    running_stat_t<Mat> _stat(K, m);

    for (Index b = 0; b < opt.nboot_var(); ++b) {
      // 1. sample theta
      loading_s = var_param(_theta_right);
      loading_s = loading_s.unaryExpr([&](const auto &v) {
        return std::sqrt(v) * static_cast<Scalar>(R::rnorm(0.0, 1.0));
      });
      loading_s += mean_param(_theta_right);

      theta_s = var_param(_theta_left);
      theta_s = theta_s.unaryExpr([&](const auto &v) {
        return std::sqrt(v) * static_cast<Scalar>(R::rnorm(0.0, 1.0));
      });

      theta_s += mean_param(_theta_left);
      theta_s = theta_s.cwiseProduct(_annot);

      // 2. select for each component,
      // z = V * D2 * _x[,c] * theta[c,]
      for (Index c = 0; c < K; ++c) {
        eta_k = _x * theta_s.col(c) * (loading_s.col(c)).transpose();
        z = Vt.transpose() * D2.asDiagonal() * eta_k;

        if (opt.scale_var_calc()) {
          z = z.cwiseProduct(effect_sqrt);
          xi = Dinv.asDiagonal() * Vt * z;
          temp.row(c) = onesSV * (xi.cwiseProduct(xi));
        } else {
          xi = Dinv.asDiagonal() * Vt * z;
          temp.row(c) = onesSV * (xi.cwiseProduct(xi)) / denom;
        }
      }
      _stat(temp.unaryExpr(log10_op));
    }
    return Rcpp::List::create(Rcpp::_["mean"] = _stat.mean(),
                              Rcpp::_["var"] = _stat.var());
  };

  auto take_eta_var = [&](auto &_eta) {
    const Index SV = Vt.rows();

    const Index m = effect_sqrt.cols();
    const Index p = Vt.cols();
    const Scalar denom = static_cast<Scalar>(SV);

    _eta.resolve();

    Mat temp(1, m);
    Mat temp_SVm(SV, m);
    running_stat_t<Mat> _stat(1, m);

    Mat onesSV = Mat::Ones(1, SV);
    Mat z(p, m);  // projected GWAS
    Mat obs;

    temp_SVm = _eta.repr_mean();
    z = Vt.transpose() * D2.asDiagonal() * temp_SVm;
    if (opt.scale_var_calc()) {
      z = z.cwiseProduct(effect_sqrt);
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesSV * (xi.cwiseProduct(xi));
    } else {
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesSV * (xi.cwiseProduct(xi)) / denom;
    }
    obs = obs.unaryExpr(log10_op);

    for (Index b = 0; b < opt.nboot_var(); ++b) {
      temp_SVm = _eta.sample(rng);
      z = Vt.transpose() * D2.asDiagonal() * temp_SVm;
      if (opt.scale_var_calc()) {
        z = z.cwiseProduct(effect_sqrt);
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesSV * (xi.cwiseProduct(xi));
      } else {
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesSV * (xi.cwiseProduct(xi)) / denom;
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
    const Index SV = Vt.rows();
    const Index m = effect_sqrt.cols();
    const Index p = Vt.cols();
    const Scalar denom = static_cast<Scalar>(SV);

    _delta.resolve();

    Mat temp(1, m);
    Mat temp_SVm(SV, m);
    running_stat_t<Mat> _stat(1, m);
    // running_stat_t<Mat> _stat_null(1, m);
    Mat onesSV = Mat::Ones(1, SV);
    Mat z(p, m);  // projected GWAS

    temp_SVm = _delta.repr_mean();
    z = Vt.transpose() * temp_SVm;
    Mat obs;
    if (opt.scale_var_calc()) {
      z = z.cwiseProduct(effect_sqrt);
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesSV * (xi.cwiseProduct(xi));
    } else {
      xi = Dinv.asDiagonal() * Vt * z;
      obs = onesSV * (xi.cwiseProduct(xi)) / denom;
    }
    obs = obs.unaryExpr(log10_op);

    for (Index b = 0; b < opt.nboot_var(); ++b) {
      temp_SVm = _delta.sample(rng);
      z = Vt.transpose() * temp_SVm;
      if (opt.scale_var_calc()) {
        z = z.cwiseProduct(effect_sqrt);
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesSV * (xi.cwiseProduct(xi));
      } else {
        xi = Dinv.asDiagonal() * Vt * z;
        temp = onesSV * (xi.cwiseProduct(xi)) / denom;
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

  Rcpp::List out_left_param = Rcpp::List::create();
  Rcpp::List out_right_param = Rcpp::List::create();
  Mat llik;
  Rcpp::List var_annotated = Rcpp::List::create();
  Rcpp::List var_annotated_each = Rcpp::List::create();

  if (opt.mf_right_nn()) {
    // use beta on the right
    auto theta_left = make_dense_spike_slab<Scalar>(Vt.cols(), K, opt);
    auto theta_right = make_dense_beta<Scalar>(Y.cols(), K, opt);

    auto eta_f = make_annotated_regression_eta(Vt, Y, theta_left, theta_right);
    eta_f.set_weight_pk(A, Vt, Y);

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
      var_annotated = take_eta_var(eta_f);
      var_annotated_each = take_eta_var_each(Vt, A, theta_left, theta_right);
    }

  } else {
    // use regular spike-slab on both sides
    auto theta_left = make_dense_spike_slab<Scalar>(Vt.cols(), K, opt);
    auto theta_right = make_dense_spike_slab<Scalar>(Y.cols(), K, opt);

    auto eta_f = make_annotated_regression_eta(Vt, Y, theta_left, theta_right);
    eta_f.set_weight_pk(A, Vt, Y);

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
      var_annotated = take_eta_var(eta_f);
      var_annotated_each = take_eta_var_each(Vt, A, theta_left, theta_right);
    }
  }

  /////////////////////////
  // Variance estimation //
  /////////////////////////

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
        Rcpp::_["annotated"] = var_annotated,
        Rcpp::_["annotated.each"] = var_annotated_each,
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
      Rcpp::_["VtCd"] = VtCd, Rcpp::_["VtC"] = VtC, Rcpp::_["D2"] = D2,
      Rcpp::_["S.inv"] = weight, Rcpp::_["param.left"] = out_left_param,
      Rcpp::_["param.right"] = out_right_param,
      Rcpp::_["conf.multi"] = param_rcpp_list(theta_c),
      Rcpp::_["conf.uni"] = param_rcpp_list(theta_c_delta),
      Rcpp::_["llik"] = llik, Rcpp::_["gwas.clean"] = clean,
      Rcpp::_["var"] = var_decomp);
}

#endif
