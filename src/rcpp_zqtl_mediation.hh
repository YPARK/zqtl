#ifndef RCPP_ZQTL_MEDIATION_HH_
#define RCPP_ZQTL_MEDIATION_HH_

struct effect_y_mat_t {
  explicit effect_y_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct effect_y_se_mat_t {
  explicit effect_y_se_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct effect_m_mat_t {
  explicit effect_m_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct effect_m_se_mat_t {
  explicit effect_m_se_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct conf_mat_t {
  explicit conf_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct geno_mat_t {
  explicit geno_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

///////////////////////////////////////////////////////////
// estimate residual effects using MR-Egger type of idea //
//                                                       //
// eta1 = D^{-2} V' Z_qtl Theta_med'                     //
// eta2 = V' Theta_dir                                   //
// eta3 = V' Theta_conf                                  //
///////////////////////////////////////////////////////////

template <typename DIRECT, typename... DATA>
Rcpp::List _bootstrap_direct(const Mat obs_lodds, DIRECT& eta_direct,
                             const options_t& opt,
                             std::tuple<DATA...>&& data_tup);

template <typename... DATA>
Rcpp::List _bootstrap_marginal(const Mat obs_lodds, const options_t& opt,
                               std::tuple<DATA...>&& data_tup);

Rcpp::List impl_fit_med_zqtl(const effect_y_mat_t& yy,        // z_y
                             const effect_y_se_mat_t& yy_se,  // z_y_se
                             const effect_m_mat_t& mm,        // z_m
                             const effect_m_se_mat_t& mm_se,  // z_m_se
                             const geno_mat_t& geno,          // genotype or ld
                             const conf_mat_t& conf,          // snp confounder
                             const options_t& opt) {
  //////////////////////
  // check dimensions //
  //////////////////////

  if (opt.with_ld_matrix() && conf.val.cols() != conf.val.rows()) {
    ELOG("geno is not a square LD matrix");
    return Rcpp::List::create();
  }

  if (yy.val.rows() != yy_se.val.rows()) {
    ELOG("Check dimensions of effect and se on y");
    return Rcpp::List::create();
  }

  if (mm.val.rows() != mm_se.val.rows()) {
    ELOG("Check dimensions of effect and se on m");
    return Rcpp::List::create();
  }

  if (mm.val.cols() != mm_se.val.cols()) {
    ELOG("Check dimensions of effect and se on m");
    return Rcpp::List::create();
  }

  if (yy.val.rows() != conf.val.rows()) {
    ELOG("Check dimensions of C");
    return Rcpp::List::create();
  }

// random seed initialization
#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  /////////////////////////////////
  // Pre-process genotype matrix //
  /////////////////////////////////

  Mat D2, Vt;
  if (opt.with_ld_matrix()) {
    std::tie(D2, Vt) = do_eigen_decomp(geno.val, opt);
    TLOG("Finished eigen-decomposition of LD matrix");
  } else {
    std::tie(D2, Vt) = do_svd(geno.val, opt);
    D2 = D2.cwiseProduct(D2);
    TLOG("Finished SVD of genotype matrix");
  }

  TLOG("GWAS sample size = " << opt.sample_size());

  Mat effect_y_z, weight_y;
  std::tie(effect_y_z, weight_y) =
      preprocess_effect(yy.val, yy_se.val, opt.sample_size());

  Mat Y = Vt * effect_y_z;

  TLOG("Mediator sample size = " << opt.m_sample_size());

  Mat effect_m_z, weight_m;
  std::tie(effect_m_z, weight_m) =
      preprocess_effect(mm.val, mm_se.val, opt.m_sample_size());

  Mat M = Vt * effect_m_z;

  zqtl_model_t<Mat> model_y(Y, D2);

  ////////////////////////////////////////////////////////////////
  // construct delta_med to capture overall (potential) mediation effect
  auto theta_med = make_dense_spike_slab<Scalar>(M.cols(), Y.cols(), opt);
  auto delta_med = make_regression_eta(M, Y, theta_med);

  // confounder
  Mat VtC = Vt * conf.val;
  auto theta_conf_y = make_dense_spike_slab<Scalar>(VtC.cols(), Y.cols(), opt);
  auto eta_conf_y = make_regression_eta(VtC, Y, theta_conf_y);

  // construct eta_conf_y to capture direct (pleiotropic) effect
  auto theta_direct = make_dense_slab<Scalar>(Vt.cols(), Y.cols(), opt);
  auto eta_direct = make_regression_eta(Vt, Y, theta_direct);
  if (opt.weight_y()) eta_direct.set_weight(weight_y);

  Mat Id = Mat::Ones(Y.rows(), 1) / static_cast<Scalar>(Vt.cols());
  auto theta_delta_rand_y =
      make_dense_spike_slab<Scalar>(Id.cols(), Y.cols(), opt);
  auto delta_rand_y = make_regression_eta(Id, Y, theta_delta_rand_y);

  ////////////////////////////////////////////////////////////////
  // Estimate observed full model
  auto llik1 = impl_fit_eta_delta(model_y, opt, rng,
                                  std::make_tuple(eta_direct, eta_conf_y),
                                  std::make_tuple(delta_med, delta_rand_y));

  // residuals
  auto resid = Rcpp::List::create();
  Mat llik_resid;

  // V' z ~ N(V' (w .* r) + ..., D^2)
  if (opt.out_resid()) {
    auto theta_resid =
        make_dense_slab<Scalar>(effect_y_z.rows(), effect_y_z.cols(), opt);
    auto delta_resid = make_regression_eta(Vt, Y, theta_resid);

    dummy_eta_t dummy;

    eta_direct.resolve();
    eta_conf_y.resolve();
    delta_med.resolve();
    delta_rand_y.resolve();

    llik_resid = impl_fit_eta_delta(model_y, opt, rng, std::make_tuple(dummy),
                                    std::make_tuple(delta_resid),
                                    std::make_tuple(eta_direct, eta_conf_y),
                                    std::make_tuple(delta_med, delta_rand_y));

    resid = param_rcpp_list(theta_resid);
    TLOG("Calibrated residual effect");
  }

  Rcpp::List finemap = Rcpp::List::create();

  // Fine-mapping QTLs
  if (opt.med_finemap()) {
    // Construct M including potential mediators
    Mat row_max_lodds = log_odds_param(theta_med).rowwise().maxCoeff();
    std::vector<Index> med_include(0);

    for (Index j = 0; j < row_max_lodds.size(); ++j) {
      if (row_max_lodds(j) > opt.med_lodds_cutoff()) med_include.push_back(j);
    }

    const Index n_med_include = med_include.size();

    if (n_med_include > 0) {
      TLOG("Fine-mapping QTLs on " << n_med_include << " mediators");

      Mat Msub(Vt.rows(), n_med_include);
      Mat weight_m_sub(Vt.cols(), n_med_include);
      Index c = 0;
      for (Index j : med_include) {
        Msub.col(c) = M.col(j);
        weight_m_sub.col(c) = weight_m.col(j);
        c++;
      }

      zqtl_model_t<Mat> model_m(Msub, D2);

      auto theta_left =
          make_dense_spike_slab<Scalar>(Vt.cols(), Msub.cols(), opt);
      auto theta_right =
          make_dense_spike_slab<Scalar>(Y.cols(), Msub.cols(), opt);
      auto eta_med =
          make_mediation_eta(Vt, Msub, Vt, Y, theta_left, theta_right);
      if (opt.weight_m()) eta_med.set_weight_pk(weight_m_sub);
      if (opt.weight_y()) eta_med.set_weight_pt(weight_y);

      auto theta_conf_m =
          make_dense_spike_slab<Scalar>(VtC.cols(), Msub.cols(), opt);
      auto eta_conf_m = make_regression_eta(VtC, Msub, theta_conf_m);

      Mat Id_m = Mat::Ones(Msub.rows(), 1) / static_cast<Scalar>(Vt.cols());
      auto theta_delta_rand_m =
          make_dense_spike_slab<Scalar>(Id_m.cols(), Msub.cols(), opt);
      auto delta_rand_m = make_regression_eta(Id_m, Msub, theta_delta_rand_m);
      dummy_eta_t dummy;

      eta_direct.resolve();

      auto llik2 =
          impl_fit_mediation(model_y, model_m, opt, rng,
                             std::make_tuple(eta_med),       // mediation
                             std::make_tuple(eta_conf_y),    // eta[y] only
                             std::make_tuple(eta_conf_m),    // eta[m] only
                             std::make_tuple(delta_rand_y),  // delta[y]
                             std::make_tuple(delta_rand_m),  // delta[m]
                             std::make_tuple(eta_direct),    // clamped eta[y]
                             std::make_tuple(dummy));        // clamped eta[m]

      std::for_each(med_include.begin(), med_include.end(),
                    [](Index& x) { ++x; });

      finemap = Rcpp::List::create(
          Rcpp::_["llik"] = llik2, Rcpp::_["mediators"] = med_include,
          Rcpp::_["param.qtl"] = param_rcpp_list(theta_left),
          Rcpp::_["param.mediated"] = param_rcpp_list(theta_right));
    }
  }
#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  TLOG("Finished joint model estimation\n\n");

  Rcpp::List boot = Rcpp::List::create();

  if (opt.nboot() > 0) {
    if (opt.do_hyper())
      WLOG("bootstrap with do_hyper() can yield invalid null");

    if (opt.bootstrap_method() == 1) {
      boot = _bootstrap_direct(log_odds_param(theta_med), eta_direct, opt,
                               std::make_tuple(Y, M, D2, weight_y, Vt));
    } else if (opt.bootstrap_method() == 2) {
      boot = _bootstrap_marginal(log_odds_param(theta_med), opt,
                                 std::make_tuple(Y, M, D2, weight_y, Vt));
    }
  }

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["M"] = M, Rcpp::_["Vt"] = Vt,
      Rcpp::_["S.inv.y"] = weight_y, Rcpp::_["S.inv.m"] = weight_m,
      Rcpp::_["D2"] = D2,
      Rcpp::_["param.mediated"] = param_rcpp_list(theta_med),
      Rcpp::_["param.direct"] = param_rcpp_list(theta_direct),
      Rcpp::_["param.covariate.eta"] = param_rcpp_list(theta_conf_y),
      Rcpp::_["param.covariate.delta"] = param_rcpp_list(theta_delta_rand_y),
      Rcpp::_["llik"] = llik1, Rcpp::_["bootstrap"] = boot,
      Rcpp::_["finemap"] = finemap, Rcpp::_["resid"] = resid,
      Rcpp::_["llik.resid"] = llik_resid);
}

////////////////////////
// bootstrap method I //
////////////////////////

template <typename DIRECT, typename... DATA>
Rcpp::List _bootstrap_direct(const Mat obs_lodds, DIRECT& eta_direct,
                             const options_t& opt,
                             std::tuple<DATA...>&& data_tup) {
  Mat Y, M, D2, weight_y, Vt;
  std::tie(Y, M, D2, weight_y, Vt) = data_tup;

  // bootstrap parameters
  auto theta_boot_med = make_dense_spike_slab<Scalar>(M.cols(), Y.cols(), opt);
  auto delta_boot_med = make_regression_eta(M, Y, theta_boot_med);

  auto theta_boot_direct = make_dense_slab<Scalar>(Vt.cols(), Y.cols(), opt);
  auto eta_boot_direct = make_regression_eta(Vt, Y, theta_boot_direct);
  if (opt.weight_y()) eta_boot_direct.set_weight(weight_y);

  Mat FD = Mat::Ones(M.cols(), Y.cols());
  Mat PVAL(M.cols(), Y.cols());

  const Scalar zero_val = 0.0;
  const Scalar one_val = 1.0;

  auto add_false_discovery = [&](const Scalar& obs, const Scalar& perm) {
    if (obs <= perm) return one_val;
    return zero_val;
  };

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  Index nboot;
  Index nmed = theta_boot_med.rows();
  Index nout = theta_boot_med.cols();
  running_stat_t<Mat> LODDS(nmed, nout);
  zqtl_model_t<Mat> boot_model(Y, D2);
  Mat lodds_boot_mat = Mat::Zero(nmed, opt.nboot() * nout);

  for (nboot = 0; nboot < opt.nboot(); ++nboot) {
    const Scalar denom = static_cast<Scalar>(nboot + 2.0);

    eta_direct.resolve();
    boot_model.sample(eta_direct.sample(rng));

    impl_fit_eta_delta(boot_model, opt, rng, std::make_tuple(eta_boot_direct),
                       std::make_tuple(delta_boot_med));

    Mat log_odds = log_odds_param(theta_boot_med);
    FD += obs_lodds.binaryExpr(log_odds, add_false_discovery);

    // start = 1 + nboot * nout, end = (nboot + 1) * nout
    for (Index j = 0; j < nout; ++j)
      lodds_boot_mat.col(j + nboot * nout) = log_odds.col(j);

    PVAL = FD / denom;
    LODDS(log_odds_param(theta_boot_med));

    TLOG("Bootstrap : " << (nboot + 1) << " / " << opt.nboot()
                        << " min p-value " << PVAL.minCoeff() << " max p-value "
                        << PVAL.maxCoeff());

    initialize_param(theta_boot_direct);
    initialize_param(theta_boot_med);
  }
  TLOG("Finished bootstrapping by marginal model\n\n");

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  return Rcpp::List::create(
      Rcpp::_["stat.mat"] = lodds_boot_mat, Rcpp::_["nboot"] = nboot + 1,
      Rcpp::_["pval"] = PVAL, Rcpp::_["fd"] = FD,
      Rcpp::_["lodds.mean"] = LODDS.mean(), Rcpp::_["lodds.var"] = LODDS.var());
}

/////////////////////////
// bootstrap method II //
/////////////////////////

template <typename... DATA>
Rcpp::List _bootstrap_marginal(const Mat obs_lodds, const options_t& opt,
                               std::tuple<DATA...>&& data_tup) {
  Mat Y, M, D2, weight_y, Vt;
  std::tie(Y, M, D2, weight_y, Vt) = data_tup;

  // Estimate marginal model
  Mat Id = Mat::Ones(Y.rows(), 1) / static_cast<Scalar>(Vt.cols());
  zqtl_model_t<Mat> model_y(Y, D2);
  auto theta_marg = make_dense_slab<Scalar>(Vt.cols(), Y.cols(), opt);
  auto eta_marg = make_regression_eta(Vt, Y, theta_marg);
  if (opt.weight_y()) eta_marg.set_weight(weight_y);

  auto theta_delta_rand_marg =
      make_dense_spike_slab<Scalar>(Id.cols(), Y.cols(), opt);
  auto delta_rand_marg = make_regression_eta(Id, Y, theta_delta_rand_marg);

  // bootstrap parameters
  auto theta_boot_med = make_dense_spike_slab<Scalar>(M.cols(), Y.cols(), opt);
  auto delta_boot_med = make_regression_eta(M, Y, theta_boot_med);

  auto theta_boot_direct = make_dense_slab<Scalar>(Vt.cols(), Y.cols(), opt);
  auto eta_boot_direct = make_regression_eta(Vt, Y, theta_boot_direct);
  if (opt.weight_y()) eta_boot_direct.set_weight(weight_y);

  Mat FD = Mat::Ones(M.cols(), Y.cols());
  Mat PVAL(M.cols(), Y.cols());

  const Scalar zero_val = 0.0;
  const Scalar one_val = 1.0;

  auto add_false_discovery = [&](const Scalar& obs, const Scalar& perm) {
    if (obs <= perm) return one_val;
    return zero_val;
  };

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  auto llik = impl_fit_eta_delta(model_y, opt, rng, std::make_tuple(eta_marg),
                                 std::make_tuple(delta_rand_marg));

  TLOG("Finished estimation of the marginal model\n\n");

  Index nboot;
  Index nmed = theta_boot_med.rows();
  Index nout = theta_boot_med.cols();
  running_stat_t<Mat> LODDS(nmed, nout);
  zqtl_model_t<Mat> boot_model(Y, D2);
  Mat lodds_boot_mat = Mat::Zero(nmed, opt.nboot() * nout);

  for (nboot = 0; nboot < opt.nboot(); ++nboot) {
    const Scalar denom = static_cast<Scalar>(nboot + 2.0);
    eta_marg.resolve();
    boot_model.sample(eta_marg.sample(rng));

    impl_fit_eta_delta(boot_model, opt, rng, std::make_tuple(eta_boot_direct),
                       std::make_tuple(delta_boot_med));

    Mat log_odds = log_odds_param(theta_boot_med);
    FD += obs_lodds.binaryExpr(log_odds, add_false_discovery);

    // start = 1 + nboot * nout, end = (nboot + 1) * nout
    for (Index j = 0; j < nout; ++j)
      lodds_boot_mat.col(j + nboot * nout) = log_odds.col(j);

    PVAL = FD / denom;
    LODDS(log_odds_param(theta_boot_med));

    TLOG("bootstrap : " << (nboot + 1) << " / " << opt.nboot()
                        << " min p-value " << PVAL.minCoeff() << " max p-value "
                        << PVAL.maxCoeff());

    initialize_param(theta_boot_direct);
    initialize_param(theta_boot_med);
  }

  TLOG("Finished bootstrapping by marginal model\n\n");

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  return Rcpp::List::create(
      Rcpp::_["stat.mat"] = lodds_boot_mat,
      Rcpp::_["marginal"] = param_rcpp_list(theta_marg),
      Rcpp::_["nboot"] = nboot + 1, Rcpp::_["pval"] = PVAL, Rcpp::_["fd"] = FD,
      Rcpp::_["lodds.mean"] = LODDS.mean(), Rcpp::_["lodds.var"] = LODDS.var(),
      Rcpp::_["llik.marg"] = llik);
}

#endif
