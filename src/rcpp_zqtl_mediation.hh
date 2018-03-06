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

struct geno_y_mat_t {
  explicit geno_y_mat_t(const Mat& _val) : val(_val) {}
  const Mat& val;
};

struct geno_m_mat_t {
  explicit geno_m_mat_t(const Mat& _val) : val(_val) {}
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

template <typename DIRECT, typename MEDIATED_D, typename MEDIATED_E,
          typename... DATA>
Rcpp::List _variance_calculation(DIRECT& eta_direct, MEDIATED_D& delta_med,
                                 MEDIATED_E theta_med, const options_t& opt,
                                 std::tuple<DATA...>&& data_tup);

template <typename MODEL_Y, typename DIRECT, typename CONF, typename MEDIATED_E,
          typename... DATA>
Rcpp::List _fine_map(MODEL_Y& model_y, DIRECT& eta_direct, CONF& eta_conf_y,
                     MEDIATED_E theta_med_org, const options_t& opt,
                     std::tuple<DATA...>&& data_tup);

Rcpp::List impl_fit_med_zqtl(const effect_y_mat_t& yy,        // z_y
                             const effect_y_se_mat_t& yy_se,  // z_y_se
                             const effect_m_mat_t& mm,        // z_m
                             const effect_m_se_mat_t& mm_se,  // z_m_se
                             const geno_y_mat_t& geno_y,      // genotype_y
                             const geno_m_mat_t& geno_m,      // genotype_m
                             const conf_mat_t& conf,          // snp confounder
                             const options_t& opt) {
  //////////////////////
  // check dimensions //
  //////////////////////

  if (opt.with_ld_matrix()) {
    ELOG("Deprecated: longer use full LD matrix.");
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

  const Scalar n = static_cast<Scalar>(opt.sample_size());
  const Scalar n1 = static_cast<Scalar>(opt.m_sample_size());

  TLOG("GWAS sample size = " << opt.sample_size());
  TLOG("Mediator sample size = " << opt.m_sample_size());

  Mat effect_y_z, effect_sqrt_y, weight_y;
  std::tie(effect_y_z, effect_sqrt_y, weight_y) =
      preprocess_effect(yy.val, yy_se.val, n);

  Mat effect_m_z, effect_sqrt_m, weight_m;
  std::tie(effect_m_z, effect_sqrt_m, weight_m) =
      preprocess_effect(mm.val, mm_se.val, n1);

  /////////////////////////////////
  // Pre-process genotype matrix //
  /////////////////////////////////

  Mat U, D, Vt;
  std::tie(U, D, Vt) = do_svd(geno_y.val, opt);
  Mat D2 = D.cwiseProduct(D);

  Mat U_m, D_m, Vt_m;
  std::tie(U_m, D_m, Vt_m) = do_svd(geno_m.val, opt);
  Mat D2_m = D_m.cwiseProduct(D_m);

  TLOG("Finished SVD of genotype matrix");

  // alpha.uni           = S1 R1 inv(S1) alpha
  //
  // alpha               = S1 inv(R1) inv(S1) alpha.uni
  //                     = S1 V1 inv(D1^2) t(V1) inv(S1) alpha.uni
  //
  // t(V) R inv(S) alpha = t(V) V D^2 t(V) inv(S) alpha
  //                     = D^2 t(V) (S1/S) V1 inv(D1^2) t(V1) inv(S1)
  //                     alpha.uni
  //                   M = D^2 t(V) inv(S) S1 (V1/D1) t(V1/D1) * Z_alpha

  Mat Vt_m_d = D_m.cwiseInverse().asDiagonal() * Vt_m;

  // un-normalized version is more stable
  effect_m_z = Vt.transpose() * D2.asDiagonal() * Vt * Vt_m_d.transpose() *
               Vt_m_d * effect_m_z;

  // Mat M(Vt.rows(), effect_m_z.cols());
  // Mat VtZ = Vt_m_d * effect_m_z;
  // Mat invS_S1 = weight_y.asDiagonal() * effect_sqrt_m;  // p x K
  // Mat stuff(Vt.rows(), 1);
  // for (Index k = 0; k < effect_m_z.cols(); ++k) {
  //   stuff = Vt_m_d.transpose() * VtZ.col(k);
  //   M.col(k) = D2.asDiagonal() * Vt * stuff.cwiseProduct(invS_S1.col(k));
  // }

  Mat Y = Vt * effect_y_z;
  Mat M = Vt * effect_m_z;

  zqtl_model_t<Mat> model_y(Y, D2);

  ////////////////////////////////////////////////////////////////
  // construct delta_med to capture overall (potential) mediation effect
  auto theta_med = make_dense_spike_slab<Scalar>(M.cols(), Y.cols(), opt);
  auto delta_med = make_regression_eta(M, Y, theta_med);

  Mat m_bias = M * Mat::Ones(M.cols(), 1) / static_cast<Scalar>(M.cols());
  auto theta_med_bias =
      make_dense_slab<Scalar>(static_cast<Index>(1), Y.cols(), opt);
  auto delta_med_bias = make_regression_eta(m_bias, Y, theta_med_bias);

  // intercept    ~ R 1 theta
  // Vt intercept ~ D2 Vt 1 theta
  Mat VtI = Vt * Mat::Ones(Vt.cols(), static_cast<Index>(1)) /
            static_cast<Scalar>(Vt.cols());
  auto theta_direct = make_dense_slab<Scalar>(VtI.cols(), Y.cols(), opt);
  auto eta_direct = make_regression_eta(VtI, Y, theta_direct);

  // confounder -- or bias
  Mat VtC = Vt * conf.val;
  auto theta_conf_y = make_dense_slab<Scalar>(VtC.cols(), Y.cols(), opt);
  auto eta_conf_y = make_regression_eta(VtC, Y, theta_conf_y);

#ifdef EIGEN_USE_MKL_ALL
  // random seed initialization
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  ////////////////////////////////////////////////////////////////
  // Match scales -- just to help inference

  if (opt.do_rescale()) {
    rescale(Y);
    rescale(M);
    rescale(VtI);
    rescale(Vt);
    rescale(VtC);
  }

  Mat var_intercept = Mat::Ones(Vt.rows(), 1);
  auto theta_var = make_dense_slab<Scalar>(var_intercept.cols(), Y.cols(), opt);
  auto zeta_var = make_regression_eta(var_intercept, Y, theta_var);

  ////////////////////////////////////////////////////////////////
  // Estimate observed full model

  auto llik1 = impl_fit_eta_delta_zeta(
      model_y, opt, rng, std::make_tuple(eta_direct, eta_conf_y),
      std::make_tuple(delta_med), std::make_tuple(zeta_var));

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  // Fine-mapping QTLs
  Rcpp::List finemap = Rcpp::List::create();
  if (opt.med_finemap()) {
    finemap =
        _fine_map(model_y, eta_direct, eta_conf_y, theta_med, opt,
                  std::make_tuple(Y, M, U, D2, Vt, VtC, weight_y, weight_m));
  }

  TLOG("Finished joint model estimation\n\n");

  Rcpp::List var_decomp = _variance_calculation(
      eta_direct, delta_med, theta_med, opt, std::make_tuple(Y, M, U, D2));

  TLOG("Finished variance decomposition\n\n");

  return Rcpp::List::create(
      Rcpp::_["Y"] = Y, Rcpp::_["U"] = U, Rcpp::_["M"] = M, Rcpp::_["Vt"] = Vt,
      Rcpp::_["S.inv.y"] = weight_y, Rcpp::_["S.inv.m"] = weight_m,
      Rcpp::_["D2"] = D2,
      Rcpp::_["param.mediated"] = param_rcpp_list(theta_med),
      Rcpp::_["param.direct"] = param_rcpp_list(theta_direct),
      Rcpp::_["param.covariate.eta"] = param_rcpp_list(theta_conf_y),
      Rcpp::_["param.var"] = param_rcpp_list(theta_var),
      Rcpp::_["llik"] = llik1, Rcpp::_["finemap"] = finemap,
      Rcpp::_["var.decomp"] = var_decomp);
}

////////////////////////
// finemapping method //
////////////////////////

template <typename MODEL_Y, typename DIRECT, typename CONF, typename MEDIATED_E,
          typename... DATA>
Rcpp::List _fine_map(MODEL_Y& model_y, DIRECT& eta_direct, CONF& eta_conf_y,
                     MEDIATED_E theta_med_org, const options_t& opt,
                     std::tuple<DATA...>&& data_tup) {
  Mat Y, M, U, D2, Vt, VtC, weight_y, weight_m;
  std::tie(Y, M, U, D2, Vt, VtC, weight_y, weight_m) = data_tup;

  // Construct M including potential mediators
  Mat row_max_lodds = log_odds_param(theta_med_org).rowwise().maxCoeff();
  std::vector<Index> med_include(0);

  for (Index j = 0; j < row_max_lodds.size(); ++j) {
    if (row_max_lodds(j) > opt.med_lodds_cutoff()) med_include.push_back(j);
  }

  const Index n_med_include = med_include.size();
  Rcpp::List finemap = Rcpp::List::create();

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
    auto theta_right = make_dense_slab<Scalar>(Y.cols(), Msub.cols(), opt);
    auto eta_med = make_mediation_eta(Vt, Msub, Vt, Y, theta_left, theta_right);
    eta_med.resolve();

    auto theta_conf_m = make_dense_slab<Scalar>(VtC.cols(), Msub.cols(), opt);
    auto eta_conf_m = make_regression_eta(VtC, Msub, theta_conf_m);

    dummy_eta_t dummy;

    eta_direct.resolve();

#ifdef EIGEN_USE_MKL_ALL
    // random seed initialization
    VSLStreamStatePtr rng;
    vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
    omp_set_num_threads(opt.nthread());
#else
    std::mt19937 rng(opt.rseed());
#endif

    auto llik2 = impl_fit_mediation(
        model_y, model_m, opt, rng,
        std::make_tuple(eta_med),                 // mediation
        std::make_tuple(dummy),                   // eta[y] only
        std::make_tuple(eta_conf_m),              // eta[m] only
        std::make_tuple(dummy),                   // delta[y]
        std::make_tuple(dummy),                   // delta[m]
        std::make_tuple(eta_direct, eta_conf_y),  // clamped eta[y]
        std::make_tuple(dummy));                  // clamped eta[m]

#ifdef EIGEN_USE_MKL_ALL
    vslDeleteStream(&rng);
#endif

    std::for_each(med_include.begin(), med_include.end(),
                  [](Index& x) { ++x; });

    finemap = Rcpp::List::create(
        Rcpp::_["llik"] = llik2, Rcpp::_["mediators"] = med_include,
        Rcpp::_["param.qtl"] = param_rcpp_list(theta_left),
        Rcpp::_["param.mediated"] = param_rcpp_list(theta_right));
  }

  return finemap;
}

template <typename DIRECT, typename MEDIATED_D, typename MEDIATED_E,
          typename... DATA>
Rcpp::List _variance_calculation(DIRECT& eta_direct, MEDIATED_D& delta_med,
                                 MEDIATED_E theta_med, const options_t& opt,
                                 std::tuple<DATA...>&& data_tup) {
  Mat Y, M, U, D2;
  std::tie(Y, M, U, D2) = data_tup;

  const Index _n = U.rows();
  const Index _T = Y.cols();
  const Scalar n = static_cast<Scalar>(U.rows());
  const Index nboot = 500;
  const Index _K = M.cols();

#ifdef EIGEN_USE_MKL_ALL
  VSLStreamStatePtr rng;
  vslNewStream(&rng, VSL_BRNG_SFMT19937, opt.rseed());
  omp_set_num_threads(opt.nthread());
#else
  std::mt19937 rng(opt.rseed());
#endif

  // direct   : X * theta_direct
  //            sqrt(n) U D Vt * theta_direct
  //            sqrt(n) U D eta_direct
  //
  // mediated : X * inv(R) * mm.val * theta_med
  //            X * V inv(D2) Vt * mm.val * theta_med
  //            sqrt(n) U inv(D) * Vt * mm.val * theta_med
  //            sqrt(n) U inv(D) * delta_med
  //
  Mat snUD = std::sqrt(n) * U * D2.cwiseSqrt().asDiagonal();
  Mat snUinvD = std::sqrt(n) * U * D2.cwiseSqrt().cwiseInverse().asDiagonal();
  Mat UinvD = U * D2.cwiseSqrt().cwiseInverse().asDiagonal();

  eta_direct.resolve();
  delta_med.resolve();

  Mat direct_ind(_n, _T);
  Mat med_ind(_n, _T);

  Mat temp(1, _T);
  running_stat_t<Mat> direct_stat(1, _T);
  running_stat_t<Mat> med_stat(1, _T);

  for (Index b = 0; b < nboot; ++b) {
    direct_ind = snUD * eta_direct.sample(rng);
    column_var(direct_ind, temp);
    direct_stat(temp / n);

    med_ind = snUinvD * delta_med.sample(rng);
    column_var(med_ind, temp);
    med_stat(temp / n);
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  Mat theta_med_mean = mean_param(theta_med);
  Mat var_med_each(_K, _T);

  for (Index k = 0; k < _K; ++k) {
    column_var(snUinvD * (M.col(k) * theta_med_mean.row(k)), temp);
    var_med_each.row(k) = temp / n;
  }

  return Rcpp::List::create(Rcpp::_["var.direct.mean"] = direct_stat.mean(),
                            Rcpp::_["var.direct.var"] = direct_stat.var(),
                            Rcpp::_["var.med.each"] = var_med_each,
                            Rcpp::_["var.med.mean"] = med_stat.mean(),
                            Rcpp::_["var.med.var"] = med_stat.var());
}

#endif
