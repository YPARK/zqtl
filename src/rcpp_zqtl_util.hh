#ifndef RCPP_ZQTL_UTIL_HH_
#define RCPP_ZQTL_UTIL_HH_

////////////////////////////////////////////////////////////////
// perform eigen decomposition of the X matrix
std::tuple<Mat, Mat> do_eigen_decomp(const Mat& X, const options_t& opt) {
  Eigen::SelfAdjointEigenSolver<Mat> es(X);
  Mat S = es.eigenvalues();
  Mat Vt = es.eigenvectors().transpose();

  const Scalar TOL = opt.eigen_tol();

  // set negative eigenvalues and vectors to zero
  for (Index j = 0; j < Vt.rows(); ++j) {
    if (S(j) <= TOL) {
      if (opt.verbose()) WLOG("Ignoring negative eigen values ... " << S(j));
      S(j) = 0.0;
      Vt.row(j) *= 0.0;
    }
  }

  return std::tuple<Mat, Mat>(S, Vt);
}

template <typename Derived>
std::tuple<Mat, Mat, Mat> do_svd(const Eigen::MatrixBase<Derived>& X,
                                 const options_t& opt) {
  using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
  // Center X matrix and divided by sqrt(n-1)
  // covariance = X'X
  Mat Xsafe = X;
  if (opt.std_ld()) {
    standardize(Xsafe);
    if (opt.verbose()) TLOG("Standardized matrix");
  } else {
    center(Xsafe);
    if (opt.verbose()) TLOG("Centered matrix");
  }

  is_obs_op<Mat> obs_op;
  const RowVec num_obs = X.unaryExpr(obs_op).colwise().sum();

  for (Index j = 0; j < Xsafe.cols(); ++j) {
    const Scalar nj = std::max(static_cast<Scalar>(2.0), num_obs(j));
    Xsafe.col(j) = Xsafe.col(j) / std::sqrt(nj - 1.0);
  }

  const Scalar n = static_cast<Scalar>(Xsafe.rows());
  Xsafe *= n;  // prevent underflow
  const Scalar TOL = opt.eigen_tol();

  Mat U_full;
  Mat V_full;
  Vec dd_full;

  if (opt.verbose()) TLOG("Start SVD ... ");

  if (opt.rand_svd()) {
    if (opt.verbose()) TLOG("Using randomized SVD ... ");

    RandomizedSVD<Mat> svd(opt.svd_rank(), opt.rand_svd_iter());
    if (opt.verbose()) svd.set_verbose();
    svd.compute(Xsafe);

    U_full = svd.matrixU();
    V_full = svd.matrixV();
    dd_full = svd.singularValues();

  } else {
    if (opt.jacobi_svd()) {
      if (opt.verbose()) TLOG("Using Jacobi SVD ... ");

      Eigen::JacobiSVD<Mat> svd;
      svd.setThreshold(TOL);
      svd.compute(Xsafe, Eigen::ComputeThinU | Eigen::ComputeThinV);

      U_full = svd.matrixU();
      V_full = svd.matrixV();
      dd_full = svd.singularValues();

    } else {
      if (opt.verbose()) TLOG("Using BCD SVD ... ");

      Eigen::BDCSVD<Mat> svd;
      svd.setThreshold(TOL);
      svd.compute(Xsafe, Eigen::ComputeThinU | Eigen::ComputeThinV);

      U_full = svd.matrixU();
      V_full = svd.matrixV();
      dd_full = svd.singularValues();
    }
  }

  if (opt.verbose()) TLOG("Done SVD");

  ///////////////////////////////////////////////////
  // Walk through the spectrum and pick components //
  ///////////////////////////////////////////////////

  Index num_comp = 0;
  Mat d2vec = dd_full / n;
  d2vec = d2vec.cwiseProduct(d2vec);

  Scalar cum = 0.0;
  const Scalar tot = d2vec.sum();
  const Scalar cutoff = tot * (1 - TOL);

  for (num_comp = 0; num_comp < V_full.cols(); ++num_comp) {
    cum += d2vec(num_comp);
    // Include as long as (cum / tot) <= (1 - TOL)
    if (cum > cutoff) break;
  }

  if (opt.verbose()) {
    TLOG("Included number of components : " << num_comp);
  }

  if (num_comp < 1) {
    ELOG("0 Number of SVD component!");
  }

  // Don't forget to rescale
  Mat dvec_out(num_comp, 1);
  dvec_out = dd_full.head(num_comp) / n;

  // Regularize to truncate very small values
  Scalar reg_value = opt.eigen_reg();
  Scalar _reg = reg_value / static_cast<Scalar>(num_comp);
  dvec_out += _reg * Mat::Ones(num_comp, 1);

  Mat Vt(num_comp, Xsafe.cols());
  Vt = V_full.leftCols(num_comp).transpose();
  Mat U(Xsafe.rows(), num_comp);
  U = U_full.leftCols(num_comp);

  return std::tuple<Mat, Mat, Mat>(U, dvec_out, Vt);
}

/////////////////////
// parsing options //
/////////////////////

void set_options_from_list(Rcpp::List& _list, options_t& opt) {
  if (_list.containsElementNamed("tau.lb"))
    opt.TAU_LODDS_LB = Rcpp::as<Scalar>(_list["tau.lb"]);

  if (_list.containsElementNamed("tau.ub"))
    opt.TAU_LODDS_UB = Rcpp::as<Scalar>(_list["tau.ub"]);

  if (_list.containsElementNamed("pi.lb"))
    opt.PI_LODDS_LB = Rcpp::as<Scalar>(_list["pi.lb"]);

  if (_list.containsElementNamed("pi.ub"))
    opt.PI_LODDS_UB = Rcpp::as<Scalar>(_list["pi.ub"]);

  if (_list.containsElementNamed("tau")) {
    opt.TAU_LODDS_LB = Rcpp::as<Scalar>(_list["tau"]);
    opt.TAU_LODDS_UB = Rcpp::as<Scalar>(_list["tau"]);
  }
  if (_list.containsElementNamed("pi")) {
    opt.PI_LODDS_LB = Rcpp::as<Scalar>(_list["pi"]);
    opt.PI_LODDS_UB = Rcpp::as<Scalar>(_list["pi"]);
  }
  if (_list.containsElementNamed("tol"))
    opt.VBTOL = Rcpp::as<Scalar>(_list["tol"]);
  if (_list.containsElementNamed("gammax"))
    opt.GAMMAX = Rcpp::as<Scalar>(_list["gammax"]);
  if (_list.containsElementNamed("decay"))
    opt.DECAY = Rcpp::as<Scalar>(_list["decay"]);
  if (_list.containsElementNamed("rate"))
    opt.RATE0 = Rcpp::as<Scalar>(_list["rate"]);
  if (_list.containsElementNamed("adam.m"))
    opt.RATE_M = Rcpp::as<Scalar>(_list["adam.m"]);
  if (_list.containsElementNamed("adam.v"))
    opt.RATE_V = Rcpp::as<Scalar>(_list["adam.v"]);
  if (_list.containsElementNamed("nsample"))
    opt.NSAMPLE = Rcpp::as<Index>(_list["nsample"]);

  if (_list.containsElementNamed("nboot"))
    opt.NBOOT = Rcpp::as<Index>(_list["nboot"]);

  if (_list.containsElementNamed("nboot.var"))
    opt.NBOOT_VAR = Rcpp::as<Index>(_list["nboot.var"]);

  if (_list.containsElementNamed("scale.var"))
    opt.SCALE_VAR_CALC = Rcpp::as<bool>(_list["scale.var"]);

  if (_list.containsElementNamed("num.duplicate.sample"))
    opt.N_DUPLICATE_SAMPLE = Rcpp::as<Index>(_list["num.duplicate.sample"]);

  if (_list.containsElementNamed("num.strat.size"))
    opt.N_STRAT_SIZE = Rcpp::as<Index>(_list["num.strat.size"]);

  if (_list.containsElementNamed("nsubmodel"))
    opt.N_SUBMODEL_MED = Rcpp::as<Index>(_list["nsubmodel"]);

  if (_list.containsElementNamed("num.submodel"))
    opt.N_SUBMODEL_MED = Rcpp::as<Index>(_list["num.submodel"]);

  if (_list.containsElementNamed("submodel.size"))
    opt.N_SUBMODEL_SIZE = Rcpp::as<Index>(_list["submodel.size"]);

  if (_list.containsElementNamed("print.interv"))
    opt.INTERV = Rcpp::as<Index>(_list["print.interv"]);

  if (_list.containsElementNamed("print.interval"))
    opt.INTERV = Rcpp::as<Index>(_list["print.interval"]);

  if (_list.containsElementNamed("nthread"))
    opt.NTHREAD = Rcpp::as<Index>(_list["nthread"]);
  if (_list.containsElementNamed("num.thread"))
    opt.NTHREAD = Rcpp::as<Index>(_list["num.thread"]);
  if (_list.containsElementNamed("k")) opt.K = Rcpp::as<Index>(_list["k"]);
  if (_list.containsElementNamed("K")) opt.K = Rcpp::as<Index>(_list["K"]);
  if (_list.containsElementNamed("re.k"))
    opt.RE_K = Rcpp::as<Index>(_list["re.k"]);
  if (_list.containsElementNamed("RE.K"))
    opt.RE_K = Rcpp::as<Index>(_list["RE.K"]);
  if (_list.containsElementNamed("vbiter"))
    opt.VBITER = Rcpp::as<Index>(_list["vbiter"]);
  if (_list.containsElementNamed("verbose"))
    opt.VERBOSE = Rcpp::as<bool>(_list["verbose"]);
  if (_list.containsElementNamed("random.effect"))
    opt.WITH_RANDOM_EFFECT = Rcpp::as<bool>(_list["random.effect"]);
  if (_list.containsElementNamed("ld.matrix"))
    opt.WITH_LD_MATRIX = Rcpp::as<bool>(_list["ld.matrix"]);
  if (_list.containsElementNamed("do.rescale"))
    opt.DO_RESCALE = Rcpp::as<bool>(_list["do.rescale"]);
  if (_list.containsElementNamed("rescale"))
    opt.DO_RESCALE = Rcpp::as<bool>(_list["rescale"]);
  if (_list.containsElementNamed("do.stdize"))
    opt.STD_LD = Rcpp::as<bool>(_list["do.stdize"]);
  if (_list.containsElementNamed("svd.init"))
    opt.MF_SVD_INIT = Rcpp::as<bool>(_list["svd.init"]);
  if (_list.containsElementNamed("mu.min"))
    opt.MU_MIN = Rcpp::as<Scalar>(_list["mu.min"]);
  if (_list.containsElementNamed("right.nn"))
    opt.MF_RIGHT_NN = Rcpp::as<bool>(_list["right.nn"]);
  if (_list.containsElementNamed("right.nonneg"))
    opt.MF_RIGHT_NN = Rcpp::as<bool>(_list["right.nonneg"]);
  if (_list.containsElementNamed("jitter"))
    opt.JITTER = Rcpp::as<Scalar>(_list["jitter"]);
  if (_list.containsElementNamed("rseed"))
    opt.RSEED = Rcpp::as<Scalar>(_list["rseed"]);
  if (_list.containsElementNamed("eigen.tol"))
    opt.EIGEN_TOL = Rcpp::as<Scalar>(_list["eigen.tol"]);
  if (_list.containsElementNamed("eigen.reg"))
    opt.EIGEN_REG = Rcpp::as<Scalar>(_list["eigen.reg"]);
  if (_list.containsElementNamed("sample.size"))
    opt.SAMPLE_SIZE = Rcpp::as<Scalar>(_list["sample.size"]);
  if (_list.containsElementNamed("med.sample.size"))
    opt.M_SAMPLE_SIZE = Rcpp::as<Scalar>(_list["med.sample.size"]);
  if (_list.containsElementNamed("med.lodds.cutoff"))
    opt.MED_LODDS_CUTOFF = Rcpp::as<Scalar>(_list["med.lodds.cutoff"]);

  if (_list.containsElementNamed("do.hyper"))
    opt.DO_HYPER = Rcpp::as<bool>(_list["do.hyper"]);

  if (_list.containsElementNamed("rand.svd"))
    opt.RAND_SVD = Rcpp::as<bool>(_list["rand.svd"]);

  if (_list.containsElementNamed("jacobi.svd"))
    opt.JACOBI_SVD = Rcpp::as<bool>(_list["jacobi.svd"]);

  if (_list.containsElementNamed("rand.svd.iter"))
    opt.RAND_SVD_ITER = Rcpp::as<Index>(_list["rand.svd.iter"]);

  if (_list.containsElementNamed("svd.rank"))
    opt.SVD_RANK = Rcpp::as<int>(_list["svd.rank"]);

  if (_list.containsElementNamed("do.finemap.unmed"))
    opt.DO_FINEMAP_UNMEDIATED = Rcpp::as<bool>(_list["do.finemap.unmed"]);

  if (_list.containsElementNamed("do.finemap.direct"))
    opt.DO_FINEMAP_UNMEDIATED = Rcpp::as<bool>(_list["do.finemap.direct"]);

  if (_list.containsElementNamed("do.var.calc"))
    opt.DO_VAR_CALC = Rcpp::as<bool>(_list["do.var.calc"]);

  if (_list.containsElementNamed("do.direct.estimation"))
    opt.DO_DIRECT_EFFECT = Rcpp::as<bool>(_list["do.direct.estimation"]);

  if (_list.containsElementNamed("do.control.backfire"))
    opt.DO_CONTROL_BACKFIRE = Rcpp::as<bool>(_list["do.control.backfire"]);

  if (_list.containsElementNamed("do.med.two.step"))
    opt.DO_MED_TWO_STEP = Rcpp::as<bool>(_list["do.med.two.step"]);

  // if (_list.containsElementNamed("de.propensity")) {
  //   opt.DO_DIRECT_EFFECT_PROPENSITY = Rcpp::as<bool>(_list["de.propensity"]);
  //   opt.DO_DIRECT_EFFECT_FACTORIZATION = false;
  // }

  if (_list.containsElementNamed("de.factorization")) {
    opt.DO_DIRECT_EFFECT_FACTORIZATION =
        Rcpp::as<bool>(_list["de.factorization"]);
    // opt.DO_DIRECT_EFFECT_PROPENSITY = false;
  }

  if (_list.containsElementNamed("factorization.model")) {
    opt.DE_FACTORIZATION_MODEL = Rcpp::as<int>(_list["factorization.model"]);
  }

  if (_list.containsElementNamed("out.resid"))
    opt.OUT_RESID = Rcpp::as<bool>(_list["out.resid"]);
  if (_list.containsElementNamed("out.residual"))
    opt.OUT_RESID = Rcpp::as<bool>(_list["out.residual"]);
  if (_list.containsElementNamed("multivar.mediator"))
    opt.MULTI_MED_EFFECT = Rcpp::as<bool>(_list["multivar.mediator"]);
}

Rcpp::List rcpp_adj_list(const Rcpp::NumericVector& d1_loc,
                         const Rcpp::NumericVector& d2_start_loc,
                         const Rcpp::NumericVector& d2_end_loc,
                         const double cis_window) {
  const auto n1 = d1_loc.size();
  const auto n2 = d2_start_loc.size();

  if (d2_start_loc.size() != d2_end_loc.size()) {
    ELOG("start and end location vectors have different size");
    return Rcpp::List::create();
  }

  std::vector<int> left;
  std::vector<int> right;

  for (auto i = 0u; i < n1; ++i) {
    const double d1 = d1_loc.at(i);
    for (auto j = 0u; j < n2; ++j) {
      const double d2_start = d2_start_loc[j];
      const double d2_end = d2_end_loc[j];
      if (d2_start > d2_end) continue;
      if (d1 >= (d2_start - cis_window) && d1 <= (d2_end + cis_window)) {
        left.push_back(i + 1);
        right.push_back(j + 1);
      }
    }
  }

  return Rcpp::List::create(Rcpp::_["d1"] = Rcpp::wrap(left),
                            Rcpp::_["d2"] = Rcpp::wrap(right));
}

template <typename Derived>
Mat calc_effect_sqrt(const Eigen::MatrixBase<Derived>& effect,
                     const Eigen::MatrixBase<Derived>& effect_se,
                     const Scalar sample_size) {
  const Scalar one_val = 1.0;
  Mat effect_sqrt = effect_se;
  if (sample_size < one_val) {
    WLOG("Ingoring summary-statistics sample size");
  } else {
    effect_sqrt = (effect.cwiseProduct(effect) / sample_size +
                   effect_se.cwiseProduct(effect_se))
                      .cwiseSqrt();
  }

  return effect_sqrt;
}

template <typename Derived, typename Derived2, typename Derived3>
Mat standardize_zscore(const Eigen::MatrixBase<Derived>& _zscore,
                       const Eigen::MatrixBase<Derived2>& Vt,
                       const Eigen::MatrixBase<Derived3>& D) {
  Mat Z = _zscore;
  Mat Y = D.cwiseInverse().asDiagonal() * Vt * Z;
  Mat xx = D.asDiagonal() * Vt * Mat::Ones(Vt.cols(), 1);
  Mat rr(Y.rows(), 1);
  Scalar xx_sum = xx.cwiseProduct(xx).sum();
  // Scalar n = Z.rows();
  Scalar denom = Y.rows();

  for (Index k = 0; k < Z.cols(); ++k) {
    Scalar xy = Y.col(k).cwiseProduct(xx).sum();
    Scalar mu = xy / xx_sum;
    rr = Y.col(k) - xx * mu;
    Scalar tau = rr.cwiseProduct(rr).sum() / denom + 1e-8;

    // TLOG("Standardize mu : " << mu << " xy : " << xy << " tau : " << tau
    //                          << " denom : " << denom);

    Z.col(k) = Vt.transpose() * D.asDiagonal() * rr / std::sqrt(tau);
  }

  return Z;
}

template <typename Derived, typename Derived2, typename Derived3>
Mat center_zscore(const Eigen::MatrixBase<Derived>& _zscore,
                  const Eigen::MatrixBase<Derived2>& Vt,
                  const Eigen::MatrixBase<Derived3>& D) {
  Mat Z = _zscore;
  Mat Y = D.cwiseInverse().asDiagonal() * Vt * Z;
  Mat xx = D.asDiagonal() * Vt * Mat::Ones(Vt.cols(), 1);
  Scalar xx_sum = xx.cwiseProduct(xx).sum();
  // Scalar denom = Y.rows();

  for (Index k = 0; k < Z.cols(); ++k) {
    Scalar xy = Y.col(k).cwiseProduct(xx).sum();
    Scalar mu = xy / xx_sum;
    // TLOG("Center mu : " << mu << " xy : " << xy);
    Z.col(k) = Vt.transpose() * D.asDiagonal() * (Y.col(k) - xx * mu);
  }

  return Z;
}

template <typename Derived>
std::tuple<Mat, Mat, Mat> preprocess_effect(
    const Eigen::MatrixBase<Derived>& _effect,
    const Eigen::MatrixBase<Derived>& _effect_se, const Scalar sample_size) {
  // 1. characterize NaN or infinite elements
  const Scalar zero_val = 0.0;
  const Scalar one_val = 1.0;
  is_obs_op<Mat> is_obs;
  Mat obs_mat =
      _effect.unaryExpr(is_obs).cwiseProduct(_effect_se.unaryExpr(is_obs));

  if (obs_mat.sum() < (obs_mat.rows() * obs_mat.cols())) {
    WLOG("Make sure all the effect size and standard errors are observed!");
    WLOG("Results may become biased due to the different level of missingness.")
  }

  Mat effect, effect_se;
  remove_missing(_effect, effect);
  remove_missing(_effect_se, effect_se);

  // effect_sqrt = sqrt(effect^2 /n + effect_se^2)
  // effect_z = effect / effect_sqrt
  // weight = 1/effect_sqrt

  Mat effect_sqrt =
      calc_effect_sqrt(effect, effect_se, sample_size).cwiseProduct(obs_mat);

  // 2. safely inverse
  auto safe_inverse = [&](const Scalar& x) {
    if (x <= zero_val) return zero_val;
    return one_val / x;
  };

  auto safe_division = [&](const Scalar& a, const Scalar& b) {
    if (b <= zero_val) return zero_val;
    return a / b;
  };

  Mat weight = effect_sqrt.unaryExpr(safe_inverse).cwiseProduct(obs_mat);
  Mat effect_z =
      effect.binaryExpr(effect_sqrt, safe_division).cwiseProduct(obs_mat);

  return std::make_tuple(effect_z, effect_sqrt, weight);
}

////////////////////////////////////////////////////////////////
// Take LD pairs for visualization
Rcpp::List take_ld_pairs(const Mat X, const float cutoff = 0.05,
                         const bool do_standardize = false) {
  using Index = Mat::Index;

  // Center X matrix and divided by sqrt(n-1)
  // covariance = X'X
  Mat Xsafe = X;
  if (do_standardize) {
    standardize(Xsafe);
  } else {
    center(Xsafe);
  }
  using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
  is_obs_op<Mat> obs_op;
  const RowVec num_obs = X.unaryExpr(obs_op).colwise().sum();

  for (Index j = 0; j < Xsafe.cols(); ++j) {
    Xsafe.col(j) = Xsafe.col(j) / std::max(1.0, std::sqrt(num_obs(j)) - 1.0);
  }

  // generate LD triangle coordinates
  const Index n = X.cols();

  std::vector<Scalar> xx;
  std::vector<Scalar> yy;
  std::vector<Scalar> xx_pos;
  std::vector<Scalar> yy_pos;
  std::vector<Scalar> cov;

  for (Index r = 0; r < n; ++r) {
    for (Index x = r + 1; x <= n; ++x) {
      const Index y = x - r;
      const Scalar _cov = Xsafe.col(x - 1).transpose() * Xsafe.col(y - 1);

      if (std::abs(_cov) < cutoff) continue;

      const Scalar _x_pos = y + static_cast<Scalar>(r) * 0.5;
      const Scalar _y_pos = -static_cast<Scalar>(r) - 1.0;

      for (Index j = 0; j < 5; ++j) {
        xx.push_back(x);
        yy.push_back(y);
        cov.push_back(_cov);
      }

      xx_pos.push_back(_x_pos - 0.5);
      xx_pos.push_back(_x_pos);
      xx_pos.push_back(_x_pos + 0.5);
      xx_pos.push_back(_x_pos);
      xx_pos.push_back(_x_pos - 0.5);

      yy_pos.push_back(_y_pos);
      yy_pos.push_back(_y_pos - 1.0);
      yy_pos.push_back(_y_pos);
      yy_pos.push_back(_y_pos + 1.0);
      yy_pos.push_back(_y_pos);
    }
  }

  Rcpp::NumericVector xx_out = Rcpp::wrap(xx);
  Rcpp::NumericVector xx_pos_out = Rcpp::wrap(xx_pos);
  Rcpp::NumericVector yy_out = Rcpp::wrap(yy);
  Rcpp::NumericVector yy_pos_out = Rcpp::wrap(yy_pos);
  Rcpp::NumericVector cov_out = Rcpp::wrap(cov);

  return Rcpp::List::create(
      Rcpp::_["x"] = xx_out, Rcpp::_["x.pos"] = xx_pos_out,
      Rcpp::_["y"] = yy_out, Rcpp::_["y.pos"] = yy_pos_out,
      Rcpp::_["cov"] = cov_out);
}

////////////////////////////////////////////////////////////////
template <typename T>
Rcpp::List param_rcpp_list(const T& param) {
  return impl_param_rcpp_list(param, sgd_tag<T>());
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param, const tag_param_spike_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param),
                            Rcpp::_["lodds"] = log_odds_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param, const tag_param_spike_gamma) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param),
                            Rcpp::_["lodds"] = log_odds_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param, const tag_param_mixture) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param),
                            Rcpp::_["lodds"] = log_odds_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param,
                                const tag_param_col_spike_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param),
                            Rcpp::_["lodds"] = log_odds_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param,
                                const tag_param_col_spike_gamma) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param),
                            Rcpp::_["lodds"] = log_odds_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param, const tag_param_col_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param, const tag_param_col_slab_zero) {
  return Rcpp::List::create(Rcpp::_["theta.var"] = var_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param, const tag_param_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param, const tag_param_beta) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param));
}

#endif
