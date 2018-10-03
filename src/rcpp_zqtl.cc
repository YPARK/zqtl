#include "rcpp_zqtl.hh"

RcppExport SEXP rcpp_zqtl(SEXP effect_sexp, SEXP effect_se_sexp, SEXP x_sexp,
                          SEXP c_sexp, SEXP c_delta_sexp, SEXP options_sexp) {
  BEGIN_RCPP

  Rcpp::traits::input_parameter<const Mat>::type effect(effect_sexp);
  Rcpp::traits::input_parameter<const Mat>::type effect_se(effect_se_sexp);
  Rcpp::traits::input_parameter<const Mat>::type C(c_sexp);
  Rcpp::traits::input_parameter<const Mat>::type Cdelta(c_delta_sexp);
  Rcpp::traits::input_parameter<const Mat>::type X(x_sexp);
  Rcpp::List options_list(options_sexp);

  options_t opt;
  set_options_from_list(options_list, opt);
  return Rcpp::wrap(impl_fit_zqtl(effect, effect_se, X, C, Cdelta, opt));
  END_RCPP
}

RcppExport SEXP rcpp_factorize(SEXP effect_sexp, SEXP effect_se_sexp,
                               SEXP x_sexp, SEXP options_sexp) {
  BEGIN_RCPP

  Rcpp::traits::input_parameter<const Mat>::type effect(effect_sexp);
  Rcpp::traits::input_parameter<const Mat>::type effect_se(effect_se_sexp);
  Rcpp::traits::input_parameter<const Mat>::type X(x_sexp);
  Rcpp::List options_list(options_sexp);

  options_t opt;
  set_options_from_list(options_list, opt);
  return Rcpp::wrap(impl_fit_factorization(effect, effect_se, X, opt));
  END_RCPP
}

RcppExport SEXP rcpp_med_zqtl(SEXP effect_sexp, SEXP effect_se_sexp,  // GWAS
                              SEXP qtl_sexp, SEXP qtl_se_sexp,        // QTL
                              SEXP x_gwas_sexp,  // genotype for gwas
                              SEXP x_med_sexp,   // genotype for mediation
                              SEXP c_sexp,       // covariates
                              SEXP cd_sexp,      // covariates
                              SEXP options_sexp) {
  BEGIN_RCPP

  Rcpp::traits::input_parameter<const Mat>::type effect(effect_sexp);
  Rcpp::traits::input_parameter<const Mat>::type effect_se(effect_se_sexp);
  Rcpp::traits::input_parameter<const Mat>::type qtl(qtl_sexp);
  Rcpp::traits::input_parameter<const Mat>::type qtl_se(qtl_se_sexp);

  Rcpp::traits::input_parameter<const Mat>::type Xgwas(x_gwas_sexp);
  Rcpp::traits::input_parameter<const Mat>::type Xmed(x_med_sexp);
  Rcpp::traits::input_parameter<const Mat>::type C(c_sexp);
  Rcpp::traits::input_parameter<const Mat>::type Cd(cd_sexp);
  Rcpp::List options_list(options_sexp);

  options_t opt;
  set_options_from_list(options_list, opt);
  return Rcpp::wrap(impl_fit_med_zqtl(effect_y_mat_t(effect),        //
                                      effect_y_se_mat_t(effect_se),  //
                                      effect_m_mat_t(qtl),           //
                                      effect_m_se_mat_t(qtl_se),     //
                                      geno_y_mat_t(Xgwas),           //
                                      geno_m_mat_t(Xmed),            //
                                      mult_conf_t(C),                 //
                                      univ_conf_t(Cd),                 //
                                      opt));
  END_RCPP
}

RcppExport SEXP rcpp_fac_med_zqtl(SEXP effect_sexp,
                                  SEXP effect_se_sexp,              // GWAS
                                  SEXP qtl_sexp, SEXP qtl_se_sexp,  // QTL
                                  SEXP x_gwas_sexp,  // genotype for gwas
                                  SEXP x_med_sexp,   // genotype for mediation
                                  SEXP c_sexp,       // covariates
                                  SEXP cd_sexp,       // covariates
                                  SEXP options_sexp) {
  BEGIN_RCPP

  Rcpp::traits::input_parameter<const Mat>::type effect(effect_sexp);
  Rcpp::traits::input_parameter<const Mat>::type effect_se(effect_se_sexp);
  Rcpp::traits::input_parameter<const Mat>::type qtl(qtl_sexp);
  Rcpp::traits::input_parameter<const Mat>::type qtl_se(qtl_se_sexp);

  Rcpp::traits::input_parameter<const Mat>::type Xgwas(x_gwas_sexp);
  Rcpp::traits::input_parameter<const Mat>::type Xmed(x_med_sexp);
  Rcpp::traits::input_parameter<const Mat>::type C(c_sexp);
  Rcpp::traits::input_parameter<const Mat>::type Cd(cd_sexp);
  Rcpp::List options_list(options_sexp);

  options_t opt;
  set_options_from_list(options_list, opt);
  return Rcpp::wrap(impl_fit_fac_med_zqtl(effect_y_mat_t(effect),        //
                                          effect_y_se_mat_t(effect_se),  //
                                          effect_m_mat_t(qtl),           //
                                          effect_m_se_mat_t(qtl_se),     //
                                          geno_y_mat_t(Xgwas),           //
                                          geno_m_mat_t(Xmed),            //
                                          mult_conf_t(C),                 //
                                          univ_conf_t(Cd),                 //
                                          opt));
  END_RCPP
}

RcppExport SEXP rcpp_fac_zqtl(SEXP effect_sexp, SEXP effect_se_sexp,
                              SEXP x_sexp, SEXP c_sexp, SEXP c_delta_sexp,
                              SEXP options_sexp) {
  BEGIN_RCPP

  Rcpp::traits::input_parameter<const Mat>::type effect(effect_sexp);
  Rcpp::traits::input_parameter<const Mat>::type effect_se(effect_se_sexp);
  Rcpp::traits::input_parameter<const Mat>::type C(c_sexp);
  Rcpp::traits::input_parameter<const Mat>::type Cdelta(c_delta_sexp);
  Rcpp::traits::input_parameter<const Mat>::type X(x_sexp);
  Rcpp::List options_list(options_sexp);

  options_t opt;
  set_options_from_list(options_list, opt);
  return Rcpp::wrap(impl_fit_fac_zqtl(effect, effect_se, X, C, Cdelta, opt));
  END_RCPP
}

RcppExport SEXP rcpp_adj(SEXP d1, SEXP d2_start, SEXP d2_end, SEXP cis) {
  BEGIN_RCPP
  Rcpp::traits::input_parameter<const Rcpp::NumericVector&>::type d1_loc(d1);
  Rcpp::traits::input_parameter<const Rcpp::NumericVector&>::type d2_start_loc(
      d2_start);
  Rcpp::traits::input_parameter<const Rcpp::NumericVector&>::type d2_end_loc(
      d2_end);
  Rcpp::traits::input_parameter<const double>::type cis_window(cis);

  return Rcpp::wrap(
      rcpp_adj_list(d1_loc, d2_start_loc, d2_end_loc, cis_window));
  END_RCPP
}

RcppExport SEXP rcpp_take_ld_pairs(SEXP x_sexp, SEXP cutoff_sexp,
                                   SEXP stdize_sexp) {
  BEGIN_RCPP

  Rcpp::traits::input_parameter<const Mat>::type X(x_sexp);
  Rcpp::traits::input_parameter<const Scalar>::type cutoff(cutoff_sexp);
  Rcpp::traits::input_parameter<const bool>::type do_stdize(stdize_sexp);

  return Rcpp::wrap(take_ld_pairs(X, cutoff, do_stdize));

  END_RCPP
}

RcppExport SEXP rcpp_take_svd_xtx(SEXP x_sexp, SEXP options_sexp) {
  BEGIN_RCPP

  Rcpp::List options_list(options_sexp);
  options_t opt;
  set_options_from_list(options_list, opt);
  Rcpp::traits::input_parameter<const Mat>::type xx(x_sexp);
  Mat U, D, Vt;
  const Mat& X = xx;
  std::tie(U, D, Vt) = do_svd(X, opt);
  return Rcpp::List::create(Rcpp::_["D"] = D, Rcpp::_["U"] = U,
                            Rcpp::_["V.t"] = Vt);

  END_RCPP
}
