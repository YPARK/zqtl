// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
// [[Rcpp::plugins(openmp)]]
#include <omp.h>

#include <typeinfo>
#include "convergence.hh"
#include "dummy.hh"
#include "sgvb_util.hh"
#include "tuple_util.hh"

#ifndef SGVB_INFERENCE_HH_
#define SGVB_INFERENCE_HH_

template <typename Model, typename Opt, typename RNG, typename... MeanEtas,
          typename... ClampedMeanEtas>
auto impl_fit_eta(Model &model, const Opt &opt, RNG &rng,
                  std::tuple<MeanEtas...> &&mean_eta_tup,
                  std::tuple<ClampedMeanEtas...> &&clamped_mean_eta_tup);

template <typename Model, typename Opt, typename RNG, typename... MeanEtas>
auto impl_fit_eta(Model &model, const Opt &opt, RNG &rng,
                  std::tuple<MeanEtas...> &&mean_eta_tup);

template <typename Model, typename Opt, typename RNG, typename... Etas,
          typename... Deltas, typename... ClampedEtas,
          typename... ClampedDeltas>
auto impl_fit_eta_delta(Model &model, const Opt &opt, RNG &rng,
                        std::tuple<Etas...> &&eta_tup,
                        std::tuple<Deltas...> &&delta_tup,
                        std::tuple<ClampedEtas...> &&clamped_eta_tup,
                        std::tuple<ClampedDeltas...> &&clamped_delta_tup);

template <typename Model, typename Opt, typename RNG, typename... Etas,
          typename... Deltas>
auto impl_fit_eta_delta(Model &model, const Opt &opt, RNG &rng,
                        std::tuple<Etas...> &&eta_tup,
                        std::tuple<Deltas...> &&delta_tup);

////////////////////////////////////////////////////////////////
// for shorter implementations (meats)

template <typename Model, typename Opt, typename RNG, typename... Etas,
          typename... Deltas>
auto impl_fit_eta_delta(Model &model, const Opt &opt, RNG &rng,
                        std::tuple<Etas...> &&eta_tup,
                        std::tuple<Deltas...> &&delta_tup) {
  dummy_eta_t dummy;
  return impl_fit_eta_delta(model, opt, rng, std::move(eta_tup),
                            std::move(delta_tup), std::make_tuple(dummy),
                            std::make_tuple(dummy));
}

////////////////////////////////////////////////////////////////
// Log-likelihood of full model after spectral transformation
//
// llik[i] = - 0.5 ln var[i]
//           - 0.5 * (y[i] - var[i] eta[i] - delta[i])^2 / var[i]
//
// var[i]   = var0
// eta[i]   = (V' * theta)[i]
// delta[i] = (C * theta)[i]
//
// X / sqrt(n) = U sqrt(var0) V'
// LD          = X' * X / n = V * var0 * V'
// Kinship     = X * X' / n = U * var0 * U'
////////////////////////////////////////////////////////////////

template <typename Model, typename Opt, typename RNG, typename... Etas,
          typename... Deltas, typename... ClampedEtas,
          typename... ClampedDeltas>
auto impl_fit_eta_delta(Model &model, const Opt &opt, RNG &rng,
                        std::tuple<Etas...> &&eta_tup,
                        std::tuple<Deltas...> &&delta_tup,
                        std::tuple<ClampedEtas...> &&clamped_eta_tup,
                        std::tuple<ClampedDeltas...> &&clamped_delta_tup) {
  using Scalar = typename Model::Scalar;
  using Index = typename Model::Index;
  using Mat = typename Model::Data;

  Eigen::setNbThreads(opt.nthread());
  TLOG("Number of threads = " << opt.nthread());

  using conv_t = convergence_t<Scalar>;
  Mat onesN = Mat::Ones(model.nobs(), 1) / static_cast<Scalar>(model.nobs());
  conv_t conv(typename conv_t::Nmodels(model.ntraits()),
              typename conv_t::Interv(opt.ninterval()));

  const Index nstoch = opt.nsample();
  const Index niter = opt.vbiter();
  Index t;

  Mat eta_sampled(model.nobs(), model.ntraits());
  Mat delta_sampled(model.nobs(), model.ntraits());

  // Must keep this progress obj; otherwise segfault will occur
  Progress prog(2 * opt.vbiter(), !opt.verbose());

  // model fitting
  Scalar rate = opt.rate0();
  bool do_hyper = false;

  auto func_resolve = [&](auto &&eta) { eta.resolve(); };

  auto sample_eta = [&](auto &&eta) { eta_sampled += eta.sample(rng); };

  auto sample_delta = [&](auto &&delta) { delta_sampled += delta.sample(rng); };

  auto update_sgd = [&](auto &&effect) {

    for (Index s = 0; s < nstoch; ++s) {
      eta_sampled.setZero();
      delta_sampled.setZero();

      func_apply(sample_eta, std::move(eta_tup));
      func_apply(sample_delta, std::move(delta_tup));

      func_apply(sample_eta, std::move(clamped_eta_tup));
      func_apply(sample_delta, std::move(clamped_delta_tup));

      model.eval_eta_delta(eta_sampled, delta_sampled);
      effect.add_sgd(model.llik());
    }
    if (do_hyper) {
      effect.eval_hyper_sgd();
      effect.update_hyper_sgd(rate);
    }
    effect.eval_sgd();
    effect.update_sgd(rate);
  };

  // initial tuning without hyperparameter optimization
  func_apply(func_resolve, std::move(eta_tup));
  func_apply(func_resolve, std::move(delta_tup));
  func_apply(func_resolve, std::move(clamped_eta_tup));
  func_apply(func_resolve, std::move(clamped_delta_tup));

  do_hyper = false;
  for (t = 0; t < niter; ++t) {
    if (Progress::check_abort()) {
      break;
    }
    prog.increment();
    rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());
    func_apply(update_sgd, std::move(eta_tup));
    func_apply(update_sgd, std::move(delta_tup));

    conv.add(model.llik().transpose() * onesN);
    bool converged = conv.converged(opt.vbtol(), opt.miniter());
    if (opt.verbose()) conv.print(Rcpp::Rcerr);
    if (converged) {
      TLOG("Converged initial log-likelihood");
      break;
    }
  }

  // hyperparameter tuning
  if (opt.do_hyper()) {
    do_hyper = true;
    for (; t < 2 * niter; ++t) {
      if (Progress::check_abort()) {
        break;
      }
      prog.increment();
      rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());
      func_apply(update_sgd, std::move(eta_tup));
      func_apply(update_sgd, std::move(delta_tup));

      conv.add(model.llik().transpose() * onesN);
      bool converged = conv.converged(opt.vbtol(), opt.miniter());
      if (opt.verbose()) conv.print(Rcpp::Rcerr);
      if (converged) {
        TLOG("Converged hyperparameter log-likelihood");
        break;
      }
    }
  }

  TLOG("Finished SGVB inference");
  return conv.summarize();
}

////////////////////////////////////////////////////////////////
// Fit multiple eta's without clamping effects
template <typename Model, typename Opt, typename RNG, typename... MeanEtas>
auto impl_fit_eta(Model &model, const Opt &opt, RNG &rng,
                  std::tuple<MeanEtas...> &&mean_eta_tup) {
  dummy_eta_t dummy;
  return impl_fit_eta(model, opt, rng, std::move(mean_eta_tup),
                      std::make_tuple(dummy));
}

////////////////////////////////////////////////////////////////
// Fit multiple eta's
template <typename Model, typename Opt, typename RNG, typename... MeanEtas,
          typename... ClampedMeanEtas>
auto impl_fit_eta(Model &model, const Opt &opt, RNG &rng,
                  std::tuple<MeanEtas...> &&mean_eta_tup,
                  std::tuple<ClampedMeanEtas...> &&clamped_mean_eta_tup) {
  using Scalar = typename Model::Scalar;
  using Index = typename Model::Index;
  using Mat = typename Model::Data;

  Eigen::setNbThreads(opt.nthread());
  TLOG("Number of threads = " << opt.nthread());

  using conv_t = convergence_t<Scalar>;
  Mat onesN = Mat::Ones(model.nobs(), 1) / static_cast<Scalar>(model.nobs());
  conv_t conv(typename conv_t::Nmodels(model.ntraits()),
              typename conv_t::Interv(opt.ninterval()));

  const Index nstoch = opt.nsample();
  const Index niter = opt.vbiter();
  Index t;

  Mat mean_sampled(model.nobs(), model.ntraits());

  // Must keep this progress obj; otherwise segfault will occur
  Progress prog(2 * opt.vbiter(), !opt.verbose());

  // model fitting
  Scalar rate = opt.rate0();
  bool do_hyper = false;

  auto func_resolve = [&](auto &&eta) { eta.resolve(); };

  auto sample_mean_eta = [&rng, &mean_sampled](auto &&eta) {
    mean_sampled += eta.sample(rng);
  };

  auto update_sgd_eta = [&do_hyper, &nstoch, &mean_eta_tup,
                         &clamped_mean_eta_tup, &sample_mean_eta, &mean_sampled,
                         &model, &rate](auto &&eta) {

    for (Index s = 0; s < nstoch; ++s) {
      mean_sampled.setZero();
      func_apply(sample_mean_eta, std::move(mean_eta_tup));
      func_apply(sample_mean_eta, std::move(clamped_mean_eta_tup));
      model.eval_eta(mean_sampled);
      eta.add_sgd(model.llik());
    }
    if (do_hyper) {
      eta.eval_hyper_sgd();
      eta.update_hyper_sgd(rate);
    }
    eta.eval_sgd();
    eta.update_sgd(rate);
  };

  // initial tuning without hyperparameter optimization
  func_apply(func_resolve, std::move(mean_eta_tup));
  func_apply(func_resolve, std::move(clamped_mean_eta_tup));

  do_hyper = false;
  for (t = 0; t < niter; ++t) {
    if (Progress::check_abort()) {
      break;
    }
    prog.increment();
    rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());
    func_apply(update_sgd_eta, std::move(mean_eta_tup));

    conv.add(model.llik().transpose() * onesN);
    bool converged = conv.converged(opt.vbtol(), opt.miniter());
    if (opt.verbose()) conv.print(Rcpp::Rcerr);
    if (converged) {
      TLOG("Converged initial log-likelihood");
      break;
    }
  }

  // hyperparameter tuning
  if (opt.do_hyper()) {
    do_hyper = true;
    for (; t < 2 * niter; ++t) {
      if (Progress::check_abort()) {
        break;
      }
      prog.increment();
      rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());
      func_apply(update_sgd_eta, std::move(mean_eta_tup));

      conv.add(model.llik().transpose() * onesN);
      bool converged = conv.converged(opt.vbtol(), opt.miniter());
      if (opt.verbose()) conv.print(Rcpp::Rcerr);
      if (converged) {
        TLOG("Converged hyperparameter log-likelihood");
        break;
      }
    }
  }

#ifdef EIGEN_USE_MKL_ALL
  vslDeleteStream(&rng);
#endif

  TLOG("Finished SGVB inference");
  return conv.summarize();
}

#endif
