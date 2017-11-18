// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
// [[Rcpp::plugins(openmp)]]
#include <omp.h>

#include "convergence.hh"
#include "dummy.hh"
#include "sgvb_util.hh"
#include "tuple_util.hh"

#ifndef SGVB_MEDIATION_INFERENCE_HH_
#define SGVB_MEDIATION_INFERENCE_HH_

template <typename ModelY, typename ModelM, typename Opt, typename RNG,
          typename... MediationEtas, typename... YEtas, typename... MEtas,
          typename... YDeltas, typename... MDeltas, typename... ClampedYEtas,
          typename... ClampedMEtas>
auto impl_fit_mediation(ModelY &modelY, ModelM &modelM, const Opt &opt,
                        RNG &rng, std::tuple<MediationEtas...> &&med_eta_tup,
                        std::tuple<YEtas...> &&y_eta_tup,
                        std::tuple<MEtas...> &&m_eta_tup,
                        std::tuple<YDeltas...> &&y_delta_tup,
                        std::tuple<MDeltas...> &&m_delta_tup,
                        std::tuple<ClampedYEtas...> &&clamped_y_eta_tup,
                        std::tuple<ClampedMEtas...> &&clamped_m_eta_tup) {
  using Scalar = typename ModelY::Scalar;
  using Index = typename ModelY::Index;
  using Mat = typename ModelY::Data;

  Eigen::setNbThreads(opt.nthread());
  TLOG("Number of threads = " << opt.nthread());

  using conv_t = convergence_t<Scalar>;
  Mat onesN = Mat::Ones(modelY.nobs(), 1) / static_cast<Scalar>(modelY.nobs());
  conv_t conv(typename conv_t::Nmodels(modelY.ntraits()),
              typename conv_t::Interv(opt.ninterval()));

  // Must keep this progress obj; otherwise segfault will occur
  Progress prog(2 * opt.vbiter(), !opt.verbose());
  Scalar rate = opt.rate0();

  // 1. sampling functions

  Mat y_eta_sampled(modelY.nobs(), modelY.ntraits());
  Mat m_eta_sampled(modelM.nobs(), modelM.ntraits());
  Mat y_delta_sampled(modelY.nobs(), modelY.ntraits());
  Mat m_delta_sampled(modelM.nobs(), modelM.ntraits());

  auto func_resolve = [&](auto &&eta) { eta.resolve(); };

  // on eta

  auto mean_y_eta = [&y_eta_sampled](auto &&eta) {
    y_eta_sampled += eta.repr_mean();
  };

  auto mean_m_eta = [&m_eta_sampled](auto &&eta) {
    m_eta_sampled += eta.repr_mean();
  };

  auto sample_y_eta = [&rng, &y_eta_sampled](auto &&eta) {
    y_eta_sampled += eta.sample(rng);
  };

  auto sample_m_eta = [&rng, &m_eta_sampled](auto &&eta) {
    m_eta_sampled += eta.sample(rng);
  };

  // on delta

  auto mean_y_delta = [&y_delta_sampled](auto &&delta) {
    y_delta_sampled += delta.repr_mean();
  };

  auto mean_m_delta = [&m_delta_sampled](auto &&delta) {
    m_delta_sampled += delta.repr_mean();
  };

  auto sample_y_delta = [&rng, &y_delta_sampled](auto &&delta) {
    y_delta_sampled += delta.sample(rng);
  };

  auto sample_m_delta = [&rng, &m_delta_sampled](auto &&delta) {
    m_delta_sampled += delta.sample(rng);
  };

  // mediation effects

  auto sample_med_m_eta = [&rng, &m_eta_sampled](auto &&eta) {
    m_eta_sampled += eta.sample_m(rng);
  };

  auto sample_med_y_eta = [&rng, &y_eta_sampled](auto &&eta) {
    y_eta_sampled += eta.sample_y(rng);
  };

  // 2. update functions

  const Index nstoch = opt.nsample();
  const Index niter = opt.vbiter();
  Index t;
  bool do_hyper = false;

  auto update_m = [&](auto &&eta) {

    for (Index s = 0; s < nstoch; ++s) {
      m_eta_sampled.setZero();
      func_apply(sample_med_m_eta, std::move(med_eta_tup));
      func_apply(sample_m_eta, std::move(m_eta_tup));
      func_apply(mean_m_eta, std::move(clamped_m_eta_tup));

      m_delta_sampled.setZero();
      func_apply(sample_m_delta, std::move(m_delta_tup));

      modelM.eval_eta_delta(m_eta_sampled, m_delta_sampled);
      eta.add_sgd(modelM.llik());
    }
    if (do_hyper) {
      eta.eval_hyper_sgd();
      eta.update_hyper_sgd(rate);
    }
    eta.eval_sgd();
    eta.update_sgd(rate);
  };

  auto update_y = [&](auto &&eta) {

    for (Index s = 0; s < nstoch; ++s) {
      y_eta_sampled.setZero();
      func_apply(sample_med_y_eta, std::move(med_eta_tup));
      func_apply(sample_y_eta, std::move(y_eta_tup));
      func_apply(mean_y_eta, std::move(clamped_y_eta_tup));

      y_delta_sampled.setZero();
      func_apply(sample_y_delta, std::move(y_delta_tup));

      modelY.eval_eta_delta(y_eta_sampled, y_delta_sampled);
      eta.add_sgd(modelY.llik());
    }
    if (do_hyper) {
      eta.eval_hyper_sgd();
      eta.update_hyper_sgd(rate);
    }
    eta.eval_sgd();
    eta.update_sgd(rate);
  };

  auto update_my_eta = [&](auto &&eta) {

    for (Index s = 0; s < nstoch; ++s) {
      // sample on Y
      y_eta_sampled.setZero();
      func_apply(sample_med_y_eta, std::move(med_eta_tup));
      func_apply(sample_y_eta, std::move(y_eta_tup));
      func_apply(mean_y_eta, std::move(clamped_y_eta_tup));

      y_delta_sampled.setZero();
      func_apply(sample_y_delta, std::move(y_delta_tup));

      modelY.eval_eta_delta(y_eta_sampled, y_delta_sampled);

      // sample on M
      m_eta_sampled.setZero();
      func_apply(sample_med_m_eta, std::move(med_eta_tup));
      func_apply(sample_m_eta, std::move(m_eta_tup));
      func_apply(mean_m_eta, std::move(clamped_m_eta_tup));

      m_delta_sampled.setZero();
      func_apply(sample_m_delta, std::move(m_delta_tup));

      modelM.eval_eta_delta(m_eta_sampled, m_delta_sampled);

      eta.add_sgd(modelM.llik(), modelY.llik());
    }
    if (do_hyper) {
      eta.eval_hyper_sgd();
      eta.update_hyper_sgd(rate);
    }
    eta.eval_sgd();
    eta.update_sgd(rate);
  };

  // initial tuning without hyperparameter optimization
  func_apply(func_resolve, std::move(med_eta_tup));
  func_apply(func_resolve, std::move(y_eta_tup));
  func_apply(func_resolve, std::move(clamped_y_eta_tup));
  func_apply(func_resolve, std::move(y_delta_tup));
  func_apply(func_resolve, std::move(med_eta_tup));
  func_apply(func_resolve, std::move(m_eta_tup));
  func_apply(func_resolve, std::move(clamped_m_eta_tup));

  do_hyper = false;
  for (t = 0; t < niter; ++t) {
    if (Progress::check_abort()) {
      break;
    }
    prog.increment();
    rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());

    func_apply(update_m, std::move(m_eta_tup));
    func_apply(update_m, std::move(m_delta_tup));
    func_apply(update_my_eta, std::move(med_eta_tup));
    func_apply(update_y, std::move(y_eta_tup));
    func_apply(update_y, std::move(y_delta_tup));

    conv.add(modelY.llik().transpose() * onesN);
    bool converged = conv.converged(opt.vbtol(), opt.miniter());
    if (opt.verbose()) conv.print(Rcpp::Rcerr);
    if (converged) {
      TLOG("Converged initial log-likelihood");
      break;
    }
  }

  if (opt.do_hyper()) {
    do_hyper = true;
    for (; t < 2 * niter; ++t) {
      if (Progress::check_abort()) {
        break;
      }
      prog.increment();
      rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());

      func_apply(update_m, std::move(m_eta_tup));
      func_apply(update_m, std::move(m_delta_tup));
      func_apply(update_my_eta, std::move(med_eta_tup));
      func_apply(update_y, std::move(y_eta_tup));
      func_apply(update_y, std::move(y_delta_tup));

      conv.add(modelY.llik().transpose() * onesN);
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

#endif
