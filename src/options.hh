#ifndef OPTIONS_HH_
#define OPTIONS_HH_

#include <string>
#include <vector>

struct options_t {
  explicit options_t() {
    VBITER = 2000;
    MINITER = 100;
    K = 1;
    RE_K = 1;
    VBTOL = 1e-6;
    NSAMPLE = 10;
    NBOOT = 0;
    NBOOT_VAR = 1;
    SCALE_VAR_CALC = false;

    NTHREAD = 1;
    JITTER = 1e-1;
    RSEED = 13;
    RATE0 = 1e-2;
    DECAY = -0.01;
    TAU_LODDS_LB = -4;
    TAU_LODDS_UB = -4;
    PI_LODDS_LB = -2;
    PI_LODDS_UB = -2;
    GAMMAX = 10000;
    INTERV = 10;
    RATE_M = 0.99;
    RATE_V = 0.9999;
    VERBOSE = true;

    WITH_LD_MATRIX = false;
    WITH_RANDOM_EFFECT = false;

    EIGEN_TOL = 0.01;
    EIGEN_REG = 0.0;

    STD_LD = true;
    SAMPLE_SIZE = 0.0;
    M_SAMPLE_SIZE = 0.0;
    MED_LODDS_CUTOFF = 0.0;

    DO_DIRECT_EFFECT = true;
    DO_CONTROL_BACKFIRE = false;
    // DO_DIRECT_EFFECT_PROPENSITY = false;
    DO_DIRECT_EFFECT_FACTORIZATION = false;
    DO_DIRECT_EFFECT_CONDITIONAL = false;

    DE_FACTORIZATION_MODEL = 0;

    MF_SVD_INIT = true;
    MF_RIGHT_NN = true;
    MU_MIN = 1e-4;
    VAR_BETA_MIN = 1e-4;

    SVD_RANK = -1;
    RAND_SVD = false;
    RAND_SVD_ITER = 5;
    JACOBI_SVD = false;

    DO_HYPER = false;
    DO_RESCALE = false;
    OUT_RESID = false;
    MULTI_MED_EFFECT = false;

    DO_FINEMAP_UNMEDIATED = false;
    DO_MED_TWO_STEP = false;
    DO_VAR_CALC = false;

    N_SUBMODEL_MED = 0;
    N_SUBMODEL_SIZE = 1;
    N_DUPLICATE_SAMPLE = 1;
    N_STRAT_SIZE = 2;
  }

  const int vbiter() const { return VBITER; };
  const int miniter() const { return MINITER; };
  const int nsample() const { return NSAMPLE; };
  const int nboot() const { return NBOOT; };
  const int nboot_var() const { return NBOOT_VAR; };
  const bool scale_var_calc() const { return SCALE_VAR_CALC; }

  const int nthread() const { return NTHREAD; };
  const int k() const { return K; };
  const int re_k() const { return RE_K; };
  const bool do_rescale() const { return DO_RESCALE; };

  const float vbtol() const { return VBTOL; };
  const float jitter() const { return JITTER; };
  const int rseed() const { return RSEED; };
  const float rate0() const { return RATE0; };
  const float decay() const { return DECAY; };
  const float tau_lodds_lb() const { return TAU_LODDS_LB; };
  const float tau_lodds_ub() const { return TAU_LODDS_UB; };
  const float pi_lodds_lb() const { return PI_LODDS_LB; };
  const float pi_lodds_ub() const { return PI_LODDS_UB; };
  const float gammax() const { return GAMMAX; };
  const int ninterval() const { return INTERV; };
  const float rate_m() const { return RATE_M; };
  const float rate_v() const { return RATE_V; };
  const bool verbose() const { return VERBOSE; }
  const bool with_ld_matrix() const { return WITH_LD_MATRIX; }
  const bool with_random_effect() const { return WITH_RANDOM_EFFECT; }
  const bool mf_svd_init() const { return MF_SVD_INIT; }
  const bool mf_right_nn() const { return MF_RIGHT_NN; }
  const float mu_min() const { return MU_MIN; }
  const float var_beta_min() const { return VAR_BETA_MIN; }
  const float eigen_tol() const { return EIGEN_TOL; };
  const float eigen_reg() const { return EIGEN_REG; };
  const bool std_ld() const { return STD_LD; }
  const float sample_size() const { return SAMPLE_SIZE; };
  const float m_sample_size() const { return M_SAMPLE_SIZE; };
  const float med_lodds_cutoff() const { return MED_LODDS_CUTOFF; }

  const bool do_direct_effect() const { return DO_DIRECT_EFFECT; }
  const bool do_control_backfire() const { return DO_CONTROL_BACKFIRE; }

  // const bool do_de_propensity() const { return DO_DIRECT_EFFECT_PROPENSITY; }
  const bool do_de_factorization() const {
    return DO_DIRECT_EFFECT_FACTORIZATION;
  }
  const int de_factorization_model() const { return DE_FACTORIZATION_MODEL; }
  const bool do_de_conditional() const { return DO_DIRECT_EFFECT_CONDITIONAL; }
  const bool do_med_two_step() const { return DO_MED_TWO_STEP; }

  const int svd_rank() const { return SVD_RANK; }
  const bool rand_svd() const { return RAND_SVD; }
  const int rand_svd_iter() const { return RAND_SVD_ITER; }
  const bool jacobi_svd() const { return JACOBI_SVD; }

  const bool do_hyper() const { return DO_HYPER; }
  void off_hyper() { DO_HYPER = false; }
  void on_hyper() { DO_HYPER = true; }

  const bool out_resid() const { return OUT_RESID; }
  const bool multi_med_effect() const { return MULTI_MED_EFFECT; }
  const bool do_finemap_unmediated() const { return DO_FINEMAP_UNMEDIATED; }
  const bool do_var_calc() const { return DO_VAR_CALC; }

  const int n_submodel_model() const { return N_SUBMODEL_MED; }
  const int n_submodel_size() const { return N_SUBMODEL_SIZE; }
  const int n_duplicate_sample() const { return N_DUPLICATE_SAMPLE; }
  const int n_strat_size() const { return N_STRAT_SIZE; }

  int VBITER;
  int MINITER;
  int NSAMPLE;
  int NBOOT;
  int NBOOT_VAR;
  int NBOOT_CUTOFF;
  bool SCALE_VAR_CALC;

  int NTHREAD;
  int K;
  int RE_K;
  bool DO_RESCALE;
  float VBTOL;
  float JITTER;
  int RSEED;
  float RATE0;
  float DECAY;

  float TAU_LODDS_LB;
  float TAU_LODDS_UB;
  float PI_LODDS_LB;
  float PI_LODDS_UB;

  float GAMMAX;
  int INTERV;
  float RATE_M;
  float RATE_V;

  bool VERBOSE;
  bool WITH_LD_MATRIX;
  bool WITH_RANDOM_EFFECT;
  bool MF_SVD_INIT;
  bool MF_RIGHT_NN;
  float MU_MIN;
  float VAR_BETA_MIN;

  float EIGEN_TOL;
  float EIGEN_REG;
  bool STD_LD;
  float SAMPLE_SIZE;
  float M_SAMPLE_SIZE;
  float MED_LODDS_CUTOFF;

  bool DO_DIRECT_EFFECT;
  bool DO_CONTROL_BACKFIRE;
  // bool DO_DIRECT_EFFECT_PROPENSITY;
  bool DO_DIRECT_EFFECT_FACTORIZATION;
  bool DO_DIRECT_EFFECT_CONDITIONAL;
  int DE_FACTORIZATION_MODEL;

  int SVD_RANK;
  bool RAND_SVD;
  int RAND_SVD_ITER;
  bool JACOBI_SVD;

  bool DO_HYPER;
  bool OUT_RESID;
  bool MULTI_MED_EFFECT;
  bool DO_MED_TWO_STEP;
  bool DO_FINEMAP_UNMEDIATED;
  bool DO_VAR_CALC;

  int N_SUBMODEL_MED;
  int N_SUBMODEL_SIZE;
  int N_DUPLICATE_SAMPLE;
  int N_STRAT_SIZE;
};

#endif
