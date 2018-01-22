#ifndef OPTIONS_HH_
#define OPTIONS_HH_

#include <string>
#include <vector>

struct options_t {
  explicit options_t() {
    VBITER = 2000;
    MINITER = 100;
    K = 1;
    VBTOL = 1e-6;
    NSAMPLE = 10;
    NBOOT = 100;
    NTHREAD = 1;
    JITTER = 1e-2;
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

    EIGEN_TOL = 0.1;
    STD_LD = true;
    SAMPLE_SIZE = 0.0;
    M_SAMPLE_SIZE = 0.0;
    MED_LODDS_CUTOFF = 0.0;
    MED_FINEMAP = false;
    MF_SVD_INIT = true;

    WEIGHT_M = false;
    WEIGHT_Y = false;
    PRETRAIN = false;
    DO_HYPER = false;
    OUT_RESID = false;

    BOOTSTRAP_METHOD = 1;
  }

  const int vbiter() const { return VBITER; };
  const int miniter() const { return MINITER; };
  const int nsample() const { return NSAMPLE; };
  const int nboot() const { return NBOOT; };
  const int nthread() const { return NTHREAD; };
  const int k() const { return K; };
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

  const float eigen_tol() const { return EIGEN_TOL; };
  const bool std_ld() const { return STD_LD; }
  const float sample_size() const { return SAMPLE_SIZE; };
  const float m_sample_size() const { return M_SAMPLE_SIZE; };
  const float med_lodds_cutoff() const { return MED_LODDS_CUTOFF; }
  const bool med_finemap() const { return MED_FINEMAP; }

  const bool weight_m() const { return WEIGHT_M; }
  const bool weight_y() const { return WEIGHT_Y; }
  const bool pretrain() const { return PRETRAIN; }
  const bool do_hyper() const { return DO_HYPER; }
  const bool out_resid() const { return OUT_RESID; }

  const int bootstrap_method() const { return BOOTSTRAP_METHOD; }

  int VBITER;
  int MINITER;
  int NSAMPLE;
  int NBOOT;
  int NBOOT_CUTOFF;
  int NTHREAD;
  int K;
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

  float EIGEN_TOL;
  bool STD_LD;
  float SAMPLE_SIZE;
  float M_SAMPLE_SIZE;
  float MED_LODDS_CUTOFF;
  bool MED_FINEMAP;

  bool WEIGHT_M;
  bool WEIGHT_Y;
  bool PRETRAIN;
  bool DO_HYPER;
  bool OUT_RESID;

  int BOOTSTRAP_METHOD;
};

#endif
