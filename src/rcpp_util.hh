#ifndef RCPPUTIL_HH_
#define RCPPUTIL_HH_

#include <cassert>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

std::string curr_time();

#include <Rcpp.h>
#include <RcppEigen.h>

#define TLOG(msg) \
  { Rcpp::Rcerr << "[" << curr_time() << "] " << msg << std::endl; }
#define ELOG(msg) \
  { Rcpp::Rcerr << "[" << curr_time() << "] [Error] " << msg << std::endl; }
#define WLOG(msg) \
  { Rcpp::Rcerr << "[" << curr_time() << "] [Warning] " << msg << std::endl; }
#define ASSERT(cond, msg)             \
  {                                   \
    if (!(cond)) {                    \
      ELOG(msg);                      \
      Rcpp::stop("assertion failed"); \
    }                                 \
  }

#endif
