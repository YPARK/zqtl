#include "dummy.hh"

#ifndef SGVB_UTIL_HH_
#define SGVB_UTIL_HH_

template <typename Func>                                     // don't do anyting
void func_apply(Func &&f, std::tuple<dummy_eta_t> &&tup) {}  // for dummy eta

#endif
