#include <algorithm>
#include <random>
#include <type_traits>
#include <vector>
#include "rcpp_util.hh"

#ifndef EIGEN_UTIL_HH_
#define EIGEN_UTIL_HH_

////////////////////////////////////////////////////////////////
// C += A * B
template <typename Derived1, typename Derived2, typename RetDerived>
void times_add(const Eigen::MatrixBase<Derived1>& A,
               const Eigen::MatrixBase<Derived2>& B,
               Eigen::MatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  Tgt += A * B;
}

template <typename Derived1, typename Derived2, typename RetDerived>
void times_add(const Eigen::MatrixBase<Derived1>& A,
               const Eigen::SparseMatrixBase<Derived2>& B,
               Eigen::MatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  Tgt += A * B;
}

// C += A * B
template <typename Derived1, typename Derived2, typename RetDerived>
void times_add(const Eigen::MatrixBase<Derived1>& A,
               const Eigen::MatrixBase<Derived2>& B,
               Eigen::SparseMatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  using Index = typename RetDerived::Index;
  // column major
  for (Index k = 0; k < Tgt.outerSize(); ++k) {
    for (typename RetDerived::InnerIterator it(Tgt, k); it; ++it) {
      Index i = it.row();
      Index j = it.col();
      Tgt.coeffRef(i, j) += A.row(i) * B.col(j);
    }
  }
}

////////////////////////////////////////////////////////////////
// A * B -> C
template <typename Derived1, typename Derived2, typename RetDerived>
void times_set(const Eigen::MatrixBase<Derived1>& A,
               const Eigen::MatrixBase<Derived2>& B,
               Eigen::MatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  Tgt = A * B;
}

template <typename Derived1, typename Derived2, typename RetDerived>
void times_set(const Eigen::MatrixBase<Derived1>& A,
               const Eigen::SparseMatrixBase<Derived2>& B,
               Eigen::MatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  Tgt = A * B;
}

// A * B -> C
template <typename Derived1, typename Derived2, typename RetDerived>
void times_set(const Eigen::MatrixBase<Derived1>& A,
               const Eigen::MatrixBase<Derived2>& B,
               Eigen::SparseMatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  using Index = typename RetDerived::Index;
  // column major
  for (Index k = 0; k < Tgt.outerSize(); ++k) {
    for (typename RetDerived::InnerIterator it(Tgt, k); it; ++it) {
      Index i = it.row();
      Index j = it.col();
      Tgt.coeffRef(i, j) = A.row(i) * B.col(j);
    }
  }
}

template <typename Derived1, typename Derived2, typename RetDerived>
void times_set(const Eigen::SparseMatrixBase<Derived1>& A,
               const Eigen::MatrixBase<Derived2>& B,
               Eigen::SparseMatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  Tgt = A * B;
}

template <typename Derived1, typename Derived2, typename RetDerived>
void times_set(const Eigen::MatrixBase<Derived1>& A,
               const Eigen::SparseMatrixBase<Derived2>& B,
               Eigen::SparseMatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  Tgt = A * B;
}

////////////////////////////////////////////////////////////////
// A' * B -> C
template <typename Derived1, typename Derived2, typename RetDerived>
void trans_times_set(const Eigen::MatrixBase<Derived1>& A,
                     const Eigen::MatrixBase<Derived2>& B,
                     Eigen::MatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  Tgt = A.transpose() * B;
}

template <typename Derived1, typename Derived2, typename RetDerived>
void trans_times_set(const Eigen::MatrixBase<Derived1>& A,
                     const Eigen::SparseMatrixBase<Derived2>& B,
                     Eigen::MatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  Tgt = A.transpose() * B;
}

// A' * B -> C
template <typename Derived1, typename Derived2, typename RetDerived>
void trans_times_set(const Eigen::MatrixBase<Derived1>& A,
                     const Eigen::MatrixBase<Derived2>& B,
                     Eigen::SparseMatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  using Index = typename RetDerived::Index;
  // column major
  for (Index k = 0; k < Tgt.outerSize(); ++k) {
    for (typename RetDerived::InnerIterator it(Tgt, k); it; ++it) {
      Index i = it.row();
      Index j = it.col();
      Tgt.coeffRef(i, j) = A.col(i).transpose() * B.col(j);
    }
  }
}

////////////////////////////////////////////////////////////////
// A' * B -> C
template <typename Derived1, typename Derived2, typename RetDerived>
void trans_times_add(const Eigen::MatrixBase<Derived1>& A,
                     const Eigen::MatrixBase<Derived2>& B,
                     Eigen::MatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  Tgt += A.transpose() * B;
}

template <typename Derived1, typename Derived2, typename RetDerived>
void trans_times_add(const Eigen::MatrixBase<Derived1>& A,
                     const Eigen::SparseMatrixBase<Derived2>& B,
                     Eigen::MatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  Tgt += A.transpose() * B;
}

// A' * B -> C
template <typename Derived1, typename Derived2, typename RetDerived>
void trans_times_add(const Eigen::MatrixBase<Derived1>& A,
                     const Eigen::MatrixBase<Derived2>& B,
                     Eigen::SparseMatrixBase<RetDerived>& C) {
  RetDerived& Tgt(C.derived());
  using Index = typename RetDerived::Index;
  // column major
  for (Index k = 0; k < Tgt.outerSize(); ++k) {
    for (typename RetDerived::InnerIterator it(Tgt, k); it; ++it) {
      Index i = it.row();
      Index j = it.col();
      Tgt.coeffRef(i, j) += A.col(i).transpose() * B.col(j);
    }
  }
}

////////////////////////////////////////////////////////////////
template <typename Derived, typename OtherDerived>
void copy_matrix(const Eigen::MatrixBase<Derived>& src,
                 Eigen::MatrixBase<OtherDerived>& tgt) {
  tgt = src.eval();
}

template <typename Derived, typename OtherDerived>
void copy_matrix(const Eigen::SparseMatrixBase<Derived>& src,
                 Eigen::SparseMatrixBase<OtherDerived>& tgt) {
  OtherDerived& Tgt = tgt.derived();
  Tgt.setZero();
  Tgt += src.eval();
}

////////////////////////////////////////////////////////////////
// initialize sparse matrix taking the same non-zeroness of Adj
template <typename Derived>
void initialize(const Eigen::SparseMatrixBase<Derived>& adj,
                Eigen::SparseMatrixBase<Derived>& mat,
                const typename Derived::Scalar value) {
  using Index = typename Derived::Index;
  const Derived& A = adj.derived();
  Derived& Mat = mat.derived();
  Mat.resize(A.rows(), A.cols());

  using Scalar = typename std::remove_reference<decltype(A)>::type::Scalar;
  using Triplet = Eigen::Triplet<Scalar>;
  std::vector<Triplet> triples;

  // column major
  for (Index k = 0; k < A.outerSize(); ++k) {
    for (typename Derived::InnerIterator it(A, k); it; ++it) {
      const Index i = it.row();
      const Index j = it.col();
      triples.push_back(Triplet(i, j, value));
    }
  }

  Mat.setFromTriplets(triples.begin(), triples.end());
}

template <typename Derived>
void initialize(const Eigen::SparseMatrixBase<Derived>& adj,
                Eigen::SparseMatrixBase<Derived>& mat) {
  initialize(adj, mat, 0.0);
}

////////////////////////////////////////////////////////////////
template <typename Derived, typename Visitor>
void visit(Eigen::SparseMatrixBase<Derived>& M, Visitor& visitor) {
  using Scalar = typename Derived::Scalar;
  using Index = typename Derived::Index;

  bool is_first = true;
  for (Index k = 0; k < M.outerSize(); ++k) {
    for (typename Derived::InnerIterator it(M, k); it; ++it) {
      const Scalar val = it.value();
      const Index i = it.row();
      const Index j = it.col();
      if (is_first) {
        visitor.init(val, i, j);
        is_first = false;
      } else {
        visitor(val, i, j);
      }
    }
  }
}

template <typename Derived, typename Visitor>
void visit(const Eigen::MatrixBase<Derived>& M, Visitor& visitor) {
  M.visit(visitor);
}

////////////////////////////////////////////////////////////////
// initialize non-zeroness by adjacency A
template <typename S>
struct column_sum_t {
  using Scalar = typename S::Scalar;
  using Index = typename S::Index;
  explicit column_sum_t(S& _ret) : ret(_ret) { ret.setZero(); }

  void init(const Scalar& value, Index r, Index c) {
    if (std::abs(value) > 0.0) ret.coeffRef(0, c) += value;
  }
  void operator()(const Scalar& value, Index r, Index c) {
    if (std::abs(value) > 0.0) ret.coeffRef(0, c) += value;
  }

  S& ret;
};

template <typename Derived, typename OtherDerived>
void column_sum(Eigen::SparseMatrixBase<Derived>& mat,
                Eigen::SparseMatrixBase<OtherDerived>& ret) {
  ret.derived.resize(1, mat.cols());
  column_sum_t<OtherDerived> column_sum_op(ret.derived());
  visit(mat, column_sum_op);
}

////////////////////////////////////////////////////////////////
// a simple visitor to keep track of statistics
template <typename real_t>
struct calc_stat_t {
  explicit calc_stat_t() { reset(); }

  void init(const real_t& val, const int i, const int j) {
    reset();
    if (std::isfinite(val)) {
      s1 = val;
      s2 = val * val;
      n = 1;
      is_first = false;
      min_val = val;
      max_val = val;
    }
  }

  void operator()(const real_t& val, const int i, const int j) {
    if (std::isfinite(val)) {
      s1 += val;
      s2 += val * val;
      ++n;
      if (!is_first) {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
      } else {
        min_val = val;
        max_val = val;
      }
    }
  }

  void reset() {
    is_first = true;
    min_val = 0.0;
    max_val = 0.0;
    s1 = 0.;
    s2 = 0.;
    n = 0.;
  }

  real_t min() const { return min_val; }

  real_t max() const { return max_val; }

  real_t mean() const {
    if (n < 1.0) return 0.0;
    return s1 / n;
  }

  real_t var() const {
    if (n < 1.0) return 0.0;
    real_t m = s1 / n;
    return s2 / n - m * m;
  }

 private:
  bool is_first = true;
  real_t min_val = 0.0;
  real_t max_val = 0.0;

  real_t s1 = 0.;
  real_t s2 = 0.;
  real_t n = 0.;
};

////////////////////////////////////////////////////////////////
template <typename Derived>
void setConstant(Eigen::SparseMatrixBase<Derived>& mat,
                 const typename Derived::Scalar val) {
  using Scalar = typename Derived::Scalar;
  auto fill_const = [val](const Scalar& x) { return val; };
  Derived& Mat = mat.derived();
  Mat = Mat.unaryExpr(fill_const);
}

template <typename Derived>
void setConstant(Eigen::MatrixBase<Derived>& mat,
                 const typename Derived::Scalar val) {
  Derived& Mat = mat.derived();
  Mat = Mat.setConstant(val);
}

////////////////////////////////////////////////////////////////
template <typename T>
struct running_stat_t {
  using Scalar = typename T::Scalar;
  using Index = typename T::Index;

  explicit running_stat_t(const Index _d1, const Index _d2) : d1{_d1}, d2{_d2} {
    Cum.resize(d1, d2);
    SqCum.resize(d1, d2);
    Mean.resize(d1, d2);
    Var.resize(d1, d2);
    reset();
  }

  void reset() {
    setConstant(SqCum, 0.0);
    setConstant(Cum, 0.0);
    setConstant(Mean, 0.0);
    setConstant(Var, 0.0);
    n = 0.0;
  }

  template <typename Derived>
  void operator()(const Eigen::MatrixBase<Derived>& X) {
    Cum += X;
    SqCum += X.cwiseProduct(X);
    ++n;
  }

  template <typename Derived>
  void operator()(const Eigen::SparseMatrixBase<Derived>& X) {
    Cum += X;
    SqCum += X.cwiseProduct(X);
    ++n;
  }

  const T& mean() {
    if (n > 0) {
      Mean = Cum / n;
    }
    return Mean;
  }

  const T& var() {
    if (n > 0) {
      Mean = Cum / n;
      Var = SqCum / n - Mean.cwiseProduct(Mean);
    }
    return Var;
  }

  const Index d1;
  const Index d2;

  T Cum;
  T SqCum;
  T Mean;
  T Var;
  Scalar n;
};

////////////////////////////////////////////////////////////////
template <typename T>
struct op_mark_nonfinite_t {
  const T operator()(const T& x) const {
    if (!std::isfinite(x)) {
      ELOG(x);
      return one_value;
    }
    return zero_value;
  }
  static constexpr T zero_value = 0;
  static constexpr T one_value = 1;
};

template <typename T>
struct op_remove_nonfinite_t {
  const T operator()(const T& x) const {
    return std::isfinite(x) ? x : zero_value;
  }
  static constexpr T zero_value = 0;
};

template <typename Derived, typename OtherDerived>
void remove_missing(const Eigen::MatrixBase<Derived>& X,
                    Eigen::MatrixBase<OtherDerived> const& ret) {
  typedef typename Derived::Scalar value_type;
  op_remove_nonfinite_t<value_type> op;

  Eigen::MatrixBase<OtherDerived>& R =
      const_cast<Eigen::MatrixBase<OtherDerived>&>(ret);

  if (X.rows() != R.rows() || X.cols() != R.cols()) {
    // WLOG("resizing the matrix of " << R.rows() << " x " << R.cols() << " -> "
    //                                << X.rows() << " x " << X.cols());
    R.derived().resize(X.rows(), X.cols());
  }

  const_cast<Eigen::MatrixBase<OtherDerived>&>(ret) = X.unaryExpr(op);
}

////////////////////////////////////////////////////////////////
// check dimensionality
template <typename Derived, typename V>
void check_dim(const Eigen::MatrixBase<Derived>& mat, const V nrows,
               const V ncols, const std::string msg) {
  ASSERT(mat.rows() == nrows, msg << " : " << nrows << " rows "
                                  << " != " << mat.rows() << " rows");
  ASSERT(mat.cols() == ncols, msg << " : " << ncols << " cols "
                                  << " != " << mat.cols() << " cols");
}

template <typename Derived, typename V>
void check_dim(const Eigen::SparseMatrixBase<Derived>& mat, const V nrows,
               const V ncols, const std::string msg) {
  ASSERT(mat.rows() == nrows, msg << " : " << nrows << " rows "
                                  << " != " << mat.rows() << " rows");
  ASSERT(mat.cols() == ncols, msg << " : " << ncols << " cols "
                                  << " != " << mat.cols() << " cols");
}

template <typename T, typename V>
void check_dim(const T& mat, const V nrows, const V ncols,
               const std::string msg) {
  ASSERT(mat.rows() == nrows, msg << " : " << nrows << " rows "
                                  << " != " << mat.rows() << " rows");
  ASSERT(mat.cols() == ncols, msg << " : " << ncols << " cols "
                                  << " != " << mat.cols() << " cols");
}

template <typename Derived>
void print_dim(const Eigen::MatrixBase<Derived>& mat,
               const std::string msg = "") {
  ELOG(msg << "\n" << mat.rows() << " x " << mat.cols());
}

template <typename Derived>
void print_dim(const Eigen::SparseMatrixBase<Derived>& mat,
               const std::string msg = "") {
  ELOG(msg << "\n" << mat.rows() << " x " << mat.cols());
}

////////////////////////////////////////////////////////////////
template <typename Derived, typename Rows, typename OtherDerived>
void subset_rows(const Eigen::MatrixBase<Derived>& X, const Rows& rows,
                 Eigen::MatrixBase<OtherDerived> const& ret) {
  Eigen::MatrixBase<OtherDerived>& R =
      const_cast<Eigen::MatrixBase<OtherDerived>&>(ret);
  using Index = typename Derived::Index;
  Index i = 0;
  if (R.rows() != static_cast<Index>(rows.size()) || X.cols() != R.cols()) {
    // WLOG("resizing the matrix of " << R.rows() << " x " << R.cols() << " -> "
    //                                << rows.size() << " x " << X.cols());
    R.derived().resize(rows.size(), X.cols());
  }

  for (Index r : rows) {
    R.row(i) = X.row(r);
    i++;
  }
}

template <typename Derived, typename Cols, typename OtherDerived>
void subset_cols(const Eigen::MatrixBase<Derived>& X, const Cols& cols,
                 Eigen::MatrixBase<OtherDerived> const& ret) {
  Eigen::MatrixBase<OtherDerived>& R =
      const_cast<Eigen::MatrixBase<OtherDerived>&>(ret);
  using Index = typename Derived::Index;
  Index i = 0;
  if (R.cols() != static_cast<Index>(cols.size()) || X.rows() != R.rows()) {
    // WLOG("resizing the matrix of " << R.rows() << " x " << R.cols() << " -> "
    //                                << X.rows() << " x " << cols.size());
    R.derived().resize(X.rows(), cols.size());
  }

  for (Index c : cols) {
    R.col(i) = X.col(c);
    i++;
  }
}

////////////////////////////////////////////////////////////////
template <typename T>
struct is_obs_op {
  using Scalar = typename T::Scalar;
  const Scalar operator()(const Scalar& x) const {
    return std::isfinite(x) ? one_val : zero_val;
  }
  static constexpr Scalar one_val = 1.0;
  static constexpr Scalar zero_val = 0.0;
};

template <typename T>
struct add_pseudo_op {
  using Scalar = typename T::Scalar;
  explicit add_pseudo_op(const Scalar pseudo_val) : val(pseudo_val) {}
  const Scalar operator()(const Scalar& x) const { return x + val; }
  const Scalar val;
};

template <typename T1, typename T2, typename Ret>
void XY_nobs(const Eigen::MatrixBase<T1>& X, const Eigen::MatrixBase<T2>& Y,
             Eigen::MatrixBase<Ret>& ret,
             const typename Ret::Scalar pseudo = 1.0) {
  is_obs_op<T1> op1;
  is_obs_op<T2> op2;
  ret.derived() = (X.unaryExpr(op1) * Y.unaryExpr(op2)).array() + pseudo;
}

template <typename T1, typename T2, typename Ret>
void XY_nobs(const Eigen::MatrixBase<T1>& X, const Eigen::MatrixBase<T2>& Y,
             Eigen::SparseMatrixBase<Ret>& ret,
             const typename Ret::Scalar pseudo = 1.0) {
  is_obs_op<T1> op1;
  is_obs_op<T2> op2;
  add_pseudo_op<Ret> op_add(pseudo);

  times_set(X.unaryExpr(op1), Y.unaryExpr(op2), ret);
  ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void XYZ_nobs(const Eigen::MatrixBase<T1>& X, const Eigen::MatrixBase<T2>& Y,
              const Eigen::MatrixBase<T3>& Z, Eigen::MatrixBase<Ret>& ret,
              const typename Ret::Scalar pseudo = 1.0) {
  is_obs_op<T1> op1;
  is_obs_op<T2> op2;
  is_obs_op<T3> op3;

  ret.derived() =
      (X.unaryExpr(op1) * Y.unaryExpr(op2) * Z.unaryExpr(op3)).array() + pseudo;
}

template <typename T1, typename T2, typename T3, typename Ret>
void XYZ_nobs(const Eigen::MatrixBase<T1>& X, const Eigen::MatrixBase<T2>& Y,
              const Eigen::MatrixBase<T3>& Z, Eigen::SparseMatrixBase<Ret>& ret,
              const typename Ret::Scalar pseudo = 1.0) {
  is_obs_op<T1> op1;
  is_obs_op<T2> op2;
  is_obs_op<T3> op3;
  add_pseudo_op<Ret> op_add(pseudo);

  auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
  times_set(X.unaryExpr(op1), YZ, ret);
  ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void XYZ_nobs(const Eigen::MatrixBase<T1>& X, const Eigen::MatrixBase<T2>& Y,
              const Eigen::SparseMatrixBase<T3>& Z, Eigen::MatrixBase<Ret>& ret,
              const typename Ret::Scalar pseudo = 1.0) {
  is_obs_op<T1> op1;
  is_obs_op<T2> op2;
  is_obs_op<T3> op3;
  add_pseudo_op<Ret> op_add(pseudo);

  auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
  times_set(X.unaryExpr(op1), YZ, ret);
  ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void XYZ_nobs(const Eigen::MatrixBase<T1>& X, const Eigen::MatrixBase<T2>& Y,
              const Eigen::SparseMatrixBase<T3>& Z,
              Eigen::SparseMatrixBase<Ret>& ret,
              const typename Ret::Scalar pseudo = 1.0) {
  is_obs_op<T1> op1;
  is_obs_op<T2> op2;
  is_obs_op<T3> op3;
  add_pseudo_op<Ret> op_add(pseudo);

  auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
  times_set(X.unaryExpr(op1), YZ, ret);
  ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void XYZ_nobs(const Eigen::SparseMatrixBase<T1>& X,
              const Eigen::SparseMatrixBase<T2>& Y,
              const Eigen::MatrixBase<T3>& Z, Eigen::SparseMatrixBase<Ret>& ret,
              const typename Ret::Scalar pseudo = 1.0) {
  is_obs_op<T1> op1;
  is_obs_op<T2> op2;
  is_obs_op<T3> op3;
  add_pseudo_op<Ret> op_add(pseudo);

  auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
  times_set(X.unaryExpr(op1), YZ, ret);
  ret.derived() = ret.unaryExpr(op_add);
}

template <typename T1, typename T2, typename T3, typename Ret>
void XYZ_nobs(const Eigen::MatrixBase<T1>& X,
              const Eigen::SparseMatrixBase<T2>& Y,
              const Eigen::MatrixBase<T3>& Z, Eigen::SparseMatrixBase<Ret>& ret,
              const typename Ret::Scalar pseudo = 1.0) {
  is_obs_op<T1> op1;
  is_obs_op<T2> op2;
  is_obs_op<T3> op3;
  add_pseudo_op<Ret> op_add(pseudo);

  auto YZ = (Y.unaryExpr(op2) * Z.unaryExpr(op3)).eval();
  times_set(X.unaryExpr(op1), YZ, ret);
  ret.derived() = ret.unaryExpr(op_add);
}

////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename Ret>
void XtY_nobs(const Eigen::MatrixBase<T1>& X, const Eigen::MatrixBase<T2>& Y,
              Eigen::MatrixBase<Ret>& ret,
              const typename Ret::Scalar pseudo = 1.0) {
  XY_nobs(X.transpose(), Y, ret, pseudo);
}

template <typename T1, typename T2, typename Ret>
void XtY_nobs(const Eigen::MatrixBase<T1>& X, const Eigen::MatrixBase<T2>& Y,
              Eigen::SparseMatrixBase<Ret>& ret,
              const typename Ret::Scalar pseudo = 1.0) {
  XY_nobs(X.transpose(), Y, ret, pseudo);
}

////////////////////////////////////////////////////////////////
template <typename Derived>
void standardize(Eigen::MatrixBase<Derived>& Xraw,
                 const typename Derived::Scalar EPS = 1e-8) {
  using Index = typename Derived::Index;
  using Scalar = typename Derived::Scalar;
  using mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RowVec = typename Eigen::internal::plain_row_type<Derived>::type;

  // const Index n = Xraw.rows();
  // const Index p = Xraw.cols();
  is_obs_op<mat_t> obs_op;

  // Remove NaN
  mat_t X = Xraw.unaryExpr(obs_op);
  const RowVec num_obs = X.colwise().sum();

  // calculate statistics
  remove_missing(Xraw, X);
  const RowVec x_mean = X.colwise().sum().cwiseQuotient(num_obs);
  const RowVec x2_mean =
      X.cwiseProduct(X).colwise().sum().cwiseQuotient(num_obs);
  const RowVec x_sd = (x2_mean - x_mean.cwiseProduct(x_mean)).cwiseSqrt();

  // standardize
  for (Index j = 0; j < X.cols(); ++j) {
    const Scalar mu = x_mean(j);
    const Scalar sd = x_sd(j) + EPS;
    auto std_op = [&mu, &sd](const Scalar& x) { return (x - mu) / sd; };

    // This must be done with original data
    X.col(j) = X.col(j).unaryExpr(std_op);
  }

  Xraw.derived() = X;
}

////////////////////////////////////////////////////////////////
template <typename Derived>
void rescale(Eigen::MatrixBase<Derived>& Xraw,
             const typename Derived::Scalar EPS = 1e-8) {
  using Index = typename Derived::Index;
  using Scalar = typename Derived::Scalar;
  using mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RowVec = typename Eigen::internal::plain_row_type<Derived>::type;

  const Index n = Xraw.rows();
  const Index p = Xraw.cols();
  is_obs_op<mat_t> obs_op;

  // Remove NaN
  mat_t X = Xraw.unaryExpr(obs_op);
  const RowVec num_obs = X.colwise().sum();

  // calculate statistics
  remove_missing(Xraw, X);
  const RowVec x_mean = X.colwise().sum().cwiseQuotient(num_obs);
  const RowVec x2_mean =
      X.cwiseProduct(X).colwise().sum().cwiseQuotient(num_obs);
  const RowVec x_sd = (x2_mean - x_mean.cwiseProduct(x_mean)).cwiseSqrt();

  // standardize
  for (Index j = 0; j < X.cols(); ++j) {
    const Scalar sd = x_sd(j) + EPS;
    X.col(j) = X.col(j) / sd;
  }

  Xraw.derived() = X;
}

template <typename Derived>
void center(Eigen::MatrixBase<Derived>& Xraw) {
  // using Index = typename Derived::Index;
  using Scalar = typename Derived::Scalar;
  using mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RowVec = typename Eigen::internal::plain_row_type<Derived>::type;

  // const Index n = Xraw.rows();
  // const Index p = Xraw.cols();
  is_obs_op<mat_t> obs_op;

  // Remove NaN
  mat_t X = Xraw.unaryExpr(obs_op);
  const RowVec num_obs = X.colwise().sum();

  // calculate statistics
  remove_missing(Xraw, X);
  const RowVec x_mean = X.colwise().sum().cwiseQuotient(num_obs);

  // standardize
  X = X.rowwise() - x_mean;
  Xraw.derived() = X;
}

////////////////////////////////////////////////////////////////
template <typename Derived, typename OtherDerived>
void column_var(Eigen::MatrixBase<Derived> const& Xraw,
                Eigen::MatrixBase<OtherDerived> const& ret) {
  using Index = typename Derived::Index;
  using Scalar = typename Derived::Scalar;
  using mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using ColVec = typename Eigen::internal::plain_col_type<Derived>::type;

  Eigen::MatrixBase<OtherDerived>& R =
      const_cast<Eigen::MatrixBase<OtherDerived>&>(ret);
  const Index p = Xraw.cols();
  R.resize(p, 1);
  is_obs_op<mat_t> obs_op;

  // Remove NaN
  mat_t X = Xraw.transpose().unaryExpr(obs_op);
  const ColVec num_obs = X.rowwise().sum();

  // calculate statistics
  remove_missing(Xraw.transpose(), X);
  const ColVec x_mean = X.rowwise().sum().cwiseQuotient(num_obs);
  const ColVec x2_mean =
      X.cwiseProduct(X).rowwise().sum().cwiseQuotient(num_obs);

  R = x2_mean - x_mean.cwiseProduct(x_mean);
}

////////////////////////////////////////////////////////////////
template <typename Derived>
auto column_mean(Eigen::MatrixBase<Derived> const& Xraw) {
  using Scalar = typename Derived::Scalar;
  using mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using ColVec = typename Eigen::internal::plain_col_type<Derived>::type;

  is_obs_op<mat_t> obs_op;

  // Remove NaN
  mat_t X = Xraw.transpose().unaryExpr(obs_op);
  const ColVec num_obs = X.rowwise().sum();

  // calculate statistics
  remove_missing(Xraw.transpose(), X);
  return X.rowwise().sum().cwiseQuotient(num_obs).eval();
}

////////////////////
// randomized SVD //
////////////////////

// Implement Alg 4.4 of Halko et al. (2009)
// Modified from https://github.com/kazuotani14/RandomizedSvd

template <typename T>
class RandomizedSVD {
 public:
  RandomizedSVD(const int _max_rank, const int _iter)
      : max_rank(_max_rank), iter(_iter), U(), D(), V(), qq() {}

  using Vec = Eigen::Matrix<typename T::Scalar, Eigen::Dynamic, 1>;

  const T matrixU() const { return U; }
  const T matrixV() const { return V; }
  const Vec singularValues() const { return D; }

  template <typename Derived>
  void compute(Eigen::MatrixBase<Derived> const& X) {
    using Index    = typename Derived::Index;
    const Index nr = X.rows();
    const Index nc = X.cols();

    int rank       = std::min(nr, nc);
    int oversample = 0;

    if (max_rank > 0 && max_rank < rank) {
      rank       = max_rank;
      oversample = 5;
    }

    ASSERT(rank > 0, "Must be at least rank = 1");

    qq.resize(nr, rank + oversample);
    qq.setZero();
    rand_subspace_iteration(X, rank + oversample);

    qq.conservativeResize(Eigen::NoChange, rank);

    T B(rank, nc);

    if (verbose) TLOG("Final svd on [" << B.rows() << "x" << B.cols() << "]");

    B = qq.transpose() * X;

    // Eigen::JacobiSVD<T> svd;
    Eigen::BDCSVD<T> svd;
    svd.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    if (verbose) TLOG("Construct U, D, V");

    U.resize(nr, rank);
    V.resize(nc, rank);
    D.resize(rank, 1);

    U = (qq * svd.matrixU()).block(0, 0, nr, rank);
    V = svd.matrixV().block(0, 0, nc, rank);
    D = svd.singularValues().head(rank);

    if (verbose) TLOG("Done: RandomizedSVD.compute()");
  }

  void set_verbose() { verbose = true; }

  const int max_rank;
  const int iter;

 private:
  T U;
  Vec D;
  T V;
  T qq;
  bool verbose;

  template <typename Derived>
  void rand_subspace_iteration(Eigen::MatrixBase<Derived> const& X, const int rank_and_oversample) {
    using Index    = typename Derived::Index;
    const Index nr = X.rows();
    const Index nc = X.cols();

    T L(nr, rank_and_oversample);
    T Q = T::Random(nc, rank_and_oversample);

    // Use LU normalization since QR is too slow in Eigen

    Eigen::FullPivLU<T> lu1(nr, rank_and_oversample);
    Eigen::FullPivLU<T> lu2(nc, nr);

    if (verbose) TLOG("Find Q in randomized svd...");

    for (int i = 0; i < iter; ++i) {
      if (verbose) TLOG("Start : LU iteration " << (i + 1));

      lu1.compute(X * Q);
      L.setIdentity();
      L.block(0, 0, nr, rank_and_oversample).template triangularView<Eigen::StrictlyLower>() =
          lu1.matrixLU();

      lu2.compute(X.transpose() * L);
      Q.setIdentity();
      Q.block(0, 0, nc, rank_and_oversample).template triangularView<Eigen::StrictlyLower>() =
          lu2.matrixLU();

      if (verbose) TLOG("Done : LU iteration " << (i + 1));
    }

    Eigen::ColPivHouseholderQR<T> qr(X * Q);
    qq = qr.householderQ() * T::Identity(nr, rank_and_oversample);
    if (verbose) TLOG("Found Q [" << qq.rows() << "x" << qq.cols() << "]");
  }
};

#endif
