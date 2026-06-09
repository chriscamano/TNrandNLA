// libincrementalqr.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <mkl_lapacke.h>
#include <mkl_cblas.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cstring>

namespace py = pybind11;

// -------------------- Workspace reuse (per-thread) --------------------
static thread_local std::vector<double> tls_work;

static inline void ensure_work_size(std::size_t need) {
    if (tls_work.size() < need) tls_work.resize(need);
}

// -------------------- Array checks --------------------
static inline double* require_f64_f2d(py::array_t<double>& A, int expected_rows, int expected_cols_min) {
    if (A.ndim() != 2) throw std::runtime_error("A must be 2D");
    if (!A.writeable()) throw std::runtime_error("A must be writable");
    if (!(A.flags() & py::array::f_style)) throw std::runtime_error("A must be Fortran-contiguous (order='F')");
    if (A.shape(0) != expected_rows || A.shape(1) < expected_cols_min) {
        throw std::runtime_error("A has wrong shape");
    }
    if (A.strides(0) != (py::ssize_t)sizeof(double)) {
        throw std::runtime_error("A must have unit stride in the first dimension (Fortran layout)");
    }
    return static_cast<double*>(A.mutable_data());
}

static inline double* require_f64_1d(py::array_t<double>& v, int expected_len_min) {
    if (v.ndim() != 1) throw std::runtime_error("array must be 1D");
    if (!v.writeable()) throw std::runtime_error("array must be writable");
    if (v.shape(0) < expected_len_min) throw std::runtime_error("array too small");
    if (v.strides(0) != (py::ssize_t)sizeof(double)) {
        throw std::runtime_error("1D array must be contiguous (stride == sizeof(double))");
    }
    return static_cast<double*>(v.mutable_data());
}

// -------------------- Stats helpers --------------------
static inline double safe_inv(double x) {
    const double tiny = 1e-300;
    if (!(x > tiny) || !std::isfinite(x)) x = tiny;
    return 1.0 / x;
}

static inline bool any_nonfinite_upper(const double* A, int m, int n) {
    for (int j = 0; j < n; ++j) {
        const double* col = A + (std::size_t)m * j;
        for (int i = 0; i <= j; ++i) {
            if (!std::isfinite(col[i])) return true;
        }
    }
    return false;
}

// Recompute sum_inv from row_norm2 in O(n). This is the key stability fix.
static inline double recompute_sum_inv_from_row_norm2(const double* row_norm2, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += safe_inv(row_norm2[i]);
    return std::isfinite(acc) ? acc : std::numeric_limits<double>::infinity();
}

// Initialize row_norm2[i] = ||U_{i,:}||^2 for U = triu(A[:n,:n]).
static inline void stats_init_from_U(const double* A, int m, int n, double* row_norm2, double* sum_inv) {
    if (n <= 0) {
        sum_inv[0] = std::numeric_limits<double>::infinity();
        return;
    }
    if (any_nonfinite_upper(A, m, n)) {
        sum_inv[0] = std::numeric_limits<double>::infinity();
        std::fill(row_norm2, row_norm2 + n, 0.0);
        return;
    }

    std::fill(row_norm2, row_norm2 + n, 0.0);

    for (int j = 0; j < n; ++j) {
        const double* colptr = A + (std::size_t)m * j;
        for (int i = 0; i <= j; ++i) {
            const double v = colptr[i];
            row_norm2[i] += v * v;
        }
    }

    sum_inv[0] = recompute_sum_inv_from_row_norm2(row_norm2, n);
}

// Update stats after expanding from old_n to new_n=old_n+k.
// Assumes A already contains U = triu(inv(R)) in the leading new_n x new_n block.
static inline void stats_update_after_addcols(const double* A, int m, int old_n, int k,
                                             double* row_norm2, double* sum_inv) {
    const int new_n = old_n + k;
    if (new_n <= 0) {
        sum_inv[0] = std::numeric_limits<double>::infinity();
        return;
    }
    if (any_nonfinite_upper(A, m, new_n)) {
        sum_inv[0] = std::numeric_limits<double>::infinity();
        return;
    }

    // Update existing rows i < old_n by adding the contribution from new columns.
    for (int j = old_n; j < new_n; ++j) {
        const double* colptr = A + (std::size_t)m * j;
        for (int i = 0; i < old_n; ++i) {
            const double v = colptr[i];
            row_norm2[i] += v * v;
        }
    }

    // New rows i in [old_n, new_n): compute their row norms over j=i..new_n-1.
    for (int i = old_n; i < new_n; ++i) {
        double s = 0.0;
        for (int j = i; j < new_n; ++j) {
            const double v = A[i + (std::size_t)m * j];
            s += v * v;
        }
        row_norm2[i] = s;
    }

    // Stable O(n) recomputation avoids cancellation bugs.
    sum_inv[0] = recompute_sum_inv_from_row_norm2(row_norm2, new_n);
}

// -------------------- Core routines (MKL/LAPACKE) --------------------
static inline int to_lwork(double x) {
    if (!std::isfinite(x) || x < 1.0) return 1;
    return (int)std::llround(x);
}

void setup(py::array_t<double> A_input, py::array_t<double> tau_input, int m, int n) {
    auto A_arr   = py::array_t<double>(A_input);
    auto tau_arr = py::array_t<double>(tau_input);

    double* A   = require_f64_f2d(A_arr, m, n);
    double* tau = require_f64_1d(tau_arr, n);

    double workq = 0.0;
    int info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, A, m, tau, &workq, -1);
    if (info != 0) throw std::runtime_error("dgeqrf_work query failed");

    const int lwork = to_lwork(workq);
    ensure_work_size((std::size_t)lwork);

    info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, A, m, tau, tls_work.data(), lwork);
    if (info != 0) throw std::runtime_error("dgeqrf_work failed");

    info = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', n, A, m);
    if (info != 0) throw std::runtime_error("dtrtri failed");
}

void setup_stats(py::array_t<double> A_input,
                 py::array_t<double> tau_input,
                 int m, int n,
                 py::array_t<double> row_norm2_input,
                 py::array_t<double> sum_inv_input) {
    auto A_arr   = py::array_t<double>(A_input);
    auto tau_arr = py::array_t<double>(tau_input);
    auto rn_arr  = py::array_t<double>(row_norm2_input);
    auto si_arr  = py::array_t<double>(sum_inv_input);

    double* A         = require_f64_f2d(A_arr, m, n);
    double* tau       = require_f64_1d(tau_arr, n);
    double* row_norm2 = require_f64_1d(rn_arr, n);
    double* sum_inv   = require_f64_1d(si_arr, 1);

    sum_inv[0] = std::numeric_limits<double>::infinity();

    double workq = 0.0;
    int info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, A, m, tau, &workq, -1);
    if (info != 0) throw std::runtime_error("dgeqrf_work query failed");

    const int lwork = to_lwork(workq);
    ensure_work_size((std::size_t)lwork);

    info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, A, m, tau, tls_work.data(), lwork);
    if (info != 0) throw std::runtime_error("dgeqrf_work failed");

    info = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', n, A, m);
    if (info != 0) {
        sum_inv[0] = std::numeric_limits<double>::infinity();
        return;
    }

    stats_init_from_U(A, m, n, row_norm2, sum_inv);
}

void add_cols(py::array_t<double> A_input, py::array_t<double> tau_input, int m, int n, int k) {
    auto A_arr   = py::array_t<double>(A_input);
    auto tau_arr = py::array_t<double>(tau_input);

    double* A   = require_f64_f2d(A_arr, m, n + k);
    double* tau = require_f64_1d(tau_arr, n + k);

    double* new_data = A + (std::size_t)m * n;
    double* new_tau  = tau + n;

    double w1q = 0.0;
    int info = LAPACKE_dormqr_work(LAPACK_COL_MAJOR, 'L', 'T',
                                  m, k, n, A, m, tau,
                                  new_data, m, &w1q, -1);
    if (info != 0) throw std::runtime_error("dormqr_work query failed");
    const int lwork1 = to_lwork(w1q);

    const int mn = m - n;
    double w2q = 0.0;
    info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, mn, k,
                              new_data + n, m, new_tau,
                              &w2q, -1);
    if (info != 0) throw std::runtime_error("dgeqrf_work query failed");
    const int lwork2 = to_lwork(w2q);

    ensure_work_size((std::size_t)std::max(1, std::max(lwork1, lwork2)));

    info = LAPACKE_dormqr_work(LAPACK_COL_MAJOR, 'L', 'T',
                              m, k, n, A, m, tau,
                              new_data, m, tls_work.data(), lwork1);
    if (info != 0) throw std::runtime_error("dormqr_work failed");

    info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, mn, k,
                              new_data + n, m, new_tau,
                              tls_work.data(), lwork2);
    if (info != 0) throw std::runtime_error("dgeqrf_work failed (bottom block)");

    info = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', k, new_data + n, m);
    if (info != 0) throw std::runtime_error("dtrtri failed (R22)");

    cblas_dtrmm(CblasColMajor, CblasLeft,  CblasUpper, CblasNoTrans, CblasNonUnit,
                n, k, -1.0, A, m, new_data, m);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                n, k,  1.0, new_data + n, m, new_data, m);
}

void add_cols_stats(py::array_t<double> A_input,
                    py::array_t<double> tau_input,
                    int m, int n, int k,
                    py::array_t<double> row_norm2_input,
                    py::array_t<double> sum_inv_input) {
    auto A_arr   = py::array_t<double>(A_input);
    auto tau_arr = py::array_t<double>(tau_input);
    auto rn_arr  = py::array_t<double>(row_norm2_input);
    auto si_arr  = py::array_t<double>(sum_inv_input);

    double* A         = require_f64_f2d(A_arr, m, n + k);
    double* tau       = require_f64_1d(tau_arr, n + k);
    double* row_norm2 = require_f64_1d(rn_arr, n + k);
    double* sum_inv   = require_f64_1d(si_arr, 1);

    double* new_data = A + (std::size_t)m * n;
    double* new_tau  = tau + n;

    double w1q = 0.0;
    int info = LAPACKE_dormqr_work(LAPACK_COL_MAJOR, 'L', 'T',
                                  m, k, n, A, m, tau,
                                  new_data, m, &w1q, -1);
    if (info != 0) throw std::runtime_error("dormqr_work query failed");
    const int lwork1 = to_lwork(w1q);

    const int mn = m - n;
    double w2q = 0.0;
    info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, mn, k,
                              new_data + n, m, new_tau,
                              &w2q, -1);
    if (info != 0) throw std::runtime_error("dgeqrf_work query failed");
    const int lwork2 = to_lwork(w2q);

    ensure_work_size((std::size_t)std::max(1, std::max(lwork1, lwork2)));

    info = LAPACKE_dormqr_work(LAPACK_COL_MAJOR, 'L', 'T',
                              m, k, n, A, m, tau,
                              new_data, m, tls_work.data(), lwork1);
    if (info != 0) throw std::runtime_error("dormqr_work failed");

    info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, mn, k,
                              new_data + n, m, new_tau,
                              tls_work.data(), lwork2);
    if (info != 0) throw std::runtime_error("dgeqrf_work failed (bottom block)");

    info = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', k, new_data + n, m);
    if (info != 0) {
        sum_inv[0] = std::numeric_limits<double>::infinity();
        return;
    }

    cblas_dtrmm(CblasColMajor, CblasLeft,  CblasUpper, CblasNoTrans, CblasNonUnit,
                n, k, -1.0, A, m, new_data, m);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                n, k,  1.0, new_data + n, m, new_data, m);

    // Always update stats; sum_inv is recomputed stably in O(n).
    stats_update_after_addcols(A, m, n, k, row_norm2, sum_inv);
}

void extract_q(py::array_t<double> A_input, py::array_t<double> tau_input, int m, int n) {
    auto A_arr   = py::array_t<double>(A_input);
    auto tau_arr = py::array_t<double>(tau_input);

    double* A   = require_f64_f2d(A_arr, m, n);
    double* tau = require_f64_1d(tau_arr, n);

    double wq = 0.0;
    int info = LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, m, n, n, A, m, tau, &wq, -1);
    if (info != 0) throw std::runtime_error("dorgqr_work query failed");

    const int lwork = to_lwork(wq);
    ensure_work_size((std::size_t)lwork);

    info = LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, m, n, n, A, m, tau, tls_work.data(), lwork);
    if (info != 0) throw std::runtime_error("dorgqr_work failed");
}
// libincrementalqr.cpp  (ADD THIS: apply_q)
// Put this near the other "Core routines" functions (before PYBIND11_MODULE)

py::array_t<double> apply_q(py::array_t<double> A_input,
                            py::array_t<double> tau_input,
                            int m, int n,
                            py::array_t<double> C_input,
                            const std::string& side_s,
                            const std::string& trans_s) {
    auto A_arr   = py::array_t<double>(A_input);
    auto tau_arr = py::array_t<double>(tau_input);
    auto C_arr   = py::array_t<double>(C_input);

    double* A   = require_f64_f2d(A_arr, m, n);
    double* tau = require_f64_1d(tau_arr, n);

    if (side_s.empty() || trans_s.empty()) throw std::runtime_error("side/trans empty");
    char side  = (char)std::toupper((unsigned char)side_s[0]);   // 'L' or 'R'
    char trans = (char)std::toupper((unsigned char)trans_s[0]);  // 'N','T','C'
    if (side != 'L' && side != 'R') throw std::runtime_error("side must be 'L' or 'R'");
    if (trans != 'N' && trans != 'T' && trans != 'C') throw std::runtime_error("trans must be 'N','T','C'");
    if (trans == 'C') trans = 'T';  // real QR path only

    auto Cbuf = C_arr.request();
    if (Cbuf.ndim != 2) throw std::runtime_error("C must be 2D");
    if (Cbuf.format != py::format_descriptor<double>::format()) throw std::runtime_error("C must be float64");

    const int Cm = (int)Cbuf.shape[0];
    const int Cn = (int)Cbuf.shape[1];

    if (side == 'L') {
        if (Cm != m) throw std::runtime_error("apply_q(side='L'): C.shape[0] must equal m");
    } else {
        if (Cn != m) throw std::runtime_error("apply_q(side='R'): C.shape[1] must equal m");
    }

    // allocate Fortran-contiguous output
    py::array_t<double> out(
        py::array::ShapeContainer{Cm, Cn},
        py::array::StridesContainer{(py::ssize_t)sizeof(double),
                                    (py::ssize_t)sizeof(double) * (py::ssize_t)Cm}
    );
    auto obuf = out.request();
    double* Cout = static_cast<double*>(obuf.ptr);

    // elementwise copy into Fortran output (handles any input strides safely)
    const char* src = static_cast<const char*>(Cbuf.ptr);
    const py::ssize_t s0 = Cbuf.strides[0];
    const py::ssize_t s1 = Cbuf.strides[1];
    for (int j = 0; j < Cn; ++j) {
        for (int i = 0; i < Cm; ++i) {
            const double* p = reinterpret_cast<const double*>(src + i * s0 + j * s1);
            Cout[i + (std::size_t)Cm * (std::size_t)j] = *p;
        }
    }

    // dormqr: apply Q from GEQRF(A,tau) to Cout in-place
    double wq = 0.0;
    int info = LAPACKE_dormqr_work(LAPACK_COL_MAJOR,
                                  side, trans,
                                  Cm, Cn, n,
                                  A, m, tau,
                                  Cout, Cm,
                                  &wq, -1);
    if (info != 0) throw std::runtime_error("dormqr_work query failed");

    const int lwork = to_lwork(wq);
    ensure_work_size((std::size_t)std::max(1, lwork));

    info = LAPACKE_dormqr_work(LAPACK_COL_MAJOR,
                              side, trans,
                              Cm, Cn, n,
                              A, m, tau,
                              Cout, Cm,
                              tls_work.data(), lwork);
    if (info != 0) throw std::runtime_error("dormqr_work failed");

    return out;
}

double get_error_estimate(py::array_t<double> A_input, int m, int n) {
    auto buf = A_input.request();
    const double* A = static_cast<const double*>(buf.ptr);

    if (n <= 0) return std::numeric_limits<double>::infinity();

    double acc = 0.0;
    for (int row = 0; row < n; ++row) {
        double rownormsq = 0.0;
        for (int col = row; col < n; ++col) {
            const double v = A[row + (std::size_t)m * col];
            if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
            rownormsq += v * v;
        }
        if (!(rownormsq > 0.0) || !std::isfinite(rownormsq))
            return std::numeric_limits<double>::infinity();
        acc += 1.0 / rownormsq;
    }
    const double out = std::sqrt(acc / (double)n);
    return std::isfinite(out) ? out : std::numeric_limits<double>::infinity();
}
#include <unordered_map>
#include <cstdint>

// cached lwork for dorgqr (per-thread)
static thread_local std::unordered_map<std::uint64_t, int> tls_dorgqr_lwork;

static inline std::uint64_t key_mn(int m, int n) {
    return (std::uint64_t)(std::uint32_t)m << 32 | (std::uint32_t)n;
}

void form_q_cached(py::array_t<double> A_input, py::array_t<double> tau_input, int m, int n) {
    auto A_arr   = py::array_t<double>(A_input);
    auto tau_arr = py::array_t<double>(tau_input);

    double* A   = require_f64_f2d(A_arr, m, n);
    double* tau = require_f64_1d(tau_arr, n);

    const std::uint64_t k = key_mn(m, n);

    int lwork = 0;
    auto it = tls_dorgqr_lwork.find(k);
    if (it != tls_dorgqr_lwork.end()) {
        lwork = it->second;
    } else {
        double wq = 0.0;
        int info = LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, m, n, n, A, m, tau, &wq, -1);
        if (info != 0) throw std::runtime_error("dorgqr_work query failed");
        lwork = to_lwork(wq);
        if (lwork < 1) lwork = 1;
        tls_dorgqr_lwork.emplace(k, lwork);
    }

    ensure_work_size((std::size_t)lwork);
    int info = LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, m, n, n, A, m, tau, tls_work.data(), lwork);
    if (info != 0) throw std::runtime_error("dorgqr_work failed");
}
PYBIND11_MODULE(libincrementalqr, m) {
    m.def("setup", &setup, "Compute in-place QR and invert R (stores inv(R) in upper triangle)");
    m.def("add_cols", &add_cols, "Append columns and update QR / inv(R)");
    m.def("extract_q", &extract_q, "Extract Q factor into A in-place");
    m.def("get_error_estimate", &get_error_estimate, "Slow scan error estimate from inv(R)");

    m.def("setup_stats", &setup_stats, "setup + initialize (row_norm2, sum_inv)");
    m.def("add_cols_stats", &add_cols_stats, "add_cols + update (row_norm2, sum_inv)");
}
