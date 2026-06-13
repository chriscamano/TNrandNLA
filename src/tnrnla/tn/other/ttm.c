/* ttm.c
   Compiled as C++ by CMake even though it has a .c name.
*/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <mkl.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>

namespace py = pybind11;

static CBLAS_TRANSPOSE parse_trans(const std::string& t) {
    if (t == "N" || t == "n") return CblasNoTrans;
    if (t == "T" || t == "t") return CblasTrans;
    if (t == "C" || t == "c") return CblasConjTrans;
    throw std::runtime_error("trans must be one of N, T, C");
}

static void require_ndim(const py::buffer_info& info, int ndim) {
    if (info.ndim != ndim) {
        throw std::runtime_error("unexpected array rank");
    }
}

static py::array_t<double> py_dgemm(
    py::array_t<double, py::array::c_style | py::array::forcecast> A,
    py::array_t<double, py::array::c_style | py::array::forcecast> B,
    std::string transA,
    std::string transB,
    double alpha,
    double beta
) {
    auto a = A.request();
    auto b = B.request();
    require_ndim(a, 2);
    require_ndim(b, 2);

    const CBLAS_TRANSPOSE ta = parse_trans(transA);
    const CBLAS_TRANSPOSE tb = parse_trans(transB);

    const MKL_INT a_rows = static_cast<MKL_INT>(a.shape[0]);
    const MKL_INT a_cols = static_cast<MKL_INT>(a.shape[1]);
    const MKL_INT b_rows = static_cast<MKL_INT>(b.shape[0]);
    const MKL_INT b_cols = static_cast<MKL_INT>(b.shape[1]);

    const MKL_INT m  = (ta == CblasNoTrans) ? a_rows : a_cols;
    const MKL_INT kA = (ta == CblasNoTrans) ? a_cols : a_rows;
    const MKL_INT kB = (tb == CblasNoTrans) ? b_rows : b_cols;
    const MKL_INT n  = (tb == CblasNoTrans) ? b_cols : b_rows;

    if (kA != kB) {
        throw std::runtime_error("inner dimensions do not match");
    }

    py::array_t<double> C({static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(n)});
    auto c = C.request();

    const MKL_INT lda = (ta == CblasNoTrans) ? a_cols : a_rows;
    const MKL_INT ldb = (tb == CblasNoTrans) ? b_cols : b_rows;
    const MKL_INT ldc = n;

    const double* Ap = static_cast<const double*>(a.ptr);
    const double* Bp = static_cast<const double*>(b.ptr);
    double* Cp = static_cast<double*>(c.ptr);

    cblas_dgemm(
        CblasRowMajor,
        ta,
        tb,
        m,
        n,
        kA,
        alpha,
        Ap,
        lda,
        Bp,
        ldb,
        beta,
        Cp,
        ldc
    );

    return C;
}

static py::array_t<double> py_dgemm_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> A_batch,
    py::array_t<double, py::array::c_style | py::array::forcecast> B_batch,
    double alpha,
    double beta
) {
    auto a = A_batch.request();
    auto b = B_batch.request();
    require_ndim(a, 3);
    require_ndim(b, 3);

    const MKL_INT batch = static_cast<MKL_INT>(a.shape[0]);
    const MKL_INT m = static_cast<MKL_INT>(a.shape[1]);
    const MKL_INT k = static_cast<MKL_INT>(a.shape[2]);

    const MKL_INT batch2 = static_cast<MKL_INT>(b.shape[0]);
    const MKL_INT k2 = static_cast<MKL_INT>(b.shape[1]);
    const MKL_INT n = static_cast<MKL_INT>(b.shape[2]);

    if (batch2 != batch) {
        throw std::runtime_error("batch dimensions do not match");
    }
    if (k2 != k) {
        throw std::runtime_error("inner dimensions do not match");
    }

    py::array_t<double> C_batch(
        {static_cast<py::ssize_t>(batch), static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(n)}
    );
    auto c = C_batch.request();

    const double* Ap0 = static_cast<const double*>(a.ptr);
    const double* Bp0 = static_cast<const double*>(b.ptr);
    double* Cp0 = static_cast<double*>(c.ptr);

    const std::int64_t strideA = static_cast<std::int64_t>(m) * static_cast<std::int64_t>(k);
    const std::int64_t strideB = static_cast<std::int64_t>(k) * static_cast<std::int64_t>(n);
    const std::int64_t strideC = static_cast<std::int64_t>(m) * static_cast<std::int64_t>(n);

    std::vector<const double*> A_ptrs(static_cast<std::size_t>(batch));
    std::vector<const double*> B_ptrs(static_cast<std::size_t>(batch));
    std::vector<double*> C_ptrs(static_cast<std::size_t>(batch));

    for (MKL_INT i = 0; i < batch; ++i) {
        A_ptrs[static_cast<std::size_t>(i)] = Ap0 + static_cast<std::int64_t>(i) * strideA;
        B_ptrs[static_cast<std::size_t>(i)] = Bp0 + static_cast<std::int64_t>(i) * strideB;
        C_ptrs[static_cast<std::size_t>(i)] = Cp0 + static_cast<std::int64_t>(i) * strideC;
    }

    const MKL_INT group_count = 1;
    MKL_INT group_size[1] = { batch };

    CBLAS_TRANSPOSE transA[1] = { CblasNoTrans };
    CBLAS_TRANSPOSE transB[1] = { CblasNoTrans };

    MKL_INT m_arr[1] = { m };
    MKL_INT n_arr[1] = { n };
    MKL_INT k_arr[1] = { k };

    double alpha_arr[1] = { alpha };
    double beta_arr[1] = { beta };

    MKL_INT lda_arr[1] = { k };
    MKL_INT ldb_arr[1] = { n };
    MKL_INT ldc_arr[1] = { n };

    cblas_dgemm_batch(
        CblasRowMajor,
        transA,
        transB,
        m_arr,
        n_arr,
        k_arr,
        alpha_arr,
        A_ptrs.data(),
        lda_arr,
        B_ptrs.data(),
        ldb_arr,
        beta_arr,
        C_ptrs.data(),
        ldc_arr,
        group_count,
        group_size
    );

    return C_batch;
}

PYBIND11_MODULE(ttm, m) {
    m.doc() = "MKL DGEMM wrappers via pybind11";

    m.def(
        "dgemm",
        &py_dgemm,
        py::arg("A"),
        py::arg("B"),
        py::arg("transA") = "N",
        py::arg("transB") = "N",
        py::arg("alpha") = 1.0,
        py::arg("beta") = 0.0
    );

    m.def(
        "dgemm_batch",
        &py_dgemm_batch,
        py::arg("A_batch"),
        py::arg("B_batch"),
        py::arg("alpha") = 1.0,
        py::arg("beta") = 0.0
    );
}