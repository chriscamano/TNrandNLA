
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <mkl.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

static inline void require_ndim(const py::buffer_info& info, int ndim) {
    if (info.ndim != ndim) throw std::runtime_error("unexpected ndim");
}

static inline void require_shape_eq(std::int64_t a, std::int64_t b, const char* msg) {
    if (a != b) throw std::runtime_error(msg);
}

static inline MKL_INT as_mkl_int(std::int64_t x) {
    if (x < 0) throw std::runtime_error("negative dimension");
    if (x > static_cast<std::int64_t>(std::numeric_limits<MKL_INT>::max())) {
        throw std::runtime_error("dimension too large");
    }
    return static_cast<MKL_INT>(x);
}

/* -------------------------------- precontract WR -------------------------------- */

static py::array_t<double> precontract_WRD(
    py::array_t<double, py::array::c_style | py::array::forcecast> W,
    py::array_t<double, py::array::c_style | py::array::forcecast> R
) {
    auto Wb = W.request();
    auto Rb = R.request();

    require_ndim(Wb, 4);
    require_ndim(Rb, 3);

    const MKL_INT wl = as_mkl_int(Wb.shape[0]);
    const MKL_INT d  = as_mkl_int(Wb.shape[1]);
    const MKL_INT wr = as_mkl_int(Wb.shape[2]);
    const MKL_INT d2 = as_mkl_int(Wb.shape[3]);
    require_shape_eq(d2, d, "W must have shape (wl, d, wr, d)");

    const MKL_INT chiR  = as_mkl_int(Rb.shape[0]);
    const MKL_INT wr2   = as_mkl_int(Rb.shape[1]);
    const MKL_INT chiR2 = as_mkl_int(Rb.shape[2]);
    require_shape_eq(wr2, wr, "R second dim must match wr");
    require_shape_eq(chiR2, chiR, "R must have shape (chiR, wr, chiR)");

    py::array_t<double> WR({(py::ssize_t)wl, (py::ssize_t)d, (py::ssize_t)d, (py::ssize_t)chiR, (py::ssize_t)chiR});
    py::array_t<double> Wmat({(py::ssize_t)(wl * d * d), (py::ssize_t)wr});
    py::array_t<double> Rmat({(py::ssize_t)wr, (py::ssize_t)(chiR * chiR)});

    const double* Wp = static_cast<const double*>(Wb.ptr);
    const double* Rp = static_cast<const double*>(Rb.ptr);

    auto Wmb = Wmat.request();
    auto Rmb = Rmat.request();
    auto WRb = WR.request();

    double* Wm = static_cast<double*>(Wmb.ptr);
    double* Rm = static_cast<double*>(Rmb.ptr);
    double* Out = static_cast<double*>(WRb.ptr);

    {
        const std::int64_t wl64 = (std::int64_t)wl;
        const std::int64_t d64  = (std::int64_t)d;
        const std::int64_t wr64 = (std::int64_t)wr;

        const std::int64_t W_stride_l = d64 * wr64 * d64;
        const std::int64_t W_stride_s = wr64 * d64;
        const std::int64_t W_stride_r = d64;

        for (std::int64_t l = 0; l < wl64; ++l) {
            for (std::int64_t s = 0; s < d64; ++s) {
                for (std::int64_t sp = 0; sp < d64; ++sp) {
                    const std::int64_t row = ((l * d64 + s) * d64 + sp);
                    double* dst = Wm + row * wr64;
                    for (std::int64_t r = 0; r < wr64; ++r) {
                        dst[r] = Wp[l * W_stride_l + s * W_stride_s + r * W_stride_r + sp];
                    }
                }
            }
        }
    }

    {
        const std::int64_t chiR64 = (std::int64_t)chiR;
        const std::int64_t wr64   = (std::int64_t)wr;
        const std::int64_t R_stride_bp = wr64 * chiR64;
        const std::int64_t R_stride_r  = chiR64;

        for (std::int64_t r = 0; r < wr64; ++r) {
            double* row = Rm + r * (chiR64 * chiR64);
            for (std::int64_t bp = 0; bp < chiR64; ++bp) {
                const double* src = Rp + bp * R_stride_bp + r * R_stride_r;
                std::memcpy(row + bp * chiR64, src, sizeof(double) * (std::size_t)chiR64);
            }
        }
    }

    {
        py::gil_scoped_release rel;

        const MKL_INT m = wl * d * d;
        const MKL_INT k = wr;
        const MKL_INT n = chiR * chiR;

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m,
            n,
            k,
            1.0,
            Wm,
            k,
            Rm,
            n,
            0.0,
            Out,
            n
        );
    }

    return WR;
}

static py::array_t<std::complex<double>> precontract_WRZ(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> W,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> R
) {
    auto Wb = W.request();
    auto Rb = R.request();

    require_ndim(Wb, 4);
    require_ndim(Rb, 3);

    const MKL_INT wl = as_mkl_int(Wb.shape[0]);
    const MKL_INT d  = as_mkl_int(Wb.shape[1]);
    const MKL_INT wr = as_mkl_int(Wb.shape[2]);
    const MKL_INT d2 = as_mkl_int(Wb.shape[3]);
    require_shape_eq(d2, d, "W must have shape (wl, d, wr, d)");

    const MKL_INT chiR  = as_mkl_int(Rb.shape[0]);
    const MKL_INT wr2   = as_mkl_int(Rb.shape[1]);
    const MKL_INT chiR2 = as_mkl_int(Rb.shape[2]);
    require_shape_eq(wr2, wr, "R second dim must match wr");
    require_shape_eq(chiR2, chiR, "R must have shape (chiR, wr, chiR)");

    py::array_t<std::complex<double>> WR({(py::ssize_t)wl, (py::ssize_t)d, (py::ssize_t)d, (py::ssize_t)chiR, (py::ssize_t)chiR});
    py::array_t<std::complex<double>> Wmat({(py::ssize_t)(wl * d * d), (py::ssize_t)wr});
    py::array_t<std::complex<double>> Rmat({(py::ssize_t)wr, (py::ssize_t)(chiR * chiR)});

    const MKL_Complex16* Wp = static_cast<const MKL_Complex16*>(Wb.ptr);
    const MKL_Complex16* Rp = static_cast<const MKL_Complex16*>(Rb.ptr);

    auto Wmb = Wmat.request();
    auto Rmb = Rmat.request();
    auto WRb = WR.request();

    MKL_Complex16* Wm = static_cast<MKL_Complex16*>(Wmb.ptr);
    MKL_Complex16* Rm = static_cast<MKL_Complex16*>(Rmb.ptr);
    MKL_Complex16* Out = static_cast<MKL_Complex16*>(WRb.ptr);

    {
        const std::int64_t wl64 = (std::int64_t)wl;
        const std::int64_t d64  = (std::int64_t)d;
        const std::int64_t wr64 = (std::int64_t)wr;

        const std::int64_t W_stride_l = d64 * wr64 * d64;
        const std::int64_t W_stride_s = wr64 * d64;
        const std::int64_t W_stride_r = d64;

        for (std::int64_t l = 0; l < wl64; ++l) {
            for (std::int64_t s = 0; s < d64; ++s) {
                for (std::int64_t sp = 0; sp < d64; ++sp) {
                    const std::int64_t row = ((l * d64 + s) * d64 + sp);
                    MKL_Complex16* dst = Wm + row * wr64;
                    for (std::int64_t r = 0; r < wr64; ++r) {
                        dst[r] = Wp[l * W_stride_l + s * W_stride_s + r * W_stride_r + sp];
                    }
                }
            }
        }
    }

    {
        const std::int64_t chiR64 = (std::int64_t)chiR;
        const std::int64_t wr64   = (std::int64_t)wr;
        const std::int64_t R_stride_bp = wr64 * chiR64;
        const std::int64_t R_stride_r  = chiR64;

        for (std::int64_t r = 0; r < wr64; ++r) {
            MKL_Complex16* row = Rm + r * (chiR64 * chiR64);
            for (std::int64_t bp = 0; bp < chiR64; ++bp) {
                const MKL_Complex16* src = Rp + bp * R_stride_bp + r * R_stride_r;
                std::memcpy(row + bp * chiR64, src, sizeof(MKL_Complex16) * (std::size_t)chiR64);
            }
        }
    }

    {
        py::gil_scoped_release rel;

        const MKL_INT m = wl * d * d;
        const MKL_INT k = wr;
        const MKL_INT n = chiR * chiR;

        const MKL_Complex16 alpha = {1.0, 0.0};
        const MKL_Complex16 beta  = {0.0, 0.0};

        cblas_zgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m,
            n,
            k,
            &alpha,
            Wm,
            k,
            Rm,
            n,
            &beta,
            Out,
            n
        );
    }

    return WR;
}

/* -------------------------------- precontract LW -------------------------------- */

static py::array_t<double> precontract_LWD(
    py::array_t<double, py::array::c_style | py::array::forcecast> L,
    py::array_t<double, py::array::c_style | py::array::forcecast> W
) {
    auto Lb = L.request();
    auto Wb = W.request();

    require_ndim(Lb, 3);
    require_ndim(Wb, 4);

    const MKL_INT chiL = as_mkl_int(Lb.shape[0]);
    const MKL_INT wl   = as_mkl_int(Lb.shape[1]);
    const MKL_INT chiL2= as_mkl_int(Lb.shape[2]);
    require_shape_eq(chiL2, chiL, "L must have shape (chiL, wl, chiL)");

    const MKL_INT wl2 = as_mkl_int(Wb.shape[0]);
    const MKL_INT d   = as_mkl_int(Wb.shape[1]);
    const MKL_INT wr  = as_mkl_int(Wb.shape[2]);
    const MKL_INT d2  = as_mkl_int(Wb.shape[3]);
    require_shape_eq(wl2, wl, "W first dim must match wl");
    require_shape_eq(d2, d, "W must have shape (wl, d, wr, d)");

    const double* Lp = static_cast<const double*>(Lb.ptr);
    const double* Wp = static_cast<const double*>(Wb.ptr);

    py::array_t<double> Lflat({(py::ssize_t)(chiL * chiL), (py::ssize_t)wl});
    py::array_t<double> Wlr({(py::ssize_t)wl, (py::ssize_t)(d * wr * d)});

    auto Lfb = Lflat.request();
    auto Wlb = Wlr.request();

    double* Lf = static_cast<double*>(Lfb.ptr);
    double* Wl = static_cast<double*>(Wlb.ptr);

    {
        const std::int64_t chiL64 = (std::int64_t)chiL;
        const std::int64_t wl64   = (std::int64_t)wl;

        const std::int64_t stride_a = wl64 * chiL64;
        const std::int64_t stride_l = chiL64;

        for (std::int64_t a = 0; a < chiL64; ++a) {
            for (std::int64_t ap = 0; ap < chiL64; ++ap) {
                const std::int64_t row = a * chiL64 + ap;
                double* dst = Lf + row * wl64;
                for (std::int64_t l = 0; l < wl64; ++l) {
                    dst[l] = Lp[a * stride_a + l * stride_l + ap];
                }
            }
        }
    }

    {
        const std::int64_t wl64 = (std::int64_t)wl;
        const std::int64_t d64  = (std::int64_t)d;
        const std::int64_t wr64 = (std::int64_t)wr;

        const std::int64_t W_stride_l = d64 * wr64 * d64;
        const std::int64_t W_stride_s = wr64 * d64;
        const std::int64_t W_stride_r = d64;

        for (std::int64_t l = 0; l < wl64; ++l) {
            double* row = Wl + l * (d64 * wr64 * d64);
            for (std::int64_t s = 0; s < d64; ++s) {
                for (std::int64_t r = 0; r < wr64; ++r) {
                    for (std::int64_t sp = 0; sp < d64; ++sp) {
                        const std::int64_t col = (s * wr64 + r) * d64 + sp;
                        row[col] = Wp[l * W_stride_l + s * W_stride_s + r * W_stride_r + sp];
                    }
                }
            }
        }
    }

    py::array_t<double> T({(py::ssize_t)chiL, (py::ssize_t)chiL, (py::ssize_t)d, (py::ssize_t)wr, (py::ssize_t)d});
    auto Tb = T.request();
    double* Tp = static_cast<double*>(Tb.ptr);

    {
        py::gil_scoped_release rel;

        const MKL_INT m = chiL * chiL;
        const MKL_INT k = wl;
        const MKL_INT n = d * wr * d;

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m,
            n,
            k,
            1.0,
            Lf,
            k,
            Wl,
            n,
            0.0,
            Tp,
            n
        );
    }

    py::array_t<double> LW({(py::ssize_t)chiL, (py::ssize_t)d, (py::ssize_t)wr, (py::ssize_t)chiL, (py::ssize_t)d});
    auto LWb = LW.request();
    double* LWp = static_cast<double*>(LWb.ptr);

    {
        const std::int64_t chiL64 = (std::int64_t)chiL;
        const std::int64_t d64    = (std::int64_t)d;
        const std::int64_t wr64   = (std::int64_t)wr;

        for (std::int64_t a = 0; a < chiL64; ++a) {
            for (std::int64_t ap = 0; ap < chiL64; ++ap) {
                for (std::int64_t s = 0; s < d64; ++s) {
                    for (std::int64_t r = 0; r < wr64; ++r) {
                        const double* src = Tp + ((((a * chiL64 + ap) * d64 + s) * wr64 + r) * d64);
                        double* dst = LWp + ((((a * d64 + s) * wr64 + r) * chiL64 + ap) * d64);
                        std::memcpy(dst, src, sizeof(double) * (std::size_t)d64);
                    }
                }
            }
        }
    }

    return LW;
}

static py::array_t<std::complex<double>> precontract_LWZ(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> L,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> W
) {
    auto Lb = L.request();
    auto Wb = W.request();

    require_ndim(Lb, 3);
    require_ndim(Wb, 4);

    const MKL_INT chiL = as_mkl_int(Lb.shape[0]);
    const MKL_INT wl   = as_mkl_int(Lb.shape[1]);
    const MKL_INT chiL2= as_mkl_int(Lb.shape[2]);
    require_shape_eq(chiL2, chiL, "L must have shape (chiL, wl, chiL)");

    const MKL_INT wl2 = as_mkl_int(Wb.shape[0]);
    const MKL_INT d   = as_mkl_int(Wb.shape[1]);
    const MKL_INT wr  = as_mkl_int(Wb.shape[2]);
    const MKL_INT d2  = as_mkl_int(Wb.shape[3]);
    require_shape_eq(wl2, wl, "W first dim must match wl");
    require_shape_eq(d2, d, "W must have shape (wl, d, wr, d)");

    const MKL_Complex16* Lp = static_cast<const MKL_Complex16*>(Lb.ptr);
    const MKL_Complex16* Wp = static_cast<const MKL_Complex16*>(Wb.ptr);

    py::array_t<std::complex<double>> Lflat({(py::ssize_t)(chiL * chiL), (py::ssize_t)wl});
    py::array_t<std::complex<double>> Wlr({(py::ssize_t)wl, (py::ssize_t)(d * wr * d)});

    auto Lfb = Lflat.request();
    auto Wlb = Wlr.request();

    MKL_Complex16* Lf = static_cast<MKL_Complex16*>(Lfb.ptr);
    MKL_Complex16* Wl = static_cast<MKL_Complex16*>(Wlb.ptr);

    {
        const std::int64_t chiL64 = (std::int64_t)chiL;
        const std::int64_t wl64   = (std::int64_t)wl;

        const std::int64_t stride_a = wl64 * chiL64;
        const std::int64_t stride_l = chiL64;

        for (std::int64_t a = 0; a < chiL64; ++a) {
            for (std::int64_t ap = 0; ap < chiL64; ++ap) {
                const std::int64_t row = a * chiL64 + ap;
                MKL_Complex16* dst = Lf + row * wl64;
                for (std::int64_t l = 0; l < wl64; ++l) {
                    dst[l] = Lp[a * stride_a + l * stride_l + ap];
                }
            }
        }
    }

    {
        const std::int64_t wl64 = (std::int64_t)wl;
        const std::int64_t d64  = (std::int64_t)d;
        const std::int64_t wr64 = (std::int64_t)wr;

        const std::int64_t W_stride_l = d64 * wr64 * d64;
        const std::int64_t W_stride_s = wr64 * d64;
        const std::int64_t W_stride_r = d64;

        for (std::int64_t l = 0; l < wl64; ++l) {
            MKL_Complex16* row = Wl + l * (d64 * wr64 * d64);
            for (std::int64_t s = 0; s < d64; ++s) {
                for (std::int64_t r = 0; r < wr64; ++r) {
                    for (std::int64_t sp = 0; sp < d64; ++sp) {
                        const std::int64_t col = (s * wr64 + r) * d64 + sp;
                        row[col] = Wp[l * W_stride_l + s * W_stride_s + r * W_stride_r + sp];
                    }
                }
            }
        }
    }

    py::array_t<std::complex<double>> T({(py::ssize_t)chiL, (py::ssize_t)chiL, (py::ssize_t)d, (py::ssize_t)wr, (py::ssize_t)d});
    auto Tb = T.request();
    MKL_Complex16* Tp = static_cast<MKL_Complex16*>(Tb.ptr);

    {
        py::gil_scoped_release rel;

        const MKL_INT m = chiL * chiL;
        const MKL_INT k = wl;
        const MKL_INT n = d * wr * d;

        const MKL_Complex16 alpha = {1.0, 0.0};
        const MKL_Complex16 beta  = {0.0, 0.0};

        cblas_zgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m,
            n,
            k,
            &alpha,
            Lf,
            k,
            Wl,
            n,
            &beta,
            Tp,
            n
        );
    }

    py::array_t<std::complex<double>> LW({(py::ssize_t)chiL, (py::ssize_t)d, (py::ssize_t)wr, (py::ssize_t)chiL, (py::ssize_t)d});
    auto LWb = LW.request();
    MKL_Complex16* LWp = static_cast<MKL_Complex16*>(LWb.ptr);

    {
        const std::int64_t chiL64 = (std::int64_t)chiL;
        const std::int64_t d64    = (std::int64_t)d;
        const std::int64_t wr64   = (std::int64_t)wr;

        for (std::int64_t a = 0; a < chiL64; ++a) {
            for (std::int64_t ap = 0; ap < chiL64; ++ap) {
                for (std::int64_t s = 0; s < d64; ++s) {
                    for (std::int64_t r = 0; r < wr64; ++r) {
                        const MKL_Complex16* src = Tp + ((((a * chiL64 + ap) * d64 + s) * wr64 + r) * d64);
                        MKL_Complex16* dst = LWp + ((((a * d64 + s) * wr64 + r) * chiL64 + ap) * d64);
                        std::memcpy(dst, src, sizeof(MKL_Complex16) * (std::size_t)d64);
                    }
                }
            }
        }
    }

    return LW;
}

/* -------------------------------- RightOracle -------------------------------- */

class RightOracleD {
public:
    RightOracleD() = default;

    RightOracleD(
        py::array_t<double, py::array::c_style | py::array::forcecast> L,
        py::array_t<double, py::array::c_style | py::array::forcecast> WR
    ) {
        update(std::move(L), std::move(WR));
    }

    void update(
        py::array_t<double, py::array::c_style | py::array::forcecast> L,
        py::array_t<double, py::array::c_style | py::array::forcecast> WR
    ) {
        L_ = std::move(L);
        WR_ = std::move(WR);

        auto Lb = L_.request();
        auto WRb = WR_.request();

        require_ndim(Lb, 3);
        require_ndim(WRb, 5);

        const MKL_INT chiL = as_mkl_int(Lb.shape[0]);
        const MKL_INT wl   = as_mkl_int(Lb.shape[1]);
        const MKL_INT chiL2 = as_mkl_int(Lb.shape[2]);
        require_shape_eq(chiL2, chiL, "L must have shape (chiL, wl, chiL)");

        const MKL_INT WR_wl = as_mkl_int(WRb.shape[0]);
        const MKL_INT WR_d1 = as_mkl_int(WRb.shape[1]);
        const MKL_INT WR_d2 = as_mkl_int(WRb.shape[2]);
        const MKL_INT chiR  = as_mkl_int(WRb.shape[3]);
        const MKL_INT chiR2 = as_mkl_int(WRb.shape[4]);

        require_shape_eq(WR_wl, wl, "WR first dim must match wl");
        require_shape_eq(WR_d2, WR_d1, "WR must have shape (wl, d, d, chiR, chiR)");
        require_shape_eq(chiR2, chiR, "WR last dims must be (chiR, chiR)");

        const MKL_INT d = WR_d1;

        const bool need_alloc =
            !initialized_ ||
            chiL != chiL_ ||
            wl != wl_ ||
            d != d_ ||
            chiR != chiR_;

        chiL_ = chiL;
        wl_   = wl;
        d_    = d;
        chiR_ = chiR;

        if (need_alloc) {
            const MKL_INT batch = wl_ * d_;

            Lmat_ = py::array_t<double>({(py::ssize_t)chiL_, (py::ssize_t)(wl_ * chiL_)});
            C_    = py::array_t<double>({(py::ssize_t)batch, (py::ssize_t)chiL_, (py::ssize_t)chiR_});
            X_    = py::array_t<double>({(py::ssize_t)(wl_ * chiL_), (py::ssize_t)(d_ * chiR_)});
            out_  = py::array_t<double>({(py::ssize_t)(chiL_ * d_ * chiR_)});

            m_ = chiL_;
            n_ = chiR_;
            k_ = d_ * chiR_;
            lda_ = k_;
            ldb_ = n_;
            ldc_ = n_;
            batch_ = batch;

            strideA_ = 0;
            strideB_ = (MKL_INT)((std::int64_t)d_ * (std::int64_t)chiR_ * (std::int64_t)chiR_);
            strideC_ = (MKL_INT)((std::int64_t)chiL_ * (std::int64_t)chiR_);

            initialized_ = true;
        }

        {
            auto Lmb = Lmat_.request();
            const double* Lp3 = static_cast<const double*>(Lb.ptr);
            double* Lp2 = static_cast<double*>(Lmb.ptr);

            const std::int64_t stride_a = (std::int64_t)wl_ * (std::int64_t)chiL_;
            const std::int64_t stride_l = (std::int64_t)chiL_;

            for (MKL_INT a = 0; a < chiL_; ++a) {
                for (MKL_INT l = 0; l < wl_; ++l) {
                    const double* src = Lp3 + (std::int64_t)a * stride_a + (std::int64_t)l * stride_l;
                    double* dst = Lp2 + (std::int64_t)a * (std::int64_t)(wl_ * chiL_) + (std::int64_t)l * (std::int64_t)chiL_;
                    std::memcpy(dst, src, sizeof(double) * (std::size_t)chiL_);
                }
            }
        }

        WRp_ = static_cast<const double*>(WRb.ptr);
    }

    py::array_t<double> operator()(py::array_t<double, py::array::c_style | py::array::forcecast> psi) {
        if (!initialized_) throw std::runtime_error("oracle not initialized");

        auto xb = psi.request();
        require_ndim(xb, 1);

        const std::int64_t expected = (std::int64_t)chiL_ * (std::int64_t)d_ * (std::int64_t)chiR_;
        if ((std::int64_t)xb.size != expected) throw std::runtime_error("psi length mismatch");

        const double* Ap = static_cast<const double*>(xb.ptr);

        auto Cb   = C_.request();
        auto Xb   = X_.request();
        auto Lmb  = Lmat_.request();
        auto outb = out_.request();

        double* Cp       = static_cast<double*>(Cb.ptr);
        double* Xp       = static_cast<double*>(Xb.ptr);
        const double* Lp = static_cast<const double*>(Lmb.ptr);
        double* Outp     = static_cast<double*>(outb.ptr);

        const double* WRp = WRp_;

        {
            py::gil_scoped_release rel;

            cblas_dgemm_batch_strided(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                m_,
                n_,
                k_,
                1.0,
                Ap,
                lda_,
                strideA_,
                WRp,
                ldb_,
                strideB_,
                0.0,
                Cp,
                ldc_,
                strideC_,
                batch_
            );
        }

        {
            const std::int64_t chiL64 = (std::int64_t)chiL_;
            const std::int64_t chiR64 = (std::int64_t)chiR_;
            const std::int64_t wl64   = (std::int64_t)wl_;
            const std::int64_t d64    = (std::int64_t)d_;

            for (std::int64_t l = 0; l < wl64; ++l) {
                for (std::int64_t a = 0; a < chiL64; ++a) {
                    double* row = Xp + (l * chiL64 + a) * (d64 * chiR64);
                    for (std::int64_t s = 0; s < d64; ++s) {
                        const std::int64_t ls = l * d64 + s;
                        const double* src = Cp + (ls * chiL64 + a) * chiR64;
                        std::memcpy(row + s * chiR64, src, sizeof(double) * (std::size_t)chiR64);
                    }
                }
            }
        }

        {
            py::gil_scoped_release rel;

            const MKL_INT M2 = chiL_;
            const MKL_INT N2 = d_ * chiR_;
            const MKL_INT K2 = wl_ * chiL_;

            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                M2,
                N2,
                K2,
                1.0,
                Lp,
                K2,
                Xp,
                N2,
                0.0,
                Outp,
                N2
            );
        }

        return out_;
    }

    py::tuple info() const {
        return py::make_tuple(chiL_, wl_, d_, chiR_);
    }

private:
    bool initialized_ = false;

    py::array_t<double, py::array::c_style | py::array::forcecast> L_;
    py::array_t<double, py::array::c_style | py::array::forcecast> WR_;

    const double* WRp_ = nullptr;

    MKL_INT chiL_ = 0;
    MKL_INT wl_   = 0;
    MKL_INT d_    = 0;
    MKL_INT chiR_ = 0;

    py::array_t<double> Lmat_;
    py::array_t<double> C_;
    py::array_t<double> X_;
    py::array_t<double> out_;

    MKL_INT m_ = 0, n_ = 0, k_ = 0, lda_ = 0, ldb_ = 0, ldc_ = 0, batch_ = 0;
    MKL_INT strideA_ = 0, strideB_ = 0, strideC_ = 0;
};

class RightOracleZ {
public:
    RightOracleZ() = default;

    RightOracleZ(
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> L,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> WR
    ) {
        update(std::move(L), std::move(WR));
    }

    void update(
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> L,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> WR
    ) {
        L_ = std::move(L);
        WR_ = std::move(WR);

        auto Lb = L_.request();
        auto WRb = WR_.request();

        require_ndim(Lb, 3);
        require_ndim(WRb, 5);

        const MKL_INT chiL = as_mkl_int(Lb.shape[0]);
        const MKL_INT wl   = as_mkl_int(Lb.shape[1]);
        const MKL_INT chiL2 = as_mkl_int(Lb.shape[2]);
        require_shape_eq(chiL2, chiL, "L must have shape (chiL, wl, chiL)");

        const MKL_INT WR_wl = as_mkl_int(WRb.shape[0]);
        const MKL_INT WR_d1 = as_mkl_int(WRb.shape[1]);
        const MKL_INT WR_d2 = as_mkl_int(WRb.shape[2]);
        const MKL_INT chiR  = as_mkl_int(WRb.shape[3]);
        const MKL_INT chiR2 = as_mkl_int(WRb.shape[4]);

        require_shape_eq(WR_wl, wl, "WR first dim must match wl");
        require_shape_eq(WR_d2, WR_d1, "WR must have shape (wl, d, d, chiR, chiR)");
        require_shape_eq(chiR2, chiR, "WR last dims must be (chiR, chiR)");

        const MKL_INT d = WR_d1;

        const bool need_alloc =
            !initialized_ ||
            chiL != chiL_ ||
            wl != wl_ ||
            d != d_ ||
            chiR != chiR_;

        chiL_ = chiL;
        wl_   = wl;
        d_    = d;
        chiR_ = chiR;

        if (need_alloc) {
            const MKL_INT batch = wl_ * d_;

            Lmat_ = py::array_t<std::complex<double>>({(py::ssize_t)chiL_, (py::ssize_t)(wl_ * chiL_)});
            C_    = py::array_t<std::complex<double>>({(py::ssize_t)batch, (py::ssize_t)chiL_, (py::ssize_t)chiR_});
            X_    = py::array_t<std::complex<double>>({(py::ssize_t)(wl_ * chiL_), (py::ssize_t)(d_ * chiR_)});
            out_  = py::array_t<std::complex<double>>({(py::ssize_t)(chiL_ * d_ * chiR_)});

            m_ = chiL_;
            n_ = chiR_;
            k_ = d_ * chiR_;
            lda_ = k_;
            ldb_ = n_;
            ldc_ = n_;
            batch_ = batch;

            strideA_ = 0;
            strideB_ = (MKL_INT)((std::int64_t)d_ * (std::int64_t)chiR_ * (std::int64_t)chiR_);
            strideC_ = (MKL_INT)((std::int64_t)chiL_ * (std::int64_t)chiR_);

            initialized_ = true;
        }

        {
            auto Lmb = Lmat_.request();
            const MKL_Complex16* Lp3 = static_cast<const MKL_Complex16*>(Lb.ptr);
            MKL_Complex16* Lp2 = static_cast<MKL_Complex16*>(Lmb.ptr);

            const std::int64_t stride_a = (std::int64_t)wl_ * (std::int64_t)chiL_;
            const std::int64_t stride_l = (std::int64_t)chiL_;

            for (MKL_INT a = 0; a < chiL_; ++a) {
                for (MKL_INT l = 0; l < wl_; ++l) {
                    const MKL_Complex16* src = Lp3 + (std::int64_t)a * stride_a + (std::int64_t)l * stride_l;
                    MKL_Complex16* dst = Lp2 + (std::int64_t)a * (std::int64_t)(wl_ * chiL_) + (std::int64_t)l * (std::int64_t)chiL_;
                    std::memcpy(dst, src, sizeof(MKL_Complex16) * (std::size_t)chiL_);
                }
            }
        }

        WRp_ = static_cast<const MKL_Complex16*>(WRb.ptr);
    }

    py::array_t<std::complex<double>> operator()(py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> psi) {
        if (!initialized_) throw std::runtime_error("oracle not initialized");

        auto xb = psi.request();
        require_ndim(xb, 1);

        const std::int64_t expected = (std::int64_t)chiL_ * (std::int64_t)d_ * (std::int64_t)chiR_;
        if ((std::int64_t)xb.size != expected) throw std::runtime_error("psi length mismatch");

        const MKL_Complex16* Ap = static_cast<const MKL_Complex16*>(xb.ptr);

        auto Cb   = C_.request();
        auto Xb   = X_.request();
        auto Lmb  = Lmat_.request();
        auto outb = out_.request();

        MKL_Complex16* Cp       = static_cast<MKL_Complex16*>(Cb.ptr);
        MKL_Complex16* Xp       = static_cast<MKL_Complex16*>(Xb.ptr);
        const MKL_Complex16* Lp = static_cast<const MKL_Complex16*>(Lmb.ptr);
        MKL_Complex16* Outp     = static_cast<MKL_Complex16*>(outb.ptr);

        const MKL_Complex16* WRp = WRp_;

        const MKL_Complex16 alpha = {1.0, 0.0};
        const MKL_Complex16 beta  = {0.0, 0.0};

        {
            py::gil_scoped_release rel;

            cblas_zgemm_batch_strided(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                m_,
                n_,
                k_,
                &alpha,
                Ap,
                lda_,
                strideA_,
                WRp,
                ldb_,
                strideB_,
                &beta,
                Cp,
                ldc_,
                strideC_,
                batch_
            );
        }

        {
            const std::int64_t chiL64 = (std::int64_t)chiL_;
            const std::int64_t chiR64 = (std::int64_t)chiR_;
            const std::int64_t wl64   = (std::int64_t)wl_;
            const std::int64_t d64    = (std::int64_t)d_;

            for (std::int64_t l = 0; l < wl64; ++l) {
                for (std::int64_t a = 0; a < chiL64; ++a) {
                    MKL_Complex16* row = Xp + (l * chiL64 + a) * (d64 * chiR64);
                    for (std::int64_t s = 0; s < d64; ++s) {
                        const std::int64_t ls = l * d64 + s;
                        const MKL_Complex16* src = Cp + (ls * chiL64 + a) * chiR64;
                        std::memcpy(row + s * chiR64, src, sizeof(MKL_Complex16) * (std::size_t)chiR64);
                    }
                }
            }
        }

        {
            py::gil_scoped_release rel;

            const MKL_INT M2 = chiL_;
            const MKL_INT N2 = d_ * chiR_;
            const MKL_INT K2 = wl_ * chiL_;

            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                M2,
                N2,
                K2,
                &alpha,
                Lp,
                K2,
                Xp,
                N2,
                &beta,
                Outp,
                N2
            );
        }

        return out_;
    }

    py::tuple info() const {
        return py::make_tuple(chiL_, wl_, d_, chiR_);
    }

private:
    bool initialized_ = false;

    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> L_;
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> WR_;

    const MKL_Complex16* WRp_ = nullptr;

    MKL_INT chiL_ = 0;
    MKL_INT wl_   = 0;
    MKL_INT d_    = 0;
    MKL_INT chiR_ = 0;

    py::array_t<std::complex<double>> Lmat_;
    py::array_t<std::complex<double>> C_;
    py::array_t<std::complex<double>> X_;
    py::array_t<std::complex<double>> out_;

    MKL_INT m_ = 0, n_ = 0, k_ = 0, lda_ = 0, ldb_ = 0, ldc_ = 0, batch_ = 0;
    MKL_INT strideA_ = 0, strideB_ = 0, strideC_ = 0;
};

/* -------------------------------- LeftOracle -------------------------------- */

class LeftOracleD {
public:
    LeftOracleD() = default;

    LeftOracleD(
        py::array_t<double, py::array::c_style | py::array::forcecast> LW,
        py::array_t<double, py::array::c_style | py::array::forcecast> R
    ) {
        update(std::move(LW), std::move(R));
    }

    void update(
        py::array_t<double, py::array::c_style | py::array::forcecast> LW,
        py::array_t<double, py::array::c_style | py::array::forcecast> R
    ) {
        LW_ = std::move(LW);
        R_ = std::move(R);

        auto LWb = LW_.request();
        auto Rb  = R_.request();

        require_ndim(LWb, 5);
        require_ndim(Rb, 3);

        const MKL_INT chiL = as_mkl_int(LWb.shape[0]);
        const MKL_INT d    = as_mkl_int(LWb.shape[1]);
        const MKL_INT wr   = as_mkl_int(LWb.shape[2]);
        const MKL_INT chiL2 = as_mkl_int(LWb.shape[3]);
        const MKL_INT d2    = as_mkl_int(LWb.shape[4]);
        require_shape_eq(chiL2, chiL, "LW must have shape (chiL, d, wr, chiL, d)");
        require_shape_eq(d2, d, "LW last dim must match d");

        const MKL_INT chiR  = as_mkl_int(Rb.shape[0]);
        const MKL_INT wr2   = as_mkl_int(Rb.shape[1]);
        const MKL_INT chiR2 = as_mkl_int(Rb.shape[2]);
        require_shape_eq(wr2, wr, "R second dim must match wr");
        require_shape_eq(chiR2, chiR, "R must have shape (chiR, wr, chiR)");

        const bool need_alloc =
            !initialized_ ||
            chiL != chiL_ ||
            d != d_ ||
            wr != wr_ ||
            chiR != chiR_;

        chiL_ = chiL;
        d_    = d;
        wr_   = wr;
        chiR_ = chiR;

        if (need_alloc) {
            LWmat_ = py::array_t<double>({(py::ssize_t)(chiL_ * d_ * wr_), (py::ssize_t)(chiL_ * d_)});
            Rmat_  = py::array_t<double>({(py::ssize_t)(wr_ * chiR_), (py::ssize_t)chiR_});

            Dbig_   = py::array_t<double>({(py::ssize_t)(chiL_ * d_ * wr_), (py::ssize_t)chiR_});
            Dstack_ = py::array_t<double>({(py::ssize_t)(chiL_ * d_), (py::ssize_t)(wr_ * chiR_)});
            out_    = py::array_t<double>({(py::ssize_t)(chiL_ * d_ * chiR_)});

            initialized_ = true;
        }

        {
            const double* LWp = static_cast<const double*>(LWb.ptr);
            auto LWmb = LWmat_.request();
            double* LWmp = static_cast<double*>(LWmb.ptr);

            const std::int64_t rows = (std::int64_t)chiL_ * (std::int64_t)d_ * (std::int64_t)wr_;
            const std::int64_t cols = (std::int64_t)chiL_ * (std::int64_t)d_;
            std::memcpy(LWmp, LWp, sizeof(double) * (std::size_t)(rows * cols));
        }

        {
            const double* Rp = static_cast<const double*>(Rb.ptr);
            auto Rmb = Rmat_.request();
            double* Rmp = static_cast<double*>(Rmb.ptr);

            const std::int64_t chiR64 = (std::int64_t)chiR_;
            const std::int64_t wr64   = (std::int64_t)wr_;

            const std::int64_t R_stride_bp = wr64 * chiR64;
            const std::int64_t R_stride_r  = chiR64;

            for (std::int64_t r = 0; r < wr64; ++r) {
                for (std::int64_t bp = 0; bp < chiR64; ++bp) {
                    const double* src = Rp + bp * R_stride_bp + r * R_stride_r;
                    double* dst = Rmp + (r * chiR64 + bp) * chiR64;
                    std::memcpy(dst, src, sizeof(double) * (std::size_t)chiR64);
                }
            }
        }
    }

    py::array_t<double> operator()(py::array_t<double, py::array::c_style | py::array::forcecast> psi) {
        if (!initialized_) throw std::runtime_error("oracle not initialized");

        auto xb = psi.request();
        require_ndim(xb, 1);

        const std::int64_t expected = (std::int64_t)chiL_ * (std::int64_t)d_ * (std::int64_t)chiR_;
        if ((std::int64_t)xb.size != expected) throw std::runtime_error("psi length mismatch");

        const double* Ap = static_cast<const double*>(xb.ptr);

        auto LWmb = LWmat_.request();
        auto Rmb  = Rmat_.request();
        auto Db   = Dbig_.request();
        auto Ds   = Dstack_.request();
        auto ob   = out_.request();

        const double* LWmp = static_cast<const double*>(LWmb.ptr);
        const double* Rmp  = static_cast<const double*>(Rmb.ptr);
        double* Dp         = static_cast<double*>(Db.ptr);
        double* Dsp        = static_cast<double*>(Ds.ptr);
        double* Outp       = static_cast<double*>(ob.ptr);

        {
            py::gil_scoped_release rel;

            const MKL_INT M = chiL_ * d_ * wr_;
            const MKL_INT K = chiL_ * d_;
            const MKL_INT N = chiR_;

            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                M,
                N,
                K,
                1.0,
                LWmp,
                K,
                Ap,
                N,
                0.0,
                Dp,
                N
            );
        }

        {
            const std::int64_t chiR64 = (std::int64_t)chiR_;
            const std::int64_t wr64   = (std::int64_t)wr_;
            const std::int64_t rows_as = (std::int64_t)chiL_ * (std::int64_t)d_;

            for (std::int64_t as = 0; as < rows_as; ++as) {
                double* dst_row = Dsp + as * (wr64 * chiR64);
                for (std::int64_t r = 0; r < wr64; ++r) {
                    const double* src_row = Dp + (as * wr64 + r) * chiR64;
                    std::memcpy(dst_row + r * chiR64, src_row, sizeof(double) * (std::size_t)chiR64);
                }
            }
        }

        {
            py::gil_scoped_release rel;

            const MKL_INT M = chiL_ * d_;
            const MKL_INT K = wr_ * chiR_;
            const MKL_INT N = chiR_;

            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                M,
                N,
                K,
                1.0,
                Dsp,
                K,
                Rmp,
                N,
                0.0,
                Outp,
                N
            );
        }

        return out_;
    }

    py::tuple info() const {
        return py::make_tuple(chiL_, d_, wr_, chiR_);
    }

private:
    bool initialized_ = false;

    py::array_t<double, py::array::c_style | py::array::forcecast> LW_;
    py::array_t<double, py::array::c_style | py::array::forcecast> R_;

    MKL_INT chiL_ = 0;
    MKL_INT d_    = 0;
    MKL_INT wr_   = 0;
    MKL_INT chiR_ = 0;

    py::array_t<double> LWmat_;
    py::array_t<double> Rmat_;
    py::array_t<double> Dbig_;
    py::array_t<double> Dstack_;
    py::array_t<double> out_;
};

class LeftOracleZ {
public:
    LeftOracleZ() = default;

    LeftOracleZ(
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> LW,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> R
    ) {
        update(std::move(LW), std::move(R));
    }

    void update(
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> LW,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> R
    ) {
        LW_ = std::move(LW);
        R_ = std::move(R);

        auto LWb = LW_.request();
        auto Rb  = R_.request();

        require_ndim(LWb, 5);
        require_ndim(Rb, 3);

        const MKL_INT chiL = as_mkl_int(LWb.shape[0]);
        const MKL_INT d    = as_mkl_int(LWb.shape[1]);
        const MKL_INT wr   = as_mkl_int(LWb.shape[2]);
        const MKL_INT chiL2 = as_mkl_int(LWb.shape[3]);
        const MKL_INT d2    = as_mkl_int(LWb.shape[4]);
        require_shape_eq(chiL2, chiL, "LW must have shape (chiL, d, wr, chiL, d)");
        require_shape_eq(d2, d, "LW last dim must match d");

        const MKL_INT chiR  = as_mkl_int(Rb.shape[0]);
        const MKL_INT wr2   = as_mkl_int(Rb.shape[1]);
        const MKL_INT chiR2 = as_mkl_int(Rb.shape[2]);
        require_shape_eq(wr2, wr, "R second dim must match wr");
        require_shape_eq(chiR2, chiR, "R must have shape (chiR, wr, chiR)");

        const bool need_alloc =
            !initialized_ ||
            chiL != chiL_ ||
            d != d_ ||
            wr != wr_ ||
            chiR != chiR_;

        chiL_ = chiL;
        d_    = d;
        wr_   = wr;
        chiR_ = chiR;

        if (need_alloc) {
            LWmat_ = py::array_t<std::complex<double>>({(py::ssize_t)(chiL_ * d_ * wr_), (py::ssize_t)(chiL_ * d_)});
            Rmat_  = py::array_t<std::complex<double>>({(py::ssize_t)(wr_ * chiR_), (py::ssize_t)chiR_});

            Dbig_   = py::array_t<std::complex<double>>({(py::ssize_t)(chiL_ * d_ * wr_), (py::ssize_t)chiR_});
            Dstack_ = py::array_t<std::complex<double>>({(py::ssize_t)(chiL_ * d_), (py::ssize_t)(wr_ * chiR_)});
            out_    = py::array_t<std::complex<double>>({(py::ssize_t)(chiL_ * d_ * chiR_)});

            initialized_ = true;
        }

        {
            const MKL_Complex16* LWp = static_cast<const MKL_Complex16*>(LWb.ptr);
            auto LWmb = LWmat_.request();
            MKL_Complex16* LWmp = static_cast<MKL_Complex16*>(LWmb.ptr);

            const std::int64_t rows = (std::int64_t)chiL_ * (std::int64_t)d_ * (std::int64_t)wr_;
            const std::int64_t cols = (std::int64_t)chiL_ * (std::int64_t)d_;
            std::memcpy(LWmp, LWp, sizeof(MKL_Complex16) * (std::size_t)(rows * cols));
        }

        {
            const MKL_Complex16* Rp = static_cast<const MKL_Complex16*>(Rb.ptr);
            auto Rmb = Rmat_.request();
            MKL_Complex16* Rmp = static_cast<MKL_Complex16*>(Rmb.ptr);

            const std::int64_t chiR64 = (std::int64_t)chiR_;
            const std::int64_t wr64   = (std::int64_t)wr_;

            const std::int64_t R_stride_bp = wr64 * chiR64;
            const std::int64_t R_stride_r  = chiR64;

            for (std::int64_t r = 0; r < wr64; ++r) {
                for (std::int64_t bp = 0; bp < chiR64; ++bp) {
                    const MKL_Complex16* src = Rp + bp * R_stride_bp + r * R_stride_r;
                    MKL_Complex16* dst = Rmp + (r * chiR64 + bp) * chiR64;
                    std::memcpy(dst, src, sizeof(MKL_Complex16) * (std::size_t)chiR64);
                }
            }
        }
    }

    py::array_t<std::complex<double>> operator()(py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> psi) {
        if (!initialized_) throw std::runtime_error("oracle not initialized");

        auto xb = psi.request();
        require_ndim(xb, 1);

        const std::int64_t expected = (std::int64_t)chiL_ * (std::int64_t)d_ * (std::int64_t)chiR_;
        if ((std::int64_t)xb.size != expected) throw std::runtime_error("psi length mismatch");

        const MKL_Complex16* Ap = static_cast<const MKL_Complex16*>(xb.ptr);

        auto LWmb = LWmat_.request();
        auto Rmb  = Rmat_.request();
        auto Db   = Dbig_.request();
        auto Ds   = Dstack_.request();
        auto ob   = out_.request();

        const MKL_Complex16* LWmp = static_cast<const MKL_Complex16*>(LWmb.ptr);
        const MKL_Complex16* Rmp  = static_cast<const MKL_Complex16*>(Rmb.ptr);
        MKL_Complex16* Dp         = static_cast<MKL_Complex16*>(Db.ptr);
        MKL_Complex16* Dsp        = static_cast<MKL_Complex16*>(Ds.ptr);
        MKL_Complex16* Outp       = static_cast<MKL_Complex16*>(ob.ptr);

        const MKL_Complex16 alpha = {1.0, 0.0};
        const MKL_Complex16 beta  = {0.0, 0.0};

        {
            py::gil_scoped_release rel;

            const MKL_INT M = chiL_ * d_ * wr_;
            const MKL_INT K = chiL_ * d_;
            const MKL_INT N = chiR_;

            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                M,
                N,
                K,
                &alpha,
                LWmp,
                K,
                Ap,
                N,
                &beta,
                Dp,
                N
            );
        }

        {
            const std::int64_t chiR64 = (std::int64_t)chiR_;
            const std::int64_t wr64   = (std::int64_t)wr_;
            const std::int64_t rows_as = (std::int64_t)chiL_ * (std::int64_t)d_;

            for (std::int64_t as = 0; as < rows_as; ++as) {
                MKL_Complex16* dst_row = Dsp + as * (wr64 * chiR64);
                for (std::int64_t r = 0; r < wr64; ++r) {
                    const MKL_Complex16* src_row = Dp + (as * wr64 + r) * chiR64;
                    std::memcpy(dst_row + r * chiR64, src_row, sizeof(MKL_Complex16) * (std::size_t)chiR64);
                }
            }
        }

        {
            py::gil_scoped_release rel;

            const MKL_INT M = chiL_ * d_;
            const MKL_INT K = wr_ * chiR_;
            const MKL_INT N = chiR_;

            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                M,
                N,
                K,
                &alpha,
                Dsp,
                K,
                Rmp,
                N,
                &beta,
                Outp,
                N
            );
        }

        return out_;
    }

    py::tuple info() const {
        return py::make_tuple(chiL_, d_, wr_, chiR_);
    }

private:
    bool initialized_ = false;

    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> LW_;
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> R_;

    MKL_INT chiL_ = 0;
    MKL_INT d_    = 0;
    MKL_INT wr_   = 0;
    MKL_INT chiR_ = 0;

    py::array_t<std::complex<double>> LWmat_;
    py::array_t<std::complex<double>> Rmat_;
    py::array_t<std::complex<double>> Dbig_;
    py::array_t<std::complex<double>> Dstack_;
    py::array_t<std::complex<double>> out_;
};

/* -------------------------------- pybind module -------------------------------- */

PYBIND11_MODULE(tdvp_oracle_prefused, m) {
    m.doc() = "TDVP interior-site matvec oracles that consume precontracted WR or LW tensors";

    m.def("precontract_WRD", &precontract_WRD, "Compute WR from W and R for float64");
    m.def("precontract_WRZ", &precontract_WRZ, "Compute WR from W and R for complex128");
    m.def("precontract_LWD", &precontract_LWD, "Compute LW from L and W for float64");
    m.def("precontract_LWZ", &precontract_LWZ, "Compute LW from L and W for complex128");

    py::class_<RightOracleD>(m, "RightOracleD")
        .def(py::init<>())
        .def(py::init<
             py::array_t<double, py::array::c_style | py::array::forcecast>,
             py::array_t<double, py::array::c_style | py::array::forcecast>
        >())
        .def("update", &RightOracleD::update)
        .def("__call__", &RightOracleD::operator())
        .def("info", &RightOracleD::info);

    py::class_<RightOracleZ>(m, "RightOracleZ")
        .def(py::init<>())
        .def(py::init<
             py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>,
             py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>
        >())
        .def("update", &RightOracleZ::update)
        .def("__call__", &RightOracleZ::operator())
        .def("info", &RightOracleZ::info);

    py::class_<LeftOracleD>(m, "LeftOracleD")
        .def(py::init<>())
        .def(py::init<
             py::array_t<double, py::array::c_style | py::array::forcecast>,
             py::array_t<double, py::array::c_style | py::array::forcecast>
        >())
        .def("update", &LeftOracleD::update)
        .def("__call__", &LeftOracleD::operator())
        .def("info", &LeftOracleD::info);

    py::class_<LeftOracleZ>(m, "LeftOracleZ")
        .def(py::init<>())
        .def(py::init<
             py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>,
             py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>
        >())
        .def("update", &LeftOracleZ::update)
        .def("__call__", &LeftOracleZ::operator())
        .def("info", &LeftOracleZ::info);
}
