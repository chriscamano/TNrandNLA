from __future__ import annotations

import os
import numpy as np
from .util import _is_complex,_c2r,_r2c

def _try_import_cpp():
    if os.environ.get("TNRNLA_INCRQR_DEV_BUILD", ""):
        this_dir = os.path.dirname(__file__)
        bdir = os.path.join(this_dir, "build")
        if bdir not in os.sys.path:
            os.sys.path.insert(0, bdir)
    try:
        import libincrementalqr as L
        return L, None
    except Exception as e:
        return None, str(e)

_L, _L_err = _try_import_cpp()

if _L is not None:
    _setup_stats = getattr(_L, "setup_stats", None)
    _add_cols_stats = getattr(_L, "add_cols_stats", None)
    _setup = getattr(_L, "setup", None)
    _add_cols = getattr(_L, "add_cols", None)
    _extract_q = getattr(_L, "extract_q", None)
    _get_err = getattr(_L, "get_error_estimate", None)
else:
    _setup_stats = _add_cols_stats = _setup = _add_cols = _extract_q = _get_err = None

class IncrementalQR:
    def __init__(self, data, size=None, use_cpp_if_available=True):
        if getattr(data, "ndim", None) != 2:
            raise ValueError("data must be 2D")

        self.use_cpp_if_available = bool(use_cpp_if_available)
        self._cplx = _is_complex(data.dtype)

        m0, n0 = data.shape
        if m0 < n0:
            raise ValueError("Require m >= n")

        self.m = 2 * m0 if self._cplx else m0
        self.n = 2 * n0 if self._cplx else n0

        self.size = int(2 * self.n if size is None else size)
        if self.size < self.n:
            raise ValueError("size must be >= n")

        self.data = np.zeros((self.m, self.size), dtype=np.float64, order="F")
        self.tau = np.zeros((self.size,), dtype=np.float64)

        self._row_norm2 = np.zeros((self.size,), dtype=np.float64)
        self._sum_inv = np.zeros((1,), dtype=np.float64)

        if self._cplx:
            _c2r(np.asarray(data), self.data[:, : self.n])
        else:
            np.copyto(self.data[:, : self.n], np.asarray(data, dtype=np.float64), casting="no")

        self._cpp = bool(
            self.use_cpp_if_available
            and _extract_q is not None
            and (_setup_stats is not None or _setup is not None)
        )
        self._cpp_stats = False

        if self._cpp and _setup_stats is not None and _add_cols_stats is not None:
            _setup_stats(self.data, self.tau, self.m, self.n, self._row_norm2, self._sum_inv)
            self._cpp_stats = True
        elif self._cpp and _setup is not None and _add_cols is not None:
            _setup(self.data, self.tau, self.m, self.n)
        else:
            self._cpp = False
            self._setup_scipy()

        self.open = True

    def _setup_scipy(self):
        import scipy as sp

        self._geqrf = sp.linalg.get_lapack_funcs("geqrf", dtype=np.float64)
        self._trtri = sp.linalg.get_lapack_funcs("trtri", dtype=np.float64)
        self._ormqr = sp.linalg.get_lapack_funcs("ormqr", dtype=np.float64)
        self._orgqr = sp.linalg.get_lapack_funcs("orgqr", dtype=np.float64)

        self._transpose_flag = "T"
        self._lwork_cache = {}

        A = self.data[:, : self.n]
        A, tau, _, _ = self._geqrf(A)
        self.data[:, : self.n] = A
        self.tau[: self.n] = tau

        R = self.data[: self.n, : self.n]
        # SciPy's LAPACK wrapper for trtri uses (lower, unitdiag) flags.
        # lower=0 => upper-triangular, unitdiag=0 => non-unit diagonal.
        Rinv, _info = self._trtri(R, 0, 0)
        self.data[: self.n, : self.n] = Rinv

    def _ensure_capacity(self, needed):
        if needed <= self.size:
            return
        new_size = self.size
        while new_size < needed:
            new_size *= 2

        new_data = np.zeros((self.m, new_size), dtype=np.float64, order="F")
        new_data[:, : self.size] = self.data
        self.data = new_data

        new_tau = np.zeros((new_size,), dtype=np.float64)
        new_tau[: self.size] = self.tau
        self.tau = new_tau

        new_rn = np.zeros((new_size,), dtype=np.float64)
        new_rn[: self.size] = self._row_norm2
        self._row_norm2 = new_rn

        self.size = new_size

    def append(self, new_data):
        if not self.open:
            raise RuntimeError("Cannot append after get_q()")
        if getattr(new_data, "ndim", None) != 2:
            raise ValueError("new_data must be 2D")

        m0, k0 = new_data.shape
        if self._cplx:
            if m0 * 2 != self.m:
                raise ValueError("new_data has wrong number of rows")
            k = 2 * k0
        else:
            if m0 != self.m:
                raise ValueError("new_data has wrong number of rows")
            k = k0

        needed = self.n + k
        if needed > self.m:
            raise ValueError("Require n+k <= m")
        self._ensure_capacity(needed)

        if self._cplx:
            _c2r(np.asarray(new_data), self.data[:, self.n : self.n + k])
        else:
            np.copyto(self.data[:, self.n : self.n + k], np.asarray(new_data, dtype=np.float64), casting="no")

        if self._cpp:
            if self._cpp_stats:
                _add_cols_stats(self.data, self.tau, self.m, self.n, k, self._row_norm2, self._sum_inv)
            else:
                _add_cols(self.data, self.tau, self.m, self.n, k)
            self.n += k
            return

        self._append_scipy(k)
        self.n += k

    def _append_scipy(self, k):
        old_n = self.n
        new_n = old_n + k
        new_block = self.data[:, old_n:new_n]

        lwork = self._lwork_cache.get(k)
        if lwork is None:
            w = self._ormqr(
                "L",
                self._transpose_flag,
                self.data[:, :old_n],
                self.tau[:old_n],
                new_block,
                lwork=-1,
            )[1][0]
            lwork = int(np.real(w))
            self._lwork_cache[k] = lwork

        self.data[:, old_n:new_n] = self._ormqr(
            "L",
            self._transpose_flag,
            self.data[:, :old_n],
            self.tau[:old_n],
            new_block,
            lwork=lwork,
        )[0]

        bottom = self.data[old_n:, old_n:new_n]
        bottom, tau2, _, _ = self._geqrf(bottom)
        self.data[old_n:, old_n:new_n] = bottom
        self.tau[old_n:new_n] = tau2

        R22 = self.data[old_n:new_n, old_n:new_n]
        R22inv, _info = self._trtri(R22, 0, 0)
        self.data[old_n:new_n, old_n:new_n] = R22inv

        Rinv_old = np.triu(self.data[:old_n, :old_n])
        top_block = self.data[:old_n, old_n:new_n]
        Rinv_new = np.triu(self.data[old_n:new_n, old_n:new_n])
        self.data[:old_n, old_n:new_n] = -(Rinv_old @ top_block @ Rinv_new)

    def error_estimate(self):
        if not self.open:
            raise RuntimeError("QR is closed")
        if self.n <= 0:
            return float("inf")

        if self._cpp and self._cpp_stats:
            s = float(self._sum_inv[0])
            if (not np.isfinite(s)) or (s <= 0.0):
                return float("inf")
            out = np.sqrt(s / float(self.n))
            return float(out) if np.isfinite(out) else float("inf")

        if self._cpp and _get_err is not None:
            return float(_get_err(self.data, self.m, self.n))

        Rinv = np.triu(self.data[: self.n, : self.n])
        row_norms = np.linalg.norm(Rinv, axis=1)
        return float(np.sqrt(np.sum(row_norms ** -2) / self.n))

    def get_q(self):
        if not self.open:
            return _r2c(self.data[:, : self.n]) if self._cplx else self.data[:, : self.n]

        if self._cpp:
            _extract_q(self.data, self.tau, self.m, self.n)
            self.open = False
            return _r2c(self.data[:, : self.n]) if self._cplx else self.data[:, : self.n]

        Q = self._orgqr(self.data[:, : self.n], self.tau[: self.n])[0]
        self.data[:, : self.n] = Q
        self.open = False
        return _r2c(Q) if self._cplx else Q

    @staticmethod
    def backend_status():
        return "cpp" if _L is not None else "scipy (cpp unavailable: {})".format(_L_err)
