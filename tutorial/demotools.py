import os
import io
import contextlib
import numpy as np
from scipy.__config__ import show as scipy_show_config

import numpy as np
from tnrnla import Cutoff, FixedDimension, no_truncation
from tnrnla import MPS
from tnrnla import relerr 

def test_mkl() -> tuple[bool, bool]:
    import io
    import contextlib
    import numpy as np
    from scipy.__config__ import show as scipy_show_config

    def _captured(fn) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn()
        return buf.getvalue().lower()

    np_cfg = _captured(np.show_config)
    sp_cfg = _captured(scipy_show_config)

    numpy_mkl = ("mkl" in np_cfg) and ("blas" in np_cfg) and ("lapack" in np_cfg)
    scipy_mkl = ("mkl" in sp_cfg) and ("blas" in sp_cfg) and ("lapack" in sp_cfg)

    print(f"NumPy: MKL installed: {numpy_mkl}")
    print(f"SciPy: MKL installed: {scipy_mkl}")


def test_incrementalqr() -> str:
    import os

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

    L, err = _try_import_cpp()

    if L is None:
        backend = "scipy"
    else:
        has_stats = hasattr(L, "setup_stats") and hasattr(L, "add_cols_stats")
        has_basic = hasattr(L, "setup") and hasattr(L, "add_cols")
        has_extract = hasattr(L, "extract_q")

        if has_extract and has_stats:
            backend = "cpp (stats)"
        elif has_extract and has_basic:
            backend = "cpp"
        else:
            backend = "scipy"

    print(f"IncrementalQR backend: {backend}")

from tnrnla import MPS, relerr

import numpy as np
from tnrnla import MPS, relerr

def mps_summary(N=20, chi=8, d=2, seed=None):
    if seed is None:
        v1_0 = MPS.rmps(N, chi, d=d, dtype=float)
        v2_0 = MPS.rmps(N, chi, d=d, dtype=float)
    else:
        rng = np.random.default_rng(seed)
        v1_0 = MPS.rmps(N, chi, d=d, dtype=float, rng=rng)
        v2_0 = MPS.rmps(N, chi, d=d, dtype=float, rng=rng)

    v1 = v1_0.copy()
    v2 = v2_0.copy()

    a = v1.to_dense()
    b = v2.to_dense()

    print("Dense dimension:", a.shape[0])

    v3 = v1 + v2
    c = v3.to_dense()
    print("rel_err(v1+v2) =", relerr(c, a + b))

    v4 = v1 - v2
    dvec = v4.to_dense()
    print("rel_err(v1-v2) =", relerr(dvec, a - b))

    alpha = 0.7
    v5 = alpha * v1
    e = v5.to_dense()
    print("rel_err(alpha*v1) =", relerr(e, alpha * a))

    beta = 1.3
    v6 = v1 / beta
    f = v6.to_dense()
    print("rel_err(v1/beta) =", relerr(f, a / beta))

    ip_mps = v1 @ v2
    ip_ref = np.vdot(a, b)
    print("rel_err(<v1,v2>) =", relerr(ip_mps, ip_ref))

    sip_mps1 = v1 @ v1
    sip_ref = np.vdot(a, a)
    print("rel_err(<v1,v1> via v1@v1) =", relerr(sip_mps1, sip_ref))

    sip_mps2 = v1.self_inner_product()
    print("rel_err(self_inner_product) =", relerr(sip_mps2, sip_ref))

    n_mps = v1.norm()
    n_ref = np.linalg.norm(a)
    print("rel_err(norm(v1)) =", relerr(n_mps, n_ref))

    v7 = v1.copy()
    v7_before = v7.to_dense()
    v7.normalize()
    g = v7.to_dense()
    g_ref = v7_before / np.linalg.norm(v7_before)
    print("rel_err(normalize(v1)) =", relerr(g, g_ref))
    print("abs_err(||normalize(v1)|| - 1) =", float(np.abs(np.linalg.norm(g) - 1.0)))

    v8 = v1.copy()
    v8_0 = v8.to_dense()
    v8 += v2
    print("rel_err(v1 += v2) =", relerr(v8.to_dense(), v8_0 + b))

    v9 = v1.copy()
    v9_0 = v9.to_dense()
    v9 -= v2
    print("rel_err(v1 -= v2) =", relerr(v9.to_dense(), v9_0 - b))

    v10 = v1.copy()
    v10_0 = v10.to_dense()
    v10 *= alpha
    print("rel_err(v1 *= alpha) =", relerr(v10.to_dense(), v10_0 * alpha))

    v11 = v1.copy()
    v11_0 = v11.to_dense()
    v11 /= beta
    print("rel_err(v1 /= beta) =", relerr(v11.to_dense(), v11_0 / beta))