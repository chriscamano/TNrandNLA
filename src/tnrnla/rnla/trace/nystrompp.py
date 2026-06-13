import numpy as np

from tnrnla.linalg.utils import sym, safe_cholesky_psd
from .randomvector import rvec
from .common import ensure_oracle_and_dim
import numpy as np
import scipy.linalg as sla
from tnrnla.rnla.lra import nystrom

def _split_total_queries(num_queries, N):
    q = num_queries
    if q < 2:
        raise ValueError("num_queries must be at least 2")
    q = min(q, 2 * N)
    q -= (q & 1)
    k = q // 2
    m = q - k
    return q, k, m


def _as_matvec(A):
    if callable(A):
        return A

    A = np.asarray(A)

    def A_mv(X):
        X = np.asarray(X, dtype=A.dtype)
        return A @ X

    return A_mv



def npp(
    A,
    num_queries=None,
    *,
    n=None,
    sampler=None,
    seed=None,
    dtype=np.float64,
    Omega=None,
    Psi=None,
    ridge=None,
    return_probes=False,
):
    A_mv = _as_matvec(A)

    if Omega is None or Psi is None:
        q, k, m = _split_total_queries(num_queries, n)
        Omega = np.asarray(sampler(int(n), int(k), seed=seed), dtype=dtype)
        Psi = np.asarray(
            sampler(int(n), int(m), seed=None if seed is None else seed + 1),
            dtype=dtype,
        )
    else:
        Omega = np.asarray(Omega, dtype=dtype)
        Psi = np.asarray(Psi, dtype=dtype)
        m = Psi.shape[1]

    U, d, _ = nystrom(A, Omega,form="svd")
    Z = A_mv(Psi)
    corr = U @ (d[:, None] * (U.conj().T @ Psi))

    t1 = np.sum(d)
    tZ = np.trace(Psi.conj().T @ Z)
    tCorr = np.trace(Psi.conj().T @ corr)

    est = np.real(t1 + (tZ - tCorr) / m)

    if return_probes:
        return est, Omega, Psi, (U, d)
    return est




def npp_gram(A_mv, Omega, Psi, ridge=None, shift_factor=1e-10):
    n, k = Omega.shape
    m = Psi.shape[1]

    Y = A_mv(Omega)
    Z = A_mv(Psi)

    G = sym(Omega.conj().T @ Omega)
    K = Omega.conj().T @ Y
    YY = sym(Y.conj().T @ Y)

    OPsi = Omega.conj().T @ Psi
    YPsi = Y.conj().T @ Psi

    eps = np.finfo(np.float64).eps
    # nu =  np.sqrt(eps) * np.linalg.norm(K, 2)
    eps = np.finfo(np.float64).eps
    Tchol = sla.cholesky(G, lower=False, check_finite=False)
    invT = sla.solve_triangular(Tchol, np.eye(k, dtype=G.dtype), lower=False, check_finite=False)
    w = np.linalg.eigvalsh(sym(invT.T.conj() @ K @ invT))
    nu = float(max(0.0, -w[0]) + np.sqrt(eps) * w[-1])
    M = sym(K + nu * G)

    C = sla.cholesky(M, lower=False, check_finite=False)

    S = sym(YY + nu * 2*(K) + (nu * nu) * G)

    L = sla.solve_triangular(C.conj().T, S, lower=True, check_finite=False)
    GB = sla.solve_triangular(C.T, L.T, lower=True, check_finite=False).T
    GB = sym(GB)

    U, s, Vh = np.linalg.svd(GB, full_matrices=False)
    w = np.real(s)
    V = U

    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]

    w = np.maximum(w, 0.0)
    d = np.maximum(w - nu, 0.0)

    YnuPsi = YPsi + nu * OPsi
    BstarPsi = sla.solve_triangular(
        C.conj().T,
        YnuPsi,
        lower=True,
        check_finite=False
    )

    T = V.conj().T @ BstarPsi

    scale = np.zeros_like(w)
    good = w > 0
    scale[good] = np.sqrt(d[good] / w[good])

    W = scale[:, None] * T
    tCorr_over_m = np.mean(np.sum(np.abs(W) ** 2, axis=0))

    t1 = np.sum(d)
    tZ_over_m = np.mean(np.real(np.sum(np.conj(Psi) * Z, axis=0)))

    return np.real(t1 + tZ_over_m - tCorr_over_m)



