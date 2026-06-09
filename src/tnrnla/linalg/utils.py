import numpy as np
import numpy.linalg as la
from scipy.special import gammainc
import scipy.linalg as sla

def safe_cholesky_psd(A, max_tries=8):
    A = sym(A)

    k = A.shape[0]
    eps = np.finfo(A.dtype).eps
    base = eps * np.real(np.trace(A)) / max(k, 1)

    shift = base

    for _ in range(max_tries):
        try:
            return sla.cholesky(A + shift * np.eye(k, dtype=A.dtype),
                                lower=False,
                                check_finite=False)
        except np.linalg.LinAlgError:
            shift *= 10.0

    raise np.linalg.LinAlgError("safe_cholesky_psd failed")

def sym(G):
    G = np.asarray(G)
    return 0.5 * (G + G.T.conj())


def hermitian_sym(A):
    return sym(A)


def product_trace(A, B):
    return float(np.sum(A * B.T))


def diagprod(X, Y):
    return np.sum(X.conj() * Y, axis=0)


def sqcolnorms(X):
    return np.sum(np.abs(X) ** 2, axis=0)


def sqrownorms(X):
    return np.sum(np.abs(X) ** 2, axis=1)


def cnormc_np(X):
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def diagprod_np(A, B):
    return np.sum(np.conj(A) * B, axis=0)


def sqcolnorms_np(X):
    return np.sum(np.conj(X) * X, axis=0)


def sqrownorms_np(X):
    return np.sum(np.conj(X) * X, axis=1)


def _inv_conj_transpose(R):
    k = R.shape[0]
    I = np.eye(k, dtype=R.dtype)
    return np.linalg.solve(R.T.conj(), I)


def _as_real_if_close(z):
    return np.real_if_close(z)


def supfind(i, d):
    i = int(i)
    d = float(d)
    if i < 1:
        raise ValueError("i must be >= 1")
    if not (0.0 < d < 1.0):
        raise ValueError("d must be in (0,1)")

    alphalist = np.arange(1.0, -1e-12, -0.01)
    for a in alphalist:
        fa = gammainc(a * i / 2.0, i / 2.0)
        if fa <= d:
            return float(a)
    return 0.0


def chol_shift_psd(G, ridge0=0.0):
    G = sym(G)
    if G.size == 0:
        return float(ridge0)

    if np.iscomplexobj(G):
        G = np.real(G)

    w = la.eigvalsh(G)
    wmin = float(w[0])
    wmax = float(w[-1])

    eps = float(np.finfo(G.dtype).eps)
    lam = float(ridge0)

    if wmax > 0.0:
        lam = max(lam, float(np.sqrt(eps) * wmax))
    if wmin < 0.0:
        lam = max(lam, float(-wmin) + float(np.sqrt(eps) * max(wmax, 1.0)))

    return float(lam)


def safe_cholesky_upper_psd(H, ridge=0.0, jitter0=0.0, max_tries=60):
    """Stable upper-triangular Cholesky for Hermitian PSD-ish matrices."""
    Hh = sym(np.asarray(H))
    n = int(Hh.shape[0])
    I = np.eye(n, dtype=Hh.dtype)

    lam = 0.0 if ridge is None else float(ridge)
    if lam != 0.0:
        Hh = Hh + lam * I

    try:
        return np.linalg.cholesky(Hh).T
    except np.linalg.LinAlgError:
        shift = max(float(jitter0), 0.0)
        for _ in range(int(max_tries)):
            try:
                return np.linalg.cholesky(Hh + shift * I).T
            except np.linalg.LinAlgError:
                shift = 2.0 * shift if shift > 0.0 else max(1e-15, chol_shift_psd(Hh, 0.0))
        return np.linalg.cholesky(Hh + shift * I).T


def pinv_hermitian(H, rcond):
    try:
        return np.linalg.pinv(H, rcond=float(rcond), hermitian=True)
    except TypeError:
        return np.linalg.pinv(H, rcond=float(rcond))
