#----------needs more testing-----------
import numpy as np
import scipy.linalg as sla

from tnrnla.linalg.utils import sqrownorms_np, sym


_EPS = 1e-15


def _ensure_acol_and_dim(Acol, n):
    if callable(Acol):
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive when Acol is callable")
        return Acol, n

    A = np.asarray(Acol)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("dense A must be square")
    nA = int(A.shape[0])

    def _acol(I):
        idx = np.asarray(I, dtype=int)
        if idx.ndim == 0:
            return A[:, int(idx)]
        idx = idx.ravel()
        return A[:, idx]

    return _acol, nA


def _ensure_asub(Asub, n):
    if callable(Asub):
        return Asub

    A = np.asarray(Asub)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("dense Asub source must be square")
    if int(A.shape[0]) != int(n):
        raise ValueError("Asub matrix size must match n")

    def _asub(I):
        idx = np.asarray(I, dtype=int).ravel()
        return A[np.ix_(idx, idx)]

    return _asub


def _get_cols(acol, I, n):
    idx = np.asarray(I, dtype=int)
    if idx.ndim == 0:
        col = np.asarray(acol(int(idx)))
        col = col.reshape(-1)
        if col.shape[0] != n:
            raise ValueError("Acol(i) must return a vector with length n")
        return col

    idx = idx.ravel()
    G = np.asarray(acol(idx))
    if G.ndim == 1:
        if idx.size != 1:
            raise ValueError("Acol(I) for multi-index must return 2D array")
        G = G.reshape(-1, 1)
    if G.shape[0] != n and G.shape[1] == n:
        G = G.T
    if G.shape[0] != n or G.shape[1] != idx.size:
        raise ValueError("Acol(I) must return shape (n, len(I))")
    return G


def _get_submatrix(asub, I):
    idx = np.asarray(I, dtype=int).ravel()
    H = np.asarray(asub(idx))
    if H.ndim != 2 or H.shape[0] != idx.size or H.shape[1] != idx.size:
        raise ValueError("Asub(I) must return shape (len(I), len(I))")
    return H


def _weighted_sample(rng, d, size, *, replace):
    w = np.asarray(d, dtype=float).copy()
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    if np.sum(w) <= 0.0:
        return rng.choice(w.size, size=int(size), replace=replace)
    p = w / np.sum(w)
    return rng.choice(w.size, size=int(size), replace=replace, p=p)


def _pivot_scale(val):
    return float(np.sqrt(max(float(np.real(val)), _EPS)))


def pivpartchol(Acol, n, pivots):
    """Pivoted partial Cholesky with a provided pivot list (Program 3.1)."""
    acol, n = _ensure_acol_and_dim(Acol, n)
    piv = np.asarray(pivots, dtype=int).ravel()
    if piv.size == 0:
        return np.zeros((n, 0), dtype=float)

    F = np.zeros((n, piv.size), dtype=np.complex128)
    for i, p in enumerate(piv):
        if p < 0 or p >= n:
            raise ValueError("pivot index out of bounds")
        ai = _get_cols(acol, int(p), n)
        if i > 0:
            ai = ai - F[:, :i] @ F[p, :i].conj()
        F[:, i] = ai / _pivot_scale(ai[p])
    return np.real_if_close(F)


def greedy_chol(Acol, d, k):
    """Greedy pivoted partial Cholesky (Program 3.2)."""
    d = np.asarray(d, dtype=float).reshape(-1)
    n = int(d.size)
    acol, n2 = _ensure_acol_and_dim(Acol, n)
    if n2 != n:
        raise ValueError("len(d) must match matrix size")

    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")

    F = np.zeros((n, k), dtype=np.complex128)
    S = np.zeros(k, dtype=int)

    for i in range(k):
        p = int(np.argmax(d))
        S[i] = p
        ai = _get_cols(acol, p, n)
        if i > 0:
            ai = ai - F[:, :i] @ F[p, :i].conj()
        F[:, i] = ai / _pivot_scale(ai[p])
        d = np.maximum(d - np.abs(F[:, i]) ** 2, 0.0)

    return np.real_if_close(F), S


def rpcholesky(Acol, d, k, *, seed=None):
    """Randomly pivoted Cholesky (Program 4.1)."""
    d = np.asarray(d, dtype=float).reshape(-1)
    n = int(d.size)
    acol, n2 = _ensure_acol_and_dim(Acol, n)
    if n2 != n:
        raise ValueError("len(d) must match matrix size")

    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    rng = np.random.default_rng(seed)

    F = np.zeros((n, k), dtype=np.complex128)
    S = np.zeros(k, dtype=int)

    for i in range(k):
        p = int(_weighted_sample(rng, d, 1, replace=True)[0])
        S[i] = p
        ai = _get_cols(acol, p, n)
        if i > 0:
            ai = ai - F[:, :i] @ F[p, :i].conj()
        F[:, i] = ai / _pivot_scale(ai[p])
        d = np.maximum(d - np.abs(F[:, i]) ** 2, 0.0)

    return np.real_if_close(F), S


def robust_block_filter(H, tau, lmax):
    """Robust block filtering used by RBRP Cholesky (Program 8.2)."""
    H = sym(np.asarray(H))
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    b = int(H.shape[0])
    lmax = int(max(0, lmax))
    if b == 0 or lmax == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 0), dtype=H.dtype)

    F = np.zeros((b, b), dtype=H.dtype)
    d = np.maximum(np.real(np.diag(H)).copy(), 0.0)
    trace0 = float(np.sum(d))
    selected = []

    for i in range(min(b, lmax)):
        p = int(np.argmax(d))
        selected.append(p)
        hi = H[:, p] - F[:, :i] @ F[p, :i].conj()
        F[:, i] = hi / _pivot_scale(hi[p])
        d = np.maximum(d - np.abs(F[:, i]) ** 2, 0.0)
        if float(np.sum(d)) <= float(tau) * max(trace0, _EPS):
            break

    T = np.asarray(selected, dtype=int)
    L = F[np.ix_(T, np.arange(T.size, dtype=int))]
    return T, L


def block_rpcholesky(Acol, d, k, b, *, seed=None):
    """Block RPCholesky (Program 8.1)."""
    d = np.asarray(d, dtype=float).reshape(-1)
    n = int(d.size)
    acol, n2 = _ensure_acol_and_dim(Acol, n)
    if n2 != n:
        raise ValueError("len(d) must match matrix size")

    k = int(k)
    b = int(b)
    if k < 1 or b < 1:
        raise ValueError("k and b must be >= 1")
    rng = np.random.default_rng(seed)

    F = np.zeros((n, k), dtype=np.complex128)
    S = np.zeros(k, dtype=int)
    shift = 4.0 * max(float(np.max(d)), 0.0) * float(np.finfo(np.float64).eps)

    i = 0
    while i < k:
        b_now = int(min(b, k - i, n))
        Snew = _weighted_sample(rng, d, b_now, replace=False)
        S[i : i + b_now] = Snew

        G = _get_cols(acol, Snew, n)
        if i > 0:
            G = G - F[:, :i] @ F[Snew, :i].conj().T

        Hblk = sym(G[Snew, :]) + shift * np.eye(b_now, dtype=G.dtype)
        R = sla.cholesky(Hblk, lower=False, check_finite=False)
        F[:, i : i + b_now] = sla.solve_triangular(R, G.T, lower=False, check_finite=False).T

        d = np.maximum(d - np.real(sqrownorms_np(F[:, i : i + b_now])), 0.0)
        i += b_now

    return np.real_if_close(F), S


def rbrp_chol(Acol, Asub, d, k, b, *, seed=None):
    """RBRP Cholesky (Program 8.3)."""
    d = np.asarray(d, dtype=float).reshape(-1)
    n = int(d.size)
    acol, n2 = _ensure_acol_and_dim(Acol, n)
    if n2 != n:
        raise ValueError("len(d) must match matrix size")
    asub = _ensure_asub(Asub, n)

    k = int(k)
    b = int(b)
    if k < 1 or b < 1:
        raise ValueError("k and b must be >= 1")
    rng = np.random.default_rng(seed)

    F = np.zeros((n, k), dtype=np.complex128)
    S = np.zeros(k, dtype=int)

    i = 0
    while i < k:
        b_now = int(min(b, n, k - i))
        Sp = _weighted_sample(rng, d, b_now, replace=False)

        H = _get_submatrix(asub, Sp)
        if i > 0:
            H = H - F[Sp, :i] @ F[Sp, :i].conj().T

        Tloc, L = robust_block_filter(H, 1.0 / float(b_now), k - i)
        if Tloc.size == 0:
            break

        T = Sp[Tloc]
        l = int(T.size)
        S[i : i + l] = T

        G = _get_cols(acol, T, n)
        if i > 0:
            G = G - F[:, :i] @ F[T, :i].conj().T

        try:
            F[:, i : i + l] = sla.solve_triangular(L.T.conj(), G.T, lower=False, check_finite=False).T
        except Exception:
            F[:, i : i + l] = G @ np.linalg.pinv(L.T.conj())

        d = np.maximum(d - np.real(sqrownorms_np(F[:, i : i + l])), 0.0)
        i += l

    return np.real_if_close(F[:, :i]), S[:i]


def rejection_sample_submatrix(H, u, lmax, *, seed=None, rng=None):
    """Rejection sampling of pivots from a residual submatrix (Program 8.5)."""
    H = sym(np.asarray(H)).astype(np.complex128, copy=True)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    b = int(H.shape[0])
    u = np.asarray(u, dtype=float).reshape(-1)
    if u.size != b:
        raise ValueError("u must have length size(H,1)")
    lmax = int(max(0, lmax))

    if rng is None:
        rng = np.random.default_rng(seed)

    accepted = []
    for j in range(b):
        hjj = max(float(np.real(H[j, j])), 0.0)
        if hjj <= _EPS:
            continue
        uj = max(float(u[j]), 0.0)
        if uj * rng.random() > hjj:
            continue

        accepted.append(j)
        H[j:, j] = H[j:, j] / np.sqrt(hjj)
        if j + 1 < b:
            c = H[j + 1 :, j].copy()
            H[j + 1 :, j + 1 :] = H[j + 1 :, j + 1 :] - np.outer(c, c.conj())

        if len(accepted) >= lmax:
            break

    T = np.asarray(accepted, dtype=int)
    if T.size == 0:
        return T, np.zeros((0, 0), dtype=H.dtype)
    L = np.tril(H[np.ix_(T, T)])
    return T, L


def acc_rpcholesky(Acol, Asub, d, k, b, *, seed=None):
    """Accelerated RPCholesky (Program 8.4)."""
    d = np.asarray(d, dtype=float).reshape(-1)
    n = int(d.size)
    acol, n2 = _ensure_acol_and_dim(Acol, n)
    if n2 != n:
        raise ValueError("len(d) must match matrix size")
    asub = _ensure_asub(Asub, n)

    k = int(k)
    b = int(b)
    if k < 1 or b < 1:
        raise ValueError("k and b must be >= 1")
    rng = np.random.default_rng(seed)

    F = np.zeros((n, k), dtype=np.complex128)
    S = np.zeros(k, dtype=int)

    i = 0
    while i < k:
        b_now = int(min(b, n, k - i))
        Sp = _weighted_sample(rng, d, b_now, replace=True)

        H = _get_submatrix(asub, Sp)
        if i > 0:
            H = H - F[Sp, :i] @ F[Sp, :i].conj().T

        Tloc, L = rejection_sample_submatrix(H, np.real(np.diag(H)), k - i, rng=rng)
        if Tloc.size == 0:
            break

        T = Sp[Tloc]
        l = int(T.size)
        S[i : i + l] = T

        G = _get_cols(acol, T, n)
        if i > 0:
            G = G - F[:, :i] @ F[T, :i].conj().T

        try:
            F[:, i : i + l] = sla.solve_triangular(L.T.conj(), G.T, lower=False, check_finite=False).T
        except Exception:
            F[:, i : i + l] = G @ np.linalg.pinv(L.T.conj())

        d = np.maximum(d - np.real(sqrownorms_np(F[:, i : i + l])), 0.0)
        i += l

    return np.real_if_close(F[:, :i]), S[:i]
def acc_rpcholesky_eps(Acol, Asub, d, k, b, *, eps_stop=0.0, seed=None):
    """
    Accelerated RPCholesky with an eps stop rule.

    Stops early when max(d) <= eps_stop, where d is the running diagonal residual
    estimate. Returns F[:, :i], S[:i] just like acc_rpcholesky.

    Parameters
    ----------
    Acol, Asub
        Same call signatures as in tnrnla.rnla.lra.cholesky.acc_rpcholesky
    d : array_like, shape (n,)
        Initial diagonal weights, typically diag(A) clipped to >= 0
    k : int
        Target rank cap
    b : int
        Block size
    eps_stop : float
        Stop when max(d) <= eps_stop
    seed : int or None
        RNG seed
    """
    d = np.asarray(d, dtype=float).reshape(-1)
    n = int(d.size)

    acol, n2 = _ensure_acol_and_dim(Acol, n)
    if n2 != n:
        raise ValueError("len(d) must match matrix size")
    asub = _ensure_asub(Asub, n)

    k = int(k)
    b = int(b)
    if k < 1 or b < 1:
        raise ValueError("k and b must be >= 1")

    eps_stop = float(eps_stop)
    rng = np.random.default_rng(seed)

    F = np.zeros((n, k), dtype=np.complex128)
    S = np.zeros(k, dtype=int)

    i = 0
    while i < k:
        if float(np.max(d)) <= eps_stop:
            break

        b_now = int(min(b, n, k - i))
        Sp = _weighted_sample(rng, d, b_now, replace=True)

        H = _get_submatrix(asub, Sp)
        if i > 0:
            H = H - F[Sp, :i] @ F[Sp, :i].conj().T

        Tloc, L = rejection_sample_submatrix(H, np.real(np.diag(H)), k - i, rng=rng)
        if Tloc.size == 0:
            break

        T = Sp[Tloc]
        l = int(T.size)
        S[i : i + l] = T

        G = _get_cols(acol, T, n)
        if i > 0:
            G = G - F[:, :i] @ F[T, :i].conj().T

        try:
            F[:, i : i + l] = sla.solve_triangular(
                L.T.conj(), G.T, lower=False, check_finite=False
            ).T
        except Exception:
            F[:, i : i + l] = G @ np.linalg.pinv(L.T.conj())

        d = np.maximum(d - np.real(sqrownorms_np(F[:, i : i + l])), 0.0)
        i += l

    return np.real_if_close(F[:, :i]), S[:i]