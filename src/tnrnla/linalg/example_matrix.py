import numpy as np


def _rand_orth(n, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(G, mode="reduced")
    return Q


def rpsdmat_poly(n=1000, alpha=2.0, seed=0, dtype=np.float64):
    """
    Random symmetric PSD matrix with polynomially decaying eigenvalues.

    Returns (A, trA) where:
      - A is (n, n)
      - trA is float(trace(A))
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    alpha = float(alpha)
    Q = _rand_orth(n, seed=seed)

    lam = np.power(np.arange(1, n + 1, dtype=np.float64), -alpha)
    A = (Q * lam[None, :]) @ Q.T
    A = 0.5 * (A + A.T)
    A = np.asarray(A, dtype=dtype)

    trA = float(np.real(np.trace(A)))
    return A, trA


def rpsdmat_flat(n=1000, seed=0, dtype=np.float64):
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    U = _rand_orth(n, seed=seed)
    d = np.linspace(1.0, 3.0, n, dtype=np.float64)
    A = U @ (d[:, None] * U.T)
    A = 0.5 * (A + A.T)
    A = np.asarray(A, dtype=dtype)
    return A, float(np.sum(d))


def rpsdmat_exp(n=1000, alpha=0.7, seed=0, dtype=np.float64):
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    alpha = float(alpha)
    U = _rand_orth(n, seed=seed)
    d = np.power(alpha, np.arange(n, dtype=np.float64))
    A = U @ (d[:, None] * U.T)
    A = 0.5 * (A + A.T)
    A = np.asarray(A, dtype=dtype)
    return A, float(np.sum(d))


def rpsdmat_step(n=1000, split=50, high=1.0, low=1e-3, seed=0, dtype=np.float64):
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    split = int(split)
    split = max(0, min(split, n))
    U = _rand_orth(n, seed=seed)
    d = np.empty(n, dtype=np.float64)
    d[:split] = float(high)
    d[split:] = float(low)
    A = U @ (d[:, None] * U.T)
    A = 0.5 * (A + A.T)
    A = np.asarray(A, dtype=dtype)
    return A, float(np.sum(d))

