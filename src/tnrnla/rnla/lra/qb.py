import numpy as np


def _ensure_oracle_and_dim(matvec_oracle, dimension):
    if callable(matvec_oracle):
        if int(dimension) < 0:
            raise ValueError("dimension must be provided when matvec_oracle is callable")
        return matvec_oracle, int(dimension)
    A = np.asarray(matvec_oracle)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix input must be square")
    n = int(A.shape[0])
    return (lambda X: A @ X), n


def _sample_probes(n, k, *, vec_type="gaussian", seed=None):
    rng = np.random.default_rng(seed)
    vt = str(vec_type).lower().strip()
    if vt in ("gaussian", "normal"):
        return rng.standard_normal((n, k))
    if vt in ("uniform", "real_uniform", "unif"):
        a = np.sqrt(3.0)
        return rng.uniform(-a, a, size=(n, k))
    if vt in ("rademacher", "rade"):
        return (2.0 * rng.integers(0, 2, size=(n, k), dtype=np.int8).astype(np.float64) - 1.0)
    if vt in ("sphere", "spherical"):
        X = rng.standard_normal((n, k))
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms = np.where(norms > 0.0, norms, 1.0)
        return X * (np.sqrt(float(n)) / norms)
    if vt in ("complex_gaussian", "cgaussian", "cnormal"):
        z1 = rng.standard_normal((n, k))
        z2 = rng.standard_normal((n, k))
        return (z1 + 1j * z2) / np.sqrt(2.0)
    if vt in ("complex_uniform", "cunif", "cuniform"):
        a = np.sqrt(3.0)
        z1 = rng.uniform(-a, a, size=(n, k))
        z2 = rng.uniform(-a, a, size=(n, k))
        return (z1 + 1j * z2) / np.sqrt(2.0)
    if vt in ("complex_rademacher", "crademacher", "crade", "complex_radehamcher", "complex_rade"):
        r1 = 2.0 * rng.integers(0, 2, size=(n, k)) - 1.0
        r2 = 2.0 * rng.integers(0, 2, size=(n, k)) - 1.0
        return (r1 + 1j * r2) / np.sqrt(2.0)
    if vt in ("steinhaus", "unitcircle"):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=(n, k))
        return np.exp(1j * theta)
    if vt in ("complex_sphere", "csphere", "cspherical"):
        z1 = rng.standard_normal((n, k))
        z2 = rng.standard_normal((n, k))
        X = (z1 + 1j * z2) / np.sqrt(2.0)
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms = np.where(norms > 0.0, norms, 1.0)
        return X * (np.sqrt(float(n)) / norms)
    raise ValueError(f"Unsupported vec_type for qb: {vec_type!r}")


def qb(matvec_oracle, n, k, *, seed=None, dtype=np.float64, vec_type="gaussian"):
    """Single-pass randomized QB step: Om -> Y -> QR.

    Returns (Q, R, Om).
    """
    A_mv, n0 = _ensure_oracle_and_dim(matvec_oracle, n)
    n = int(n0)
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")

    Om = _sample_probes(n, k, vec_type=vec_type, seed=seed).astype(dtype, copy=False)
    Y = A_mv(Om)
    Q, R = np.linalg.qr(Y, mode="reduced")
    return Q, R, Om
