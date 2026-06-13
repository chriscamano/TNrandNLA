#----------needs more testing-----------
import numpy as np


def _ensure_oracle_pair(B, Bt, m, n):
    if callable(B):
        if not callable(Bt):
            raise ValueError("Bt must be callable when B is callable")
        m = int(m)
        n = int(n)
        if m <= 0 or n <= 0:
            raise ValueError("m and n must be positive when B is callable")
        return B, Bt, m, n

    A = np.asarray(B)
    if A.ndim != 2:
        raise ValueError("dense B must be 2D")
    mA, nA = int(A.shape[0]), int(A.shape[1])
    return (lambda X: A @ X), (lambda X: A.T.conj() @ X), mA, nA


def rsi(B, Bt, m, n, k, q, *, seed=None, dtype=np.float64):
    """Randomized SVD with subspace iteration (Program 2.2)."""
    B_mv, Bt_mv, m, n = _ensure_oracle_pair(B, Bt, m, n)
    k = int(k)
    q = int(q)
    if k < 1:
        raise ValueError("k must be >= 1")
    if q < 0:
        raise ValueError("q must be >= 0")

    rng = np.random.default_rng(seed)

    if q % 2 == 1:
        Om = rng.standard_normal((m, k)).astype(dtype, copy=False)
        Om = Bt_mv(Om)
    else:
        Om = rng.standard_normal((n, k)).astype(dtype, copy=False)

    for _ in range(q // 2):
        Om = Bt_mv(B_mv(Om))

    Y = B_mv(Om)
    Q, _ = np.linalg.qr(Y, mode="reduced")
    C = Bt_mv(Q)

    UU, svals, Vh = np.linalg.svd(C.T.conj(), full_matrices=False)
    U = Q @ UU
    S = np.diag(svals)
    V = Vh.T.conj()
    return U, S, V
