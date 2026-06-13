import numpy as np
from tnrnla.linalg.utils import sym


def rsvd_sparse(A, k, zeta=4, sketch="sparsestack"):
    """Randomized eigenvalue approximation for Hermitian PSD-like operators."""
    _ = sketch
    n = A.shape[0]
    ell = int(min(n, max(int(k) + int(zeta), int(k))))
    omega = np.random.standard_normal((n, ell))
    Y = A @ omega
    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = sym(Q.conj().T @ (A @ Q))
    evals = np.linalg.eigvalsh(B)
    evals = np.maximum(evals[-int(k):], 0.0)
    return evals
