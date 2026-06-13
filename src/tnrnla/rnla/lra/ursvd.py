import numpy as np

from tnrnla.linalg.utils import cnormc_np, diagprod_np
from .qb import qb


def _ensure_oracle_and_dim_pair(B, Bt, dimension):
    if not callable(B) or not callable(Bt):
        raise ValueError("B and Bt must be callables")
    n = int(dimension)
    if n <= 0:
        raise ValueError("dimension must be positive")
    return B, Bt, n


def ursvd(B, Bt, n, k, *, seed=None):
    """Unbiased randomized SVD (Program 16.4 style)."""
    B_mv, Bt_mv, n = _ensure_oracle_and_dim_pair(B, Bt, n)
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")

    Q, R, Om = qb(B_mv, n=n, k=k, seed=seed, vec_type="rademacher")
    C = Bt_mv(Q)

    S = cnormc_np(np.linalg.solve(R[:k, :k].T.conj(), np.eye(k, dtype=R.dtype)))

    F = (np.eye(k, dtype=C.dtype) - (S @ S.T.conj()) / float(k)) @ C.T.conj() + (
        (S * diagprod_np(S, R[:k, :k])[None, :]) @ Om.T.conj()
    ) / float(k)

    UU, Ss, Vh = np.linalg.svd(F, full_matrices=False)
    U = Q @ UU
    Sdiag = np.diag(Ss)
    V = Vh.T.conj()
    return U, Sdiag, V
