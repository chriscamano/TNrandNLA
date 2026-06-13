import numpy as np

from tnrnla.linalg.utils import (
    cnormc_np,
    diagprod_np,
    sqcolnorms_np,
    _inv_conj_transpose,
    _as_real_if_close,
)
from tnrnla.rnla.lra.qb import qb
from .common import ensure_oracle_and_dim


def xtrace(matvec_oracle, num_queries, dimension=-1, *, vec_type="gaussian", seed=None, resphere=False):
    """XTrace estimator with optional resphering (default True).

    When resphere=True, this matches Program 14.4 xtrace_resphere.m.
    """
    A_mv, n = ensure_oracle_and_dim(matvec_oracle, dimension)
    n = int(n)
    s = int(num_queries)
    if s < 2:
        raise ValueError("s must be >= 2")

    k = int(np.floor(s / 2))
    k = max(1, min(k, n))

    Q, R, Om = qb(A_mv, n=n, k=k, seed=seed, dtype=np.float64, vec_type=vec_type)
    k_red = int(Q.shape[1])
    R = R[:k_red, :k_red]
    Om = Om[:, :k_red]

    S = cnormc_np(_inv_conj_transpose(R))

    Z = A_mv(Q)
    H = Q.T.conj() @ Z
    W = Q.T.conj() @ Om
    T = Z.T.conj() @ Om
    X = W - S * diagprod_np(W, S)[None, :]

    core = (
        np.trace(H) * np.ones(k_red, dtype=H.dtype)
        - diagprod_np(S, H @ S)
        - diagprod_np(T, X)
        + diagprod_np(X, H @ X)
        + diagprod_np(W, S) * diagprod_np(S, R)
    )

    if bool(resphere):
        denom = sqcolnorms_np(Om) - sqcolnorms_np(X)
        denom = denom + 1e-15
        alpha = (float(n) - float(k_red) + 1.0) / denom
        tr_vec = (
            np.trace(H) * np.ones(k_red, dtype=H.dtype)
            - diagprod_np(S, H @ S)
            + alpha
            * (
                -diagprod_np(T, X)
                + diagprod_np(X, H @ X)
                + diagprod_np(W, S) * diagprod_np(S, R)
            )
        )
    else:
        tr_vec = core

    tr = np.mean(tr_vec)
    est = (np.std(tr_vec, ddof=1) / np.sqrt(float(k_red))) if k_red > 1 else 0.0
    return _as_real_if_close(tr), float(_as_real_if_close(est))
