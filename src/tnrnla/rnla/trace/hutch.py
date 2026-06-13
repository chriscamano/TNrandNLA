import numpy as np

from .randomvector import rvec
from .common import ensure_oracle_and_dim


def hutch(matvec_oracle, num_queries, dimension=-1, *, vec_type="gaussian", seed=None):
    A_mv, n = ensure_oracle_and_dim(matvec_oracle, dimension)
    q = int(num_queries)
    if q < 1:
        raise ValueError("num_queries must be >= 1")

    G = rvec(n, mode=vec_type, seed=seed).sample(q)
    val = np.trace(G.T.conj() @ A_mv(G)) / float(q)
    return float(np.real(val))
