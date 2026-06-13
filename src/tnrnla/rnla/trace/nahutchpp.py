import numpy as np

from .randomvector import rvec
from .common import ensure_oracle_and_dim


def _as_col_matrix(X):
    X = np.asarray(X)
    if X.ndim == 1:
        return X[:, None]
    return X


def nahutchpp(
    matvec_oracle,
    num_queries,
    dimension=-1,
    *,
    vec_type="gaussian",
    resphere=False,
    seed=None,
):
    Afun, n = ensure_oracle_and_dim(matvec_oracle, dimension)
    m = int(num_queries)
    if m < 4:
        raise ValueError("num_queries must be >= 4")

    rng = np.random.default_rng(seed)

    s = int(round(m / 4))
    r = int(round(m / 2))
    g = int(round(m / 4))

    if bool(resphere):
        raise ValueError("nahutchpp resphere mode has been removed")

    rv = rvec(n, mode=vec_type, seed=rng)

    S = _as_col_matrix(rv.sample(s))
    R = _as_col_matrix(rv.sample(r))

    Z = _as_col_matrix(Afun(R))
    W = _as_col_matrix(Afun(S))

    M = S.T.conj() @ Z
    Qm, U = np.linalg.qr(M.T.conj(), mode="reduced")

    X = Z @ Qm
    Y = (W @ np.linalg.inv(U)).T.conj()

    trest1 = float(np.real(np.trace(Y @ X)))

    # Hutch correction on residual (A - X Y)
    G = _as_col_matrix(rv.sample(g))

    GG = _as_col_matrix(Afun(G))
    termA = float(np.real(np.trace(G.T.conj() @ GG)))
    termB = float(np.real(np.trace((G.T.conj() @ X) @ (Y @ G))))

    return trest1 + 4.0 * (termA - termB) / m
