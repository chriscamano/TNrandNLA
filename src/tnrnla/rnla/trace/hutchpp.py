import numpy as np

from .randomvector import rvec
from .common import ensure_oracle_and_dim, resphere_columns


def _as_col_matrix(X):
    X = np.asarray(X)
    if X.ndim == 1:
        return X[:, None]
    return X


def hutchpp(
    matvec_oracle,
    num_queries,
    dimension=-1,
    *,
    vec_type="rademacher",
    sketch_vec_type=None,
    probe=None,
    sketch_probe=None,
    sketch_frac=2 / 3,
    resphere=False,
    seed=None,
):
    oracle, n = ensure_oracle_and_dim(matvec_oracle, dimension)
    rng = np.random.default_rng(seed)

    if bool(resphere):
        s = int(num_queries)
        if s < 3:
            raise ValueError("num_queries must be >= 3 when resphere=True")

        # Program: k=floor(s/3), k2=s-2k
        k = int(np.floor(s / 3))
        k = max(1, min(k, n))
        k2 = int(s - 2 * k)
        if k2 < 1:
            raise ValueError("num_queries too small for re-isotropic Hutch++")

        # Match reference algorithm: Gaussian Ω, Γ
        rv = rvec(n, mode="gaussian", seed=rng)
        Om = _as_col_matrix(rv.sample(k))
        Y = _as_col_matrix(oracle(Om))
        Q, _ = np.linalg.qr(Y, mode="reduced")
        k_eff = int(Q.shape[1])

        BQ = oracle(Q)

        Ga = _as_col_matrix(rv.sample(k2))
        X = Ga - Q @ (Q.T.conj() @ Ga)
        X = resphere_columns(X, np.sqrt(float(max(n - k_eff, 1))))

        BX = _as_col_matrix(oracle(X))
        tr = np.trace(Q.T.conj() @ BQ) + np.trace(X.T.conj() @ BX) / float(k2)
        return float(np.real(tr))

    if sketch_vec_type is None:
        sketch_vec_type = vec_type

    if probe is None:
        probe = rvec(n, mode=vec_type, seed=rng)
    if sketch_probe is None:
        sketch_probe = rvec(n, mode=sketch_vec_type, seed=rng)

    S_num_queries = int(np.round(num_queries * sketch_frac / 2.0))
    Hutch_num_queries = int(num_queries - S_num_queries)
    if S_num_queries < 1:
        S_num_queries = 1
        Hutch_num_queries = int(num_queries - S_num_queries)
    if Hutch_num_queries < 1:
        Hutch_num_queries = 1
        S_num_queries = int(num_queries - Hutch_num_queries)

    S = _as_col_matrix(sketch_probe.sample(S_num_queries))
    Q, _ = np.linalg.qr(_as_col_matrix(oracle(S)), mode="reduced")

    G = _as_col_matrix(probe.sample(Hutch_num_queries))
    G = G - Q @ (Q.T.conj() @ G)

    AQ = _as_col_matrix(oracle(Q))
    AG = _as_col_matrix(oracle(G))

    return float(np.real(np.trace(Q.T.conj() @ AQ) + np.trace(G.T.conj() @ AG) / float(Hutch_num_queries)))
