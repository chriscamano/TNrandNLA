import numpy as np

import scipy.linalg as sla
from scipy.linalg import solve_triangular
from tnrnla.linalg.utils import safe_cholesky_psd,sqrownorms,sqcolnorms,diagprod,sym
from tnrnla.rnla.lra.nystrom import nystrom

def xnystrace(A, n, s, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    F, mu, Om, R = nystrom(A, n=n, k=s, rng=rng, form="factor")
    invR = solve_triangular(R, np.eye(s, dtype=R.dtype), lower=False)
    Z0 = solve_triangular(R, F.T, lower=False).T
    Z = Z0 * (sqrownorms(invR) ** (-0.5))[None, :]
    tr_vec = np.linalg.norm(F, "fro") ** 2 - sqcolnorms(Z) + np.abs(diagprod(Z, Om)) ** 2 - mu * n
    tr = np.mean(tr_vec).real
    est = (np.std(tr_vec, ddof=1) / np.sqrt(s)).real
    return tr, est


def xnystrace_gram(
    A,
    n,
    s,
    rng=None,
    *,
    denom_eps=1e-300,
    return_debug=False,
):
    rng = np.random.default_rng() if rng is None else rng

    Om = rng.standard_normal((n, s))
    Y0 = A(Om)

    G = sym(Om.T.conj() @ Om)
    K0 = sym(Om.T.conj() @ Y0)
    YY0 = sym(Y0.T.conj() @ Y0)

    eps = np.finfo(np.float64).eps
    T = sla.cholesky(G, lower=False, check_finite=False)
    invT = sla.solve_triangular(
        T, np.eye(s, dtype=G.dtype), lower=False, check_finite=False
    )
    w = np.linalg.eigvalsh(sym(invT.T.conj() @ K0 @ invT))
    mu = max(0.0, -w[0]) + np.sqrt(eps) * w[-1]

    H = sym(K0 + mu * G)
    R = safe_cholesky_psd(H)
    S = sym(YY0 + 2.0 * mu * K0 + (mu * mu) * G)

    invH = sym(sla.cho_solve((R, False), np.eye(s, dtype=H.dtype), check_finite=False))
    t1 = np.trace(S @ invH)
    M = sym(invH @ S @ invH)

    d = np.maximum(np.diag(invH), denom_eps)
    tr_vec = t1 - (np.diag(M) / d) + (1.0 / d) - mu * n

    tr = np.mean(tr_vec)
    est = np.std(tr_vec, ddof=1) / np.sqrt(s) if s > 1 else 0.0

    if not return_debug:
        return tr, est

    return tr, est, {"mu": mu, "lam_min": w[0], "lam_max": w[-1]}

