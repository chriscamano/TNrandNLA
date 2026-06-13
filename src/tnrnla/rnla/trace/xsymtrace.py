#----------needs more testing-----------
import numpy as np
import scipy.linalg as sla

from tnrnla.linalg.utils import diagprod_np, sqcolnorms_np, sym, pinv_hermitian
from .randomvector import rvec
from .common import ensure_oracle_and_dim


def xsymtrace(
    matvec_oracle,
    num_queries,
    dimension=-1,
    *,
    vec_type="rademacher",
    seed=None,
    eps_denom=1e-15,
    inv_rcond=None,
    return_debug=False,
):
    """XSymTrace for Hermitian indefinite operators (Program 20.1 style)."""
    A_mv, n = ensure_oracle_and_dim(matvec_oracle, dimension)
    s = int(num_queries)
    if s < 1:
        raise ValueError("num_queries must be >= 1")

    Om = rvec(n, mode=vec_type, seed=seed).sample(s)
    Y = A_mv(Om)
    H = sym(Om.T.conj() @ Y)

    if inv_rcond is None:
        inv_rcond = float(np.finfo(np.float64).eps) * max(1, s)

    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Hinv = pinv_hermitian(H, inv_rcond)

    # Program 20.1 uses LDL^*; we follow that path and keep a stable fallback.
    try:
        L, D, _ = sla.ldl(H, lower=True, hermitian=True, check_finite=False)
        F = np.linalg.solve(L.T.conj(), Y.T).T
        try:
            Dinv = np.linalg.inv(D)
        except np.linalg.LinAlgError:
            Dinv = pinv_hermitian(sym(D), inv_rcond)
        Z = np.linalg.solve(L, (F @ Dinv).T).T
        try:
            first = np.trace(np.linalg.solve(D, F.T.conj() @ F))
        except np.linalg.LinAlgError:
            first = np.trace(Dinv @ (F.T.conj() @ F))
    except Exception:
        Z = Y @ Hinv
        first = np.trace(Hinv @ (Y.T.conj() @ Y))

    d = np.diag(Hinv)

    eps_denom = float(eps_denom)
    absd = np.abs(d)
    phase = np.ones_like(d, dtype=Hinv.dtype)
    nz = absd > 0.0
    phase[nz] = d[nz] / absd[nz]
    d_safe = np.where(absd > eps_denom, d, phase * eps_denom)

    tr_vec = first * np.ones(s, dtype=Hinv.dtype) - sqcolnorms_np(Z) / d_safe + (np.abs(diagprod_np(Om, Z)) ** 2) / d_safe
    tr_vec = np.real(np.real_if_close(tr_vec))

    tr = float(np.mean(tr_vec))
    est = float(np.std(tr_vec, ddof=1) / np.sqrt(float(s))) if s > 1 else 0.0

    if not bool(return_debug):
        return tr, est

    dbg = {
        "n": int(n),
        "s": int(s),
        "min_abs_diag_invH": float(np.min(np.abs(d))),
        "max_abs_diag_invH": float(np.max(np.abs(d))),
    }
    return tr, est, dbg
