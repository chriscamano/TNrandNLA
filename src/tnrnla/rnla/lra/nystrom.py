import numpy as np
import scipy.linalg as sla
from tnrnla.linalg.utils import sym, safe_cholesky_psd
from scipy.linalg import solve_triangular


def _as_matvec(A):
    return A if callable(A) else (lambda X: A @ X)


def nystrom(A, Omega=None, *, n=None, k=None, rng=None, form="factor", ridge=None):
    """
    Nyström sketch with two output forms.

    form="factor"
        returns F, nu, Omega, C
        where F = Ynu @ M^{-1/2} and M = Omega^* Ynu = C^* C

    form="svd"
        returns U, d, nu
        where A ≈ U diag(d) U^*
    """
    A_mv = _as_matvec(A)

    if Omega is None:
        if n is None or k is None:
            raise ValueError("provide either Omega or both n and k")
        rng = np.random.default_rng() if rng is None else rng
        Omega = rng.standard_normal((n, k))
    else:
        Omega = np.asarray(Omega)
        n, k = Omega.shape

    Y0 = A_mv(Omega)

    if ridge is None:
        nu = np.finfo(Y0.dtype).eps * np.linalg.norm(Y0, "fro") / np.sqrt(n)
    else:
        nu = ridge(Y0, Omega) if callable(ridge) else ridge

    Ynu = Y0 + nu * Omega
    M = sym(Omega.conj().T @ Ynu)
    C = safe_cholesky_psd(M)

    F = solve_triangular(C.T, Ynu.T, lower=True).T

    if form == "factor":
        return F, nu, Omega, C

    if form == "svd":
        U, s, _ = np.linalg.svd(F, full_matrices=False)
        d = np.maximum(s * s - nu, 0.0)
        return U, d, nu

    raise ValueError("form must be 'factor' or 'svd'")

