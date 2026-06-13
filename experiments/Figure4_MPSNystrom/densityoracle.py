import numpy as np

from tnrnla import Cutoff, MPS
from tnrnla.tn.trp import TRP


def trp_round(X, tol):
    if tol is None:
        return X
    if hasattr(X, "src_round"):
        X.src_round(stop=Cutoff(float(tol)))
    else:
        X.round(stop=Cutoff(float(tol)))
    return X


def prep_halfchain(groundstate, ell):
    psi = groundstate.copy()
    n = int(len(psi))
    ell = int(ell)
    if not (0 < ell < n):
        raise ValueError(
            f"ell must satisfy 0 < ell < {n}, got {ell}. "
            "Here ell is the number of left sites."
        )
    psi.move_pivot(ell - 1)
    return psi


def w_star_omega(psi, ell, Omega):
    ell = int(ell)
    rcut = int(np.asarray(psi[ell - 1]).shape[-1])
    Z = np.empty(
        (rcut, int(Omega.k)),
        dtype=np.result_type(psi.dtype, Omega.dtype),
    )

    for j, om in enumerate(Omega.cols):
        if ell == 1:
            A = np.asarray(psi[0])
            B = np.asarray(om[0])
            Z[:, j] = A.reshape(-1, A.shape[-1]).conj().T @ B.reshape(-1)
            continue

        env = psi[0].conj().T @ om[0]

        for i in range(1, ell - 1):
            A = psi[i]
            B = om[i]
            tmp = env @ B.reshape(B.shape[0], -1)
            tmp = tmp.reshape(env.shape[0], B.shape[1], B.shape[2])
            tmp = tmp.transpose(2, 0, 1).reshape(B.shape[2], -1)
            env = (tmp @ A.conj().reshape(-1, A.shape[-1])).T

        A = psi[ell - 1]
        B = om[ell - 1]
        tmp = env @ B.reshape(B.shape[0], -1)
        Z[:, j] = (tmp.reshape(1, -1) @ A.conj().reshape(-1, A.shape[-1])).ravel()

    return Z


def w_times_coeff(psi, ell, Z):
    Z = np.asarray(Z)
    last = psi[ell - 1]
    r = last.shape[-1]
    if Z.shape[0] != r:
        raise ValueError(
            f"Dimension mismatch: last.shape[-1] = {r}, but Z.shape[0] = {Z.shape[0]}."
        )
    out_dtype = np.result_type(last.dtype, Z.dtype)
    prefix = list(psi[:ell - 1])
    last_cols = last.reshape(-1, r) @ Z
    last_cols = last_cols.reshape(*last.shape[:-1], Z.shape[1])
    cols = [
        MPS(
            prefix + [last_cols[..., j]],
            orthform="Left",
            rounded=False,
            dtype=out_dtype,
        )
        for j in range(Z.shape[1])
    ]
    return TRP(cols, dtype=out_dtype, orthform="up")


def make_halfchain_oracle(groundstate, ell, *, tol=None):
    psi = prep_halfchain(groundstate, ell)

    def oracle(Omega):
        Z = w_star_omega(psi, ell, Omega)
        Y = w_times_coeff(psi, ell, Z)
        return Y

    return oracle
