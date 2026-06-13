import numpy as np
import scipy.linalg as sla

from tnrnla.linalg.utils import sym
from tnrnla.tn.stopping import Cutoff
from tnrnla.tn.trp import TRP

def truncated_eigh(A, *, tol):
    A = sym(np.asarray(A))
    lam, V = np.linalg.eigh(A)
    lam = np.real(lam[::-1])
    V = V[:, ::-1]

    cutoff = float(tol) * float(np.max(np.abs(lam)))
    keep = lam > cutoff
    if not np.any(keep):
        raise np.linalg.LinAlgError("matrix is numerically rank deficient")

    return lam[keep], V[:, keep]



def trp_round(X, tol):
    if tol is None:
        return X
    if hasattr(X, "src_round"):
        X.src_round(stop=Cutoff(float(tol)))
    else:
        X.round(stop=Cutoff(float(tol)))
    return X

def mps_nystrom(
    A,
    n,
    k,
    chi,
    seed=0,
    Om=None,
    *,
    tol=None,
    just_eigs=False,
):
    Omega = Om if Om is not None else TRP.gaussian(
        n_sites=n,k=k,chi=chi,seed=int(seed))

    Y = A @ Omega
    if tol is not None:
        trp_round(Y, tol)

    #==================== Compute Gram matrices ====================
    G = sym(np.asarray(Omega.gram(assume_hermitian=True)))
    C = sym(np.asarray(Omega.gram(Y, assume_hermitian=False)))
    Q = sym(np.asarray(Y.gram(Y, assume_hermitian=True)))

    #==================== Find best shift nu ====================
    root_eps = np.sqrt(np.finfo(np.real(G).dtype).eps)
    T = sla.cholesky(G, lower=False, check_finite=False)
    Ik = np.eye(k, dtype=T.dtype)
    Tinv = sla.solve_triangular(T, Ik, lower=False, check_finite=False)
    B = sym(Tinv.conj().T @ C @ Tinv)
    evals = np.real(np.linalg.eigvalsh(B))
    nu = max(0.0, root_eps * evals[-1] - evals[0])

    #==================== Gram access to Y_nu^*Y_nu ====================
    R = sla.cholesky(sym(C + nu * G), lower=False, check_finite=False)
    S = sym(Q + 2.0 * nu * C + (nu * nu) * G)

    # ==================== Extract eigeninformation ====================
    Rinv = sla.solve_triangular(R, Ik, lower=False, check_finite=False)
    E = sym(Rinv.conj().T @ S @ Rinv)

    Lambda_nu, V = np.linalg.eigh(E)
    Lambda_nu = np.real(Lambda_nu[::-1])
    V = V[:, ::-1]

    Lambda = np.maximum(Lambda_nu - nu, 0.0)

    if just_eigs:
        return Lambda

    # ==================== Compute eigenvectors ====================
    Y_nu = Y + nu * Omega

    W = sla.solve_triangular(R, V, lower=False, check_finite=False)
    W = W / np.sqrt(Lambda_nu)[None, :]

    U = Y_nu @ W

    if tol is not None:
        trp_round(U, tol)

    return U, Lambda


def mps_nystrom_gram_eigh(
    A_mpo,
    n,
    k,
    chi,
    seed=0,
    Om=None,
    *,
    round_tol=1e-14,
    just_eigs=False,
):
    Omega = Om if Om is not None else TRP.gaussian(
        n_sites=n,
        k=k,
        chi=chi,
        d=2,
        seed=int(seed),
    )

    Y = A_mpo @ Omega
    if round_tol is not None:
        Y.round(stop=Cutoff(float(round_tol)))

    S = sym(np.asarray(Omega.gram(assume_hermitian=True)))
    H = sym(np.asarray(Omega.gram(Y, assume_hermitian=True)))
    T = sym(np.asarray(Y.gram(Y, assume_hermitian=True)))

    root_eps = np.sqrt(np.finfo(np.real(S).dtype).eps)

    Lambda_S, V_S = truncated_eigh(S, tol=root_eps)
    R = V_S / np.sqrt(Lambda_S)[None, :]

    H_tilde = sym(R.conj().T @ H @ R)
    Lambda_H, V_H = truncated_eigh(H_tilde, tol=root_eps)

    RH = R @ V_H
    RH = RH / np.sqrt(Lambda_H)[None, :]

    C = sym(RH.conj().T @ T @ RH)
    Lambda, V_C = truncated_eigh(C, tol=root_eps)

    r = min(int(k), Lambda.size)
    d = np.full(int(k), root_eps, dtype=np.float64)
    d[:r] = np.maximum(Lambda[:r], root_eps)

    if just_eigs:
        return d

    M = RH @ (V_C[:, :r] / np.sqrt(Lambda[:r])[None, :])
    U = Y @ M
    return U, d
