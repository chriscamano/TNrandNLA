#----------needs more testing-----------
import numpy as np

from tnrnla.linalg.utils import cnormc_np, diagprod_np, _inv_conj_transpose


def _ensure_oracle_pair(B, Bt, n):
    if callable(B):
        if not callable(Bt):
            raise ValueError("Bt must be callable when B is callable")
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive when B is callable")
        return B, Bt, n

    A = np.asarray(B)
    if A.ndim != 2:
        raise ValueError("dense B must be 2D")
    nA = int(A.shape[1])
    return (lambda X: A @ X), (lambda X: A.T.conj() @ X), nA


def rsi_errest(B, Bt, n, k, q, *, seed=None, dtype=np.float64):
    """RSI with leave-one-out style error estimate (Program 20.2)."""
    B_mv, Bt_mv, n = _ensure_oracle_pair(B, Bt, n)
    k = int(k)
    q = int(q)
    if k < 1:
        raise ValueError("k must be >= 1")
    if q < 0 or (q % 2) != 0:
        raise ValueError("q must be a nonnegative even integer")

    rng = np.random.default_rng(seed)
    Om = rng.standard_normal((n, k)).astype(dtype, copy=False)

    Z = B_mv(Om)
    Y = Z

    for _ in range(max(q // 2 - 1, 0)):
        Y = B_mv(Bt_mv(Y))

    Q, R = np.linalg.qr(Y, mode="reduced")
    k_eff = int(min(Q.shape[1], R.shape[0], R.shape[1], Z.shape[1]))
    if k_eff < 1:
        raise ValueError("effective rank is zero in rsi_errest")

    Qk = Q[:, :k_eff]
    Rk = R[:k_eff, :k_eff]
    Zk = Z[:, :k_eff]

    C = Bt_mv(Qk)
    UU, svals, Vh = np.linalg.svd(C.T.conj(), full_matrices=False)
    U = Qk @ UU
    D = np.diag(svals)
    V = Vh.T.conj()

    S = cnormc_np(_inv_conj_transpose(Rk))
    QtZ = Qk.T.conj() @ Zk
    est_mat = Zk - Qk @ QtZ + Qk @ (S * diagprod_np(S, QtZ)[None, :])
    est = float(np.real(np.linalg.norm(est_mat, ord="fro") / np.sqrt(float(k_eff))))

    return U, D, V, est
