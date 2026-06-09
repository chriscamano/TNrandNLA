import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

def relerr(x, y, eps=1e-300):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 0 and y.ndim == 0:
        return float(np.abs(x - y) / max(np.abs(y), eps))
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), eps))


def nuc_sym(M):
    M = sym(M)
    return float(np.sum(np.abs(la.eigvalsh(M))))


def rel_fro(A_hat, A_true):
    return float(la.norm(A_hat - A_true, "fro") / (la.norm(A_true, "fro") + 1e-300))


def rel_nuclear(A_hat, A_true):
    s_diff = sla.svdvals(A_hat - A_true)
    s_true = sla.svdvals(A_true)
    return float(np.sum(s_diff) / (np.sum(s_true) + 1e-300))


def rel_trace(z_hat, z_true):
    return float(abs(z_hat - z_true) / (abs(z_true) + 1e-300))


def rel_fro_dense_over_mps(A_mps, B_dense):
    A = A_mps.to_dense()
    den = la.norm(A, "fro")
    num = la.norm(A - B_dense, "fro")
    return float(num / den) if den != 0.0 else float(num)


def rel_nuc_dense_over_mps(A_mps, B_dense):
    A = A_mps.to_dense()
    den = nuc_sym(A)
    num = nuc_sym(A - B_dense)
    return float(num / den) if den != 0.0 else float(num)
