import numpy as np
import numpy.linalg as la

def lq(A):
    """
    Thin LQ factorization for complex arrays.

    For A with shape (m, n), returns L, Q such that
    A = L @ Q
    Q has orthonormal rows  Q @ Q.conj().T = I_r
    L is lower trapezoidal
    """
    Qc, Rc = np.linalg.qr(A.conj().T, mode="reduced")
    L = Rc.conj().T
    Q = Qc.conj().T
    return L, Q