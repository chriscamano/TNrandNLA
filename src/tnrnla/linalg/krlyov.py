import numpy as np
import numpy.linalg as la
from tnrnla.tn.stopping import Cutoff,FixedDimension
import scipy.linalg as sla

def arnoldi(A, v0, k):
    """
    Arnoldi's method with partial reorthogonalization
    """
    n = A.shape[0]
    V = np.zeros((n, k + 1), dtype=A.dtype)
    H = np.zeros((k + 1, k), dtype=A.dtype)
    
    V[:, 0] = v0 / np.linalg.norm(v0)
    for m in range(k):
        vt = A @ V[:, m]
        
        # orthogonalize vt against all previous vectors in V
        for j in range(m + 1):
            H[j, m] = np.vdot(V[:, j], vt)  #  np.vdot invokes complex conjugate transpose here
            vt -= H[j, m] * V[:, j]
        
        H[m + 1, m] = np.linalg.norm(vt)
        
        # reorthogonalize 
        for j in range(m + 1):
            correction = np.vdot(V[:, j], vt)
            vt -= correction * V[:, j]
            H[j, m] += correction
        
        H[m + 1, m] = np.linalg.norm(vt)
        
        # update basis 
        if H[m + 1, m] > 1e-10 and m != k - 1:
            V[:, m + 1] = vt / H[m + 1, m]
    
    return V, H

