import numpy as np
import numpy.linalg as la
from ..tn.stopping import Cutoff,FixedDimension
import scipy.linalg as sla

def truncated_svd(A, stop=Cutoff(1e-14), abstol=0.0, *, check_finite=False):
    """
    Always use SciPy SVD with LAPACK driver 'gesdd' (divide-and-conquer).
    """
    U, s, Vt = sla.svd(
        A,
        full_matrices=False,
        lapack_driver="gesdd",
        check_finite=check_finite,
    )

    if (stop.cutoff is None) and (stop.outputdim is None) and (abstol == 0.0):
        return U, s, Vt

    if getattr(stop, "maxdim", None) is None or stop.maxdim is None:
        maxdim_eff = len(s)
    else:
        try:
            maxdim_eff = int(stop.maxdim)
        except Exception:
            maxdim_eff = len(s)
        maxdim_eff = max(0, min(maxdim_eff, len(s)))

    idx = len(s)

    if stop.outputdim is None:
        snorm = np.linalg.norm(s)
        cutoff = (snorm * (0.0 if stop.cutoff is None else stop.cutoff) + abstol) ** 2

        tail_norm_sq = 0.0
        cutoff_met = False
        for i in range(len(s) - 1, -1, -1):
            tail_norm_sq += float(s[i] * s[i])
            if tail_norm_sq > cutoff:
                idx = i + 1
                cutoff_met = True
                break

        if cutoff_met:
            idx = min(idx, maxdim_eff)
        else:
            idx = 0
    else:
        idx = int(min(idx, stop.outputdim, maxdim_eff))

    # --- Safeguard:---
    if idx == 0:
        if np.any(np.abs(s) > 0):  # only if A is not truly zero
            idx = 1

    return U[:, :idx], s[:idx], Vt[:idx, :]


def truncated_eig(A, stop = Cutoff(1e-14)):
    '''
    Perform a truncated eigendecomposition on matrix A such that the sum of 
    the eigenvalues beyond the truncation point is less than the specified cutoff.
    '''
 
    eigenvalues, eigenvectors = la.eigh(A)
    idx = np.abs(eigenvalues).argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if (stop.cutoff is None) and (stop.outputdim is None):    
        return eigenvalues, eigenvectors
    
    truncation_idx = len(eigenvalues)  # Default to no truncation.
    if stop.outputdim is None:
        total_frobenius_norm = np.sqrt(np.sum(eigenvalues**2))
        frobenius_norm_cutoff = total_frobenius_norm * stop.cutoff**2
        tail_frobenius_norm = 0.0
        
        for i in range(len(eigenvalues) - 1, -1, -1):
            tail_frobenius_norm += eigenvalues[i]**2
            if np.sqrt(tail_frobenius_norm) > frobenius_norm_cutoff:
                truncation_idx = i + 1
                break
        truncation_idx = min(truncation_idx, stop.maxdim)
    else:
        truncation_idx = min(truncation_idx, stop.outputdim)

    return eigenvalues[:truncation_idx], eigenvectors[:, :truncation_idx], np.conj(eigenvectors[:, :truncation_idx]).T
