import numpy as np 


def relerr(x, y, eps=1e-300):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 0 and y.ndim == 0:
        return float(np.abs(x - y) / max(np.abs(y), eps))
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), eps))

def _is_complex(dt):
    return np.issubdtype(dt, np.complexfloating)

def _c2r(A, out): #complex to real 
    m, n = A.shape
    if out.shape != (2 * m, 2 * n) or out.dtype != np.float64:
        raise ValueError("bad out for complex embedding")
    re = A.real
    im = A.imag
    out[0::2, 0::2] = re
    out[0::2, 1::2] = -im
    out[1::2, 0::2] = im
    out[1::2, 1::2] = re

def _r2c(B): #Real to Complex 
    if (B.shape[0] % 2) or (B.shape[1] % 2):
        raise ValueError("real embedding must be even x even")
    return B[0::2, 0::2] + 1j * B[1::2, 0::2]