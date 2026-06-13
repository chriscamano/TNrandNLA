from tnrnla.linalg.orth import lq
import numpy as np 

def flatten_right(tensor: np.ndarray) -> np.ndarray:
    A = np.asarray(tensor)
    if not A.flags["C_CONTIGUOUS"]:
        A = np.ascontiguousarray(A)
    return A.reshape(A.shape[0], -1)


def flatten_left(tensor: np.ndarray) -> np.ndarray:
    A = np.asarray(tensor)
    if not A.flags["C_CONTIGUOUS"]:
        A = np.ascontiguousarray(A)
    return A.reshape(-1, A.shape[-1])


def _flatten_last2(tensor: np.ndarray) -> np.ndarray:
    A = np.asarray(tensor)
    if not A.flags["C_CONTIGUOUS"]:
        A = np.ascontiguousarray(A)
    return A.reshape(-1, A.shape[-2] * A.shape[-1])


def _ascontig(tensor: np.ndarray) -> np.ndarray:
    A = np.asarray(tensor)
    if not A.flags["C_CONTIGUOUS"]:
        A = np.ascontiguousarray(A)
    return A


def c_reshape(a: np.ndarray, *shape: int) -> np.ndarray:
    return np.ascontiguousarray(a).reshape(*shape)

def batch_rightmul(A3: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """
    Batched right-multiplication:
      A3: (Z, m, k)
      B2: (k, n)
    Returns: (Z, m, n)
    """
    Z, m, k = A3.shape
    return (A3.reshape(Z * m, k) @ B2).reshape(Z, m, -1)


def ttm_old(tensor, mat, k):
    axes = list(range(tensor.ndim))
    perm = [k] + [i for i in axes if i != k]
    tensor_perm = np.transpose(tensor, perm)
    tmp = flatten_right(tensor_perm)
    tmp = mat @ tmp
    new_first_dim = mat.shape[0]
    other_dims = tensor_perm.shape[1:]
    tensor_perm_new = tmp.reshape(new_first_dim, *other_dims)
    inv_perm = np.argsort(perm)
    return np.transpose(tensor_perm_new, inv_perm)


def ttm3(X, U, k):
    """
    X: (n0,n1,n2)
    U: (r,nk) where nk matches X.shape[k]
    returns tensor with mode-k dimension replaced by r.
    """
    X = np.asarray(X)
    U = np.asarray(U)
    if X.ndim != 3:
        raise ValueError(f"ttm3 expects rank-3 tensor X, got ndim={X.ndim}")
    if U.ndim != 2:
        raise ValueError(f"ttm3 expects rank-2 matrix U, got ndim={U.ndim}")

    n0, n1, n2 = X.shape
    r, nk = U.shape
    if k < 0 or k > 2:
        raise ValueError(f"ttm3 mode k must be 0, 1, or 2; got {k}")
    if nk != X.shape[k]:
        raise ValueError(
            f"ttm3 shape mismatch: U has shape {U.shape}, but X.shape[{k}]={X.shape[k]}"
        )

    if k == 0:
        # (r,n0) @ (n0, n1*n2) -> (r, n1*n2)
        Xk = X.reshape(n0, n1 * n2)
        Yk = U @ Xk
        return Yk.reshape(r, n1, n2)

    if k == 2:
        # (n0*n1, n2) @ (n2, r) -> (n0*n1, r)
        # This is the hot path in right-to-left orthogonalization.
        X2 = X.reshape(n0 * n1, n2)
        UT = U.T  # (n2, r)
        Y2 = X2 @ UT
        return Y2.reshape(n0, n1, r)

    if k == 1:
        # Vectorized mode-1 contraction:
        # (n0,n1,n2) -> (n0*n2,n1), GEMM with U.T, then reshape back.
        Xt = X.transpose(0, 2, 1).reshape(n0 * n2, n1)
        Yt = Xt @ U.T
        return Yt.reshape(n0, n2, r).transpose(0, 2, 1)

def contract_blas(A, axesA, B, axesB):
    axesA = (axesA,) if isinstance(axesA, int) else tuple(axesA)
    axesB = (axesB,) if isinstance(axesB, int) else tuple(axesB)

    freeA = tuple(i for i in range(A.ndim) if i not in axesA)
    freeB = tuple(i for i in range(B.ndim) if i not in axesB)

    A_t = A.transpose(freeA + axesA)
    B_t = B.transpose(axesB + freeB)

    kA = int(np.prod([A.shape[i] for i in axesA], dtype=np.int64)) if axesA else 1
    kB = int(np.prod([B.shape[i] for i in axesB], dtype=np.int64)) if axesB else 1
    if kA != kB:
        raise ValueError(f"Contracted dimensions mismatch: {kA} vs {kB}")
    k = kA

    C2 = (
        np.ascontiguousarray(A_t).reshape(-1, k)
        @ np.ascontiguousarray(B_t).reshape(k, -1)
    )

    out_shape = tuple(A.shape[i] for i in freeA) + tuple(B.shape[i] for i in freeB)
    return C2.reshape(out_shape)

def qr_right(tensor):
    ndim = tensor.ndim

    if ndim == 2:  # MPS left boundary
        Q, R = np.linalg.qr(tensor, mode="reduced")
        return Q, R

    if ndim == 3:  # MPS middle site
        Xl, d, Xr = tensor.shape
        mat = flatten_left(tensor)                 # (Xl*d, Xr)
        Q, R = np.linalg.qr(mat, mode="reduced")   # Q: (Xl*d, r), R: (r, Xr)
        Q = Q.reshape(Xl, d, -1)                   # (Xl, d, r)
        return Q, R

    raise ValueError(f"qr_right only implemented for rank-2/3 tensors, got ndim={ndim}")


def lq_left(tensor):
    ndim = tensor.ndim
    if ndim == 2:
        # (Xl, d)
        L, Q = lq(tensor)
        return L,Q

    if ndim == 3:
        Xl, d, Xr = tensor.shape
        mat = flatten_right(tensor)
        L,Q = lq(mat)
        Q = Q.reshape(-1, d, Xr)                 #(Xl,d,r)

        return L, Q

    raise ValueError("lq_left only implemented for rank-2/3 tensors")
