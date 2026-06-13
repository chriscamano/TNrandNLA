import numpy as np


def is_scalar_like(x):
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)


def ensure_oracle_and_dim(matvec_oracle, dimension):
    if callable(matvec_oracle):
        if int(dimension) < 0:
            raise ValueError("dimension must be provided when matvec_oracle is callable")
        return matvec_oracle, int(dimension)
    A = np.asarray(matvec_oracle)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix input must be square")
    n = int(A.shape[0])
    return (lambda X: A @ X), n


def ensure_oracle_pair_and_dim(B, Bt, dimension):
    if callable(B):
        if not callable(Bt):
            raise ValueError("Bt must be callable when B is callable")
        n = int(dimension)
        if n <= 0:
            raise ValueError("dimension must be positive")
        return B, Bt, n

    A = np.asarray(B)
    if A.ndim != 2:
        raise ValueError("matrix input must be 2D")
    nA = int(A.shape[1])
    n = int(nA) if dimension is None else int(dimension)
    if n != nA:
        raise ValueError("dimension must match matrix column count")
    return (lambda X: A @ X), (lambda X: A.T.conj() @ X), n


def resphere_columns(X, target_norm):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    return float(target_norm) * (X / norms)


def resphere_in_orthogonal_complement(G, Q, ambient_dim):
    if Q.size == 0:
        k_eff = 0
        G_perp = G
    else:
        k_eff = int(Q.shape[1])
        G_perp = G - Q @ (Q.T.conj() @ G)
    return resphere_columns(G_perp, np.sqrt(float(max(int(ambient_dim) - k_eff, 1))))
