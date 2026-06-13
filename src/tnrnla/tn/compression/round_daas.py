import copy
import math

import numpy as np
import numpy.linalg as la
from tnrnla.linalg.orth import lq
from tnrnla.linalg.lra import truncated_svd
from ..mpo import MPO
from ..mps import MPS
from ..contraction.src import SRC
from ..stopping import Cutoff, no_truncation


# ======================================================================
"""
Randomized MPS Rounding: 
This file contains implementations of the methods given in arxiv.org/abs/2110.04393
from the paper "Randomized algorithms for rounding in the Tensor-Train format" by: 
Hussam Al Daas, Grey Ballard, Paul Cazeaux, Eric Hallman, 
Agnieszka Miedlar, Mirjeta Pasha, Tim W. Reid, and Arvind K. Saibaba

Author: Chris Camaño Circa: 2023 
"""
# ======================================================================


# ============ Randomized MPS Rounding from arxiv.org/abs/2110.04393 (einsum version) ============
def TTpartialContractionsLR(X, Y):
    """
    Left-to-right partial contractions that build WL used by two-sided randomization.
    This matches the PartialContractionsLR routine used alongside Algorithm 3.3 in Al Dass et al https://arxiv.org/abs/2110.04393.
    """
    WL = [None] * (Y.N - 1)
    WL[0] = X[0].T @ Y[0]
    for j in range(1, Y.N - 1):
        WL[j] = np.tensordot(Y[j], WL[j - 1], axes=(0, 1))
        WL[j] = np.tensordot(X[j], WL[j], axes=((1, 0), (0, 2)))
    return WL


def TTpartialContractionsRL(X, Y):
    """
    Right-to-left partial contractions that build WR and correspond to Algorithm 2.3 in Al Dass et al https://arxiv.org/abs/2110.04393.
    These contractions supply the sketch matrices used in Algorithm 3.2.
    """
    N = len(X)
    WR = [None] * (N - 1)
    WR[N - 2] = X[-1] @ Y[-1].T

    for j in range(N - 2, 0, -1):
        WR[j - 1] = np.tensordot(X[j], WR[j], axes=(2, 0))
        WR[j - 1] = np.tensordot(WR[j - 1], Y[j], axes=((1, 2), (1, 2)))
    return WR


def orth_then_rand(mps, r):
    """
    Randomized TT rounding in the spirit of Algorithm 3.1 in Al Dass et al https://arxiv.org/abs/2110.04393.
    This orthogonalizes first and then uses a Gaussian sketch of width r to build reduced bases.
    """
    mps.roundRL()

    Z = mps[0]
    sketch = Z @ np.random.randn(Z.shape[1], r)

    Q, _ = la.qr(sketch, mode="reduced")
    mps[0] = Q

    M = Q.T @ Z
    mps[1] = np.tensordot(M, mps[1], axes=(1, 0))

    for i in range(1, mps.N - 1):
        Z = mps[i]
        sketch = np.tensordot(Z, np.random.randn(Z.shape[2], r), axes=(2, 0))

        Q, _ = la.qr(np.reshape(sketch, (sketch.shape[0] * sketch.shape[1], sketch.shape[2])), mode="reduced")
        mps[i] = np.reshape(Q, (mps[i - 1].shape[-1], mps[i].shape[1], Q.shape[-1]))

        M = Q.T @ np.reshape(Z, (Z.shape[0] * Z.shape[1], Z.shape[2]))
        mps[i + 1] = np.tensordot(M, mps[i + 1], axes=(1, 0))


def rand_then_orth(mps, r, finalround=False, stop=Cutoff(1e-14)):
    """
    Randomize then orthogonalize following Algorithm 3.2 in Al Dass et al https://arxiv.org/abs/2110.04393.
    It forms right-to-left partial contractions with a random TT tensor and then QR orthogonalizes sketched unfoldings.
    """
    if r is None:
        r = mps[0].shape[1]
    # ==================Environment tensors ==================
    N = mps.N
    R = MPS.rmps(N, r, mps[1].shape[1])
    W = TTpartialContractionsRL(mps, R)
    # ==================Random Section==================
    Q, _ = la.qr(np.tensordot(mps[0], W[0], axes=(1, 0)), mode="reduced")
    M = np.tensordot(Q.T, mps[0], axes=(1, 0))
    mps[0] = Q

    for j in range(1, N - 1):
        mps[j] = np.tensordot(M, mps[j], axes=(1, 0))
        Z = np.reshape(mps[j], (mps[j].shape[0] * mps[j].shape[1], mps[j].shape[2]))
        sketch = np.tensordot(Z, W[j], axes=(1, 0))

        Q, _ = la.qr(sketch, mode="reduced")
        mps[j] = np.reshape(Q, (mps[j - 1].shape[-1], mps[j].shape[1], Q.shape[1]))
        M = np.tensordot(Q.T, Z, axes=(1, 0))

    mps[-1] = np.tensordot(M, mps[-1], axes=(1, 0))
    mps.rounded = True
    # ==================Final rounding==================
    if finalround:
        if stop.cutoff != None:
            normY = np.linalg.norm(mps[-1], "fro")
            tau = stop.cutoff * normY / math.sqrt(mps.N - 1)
            stop = Cutoff(tau)

        Q, R = la.qr(mps[0], mode="reduced")
        U, s, Vt = truncated_svd(R, stop=stop)
        mps[0] = Q @ U

        for i in range(1, mps.N - 1):
            A = np.tensordot(s[:, np.newaxis] * Vt, mps[i], (1, 0))
            Q, R = la.qr(np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2])), mode="reduced")
            U, s, Vt = truncated_svd(R, stop=stop)
            mps[i] = np.reshape(Q @ U, (A.shape[0], A.shape[1], len(s)))

        mps[-1] = np.tensordot(s[:, np.newaxis] * Vt, mps[-1], (1, 0))


def nystrom_round(mps, l, stop=no_truncation()):
    """
    Two-sided randomized rounding aligned with Algorithm 3.3 in Al Dass et al https://arxiv.org/abs/2110.04393.
    It uses left and right sketches and an SVD on WL_n WR_n to propagate two-sided correction factors.
    """
    if l is None:
        l = mps[0].shape[1]
    # ================== Environment tensors ==================
    L = MPS.rmps(mps.N, l, mps[1].shape[1])
    rho = int(np.ceil(1.5 * l))
    # Heuristic choice of oversampling ranks based on Yuji- Fast and stable randomized low rank matrix approximation 2020
    R = MPS.rmps(mps.N, rho, mps[1].shape[1])

    WL = TTpartialContractionsLR(L, mps)
    WR = TTpartialContractionsRL(mps, R)

    # ================== Nystrom Sketch ==================
    U, s, Vt = truncated_svd(WL[0] @ WR[0], stop=stop)
    S_inv_sqrt = np.conj(np.diag(1 / np.sqrt(s)))
    left_factor = WR[0] @ Vt.T @ S_inv_sqrt
    right_factor = S_inv_sqrt @ U.T @ WL[0]

    mps[0] = mps[0] @ left_factor

    for j in range(1, mps.N - 1):
        U, s, Vt = truncated_svd(WL[j] @ WR[j], stop=stop)
        S_inv_sqrt = np.conj(np.diag(1 / np.sqrt(s)))
        left_factor = WR[j] @ Vt.T @ S_inv_sqrt

        mps[j] = np.tensordot(mps[j], left_factor, axes=(2, 0))
        mps[j] = np.tensordot(right_factor, mps[j], axes=(1, 0))
        mps[j] = np.reshape(mps[j], (right_factor.shape[0], mps[j].shape[1], left_factor.shape[1]))

        right_factor = S_inv_sqrt @ U.T @ WL[j]

    mps[-1] = right_factor @ mps[-1]


# ============ Randomized MPS Rounding Techniques from arxiv.org/abs/2110.04393 (BLAS version) ============
def TTpartialContractionsLR_blas(X, Y):
    """
    BLAS-oriented left-to-right partial contractions for WL that match the PartialContractionsLR role in Algorithm 3.3 of Al Dass et al https://arxiv.org/abs/2110.04393.
    The implementation favors reshapes and matrix multiplies over higher-order contractions.
    """
    WL = [None] * (Y.N - 1)
    WL[0] = X[0].T @ Y[0]
    for j in range(1, Y.N - 1):
        # WL[j] = np.tensordot(Y[j], WL[j - 1], axes=(0, 1))
        Y_reshaped = Y[j].reshape(Y[j].shape[0], -1)
        temp = WL[j - 1] @ Y_reshaped
        temp = temp.reshape(WL[j - 1].shape[0], Y[j].shape[1], Y[j].shape[2])
        WL[j] = temp.transpose(1, 2, 0)

        # WL[j] = np.tenspordot(X[j], WL[j], axes=((1, 0), (0, 2)))
        X_reshaped = X[j].reshape(-1, X[j].shape[2])
        WL_transposed = WL[j].transpose(1, 2, 0)
        WL_transposed = np.ascontiguousarray(WL_transposed)  # Resolves extra view
        WL_reshaped = WL_transposed.reshape(WL_transposed.shape[0], -1)
        temp = WL_reshaped @ X_reshaped
        WL[j] = temp.T
    return WL


def TTpartialContractionsRL_blas(X, Y):
    """
    BLAS-oriented right-to-left partial contractions for WR that match Algorithm 2.3 in Al Dass et al https://arxiv.org/abs/2110.04393.
    These contractions are used to build the sketches in Algorithm 3.2 and Algorithm 3.3.
    """
    N = len(X)
    WR = [None] * (N - 1)
    WR[N - 2] = X[-1] @ Y[-1].T

    for j in range(N - 2, 0, -1):
        # WR[j-1]  = np.tensordot(X[j], WR[j], axes=(2, 0))
        X_reshaped = X[j].reshape(-1, X[j].shape[2])
        temp = X_reshaped @ WR[j]
        WR[j - 1] = temp.reshape(X[j].shape[0], X[j].shape[1], WR[j].shape[1])

        # WR[j-1] = np.tensordot(WR[j-1], Y[j], axes=((1, 2), (1, 2)))
        WR_reshaped = WR[j - 1].reshape(WR[j - 1].shape[0], -1)
        Y_transposed = Y[j].transpose(1, 2, 0)
        Y_reshaped = Y_transposed.reshape(-1, Y_transposed.shape[2])
        WR[j - 1] = WR_reshaped @ Y_reshaped
    return WR


def daas_round_blas(mps, stop=Cutoff(1e-14)):
    """
    BLAS-oriented rounding that mirrors the QR plus truncated SVD sweep used in deterministic TT rounding.
    It uses reshape based GEMM patterns to reduce contraction overhead in the same rounding setting as Al Dass et al https://arxiv.org/abs/2110.04393.
    """
    if mps.rounded == False:
        mps.orthL()
    # if stop.cutoff != None:
    #     normY = mps.norm()
    #     tau = stop.cutoff * normY / math.sqrt(mps.N - 1)
    #     stop = Cutoff(tau)

    Q, R = np.linalg.qr(mps[0], mode="reduced")

    U, s, Vt = truncated_svd(R, stop=stop)
    mps[0] = Q @ U
    for i in range(1, mps.N - 1):
        temp = s[:, np.newaxis] * Vt
        # A = np.einsum("ik,klY->ilY",temp,mps[i])
        mps_reshaped = mps[i].reshape(mps[i].shape[0], -1)  # Reshape to (DX,l(EY))
        A = temp @ mps_reshaped  # resulting shape(p,l(EY))
        A = A.reshape(A.shape[0], mps[i].shape[1], -1)  # reshape to (p,l,EY)

        Q, R = np.linalg.qr(np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2])), mode="reduced")
        U, s, Vt = truncated_svd(R, stop=stop)
        mps[i] = np.reshape(Q @ U, (A.shape[0], A.shape[1], len(s)))

    mps[-1] = (s[:, np.newaxis] * Vt) @ mps[-1]
    mps.canform = "Right"
    mps.rounded = True


def orth_then_rand_blas(mps, r):
    """
    BLAS-oriented variant of Orthogonalize-then-Randomize that keeps the same math as orth_then_rand.
    It is a performance rewrite that leans on reshaped matrix multiplies in the randomized TT rounding context of Al Dass et al https://arxiv.org/abs/2110.04393.
    """
    mps.roundRL()

    Z = mps[0]
    sketch = Z @ np.random.randn(Z.shape[1], r)

    Q, _ = la.qr(sketch, mode="reduced")
    mps[0] = Q

    M = Q.T @ Z
    # mps[1] = np.tensordot(M,mps[1],axes=(1,0))
    mps_reshaped = mps[1].reshape(mps[1].shape[0], -1)
    temp = M @ mps_reshaped
    mps[1] = temp.reshape(M.shape[0], mps[1].shape[1], mps[1].shape[2])

    for i in range(1, mps.N - 1):
        Z = mps[i]
        # sketch = np.tensordot(Z ,np.random.randn(Z.shape[2],r),axes=(2,0))
        Z_reshaped = Z.reshape(Z.shape[0] * Z.shape[1], Z.shape[2])
        sketch = Z_reshaped @ np.random.randn(Z.shape[2], r)

        # print(sketch.shape)
        Q, _ = la.qr(sketch, mode="reduced")
        mps[i] = np.reshape(Q, (mps[i - 1].shape[-1], mps[i].shape[1], Q.shape[-1]))

        M = Q.T @ np.reshape(Z, (Z.shape[0] * Z.shape[1], Z.shape[2]))
        mps[i + 1] = np.tensordot(M, mps[i + 1], axes=(1, 0))


def rand_then_orth_blas(mps, r, finalround=False, stop=Cutoff(1e-14)):
    """
    BLAS-oriented variant of Algorithm 3.2 Randomize-then-Orthogonalize in Al Dass et al https://arxiv.org/abs/2110.04393.
    It preserves the same sketching and QR structure as rand_then_orth while replacing contractions with reshaped GEMMs.
    """
    if r is None:
        if stop.cutoff is None and stop.outputdim is not None:
            r = stop.outputdim
        else:
            r = mps[0].shape[1]
    # ==================Environment tensors ==================
    N = mps.N
    R = MPS.rmps(N, r, mps[1].shape[1])
    W = TTpartialContractionsRL_blas(mps, R)
    # ==================Random Section==================
    Q, _ = la.qr(np.tensordot(mps[0], W[0], axes=(1, 0)), mode="reduced")
    # M = np.tensordot(Q.T, mps[0], axes=(1, 0))
    M = Q @ mps[0]
    mps[0] = Q

    for j in range(1, N - 1):
        # mps[j] = np.tensordot(M,mps[j],axes=(1,0))
        mps_reshaped = mps[j].reshape(mps[j].shape[0], -1)
        temp = M @ mps_reshaped
        mps[j] = temp.reshape(M.shape[0], mps[j].shape[1], mps[j].shape[2])

        Z = np.reshape(mps[j], (mps[j].shape[0] * mps[j].shape[1], mps[j].shape[2]))
        # sketch = np.tensordot(Z,W[j],axes=(1,0))
        sketch = Z @ W[j]
        Q, _ = la.qr(sketch, mode="reduced")
        mps[j] = np.reshape(Q, (mps[j - 1].shape[-1], mps[j].shape[1], Q.shape[1]))
        M = Q.T @ Z

    mps[-1] = M @ mps[-1]
    mps.rounded = True
    # #==================Final rounding==================
    # if finalround:
    #     if stop.cutoff != None:
    #         normY = np.linalg.norm(mps[-1],'fro')
    #         tau = stop.cutoff * normY / math.sqrt(mps.N - 1)
    #         stop= Cutoff(tau)

    #     Q,R = la.qr(mps[0], mode='reduced')
    #     U, s, Vt = truncated_svd(R, stop=stop)
    #     mps[0] = Q @ U
    #     #Update to use dass round blas
    #     for i in range(1, mps.N - 1):
    #         A = np.tensordot(s[:,np.newaxis] * Vt, mps[i], (1,0))
    #         Q,R = la.qr(np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2])), mode='reduced')
    #         U, s, Vt = truncated_svd(R, stop=stop)
    #         mps[i] = np.reshape(Q @ U, (A.shape[0], A.shape[1], len(s)))

    #     mps[-1] = np.tensordot(s[:,np.newaxis] * Vt, mps[-1], (1,0))

def nystrom_round_blas(mps, l, stop=no_truncation()):
    """
    BLAS-oriented variant of Algorithm 3.3 Two-Sided-Randomization in Al Dass et al https://arxiv.org/abs/2110.04393.
    It keeps the same WL and WR sketches and updates cores using left and right factors while favoring GEMM-friendly reshapes.
    """
    if l is None:
        l = mps[0].shape[1]
    # ================== Environment tensors ==================
    L = MPS.rmps(mps.N, l, mps[1].shape[1])
    rho = int(np.ceil(1.5 * l))
    # Heuristic choice of oversampling ranks based on Yuji- Fast and stable randomized low rank matrix approximation 2020
    R = MPS.rmps(mps.N, rho, mps[1].shape[1])

    WL = TTpartialContractionsLR_blas(L, mps)
    WR = TTpartialContractionsRL_blas(mps, R)

    # ================== Nystrom Sketch ==================
    U, s, Vt = truncated_svd(WL[0] @ WR[0], stop=stop)
    S_inv_sqrt = np.conj(np.diag(1 / np.sqrt(s)))
    left_factor = WR[0] @ Vt.T @ S_inv_sqrt
    right_factor = S_inv_sqrt @ U.T @ WL[0]

    mps[0] = mps[0] @ left_factor

    for j in range(1, mps.N - 1):
        U, s, Vt = truncated_svd(WL[j] @ WR[j], stop=stop)
        S_inv_sqrt = np.conj(np.diag(1 / np.sqrt(s)))
        left_factor = WR[j] @ Vt.T @ S_inv_sqrt

        # mps[j] = np.tensordot(mps[j], left_factor, axes=(2, 0))
        mps_reshaped = mps[j].reshape(-1, mps[j].shape[2])
        temp = mps_reshaped @ left_factor
        mps[j] = temp.reshape(mps[j].shape[0], mps[j].shape[1], left_factor.shape[1])

        # mps[j] = np.tensordot(right_factor, mps[j], axes=(1,0))
        mps_reshaped = mps[j].reshape(mps[j].shape[0], -1)
        temp = right_factor @ mps_reshaped
        mps[j] = temp.reshape(right_factor.shape[0], mps[j].shape[1], mps[j].shape[2])

        mps[j] = np.reshape(mps[j], (right_factor.shape[0], mps[j].shape[1], left_factor.shape[1]))

        right_factor = S_inv_sqrt @ U.T @ WL[j]

    mps[-1] = right_factor @ mps[-1]