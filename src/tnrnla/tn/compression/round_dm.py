import copy
import math

import numpy as np
import numpy.linalg as la
from tnrnla.linalg.orth import lq
from tnrnla.linalg.lra import truncated_svd, truncated_eig
from ..mpo import MPO
from ..mps import MPS
from ..contraction.src import SRC
from ..stopping import Cutoff, no_truncation


# ======================================================================
"""
Density Matrix MPS Rounding: 
This file contains and implementation of using the density matrix algorithm to round a MPS 
a description with Penrose diagrams is given here  https://tensornetwork.org/mps/index.html#Perez-Garcia:2007_6 

NOTE: We do not recommend using this method as it is known to have some numerical stability issues. 
Author: Chris Camaño Circa: 2024 
"""
# ======================================================================


# ============ Density Matrix MPS Rounding from https://tensornetwork.org/mps/index.html#Perez-Garcia:2007_6 ===========
def density_matrix_rounding(mps, stop=Cutoff(1e-14)):
    """
    Density-matrix rounding that constructs reduced density matrices from partial environments.
    This returns a new MPS rather than modifying the input in-place.
    """
    mps_c = mps.dagger()
    N = mps.N
    mps_out = [None] * N
    E = [None] * (N - 1)

    # ==========================================
    # 0. [Envs]
    # ==========================================
    # E[0] = np.einsum("dX,dA->XA",mps[0],mps_c[0])
    E[0] = mps[0].T @ mps_c[0]  # Resulting shape (XA)
    for i in range(1, N - 1):
        # E[i] = np.einsum("XA,XdY->AdY",E[i-1],mps[i])
        E[i] = E[i - 1].T @ mps[i].reshape(mps[i].shape[0], -1)  # Resulting shape A,dY
        E[i] = E[i].reshape(E[i].shape[0], mps[i].shape[1], mps[i].shape[2])  # Reshape to (A,d,Y)

        # E[i] = np.einsum("AdY,AdB->YB",E[i],mps_c[i])
        E_transposed = E[i].transpose(2, 0, 1)
        E[i] = E_transposed.reshape(E_transposed.shape[0], -1) @ mps_c[i].reshape(-1, mps_c[i].shape[-1])

    # ==========================================
    # 3. [First DM]
    # ==========================================

    # rho = np.einsum("XA,Xd->dA",E[-1],mps[-1])
    rho = (E[-1].T @ mps[-1]).T
    # rho = np.einsum("dA,Ak->dk",rho,mps_c[-1])
    rho = rho @ mps_c[-1]
    _, U, Udag = truncated_eig(rho, stop=stop)
    mps_out[-1] = U

    # ==========================================
    # 3. [First Cap]
    # ==========================================
    # M_top = np.einsum("dk,Xd->Xk", Udag, mps[-1])
    M_top = mps[-1] @ Udag

    for j in reversed(range(1, N - 1)):
        # top = np.einsum("Yk,XdY->Xdk",M_top,mps[j])
        top = mps[j].reshape(-1, mps[j].shape[-1]) @ M_top  # Resuting shape (Xd,k)
        top = top.reshape(mps[j].shape[0], mps[j].shape[1], M_top.shape[1])

        bottom = np.conj(top)

        # rho = np.einsum("XA,Xdk->Adk",E[j-1],top)
        rho = E[j - 1].T @ top.reshape(top.shape[0], -1)  # Resulting shape (A,dk)
        rho = rho.reshape(rho.shape[0], top.shape[1], top.shape[2])

        # rho = np.einsum("Adk,Alj->dklj",rho,bottom)
        rho_transposed = rho.transpose(1, 2, 0)
        rho = rho_transposed.reshape(-1, rho_transposed.shape[2]) @ bottom.reshape(bottom.shape[0], -1)  # Resulting shape dk,lj

        _, U, Udag = truncated_eig(rho, stop=stop)  # U is (dk,x)

        U = U.reshape(top.shape[1], top.shape[2], U.shape[-1])  # (d,k,x)
        U = U.transpose(2, 0, 1)  # (x,d,k)
        mps_out[j] = U

        # M_top = np.einsum("xdk,Xdk->Xx",U,top)
        top_transposed = top.transpose(1, 2, 0)
        M_top = U.reshape(U.shape[0], -1) @ top_transposed.reshape(-1, top_transposed.shape[-1])
        M_top = M_top.T

    # mps_out[0]= np.einsum("dX,Xx->dx",mps[0],M_top)
    mps_out[0] = mps[0] @ M_top
    return MPS(mps_out)

