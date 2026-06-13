import numpy as np
import numpy.linalg as la 
import random

from ..mps import MPS
from tnrnla.linalg.lra  import truncated_svd
from ..stopping import Cutoff

"""
The zipup algorithm for the compressed MPO-MPS product: 
This file contains BLAS3 level and einsum implementations of the zipup algorithm of https://arxiv.org/abs/1002.1305

NOTE: this algorithm is known to have accuracy issues due to the nature of the algorithm. 
We do not recommend using this method unless high accuracy is not important. 
"""

def zipup(mpo, mps, stop=Cutoff(1e-14), finalround=False, conditioning=False):
    if conditioning:
        mpo.orthLR()
        mps.orthLR()

    n = mpo.N
    mps_out = [None] * n
    physdim = mpo[0].shape[0]

    # =============================== First tensor ===============================
    C = np.tensordot(mpo[0], mps[0], axes=(2, 0))
    mps_out[0], S, Vt = truncated_svd(np.reshape(C, (C.shape[0], C.shape[1] * C.shape[2])), stop=stop)
    Z = np.diag(S) @ Vt
    Z = Z.reshape(Z.shape[0], mpo[1].shape[0], mps[1].shape[0])

    # =============================== Middle tensors =============================
    for i in range(1, n - 2):
        #C = np.einsum("pDX,XlY->pDlY", Z, mps[i])
        C = Z @ mps[i].reshape(mps[i].shape[0], -1)
        C = C.reshape(Z.shape[0], mpo[i].shape[0], mps[i].shape[1], mps[i].shape[2])
        
        #C = np.einsum('pDlY,DdEl->pdEY', C, mpo[i])
        C_transposed = C.transpose(0, 3, 1, 2)                                                     # Transpose to (p, Y, D, l)
        mpo_transposed = mpo[i].transpose(0,3, 1, 2)                                               # Transpose to (D, l, d, E)

        C_reshaped = C_transposed.reshape(-1, C_transposed.shape[2]*C_transposed.shape[3])         # Reshape to (pY, Dl)
        mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1)  # Reshape to (Dl, dE)

        temp = (C_reshaped @ mpo_reshaped)
        temp=temp.reshape(C.shape[0], C.shape[3],mpo[i].shape[1], mpo[i].shape[2])                 # Reshape to (p, Y, d, E)
        C=temp.transpose(0,2,3,1)                                                                  # Transpose to (p, d, E, Y)
                
        U, S, Vt = truncated_svd(C.reshape(C.shape[0] * C.shape[1], C.shape[2] * C.shape[3]), stop=stop)
        U = U.reshape(C.shape[0], physdim, U.shape[1])
        mps_out[i] = U
        Z = np.diag(S) @ Vt
        Z = Z.reshape(Z.shape[0], mpo[i + 1].shape[0], mps[i + 1].shape[0])

    # =============================== Last tensor ================================
    
    #C = np.einsum("pDX,XlY->pDlY", Z, mps[n-2])
    Z_reshaped = Z.reshape(Z.shape[0], Z.shape[1], -1)                                             # Reshape to (p, D, X)
    mps_reshaped = mps[n-2].reshape(mps[n-2].shape[0], -1)                                         # Reshape to (X, lY)
    temp = Z_reshaped @ mps_reshaped 
    C = temp.reshape(Z.shape[0], Z.shape[1], mps[n-2].shape[1], mps[n-2].shape[2])                 # Reshape to (p, D, l, Y)
    
    #C = np.einsum("pDlY,DdEl->pdEY", C, mpo[n-2])
    C_transposed = C.transpose(0, 3, 1, 2)                                                         # Transpose to (p, Y, D, l)
    mpo_transposed = mpo[n-2].transpose(0, 3, 1, 2)                                                # Transpose to (D, l, d, E)
    C_reshaped = C_transposed.reshape(-1, C_transposed.shape[2]*C_transposed.shape[3])             # Reshape to (pY, Dl)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1], -1)     # Reshape to (Dl, dE)
    temp = C_reshaped @ mpo_reshaped 
    temp = temp.reshape(C.shape[0], C.shape[3], mpo[n-2].shape[1], mpo[n-2].shape[2])              # Reshape to (p, Y, d, E)
    C = temp.transpose(0, 2, 3, 1)                                                                 # Transpose to (p, d, E, Y)
    
    #C = np.einsum("pdDX,Xl->pdDl", C, mps[n-1])
    C_reshaped = C.reshape(C.shape[0], C.shape[1], C.shape[2], -1)                                 # Reshape to (p, d, D, X)
    mps_reshaped = mps[n-1].reshape(-1, mps[n-1].shape[1])                                         # Reshape to (X, l)
    temp = C_reshaped @ mps_reshaped 
    C = temp.reshape(C.shape[0], C.shape[1], C.shape[2], mps[n-1].shape[1])                        # Reshape to (p, d, D, l)
    
    #C = np.einsum("pdDl,Dul->pdu", C, mpo[n-1])
    mpo_transposed = mpo[n-1].transpose(0, 2, 1)                                                   # Transpose to (D, l, u)
    C_reshaped = C.reshape(-1, C.shape[2]*C.shape[3])                                              # Reshape to (pd, Dl)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1)      # Reshape to (Dl, u)
    temp = C_reshaped @ mpo_reshaped
    C = temp.reshape(C.shape[0], C.shape[1], mpo[n-1].shape[1])                                    # Reshape to (p, d, u)

    #print(C.shape)
    U, S, Vt = truncated_svd(np.reshape(C, (C.shape[0] * C.shape[1], C.shape[1])), stop=stop)
    mps_out[n-2] = np.reshape(U, (C.shape[0], C.shape[2], U.shape[1]))
    Z = np.diag(S) @ Vt
    mps_out[-1] = Z

    mps_out = MPS(mps_out)
    if finalround:
        mps_out.round(stop=stop)

    return MPS(mps_out)



def zipup_einsum(mpo, mps, stop=Cutoff(1e-14), finalround=False, conditioning=False):
    # Einsum conventions:
    #   MPS[i] bond dimension X_i            = X
    #   MPS[i] bond dimension X_{i+1}        = Y
    #   MPO[i] bond dimension D_i            = D
    #   MPO[i] bond dimension D_{i+1}        = E
    #   MPO[i] physical dimension d_(top)    = d
    #   MPO[i] physical dimension d_(bottom) = l
    #   DX rank trunction p_(current)        = p
    #   DX rank trunction p_(previous)       = x
    
    if conditioning:
        mpo.orthLR()
        mps.orthLR()

    n = mpo.N
    mps_out = [None] * n
    physdim = mpo[0].shape[0]

    # =============================== First tensor ===============================
    C = np.tensordot(mpo[0], mps[0], axes=(2, 0))
    mps_out[0], S, Vt = truncated_svd(np.reshape(C, (C.shape[0], C.shape[1] * C.shape[2])), stop=stop)
    Z = np.diag(S) @ Vt
    Z = Z.reshape(Z.shape[0], mpo[1].shape[0], mps[1].shape[0])

    # =============================== Middle tensors =============================
    for i in range(1, n - 2):
        # Absorb MPOS site and MPS site into Z
        C = np.einsum("pDX,XlY->pDlY", Z, mps[i])
        C = np.einsum('pDlY,DdEl->pdEY', C, mpo[i])
        U, S, Vt = truncated_svd(C.reshape(C.shape[0] * C.shape[1], C.shape[2] * C.shape[3]), stop=stop)
        U = U.reshape(C.shape[0], physdim, U.shape[1])
        mps_out[i] = U
        Z = np.diag(S) @ Vt
        Z = Z.reshape(Z.shape[0], mpo[i + 1].shape[0], mps[i + 1].shape[0])

    # =============================== Last tensor ================================
    C = np.einsum("pDX,XlY->pDlY", Z, mps[n-2])
    C = np.einsum("pDlY,DdEl->pdEY", C, mpo[n-2])
    C = np.einsum("pdDX,Xl->pdDl", C, mps[n-1])
    C = np.einsum("pdDl,Dul->pdu", C, mpo[n-1])
    U, S, Vt = truncated_svd(np.reshape(C, (C.shape[0] * C.shape[1], C.shape[1])), stop=stop)
    mps_out[n-2] = np.reshape(U, (C.shape[0], C.shape[2], U.shape[1]))
    Z = np.diag(S) @ Vt
    mps_out[-1] = Z

    mps_out = MPS(mps_out)
    if finalround:
        mps_out.round(stop=stop)

    return MPS(mps_out)
