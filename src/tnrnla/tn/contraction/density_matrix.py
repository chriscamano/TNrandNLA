import numpy as np
import numpy.linalg as la 
import random

from ..mps import MPS
from tnrnla.linalg.lra import truncated_eig
from ..stopping import Cutoff
# ======================================================================
"""
The Density matrix algorithm for the compressed MPO-MPS product: 
This file contains an accelerated BLAS-3 level and einsum level implementation of the density matrix algorithm 
for computing the compressed MPO-MPS product (https://arxiv.org/abs/2406.09769) 

NOTE: We do not recommend using this method as it is known to have some numerical stability issues. 
Author: Chris Camaño Circa: 2024
"""
# ======================================================================



def density_matrix(mpo, mps, stop=Cutoff(1e-14), maxdim=None, normalize=False,finalround=False):
    """
    Einsum Indexing convention: 
     ______
    |      |    _______
    |      |-Z-|mps_c_j|-W-...
    |      |   |_______|
    |      |    __|l___
    |      |-F-|mpo_c_j|-G-...
    |      |   |_______|
    | L[j] |      |d  
    |      |
    |      |    __|d___
    |      |-D-| mpo_j |-E-...
    |      |   |_______|
    |      |    __|l___
    |      |-X-| mps_j |-Y-...
    |______|   |_______|
    
    Comments:
    - In cases where the same index is duplicated in a contraction ie: dxd density matrix the 
      surrogate index will be promoted one letter: dxd -> exd.
    - truncation variable produced by eig is k.
    """
    # ==========================================
    # 0. [Initialization]
    # ==========================================
    if len(mpo) != len(mps):
        raise ValueError("MPO and MPS must have the same length.")
    if maxdim is None:
        requested_maxdim = mpo.max_bond() * mps.max_bond()


    n = len(mpo)
    mps_out = mps.copy()
    mps_c = mps.dagger()
    mpo_c = mpo.dagger()
    L = [None] * (n - 1)


    # ==========================================
    # 1. [Environment Tensor Construction]          
    # 1. [Environment Tensor Construction]          
    # ==========================================
    
    # L[0] = np.einsum('dDl,lX->dDX', mpo[0], mps[0]) 
    mpo_reshaped = mpo[0].reshape(mpo[0].shape[0] * mpo[0].shape[1], -1)                    # Reshape to (dD, l)
    mps_reshaped = mps[0].reshape(-1, mps[0].shape[1])                                      # Reshape to (l, X)
    temp = mpo_reshaped @ mps_reshaped                                      
    L[0] = temp.reshape(mpo[0].shape[0], mpo[0].shape[1], mps[0].shape[1])   

    # temp = np.einsum('dFl,lZ->ZFd', mpo_c[0], mps_c[0])
    mpo_reshaped = mpo_c[0].reshape(mpo_c[0].shape[0] * mpo_c[0].shape[1], -1)              # Reshape to (dF, l)
    mps_reshaped = mps_c[0].reshape(-1, mps_c[0].shape[1])                                  # Reshape to (l, Z)
    temp = mpo_reshaped @ mps_reshaped                                        
    temp = temp.reshape(mpo_c[0].shape[0], mpo_c[0].shape[1], mps_c[0].shape[1])            # Reshape to (d, F, Z)
    temp = temp.transpose(2, 1, 0)                                                          # Transpose to (Z, F, d)

    # L[0] = np.einsum('dDX,ZFd->XDFZ', L[0], temp)
    L_transposed = L[0].transpose(2, 1, 0)                                                  # Transpose to (X, D, d)
    L_reshaped = L_transposed.reshape(-1, L_transposed.shape[2])                            # Reshape to (XD, d)
    temp_transposed = temp.transpose(2, 1, 0)                                               # Transpose to (d, F, Z)
    temp_reshaped = temp_transposed.reshape(temp_transposed.shape[0], -1)                   # Reshape to (d, FZ)
    L[0] = L_reshaped @ temp_reshaped                                                       # Resulting shape (XD, FZ)
    L[0] = L[0].reshape(L_transposed.shape[0], L_transposed.shape[1], 
                        temp_transposed.shape[1], temp_transposed.shape[2])                 # Reshape to (X, D, F, Z)

    for j in range(1, n-1):
        # L[j] = np.einsum('XDFZ,ZlW->XDFlW', L[j-1], mps_c[j])
        L_reshaped = L[j-1].reshape(-1, L[j-1].shape[3])                                    # Reshape to (XDF, Z)
        mps_reshaped = mps_c[j].reshape(mps_c[j].shape[0], -1)                              # Reshape to (Z, lW)
        intermediate = L_reshaped @ mps_reshaped                                     
        intermediate_reshaped = intermediate.reshape(L[j-1].shape[0], L[j-1].shape[1], 
                                                    L[j-1].shape[2], mps_c[j].shape[1], 
                                                    mps_c[j].shape[2])                      # Reshape to (X, D, F, l, W)
        L[j] = intermediate_reshaped.transpose(0, 1, 2, 3, 4)                               # Transpose to (XDFlW)

        # L[j] = np.einsum('XDFlW,FdGl->XDdGW', L[j], mpo_c[j])
        L_transposed = L[j].transpose(0, 1, 4, 2, 3)                                        # Transpose to (X, D, W, F, l)
        mpo_transposed = mpo_c[j].transpose(0, 3, 1, 2)                                     # Transpose to (F, l, d, G)
        L_reshaped = L_transposed.reshape(L_transposed.shape[0] * L_transposed.shape[2] * L_transposed.shape[3], -1)   # Reshape to (XDW, Fl)
        mpo_reshaped = mpo_transposed.reshape(-1, mpo_transposed.shape[2] * mpo_transposed.shape[3])                   # Reshape to (Fl, dG)
        L[j] = L_reshaped @ mpo_reshaped                                                    # Resulting shape (XDW, dG)
        L[j] = L[j].reshape(L_transposed.shape[0], L_transposed.shape[1], L_transposed.shape[2], 
                            mpo_transposed.shape[2], mpo_transposed.shape[3])               # Reshape to (X, D, W, d, G)
        L[j] = L[j].transpose(0, 1, 3, 4, 2)                                                # Transpose to (X, D, d, G, W)

        
        # L[j] = np.einsum('XDdGW,DdEl->XlEGW', L[j], mpo[j]) 
        L_transposed = L[j].transpose(0, 3, 4, 1, 2)                                        # Transpose to (X, G, W, D, d)
        L_reshaped = L_transposed.reshape(L_transposed.shape[0] * L_transposed.shape[1] * L_transposed.shape[2], -1)  # Reshape to (XGW, Dd)
        mpo_reshaped = mpo[j].reshape(mpo[j].shape[0] * mpo[j].shape[1], mpo[j].shape[2] * mpo[j].shape[3])           # Reshape to (Dd, El)
        L[j] = L_reshaped @ mpo_reshaped                                                    # Resulting shape (XGW, El)
        L[j] = L[j].reshape(L_transposed.shape[0], L_transposed.shape[1], L_transposed.shape[2], 
                            mpo[j].shape[2], mpo[j].shape[3])                               # Reshape to (X, G, W, E, l)
        L[j] = L[j].transpose(0, 4, 3, 1, 2)                                                # Transpose to (X, l, E, G, W)
        
        # L[j] = np.einsum('XlEGW,XlY->YEGW', L[j], mps[j]) 
        L_transposed = L[j].transpose(2,3,4,0,1)                                            # Transpose to ( E,G,W,X,l)
        L_reshaped = L_transposed.reshape(-1,L_transposed.shape[3]*L_transposed.shape[4])   # Reshape to (EGW,Xl)
        mps_reshaped = mps[j].reshape(mps[j].shape[0]*mps[j].shape[1],mps[j].shape[2])      # Reshape to (Xl,Y)
        L[j] = L_reshaped @ mps_reshaped                                                    # Resulting shape (EGW, Y)
        L[j] = L[j].reshape(L_transposed.shape[0],L_transposed.shape[1],L_transposed.shape[2],-1) # Reshape to (E,G,W,Y)
        L[j] = L[j].transpose(3,0,1,2) #Tranpose to (Y,E,G,W)
    
    # L[0] = np.einsum('dDl,lX->dDX', mpo[0], mps[0]) 
    mpo_reshaped = mpo[0].reshape(mpo[0].shape[0] * mpo[0].shape[1], -1)                    # Reshape to (dD, l)
    mps_reshaped = mps[0].reshape(-1, mps[0].shape[1])                                      # Reshape to (l, X)
    temp = mpo_reshaped @ mps_reshaped                                      
    L[0] = temp.reshape(mpo[0].shape[0], mpo[0].shape[1], mps[0].shape[1])   

    # temp = np.einsum('dFl,lZ->ZFd', mpo_c[0], mps_c[0])
    mpo_reshaped = mpo_c[0].reshape(mpo_c[0].shape[0] * mpo_c[0].shape[1], -1)              # Reshape to (dF, l)
    mps_reshaped = mps_c[0].reshape(-1, mps_c[0].shape[1])                                  # Reshape to (l, Z)
    temp = mpo_reshaped @ mps_reshaped                                        
    temp = temp.reshape(mpo_c[0].shape[0], mpo_c[0].shape[1], mps_c[0].shape[1])            # Reshape to (d, F, Z)
    temp = temp.transpose(2, 1, 0)                                                          # Transpose to (Z, F, d)

    # L[0] = np.einsum('dDX,ZFd->XDFZ', L[0], temp)
    L_transposed = L[0].transpose(2, 1, 0)                                                  # Transpose to (X, D, d)
    L_reshaped = L_transposed.reshape(-1, L_transposed.shape[2])                            # Reshape to (XD, d)
    temp_transposed = temp.transpose(2, 1, 0)                                               # Transpose to (d, F, Z)
    temp_reshaped = temp_transposed.reshape(temp_transposed.shape[0], -1)                   # Reshape to (d, FZ)
    L[0] = L_reshaped @ temp_reshaped                                                       # Resulting shape (XD, FZ)
    L[0] = L[0].reshape(L_transposed.shape[0], L_transposed.shape[1], 
                        temp_transposed.shape[1], temp_transposed.shape[2])                 # Reshape to (X, D, F, Z)

    for j in range(1, n-1):
        # L[j] = np.einsum('XDFZ,ZlW->XDFlW', L[j-1], mps_c[j])
        L_reshaped = L[j-1].reshape(-1, L[j-1].shape[3])                                    # Reshape to (XDF, Z)
        mps_reshaped = mps_c[j].reshape(mps_c[j].shape[0], -1)                              # Reshape to (Z, lW)
        intermediate = L_reshaped @ mps_reshaped                                     
        intermediate_reshaped = intermediate.reshape(L[j-1].shape[0], L[j-1].shape[1], 
                                                    L[j-1].shape[2], mps_c[j].shape[1], 
                                                    mps_c[j].shape[2])                      # Reshape to (X, D, F, l, W)
        L[j] = intermediate_reshaped.transpose(0, 1, 2, 3, 4)                               # Transpose to (XDFlW)

        # L[j] = np.einsum('XDFlW,FdGl->XDdGW', L[j], mpo_c[j])
        L_transposed = L[j].transpose(0, 1, 4, 2, 3)                                        # Transpose to (X, D, W, F, l)
        mpo_transposed = mpo_c[j].transpose(0, 3, 1, 2)                                     # Transpose to (F, l, d, G)
        L_reshaped = L_transposed.reshape(L_transposed.shape[0] * L_transposed.shape[2] * L_transposed.shape[3], -1)   # Reshape to (XDW, Fl)
        mpo_reshaped = mpo_transposed.reshape(-1, mpo_transposed.shape[2] * mpo_transposed.shape[3])                   # Reshape to (Fl, dG)
        L[j] = L_reshaped @ mpo_reshaped                                                    # Resulting shape (XDW, dG)
        L[j] = L[j].reshape(L_transposed.shape[0], L_transposed.shape[1], L_transposed.shape[2], 
                            mpo_transposed.shape[2], mpo_transposed.shape[3])               # Reshape to (X, D, W, d, G)
        L[j] = L[j].transpose(0, 1, 3, 4, 2)                                                # Transpose to (X, D, d, G, W)

        
        # L[j] = np.einsum('XDdGW,DdEl->XlEGW', L[j], mpo[j]) 
        L_transposed = L[j].transpose(0, 3, 4, 1, 2)                                        # Transpose to (X, G, W, D, d)
        L_reshaped = L_transposed.reshape(L_transposed.shape[0] * L_transposed.shape[1] * L_transposed.shape[2], -1)  # Reshape to (XGW, Dd)
        mpo_reshaped = mpo[j].reshape(mpo[j].shape[0] * mpo[j].shape[1], mpo[j].shape[2] * mpo[j].shape[3])           # Reshape to (Dd, El)
        L[j] = L_reshaped @ mpo_reshaped                                                    # Resulting shape (XGW, El)
        L[j] = L[j].reshape(L_transposed.shape[0], L_transposed.shape[1], L_transposed.shape[2], 
                            mpo[j].shape[2], mpo[j].shape[3])                               # Reshape to (X, G, W, E, l)
        L[j] = L[j].transpose(0, 4, 3, 1, 2)                                                # Transpose to (X, l, E, G, W)
        
        # L[j] = np.einsum('XlEGW,XlY->YEGW', L[j], mps[j]) 
        L_transposed = L[j].transpose(2,3,4,0,1)                                            # Transpose to ( E,G,W,X,l)
        L_reshaped = L_transposed.reshape(-1,L_transposed.shape[3]*L_transposed.shape[4])   # Reshape to (EGW,Xl)
        mps_reshaped = mps[j].reshape(mps[j].shape[0]*mps[j].shape[1],mps[j].shape[2])      # Reshape to (Xl,Y)
        L[j] = L_reshaped @ mps_reshaped                                                    # Resulting shape (EGW, Y)
        L[j] = L[j].reshape(L_transposed.shape[0],L_transposed.shape[1],L_transposed.shape[2],-1) # Reshape to (E,G,W,Y)
        L[j] = L[j].transpose(3,0,1,2) #Tranpose to (Y,E,G,W)

    # ==========================================
    # 2. [First Density Matrix]
    # ==========================================

    # rho = np.einsum('XDFZ,Zl->XDFl', L[n-2], mps_c[n-1])
    L_reshaped = L[n-2].reshape(-1, L[n-2].shape[3])                                        # Reshape to (XDF, Z)
    mps_reshaped = mps_c[n-1].reshape(mps_c[n-1].shape[0], -1)                              # Reshape to (Z, l)
    intermediate = L_reshaped @ mps_reshaped                                   
    rho = intermediate.reshape(L[n-2].shape[0], L[n-2].shape[1], L[n-2].shape[2], mps_c[n-1].shape[1])  # Reshape to (X, D, F, l)

    # rho = np.einsum('XDFl,Fdl->XDd', rho, mpo_c[n-1])
    rho_reshaped = rho.reshape(rho.shape[0] * rho.shape[1], rho.shape[2] * rho.shape[3])    # Reshape to (XD, Fl)
    mpo_transposed = mpo_c[n-1].transpose(0, 2, 1)                                          # Transpose to (F, l, d)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0] * mpo_transposed.shape[1], -1)      # Reshape to (Fl, d)
    temp = rho_reshaped @ mpo_reshaped                                                      # Resulting shape (XD, d)
    rho = temp.reshape(rho.shape[0], rho.shape[1], -1)                                      # Reshape to (X, D, d)

    # rho = np.einsum('XDd,Xl->lDd', rho, mps[n-1])
    rho_reshaped = rho.reshape(rho.shape[0], -1)                                            # Reshape to (X, Dd)
    mps_reshaped = mps[n-1].reshape(mps[n-1].shape[0], -1)                                  # Reshape to (X, l)
    intermediate = rho_reshaped.T @ mps_reshaped                                                 
    rho = intermediate.T.reshape(mps[n-1].shape[1], rho.shape[1], rho.shape[2])             # Reshape to (l, D, d)
    
    # rho = np.einsum('lDd,Dol->od', rho, mpo[n-1])
    rho_transposed = rho.transpose(2, 1, 0)                                                       # Transpose to (d, D, l)
    rho_reshaped = rho_transposed.reshape(-1, rho_transposed.shape[1] * rho_transposed.shape[2])  # Reshape to (d, Dl)
    mpo_transposed = mpo[n-1].transpose(0, 2, 1)                                                  # Transpose to (D, l, o)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0] * mpo_transposed.shape[1], -1)  # Reshape to (Dl, o)
    rho = rho_reshaped @ mpo_reshaped                                                             # Resulting shape (d, o)
    rho = rho.T                                                                                   # Transpose to (o, d)

    # rho = np.einsum('XDFZ,Zl->XDFl', L[n-2], mps_c[n-1])
    L_reshaped = L[n-2].reshape(-1, L[n-2].shape[3])                                        # Reshape to (XDF, Z)
    mps_reshaped = mps_c[n-1].reshape(mps_c[n-1].shape[0], -1)                              # Reshape to (Z, l)
    intermediate = L_reshaped @ mps_reshaped                                   
    rho = intermediate.reshape(L[n-2].shape[0], L[n-2].shape[1], L[n-2].shape[2], mps_c[n-1].shape[1])  # Reshape to (X, D, F, l)

    # rho = np.einsum('XDFl,Fdl->XDd', rho, mpo_c[n-1])
    rho_reshaped = rho.reshape(rho.shape[0] * rho.shape[1], rho.shape[2] * rho.shape[3])    # Reshape to (XD, Fl)
    mpo_transposed = mpo_c[n-1].transpose(0, 2, 1)                                          # Transpose to (F, l, d)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0] * mpo_transposed.shape[1], -1)      # Reshape to (Fl, d)
    temp = rho_reshaped @ mpo_reshaped                                                      # Resulting shape (XD, d)
    rho = temp.reshape(rho.shape[0], rho.shape[1], -1)                                      # Reshape to (X, D, d)

    # rho = np.einsum('XDd,Xl->lDd', rho, mps[n-1])
    rho_reshaped = rho.reshape(rho.shape[0], -1)                                            # Reshape to (X, Dd)
    mps_reshaped = mps[n-1].reshape(mps[n-1].shape[0], -1)                                  # Reshape to (X, l)
    intermediate = rho_reshaped.T @ mps_reshaped                                                 
    rho = intermediate.T.reshape(mps[n-1].shape[1], rho.shape[1], rho.shape[2])             # Reshape to (l, D, d)
    
    # rho = np.einsum('lDd,Dol->od', rho, mpo[n-1])
    rho_transposed = rho.transpose(2, 1, 0)                                                       # Transpose to (d, D, l)
    rho_reshaped = rho_transposed.reshape(-1, rho_transposed.shape[1] * rho_transposed.shape[2])  # Reshape to (d, Dl)
    mpo_transposed = mpo[n-1].transpose(0, 2, 1)                                                  # Transpose to (D, l, o)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0] * mpo_transposed.shape[1], -1)  # Reshape to (Dl, o)
    rho = rho_reshaped @ mpo_reshaped                                                             # Resulting shape (d, o)
    rho = rho.T                                                                                   # Transpose to (o, d)

    # ==========================================
    # 3. [Eigendecomposition]
    # ==========================================
    _, U, _ = truncated_eig(rho, stop)
    mps_out[n-1] = U.T 

    # ==========================================
    # 4. [Cap Construction II]
    # ==========================================
    # M_top = np.einsum("dk,Gdl->Gkl", U, mpo_c[n-1])  
    Ut = U.T                                                                                       # Transpose to (k, d)
    mpo_transposed = mpo_c[n-1].transpose(1, 0, 2)                                                 # Transpose to (d, G, l)
    mpo_reshaped = mpo_transposed.reshape(-1, mpo_transposed.shape[1] * mpo_transposed.shape[2])   # Reshape to (d, GL)
    M_top = Ut @ mpo_reshaped                                                                      # Resulting shape (k, GL)
    M_top = M_top.reshape(-1, mpo_transposed.shape[1], mpo_transposed.shape[2])                    # Reshape to (k, G, l)
    M_top = M_top.transpose(1, 0, 2)                                                               # Transpose to (G, k, l)

    # M_top = np.einsum("Gkl,Wl->WGk", M_top, mps_c[n-1]) 
    M_top_reshaped = M_top.reshape(M_top.shape[0] * M_top.shape[1], -1)                            # Reshape to (Gk, l)
    mps_transposed = mps_c[n-1].T                                                                  # Transpose to (l, W)
    temp = M_top_reshaped @ mps_transposed                                                         # Resulting shape (Gk, W)
    M_top = temp.reshape(M_top.shape[0], M_top.shape[1], -1)                                       # Reshape to (G, k, W)
    M_top = M_top.transpose(2, 0, 1)                                                               # Transpose to (W, G, k)

    _, U, _ = truncated_eig(rho, stop)
    mps_out[n-1] = U.T 

    # ==========================================
    # 4. [Cap Construction II]
    # ==========================================
    # M_top = np.einsum("dk,Gdl->Gkl", U, mpo_c[n-1])  
    Ut = U.T                                                                                       # Transpose to (k, d)
    mpo_transposed = mpo_c[n-1].transpose(1, 0, 2)                                                 # Transpose to (d, G, l)
    mpo_reshaped = mpo_transposed.reshape(-1, mpo_transposed.shape[1] * mpo_transposed.shape[2])   # Reshape to (d, GL)
    M_top = Ut @ mpo_reshaped                                                                      # Resulting shape (k, GL)
    M_top = M_top.reshape(-1, mpo_transposed.shape[1], mpo_transposed.shape[2])                    # Reshape to (k, G, l)
    M_top = M_top.transpose(1, 0, 2)                                                               # Transpose to (G, k, l)

    # M_top = np.einsum("Gkl,Wl->WGk", M_top, mps_c[n-1]) 
    M_top_reshaped = M_top.reshape(M_top.shape[0] * M_top.shape[1], -1)                            # Reshape to (Gk, l)
    mps_transposed = mps_c[n-1].T                                                                  # Transpose to (l, W)
    temp = M_top_reshaped @ mps_transposed                                                         # Resulting shape (Gk, W)
    M_top = temp.reshape(M_top.shape[0], M_top.shape[1], -1)                                       # Reshape to (G, k, W)
    M_top = M_top.transpose(2, 0, 1)                                                               # Transpose to (W, G, k)

    for j in reversed(range(1, n-1)):
        # top = np.einsum("WGk,ZlW->ZlGk", M_top, mps_c[j])
        M_top_transposed = M_top.transpose(1, 2, 0)                                                # Transpose to (G, k, W)
        M_top_reshaped = M_top_transposed.reshape(-1, M_top_transposed.shape[2])                   # Reshape to (Gk, W)
        mps_transposed = mps_c[j].transpose(2, 0, 1)                                               # Transpose to (W, Z, l)
        mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0], -1)                         # Reshape to (W, Zl)
        top = M_top_reshaped @ mps_reshaped                                                        # Resulting shape (Gk, Zl)
        top = top.T                                                                                # Transpose to (Zl, Gk)
        top = top.reshape(mps_transposed.shape[1], mps_transposed.shape[2], 
                        M_top_transposed.shape[0], M_top_transposed.shape[1])                      # Reshape to (Z, l, G, k)

        # top = np.einsum("ZlGk,FdGl->ZFdk", top, mpo_c[j])
        top_transposed = top.transpose(0, 3, 2, 1)                                                 # Transpose to (Z, k, G, l)
        top_reshaped = top_transposed.reshape(top_transposed.shape[0] * top_transposed.shape[1], -1)  # Reshape to (Zk, Gl)
        mpo_transposed = mpo_c[j].transpose(2, 3, 0, 1)                                            # Transpose to (G, l, F, d)
        mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0] * mpo_transposed.shape[1], -1)  # Reshape to (Gl, Fd)
        top = top_reshaped @ mpo_reshaped                                                          # Resulting shape (Zk, Fd)
        top = top.reshape(top_transposed.shape[0], top_transposed.shape[1], 
                        mpo_transposed.shape[2], mpo_transposed.shape[3])                          # Reshape to (Z, k, F, d)
        top = top.transpose(0, 2, 3, 1)                                                            # Transpose to (Z, F, d, k)
        bottom = np.conj(top)
        
        # ==========================================
        # 5. [Density Matrix j]
        # ==========================================
        # rho = np.einsum("XDFZ,XDdk->kdFZ", L[j-1], bottom) 
        L_transposed = L[j-1].transpose(2,3,0,1) # Transpose to (F,Z,X,D)
        L_reshaped = L_transposed.reshape(L_transposed.shape[0]*L_transposed.shape[1],-1) #Reshape to (FZ,XD)
        bottom_reshaped = bottom.reshape(bottom.shape[0]*bottom.shape[1],-1) #Reshape to (XD ,dk)
        temp = L_reshaped @ bottom_reshaped # Resulting shape (FZ,dk)
        rho=temp.reshape(L_transposed.shape[0],L_transposed.shape[1],bottom.shape[2],bottom.shape[3]) #Reshape to (F,Z,d,k)
        rho = rho.transpose(3,2,0,1) #Transpose to (k,d,F,Z)
        
        # rho = np.einsum("kdFZ,ZFej->kdje", rho, top)
        rho_reshaped = rho.reshape(-1, rho.shape[2] * rho.shape[3])                                # Reshape to (kd, FZ)
        top_transposed = top.transpose(1, 0, 2, 3)                                                 # Transpose to (F, Z, e, j)
        top_reshaped = top_transposed.reshape(-1, top_transposed.shape[2] * top_transposed.shape[3])  # Reshape to (FZ, ej)
        temp = rho_reshaped @ top_reshaped                                                         # Resulting shape (kd, ej)
        rho = temp.reshape(rho.shape[0], rho.shape[1], top.shape[2], top.shape[3])                 # Reshape to (k, d, e, j)
        rho = rho.transpose(0, 1, 3, 2)                                                            # Transpose to (k, d, j, e)
         
        k_old = rho.shape[0]
        physdim = rho.shape[1]
        rho = rho.reshape(rho.shape[0] * rho.shape[1], rho.shape[2] * rho.shape[3])
        _, U, _ = truncated_eig(rho, stop)
        U = U.reshape(k_old,physdim,U.shape[1])
        
        # U = np.einsum("kdl->ldk",U)
        U = U.transpose(2, 1, 0)  # Transpose the dimensions from (k, d, l) to (l, d, k)
        mps_out[j] = U

        # ==========================================
        # 6. [Cap Construction j]
        # ==========================================
        # M_top = np.einsum("ZFdk,ldk->ZFl", top, U)
        top_reshaped = top.reshape(top.shape[0], top.shape[1], -1)                    # Reshape to (ZF, dk)
        U_reshaped = U.reshape(U.shape[0], -1)                                        # Reshape to (l, dk)
        intermediate = top_reshaped @ U_reshaped.T                                    
        M_top = intermediate.reshape(top.shape[0], top.shape[1], U.shape[0])          # Reshape to (Z, F, l)

    # mps_out[0] = np.einsum('lZ,ZFk->lFk', mps_c[0], M_top)
    mps_reshaped = mps_c[0].reshape(-1, mps_c[0].shape[1])                                    # Reshape to (l, Z)
    M_top_reshaped = M_top.reshape(M_top.shape[0], -1)                                        # Reshape to (Z, Fk)
    intermediate = mps_reshaped @ M_top_reshaped                                             
    mps_out[0] = intermediate.reshape(mps_c[0].shape[0], M_top.shape[1], M_top.shape[2])      # Reshape to (l, F, k)

    # mps_out[0] = np.einsum('lFk,dFl->dk', mps_out[0], mpo_c[0])
    mps_out_transposed = mps_out[0].transpose(2, 1, 0)                                        # Transpose to (k, F, l)
    mps_out_reshaped = mps_out_transposed.reshape(-1, mps_out_transposed.shape[1] * mps_out_transposed.shape[2])  # Reshape to (k, Fl)
    mpo_transposed = mpo_c[0].transpose(1, 2, 0)                                              # Transpose to (F, l, d)
    mpo_reshaped = mpo_transposed.reshape(-1, mpo_transposed.shape[2])                        # Reshape to (Fl, d)
    mps_out[0] = mps_out_reshaped @ mpo_reshaped                                              # Resulting shape (k, d)
    mps_out[0] = mps_out[0].T                                                                 # Transpose to (d, k)
    out = MPS(mps_out)
    if finalround: 
        out.round(stop=stop)
    return out


def density_matrix_einsum(mpo, mps, stop=Cutoff(1e-14), maxdim=None, normalize=False):
    """
    Einsum Indexing convention: 
     ______
    |      |    _______
    |      |-Z-|mps_c_j|-W-...
    |      |   |_______|
    |      |    __|l___
    |      |-F-|mpo_c_j|-G-...
    |      |   |_______|
    | L[j] |      |d  
    |      |
    |      |    __|d___
    |      |-D-| mpo_j |-E-...
    |      |   |_______|
    |      |    __|l___
    |      |-X-| mps_j |-Y-...
    |______|   |_______|
    
    Comments:
    - In cases where the same index is duplicated in a contraction ie: dxd density matrix the 
      surrogate index will be promoted one letter: dxd -> exd.
    - truncation variable produced by eig is k.
    """
    # ==========================================
    # 0. [Initialization]
    # ==========================================
    if len(mpo) != len(mps):
        raise ValueError("MPO and MPS must have the same length.")
    if maxdim is None:
        requested_maxdim = maxlinkdim(mpo) * maxlinkdim(mps)

    n = len(mpo)
    mps_out = mps.copy()
    mps_c = mps.dagger()
    mpo_c = mpo.dagger()
    physdim = mpo[0].shape[0]
    L = [None] * (n - 1)

    # ==========================================
    # 1. [Environment Tensor Construction]          
    # ==========================================
    L[0] = np.einsum('dDl,lX->dDX', mpo[0], mps[0]) 
    temp = np.einsum('dFl,lZ->ZFd', mpo_c[0], mps_c[0])
    L[0] = np.einsum('dDX,ZFd->XDFZ', L[0], temp)

    for j in range(1, n-1):
        L[j] = np.einsum('XDFZ,ZlW->XDFlW', L[j-1], mps_c[j])
        L[j] = np.einsum('XDFlW,FdGl->XDdGW', L[j], mpo_c[j])
        L[j] = np.einsum('XDdGW,DdEl->XlEGW', L[j], mpo[j])
        L[j] = np.einsum('XlEGW,XlY->YEGW', L[j], mps[j])

    # ==========================================
    # 2. [First Density Matrix]
    # ==========================================
    rho = np.einsum('XDFZ,Zl->XDFl', L[n-2], mps_c[n-1])
    rho = np.einsum('XDFl,Fdl->XDd', rho, mpo_c[n-1])
    rho = np.einsum('XDd,Xl->lDd', rho, mps[n-1])
    rho = np.einsum('lDd,Dol->od', rho, mpo[n-1])
    # ==========================================
    # 3. [Eigendecomposition]
    # ==========================================
    _, U, _ = truncated_eig(rho, stop)
    # print(U.shape)
    mps_out[n-1] = U.T 

    # ==========================================
    # 4. [Cap Construction II]
    # ==========================================
    M_top = np.einsum("dk,Gdl->Gkl", U, mpo_c[n-1])
    M_top = np.einsum("Gkl,Wl->WGk", M_top, mps_c[n-1])
    
    for j in reversed(range(1, n-1)):
        top = np.einsum("WGk,ZlW->ZlGk", M_top, mps_c[j])  
        top = np.einsum("ZlGk,FdGl->ZFdk", top, mpo_c[j])
        bottom = np.conj(top)

        # ==========================================
        # 5. [Density Matrix j]
        # ==========================================
        rho = np.einsum("XDFZ,XDdk->kdFZ", L[j-1], bottom)
        rho = np.einsum("kdFZ,ZFej->kdje", rho, top) #consider ej
        k_old = rho.shape[0]
        physdim = rho.shape[1]
        rho = rho.reshape(rho.shape[0] * rho.shape[1], rho.shape[2] * rho.shape[3])
        _, U, _ = truncated_eig(rho, stop)
        U = U.reshape(k_old,physdim,U.shape[1])
        U = np.einsum("kdl->ldk",U)
        mps_out[j] = U

        # ==========================================
        # 6. [Cap Construction j]
        # ==========================================
        M_top = np.einsum("ZFdk,ldk->ZFl", top, U)
        # M_bot = np.einsum("XDdk,ldk->XDl", bottom, Ut)

    mps_out[0] = np.einsum('lZ,ZFk->lFk', mps_c[0], M_top) 
    mps_out[0] = np.einsum('lFk,dFl->dk', mps_out[0], mpo_c[0])       
    return MPS(mps_out)
