import numpy as np
import numpy.linalg as la 
import random

from ..mps import MPS
from ..stopping import Cutoff
from tnrnla.linalg.lra import truncated_svd

"""
The Fitting algorithm for the compressed MPO-MPS product: 
This file contains BLAS3 and einsum level implementations of the Fitting algorithm for variationally 
approximating a MPO-MPS product. This method was originally proposed by in Verstraete and Cira "Renormalization algorithms for Quantum-Many Body Systems in two and higher dimensions
https://arxiv.org/abs/cond-mat/0407066

NOTE: this method is iterative by design and may take longer to converge compared to one shot methods (all others in this library)
Author: Chris Camaño Circa: 2023
"""

def fit(mpo, mps, max_sweeps=4, stop=Cutoff(1e-14), random_tensor=np.random.randn, guess=None):
    """
    Einsum Indexing convention L->R:
     ______                 ______
    |      |    _______    |      |   
    |      |-Z-|mps_c_j|-W-|      |
    |      |   |_______|   |      |
    |      |    __|d___    |      |
    | L[j] |-D-| mpo_j |-E-|R[N-j]|
    |      |   |_______|   |      |
    |      |    __|l___    |      |
    |      |-X-| mps_j |-Y-|      |
    |______|   |_______|   |______|
    """
    
    def right_sweep(mps, mpo, L, final_site=None,stop=Cutoff(1e-14)):
        mps_out = [None] * mps.N
        R = [None] * (mps.N - 2)

        if final_site is None:
            # site = np.einsum("XjY,Yl->Xjl", mps[-2], mps[-1]) 
            mps_reshaped = mps[-2].reshape(-1,mps[-2].shape[-1]) #Reshape to (Xj,Y)
            temp = mps_reshaped @ mps[-1] #Resulting shape (Xj,l)
            site = temp.reshape(mps[-2].shape[0],mps[-2].shape[1],temp.shape[-1]) #Reshape to (X,j,l)
            
            # site = np.einsum("Xjl,Edl->XjEd", site, mpo[-1])
            site_reshaped =site.reshape(-1,site.shape[-1]) #Reshape to (Xj,l)
            mpo_transposed = mpo[-1].transpose(2,0,1) #Transpose to (l,E,d)
            mpo_reshaped = mpo_transposed.reshape(mpo[-1].shape[0],-1) #Reshape to (l,Ed)
            temp = site_reshaped @ mpo_reshaped #Resulting shape (Xj,Ed)
            site = temp.reshape(site.shape[0],site.shape[1],mpo[-1].shape[0],mpo[-1].shape[1])
            
            
            # site = np.einsum("XjEk,DdEj->XDkd", site, mpo[-2])
            site_transposed = site.transpose(0,3,1,2) #Transpose to (X,k,j,E)
            site_reshaped = site_transposed.reshape(-1,site_transposed.shape[2]*site_transposed.shape[3]) #reshape to (Xk,jE)
            mpo_transposed = mpo[-2].transpose(3,2,0,1) #transpose to (j,E,D,d)
            mpo_reshaped =  mpo_transposed.reshape(-1,mpo_transposed.shape[2]*mpo_transposed.shape[3]) #reshape to (jE,Dd)
            temp = site_reshaped @ mpo_reshaped #resulting shape (Xk,Dd)
            temp = temp.reshape(site.shape[0],site.shape[3],mpo[-2].shape0,mpo[-2].shape[1]) #Reshape to (X,k,D,d)
            site = temp.transpose(0,2,1,3) #Transpose to (X,D,k,d)
            
            # site = np.einsum("pDX,XDkd->pdk", L[-1], site)
            L_transposed = L[k].transpose(0,2,1) #Transpose to p,X,D
            L_reshaped = L_transposed.reshape(L_transposed.shape[0],-1) #Reshape to (p,XD)
            site_reshaped = site.reshape (-1,site.shape[2]*site.shape[3] )#Reshape to (XD,kd)
            temp = L_reshaped @ site_reshaped #resulting shape p,kd
            temp = temp.reshape(temp.shape[0],site.shape[2],site.shape[3]) #Reshape to (p,k,d)
            site = temp.transpose(0,2,1) #Transpose to (p,d,k)
            
        else:
            site = final_site

        U, S, Vt = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2]), stop=stop)
        mps_out[-1] = (Vt).reshape(U.shape[1], mps[-1].shape[1])

        # R[-1] = np.einsum("Ddl,Xl->XDd", mpo[-1], mps[-1])
        mpo_reshaped = mpo[-1].reshape(-1,mpo[-1].shape[-1]) #reshape to (Dd,l)
        temp = mpo_reshaped @ mps[-1].T #Resulting shape (Dd,X)
        temp = temp.reshape (mpo[-1].shape[0],mps[-1].shape[1],temp.shape[1])#Reshape to (D,d,X)
        R[-1] = temp.transpose (2,0,1) # Transpose to (X,D,d)
        
        # R[-1] = np.einsum("XDd,pd->XDp", R[-1], mps_out[-1])
        R_reshaped = R[-1].reshape(-1,R[-1].shape[-1]) #Reshape to (XD,d)
        temp = R_reshaped @ mps_out[-1].T #Resulting shape (XD,p)
        R[-1] = temp.reshape (R[-1].shape[0],R[-1].shape[1],temp.shape[-1]) #Reshape to (X,D,p)

        # Middle sites.
        for k in range(mps.N - 2, 1, -1):
            # site = np.einsum("XDp,YlX->YlDp", R[k - 1], mps[k])
            R_transposed = R[k-1].transpose(1,2,0) # transpose to (D,p,X)
            R_reshaped = R_transposed.reshape (-1,R_transposed.shape[-1]) #Reshape to (Dp,X)
            mps_transposed =mps[k].transpose(2,0,1) #Transpose to (X,Y,l)
            mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0],-1) #Reshape to (X, Yl)
            temp = R_reshaped @ mps_reshaped #Resulting shape (Dp,Yl)
            temp = temp.reshape(R[k-1].shape[1],R[k-1].shape[2],mps[k].shape[0],mps[k].shape[1]) #Reshape to (D,p,Y,l)
            site = temp.transpose(2,3,0,1) #Transpose to (Y,l,D,p)
            
            # site = np.einsum("YlDp,EdDl->YEdp", site, mpo[k])
            site_transposed = site.transpose(0,3,1,2) #Transpose to (Y,p,l,D)
            site_reshaped = site_transposed.reshape(-1,site_transposed.shape[2]*site_transposed.shape[3]) #Reshape to (Yp,lD)
            mpo_transposed = mpo[k].transpose (3,2,0,1) #Transpose to (l,D,E,d)
            mpo_reshaped = mpo_transposed.reshape(-1,mpo_transposed.shape[2]*mpo_transposed.shape[3]) #Reshape to (lD,Ed)
            temp = site_reshaped @ mpo_reshaped #resulting shape (Yp,Ed)
            temp = temp.reshape(site.shape[0],site.shape[3],mpo[k].shape[0],mpo[k].shape[1]) #reshape to (Y,p,E,d)
            site = temp.transpose (0,2,3,1) #Transpose to ( Y,E,d,p)
            
            # site = np.einsum("YEdp,ZlY->ZlEdp", site, mps[k - 1])
            site_transposed = site.transpose(1,2,3,0) #Transpose to (E,d,p,Y)
            site_reshaped = site_transposed.reshape(-1,site_transposed.shape[-1]) #Reshape to (Edp,Y)
            mps_transposed = mps[k-1].transpose(2,0,1) #transpose to (Y,Z,l)
            mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0],-1) #reshape to (Y,Zl)
            temp = site_reshaped @ mps_reshaped #resulting shape (Edp,Zl)
            temp = temp.reshape(site.shape[1],site.shape[2],site.shape[3],mps[k-1].shape[0],mps[k-1].shape[1]) #Reshape to (E,d,p,Z,l)
            site = temp.transpose(3,4,0,1,2) #Transpose to (Z,l,E,d,p)
            
            # site = np.einsum("ZlEdp,FkEl->ZFkdp", site, mpo[k - 1])
            site_transposed = site.transpose(0,3,4,1,2) #Transpose to (Z,d,p,l,E)
            site_reshaped = site_transposed.reshape(-1,site_transposed.shape[3]*site_transposed.shape[4]) # Reshape to (Zdp,lE)
            mpo_transposed = mpo[k-1].transpose(3,2,0,1) #Transpose to (l,E,F,k)
            mpo_reshaped = mpo_transposed.reshape(-1,mpo_transposed.shape[2]*mpo_transposed.shape[3]) #Reeshape to (le,Fk)
            temp = site_reshaped @ mpo_reshaped #Resulting shape (Zdp,Fk)
            temp = temp.reshape(site.shape[0],site.shape[3],site.shape[4],mpo[k-1].shape[0],mpo[k-1].shape[1]) #Reshape to (Z,d,p,F,k)
            site = temp.transpose (0,3,4,1,2) #Transpose to (Z,F,k,d,p)
            
     
            # check = np.einsum("ZFkdp,qFZ->qkdp", site, L[k - 2])
            site_transposed = site.transpose(2,3,4,1,0) #Transpose (k,d,p,F,Z)
            site_reshaped = site_transposed.reshape(-1,site_transposed.shape[3]*site_transposed.shape[4]) #reshape to (kdp,FZ)
            L_transposed = L[k-2].transpose(1,2,0) #Transpose to (F,K,q)
            L_reshaped = L_transposed.reshape(-1,L_transposed.shape[2]) #Reshape to (FK,q)
            temp = site_reshaped @ L_reshaped # Resulting shape (kdp,q)
            temp = temp.reshape(site.shape[2],site.shape[3],site.shape[4],L[k-2].shape[0]) # (k,d,p,q)
            site = temp.transpose(3,0,1,2) #Transpose to (q,k,d,p)

            U, S, Vt = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2] * site.shape[3]), stop=stop)
            mps_out[k] = (Vt).reshape(U.shape[1], mps[-1].shape[1], site.shape[3])

            
            # R[k - 2] = np.einsum("XDp,qdp->XDdq", R[k - 1], mps_out[k])
            R_reshaped = R[k-1].reshape(-1,R[k-1].shape[-1]) #Reshape to (XD,p)
            mps_transposed = mps_out[k].transpose(2,0,1) #Transpose to (p,q,d)
            mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0],-1) #Reshape to (p,qd)
            temp = R_reshaped @ mps_reshaped #Resulting shape (XD,qd)
            temp = temp.reshape(R[k-1].shape[0],R[k-1].shape[1],mps_out[k].shape[0],mps_out[k].shape[1]) #Reshape to (X,D,q,d)
            R[k-2] = temp.transpose(0,1,3,2) #Transpose tp (X,D,d,q)
            
            # R[k - 2] = np.einsum("XDdq,EdDl->XlEq", R[k - 2], mpo[k])
            R_transposed = R[k-2].transpose(0,3,1,2) #Transpose to (X,q,D,d)
            R_reshaped = R_transposed.reshape(-1,R_transposed.shape[2]*R_transposed.shape[3]) #Reshape to (Xq,Dd)
            mpo_transposed = mpo[k].transpose(2,1,0,3) # Transpose to (D,d, E,l)
            mpo_reshaped = mpo_transposed.reshape (-1, mpo_transposed.shape[2]*mpo_transposed.shape[3]) #Reshape tp (Dd,El)
            temp = R_reshaped @ mpo_reshaped #Resulting shape (Xq,El)
            temp = temp.reshape(R[k-2].shape[0],R[k-2].shape[3],mpo[k].shape[0],mpo[k].shape[3]) #Reshape tp (X,q,E,l)
            R[k-2] = temp.transpose(0,3,2,1) #Transpose to (X,l,E,q)
            
            # R[k - 2] = np.einsum("XlEq,YlX->YEq", R[k - 2], mps[k])
            R_transposed = R[k-2].transpose(2,3,0,1) #Transpose to (E,q,X,l)
            R_reshaped = R_transposed.reshape(-1,R_transposed.shape[2]*R_transposed.shape[3]) #Reshape to (Eq,Xl)
            mps_transposed = mps[k].transpose(2,1,0) #Transpose to (X,l,Y)
            mpo_reshaped = mps_transposed.reshape(-1,mps_transposed.shape[-1]) #reshape to (Xl,Y)
            temp = R_reshaped @mpo_reshaped # Resulting shape (Eq,Y)
            temp = temp.reshape(R[k-2].shape[2],R[k-2].shape[3],mps[k].shape[0]) #Reshape to (E,q,Y)
            R[k-2] = temp.transpose (2,0,1) #Transpose to (Y,E,q)

        # site = np.einsum("lX,XkY->lkY",mps[0],mps[1])
        mps_reshaped = mps[1].reshape(mps[1].shape[0],-1) #Reshape to ( X,kY)
        temp = mps[0] @ mps_reshaped #Resulting shape (l,kY)
        site = temp.reshape(mps[0].shape[0],mps[1].shape[1],mps[1].shape[2]) #Reshape to (l,k,Y)
        
        # site = np.einsum("lkY,dDl->dDkY",site,mpo[0])
        site_transposed = site.transpose(1,2,0) #Transpose to (k,Y,l)
        site_reshaped = site_transposed.reshape (-1,site_transposed.shape[-1]) #Reshape to (kY,l)
        mpo_transposed = mpo[0].transpose(2,0,1) #Transpose to (l,d,D)
        mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0],-1)#Reshape to (l,dD)
        temp = site_reshaped @ mpo_reshaped #resulting shape (kY,dD)
        temp = temp.reshape(site.shape[1],site.shape[2],mpo[0].shape[0],mpo[0].shape[1]) #Reshape to (k,Y,d,D)
        site = temp.transpose(2,3,0,1) #Transpose to (d,D,k,Y)
        
        # site = np.einsum("kDlY,DdEl->kdEY",site,mpo[1])
        site_transposed = site.transpose(0,3,1,2) #Transpose to (k,Y,D,l)
        mpo_transposed = mpo[1].transpose(0,3,1,2) #Transpose to (D,l,d,E)
        site_reshaped = site_transposed.reshape(site_transposed.shape[0]*site_transposed.shape[1],-1) #Reshape to (kY,Dl)
        mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1) #Reshape tp (Dl,dE)
        temp = site_reshaped @ mpo_reshaped #Resulting shape (kY,dE)
        temp = temp.reshape(site.shape[0],site.shape[3],mpo[1].shape[1],mpo[1].shape[2]) #Reshape to (k,Y,d,E)
        site = temp.transpose(0,2,3,1) #Transpose to (k,d,E,Y)
        
        # final_site = np.einsum("kdEY,YEq->kdq",site,R[0])
        site_reshaped= site.reshape(site.shape[0]*site.shape[1],-1) #reshape to (kd,EY)
        R_transposed = R[0].transpose(1,0,2) #Transpose to ( E,Y,q)
        R_reshaped = R_transposed.reshape(-1,R_transposed.shape[-1]) #Reshape to (EY,q)
        temp  = site_reshaped @ R_reshaped #Reusulting shape (kd,q)
        final_site = temp.reshape(site.shape[0],site.shape[1],temp.shape[-1]) #Reshape tp (k,d,q)
        
        U, S, Vt = truncated_svd(final_site.reshape(final_site.shape[0], final_site.shape[1] * final_site.shape[2]), stop=stop)
        mps_out[1] = Vt.reshape(Vt.shape[0], mps[0].shape[0], R[0].shape[2])
        mps_out[0] = U @ np.diag(S)

        return R, final_site, MPS(mps_out)
    
    def left_sweep(mps, mpo, R, final_site=None,stop=Cutoff(1e-14)):
            mps_out = [None] * mps.N
            L = [None] * (mps.N - 2)

            # First Local Site from the left 
            if final_site is None:
                # site = np.einsum("lX,XjY->ljY", mps[0], mps[1])
                mps_reshaped = mps[1].reshape(mps[1].shape[0],-1)# Reshape to (X,jY)
                temp = mps[0] @ mps_reshaped                        #Resulting shape (l,jY)
                site = temp.reshape(temp.shape[0],mps[1].shape[1],mps[1].shape[2])# Reshape to l,j,Y
                
                # site = np.einsum("ljY,dDl->dDjY", site, mpo[0])
                site_transposed = site.transpose(1,2,0) #Transpose to (j,Y,l)
                site_reshaped = site_transposed.reshape(-1,site_transposed.shape[-1]) # reshape to (jY,l)
                mpo_transposed = mpo[0].transpose(2,0,1) # Transpose to (l,d,D)
                mpo_reshaped = mpo_transposed.reshape (mpo_transposed.shape[0],-1) #reshape to (l,dD)
                temp = site_reshaped @ mpo_reshaped #Resulting shape (jY,dD)
                temp = temp.reshape (site.shape[1],site.shape[2],mpo[0].shape[0],mpo[0].shape[1]) #Reshape to (j,Y,d,D)
                site = temp.transpose (2,3,0,1) # Transpose to (d,D,j,Y)
            
                # site = np.einsum("dDjY,DkEj->dkEY", site, mpo[1]) #TODO fix me 
                site_transposed = site.transpose(0,3,1,2) #Transpose to (d,Y,D,j)
                mpo_transposed = mpo[1].transpose(0,3,1,2) #Transpose to (D,j,k,E)
                site_reshaped = site_transposed.reshape(-1,site_transposed.shape[2]*site_transposed.shape[3]) #Reshape to (dY,Dj)
                mpo_reshaped = mpo_transposed.reshape(-1,mpo_transposed.shape[2]*mpo_transposed.shape[3]) #Reshape to (Dj,kE)
                temp = site_reshaped @ mpo_reshaped #Resulting shape (dY,kE)
                temp = temp.reshape(site.shape[0],site.shape[3],mpo[1].shape[1],mpo[1].shape[2]) #Reshape to (d,Y,k,E)
                site = temp.transpose(0,2,3,1) # transpose to (d,k,E,Y)
                
                # site = np.einsum("dkEY,YEW->dkW", site, R[0])
                site_transposed = site.transpose(0,1,3,2) #transpose to (d,k,Y,E)
                site_reshaped = site_transposed.reshape(-1,site_transposed.shape[2]*site_transposed.shape[3]) #Reshape to (dk,YE)
                R_reshaped = R[0].reshape(R[0].shape[0]*R[0].shape[1],-1)# Reshape to (YE,W)
                temp = site_reshaped @ R_reshaped #Resulting shape (dk,W)
                site = temp.reshape(site.shape[0],site.shape[1],R[0].shape[2])
                
            else:
                site = final_site
            U, S, Vt = truncated_svd(site.reshape(site.shape[0], site.shape[1] * site.shape[2]), stop=stop)
            mps_out[0] = U

            # L[0] = np.einsum("dp,dDl->pDl", U, mpo[0])
            mpo_reshaped =mpo[0].reshape(mpo[0].shape[0],-1) #reshape to (d,Dl)
            temp = U.T @ mpo_reshaped #Resulting shape (p,Dl)
            L[0] = temp.reshape(temp.shape[0],mpo[0].shape[1],mpo[0].shape[2])
            
            # L[0] = np.einsum("pDl,lX->pDX", L[0], mps[0])
            L_reshaped = L[0].reshape(-1,L[0].shape[2]) #Reshape to (pD,l)
            temp = L_reshaped @ mps[0] #Resulting shape (pD,X)
            L[0] = temp.reshape(L[0].shape[0],L[0].shape[1],mps[0].shape[1]) # Reshape to (p,D,X)

            # Sweep through the remaining sites
            for k in range(1, len(mps) - 2):
                # site = np.einsum("pDX,XlY->pDlY", L[k - 1], mps[k]) 
                L_reshaped = L[k-1].reshape(-1,L[k-1].shape[2]) # Reshape to (pD,X)
                mps_reshaped = mps[k].reshape(mps[k].shape[0],-1) #Reshape to (X,lY)
                temp = L_reshaped @mps_reshaped #Resulting shape (pD,lY)
                site= temp.reshape (L[k-1].shape[0],L[k-1].shape[1],mps[k].shape[1],mps[k].shape[2]) #Reshape to (p,D,l,Y)
                
                # site = np.einsum("pDlY,DdEl->pdEY", site, mpo[k])
                site_transposed = site.transpose(0,3,1,2) #Transpose to ( p,Y,D,l)
                site_reshaped = site_transposed.reshape(-1,site_transposed.shape[2]*site_transposed.shape[3]) #Reshape to ( pY,Dl)
                mpo_transposed = mpo[k].transpose(0,3,1,2) #Transpose to (D,l,d,E)
                mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1) # Reshape to (Dl,dE)
                temp = site_reshaped @ mpo_reshaped #Resulting shape (pY,dE)
                temp = temp.reshape(site.shape[0],site.shape[3],mpo[k].shape[1],mpo[k].shape[2]) #Reshape to ( p,Y,d,E)
                site = temp.transpose ( 0,2,3,1) #Trannspose to (p,d,E,Y)
                
                
                # site = np.einsum("pdEY,YlZ->pdElZ", site, mps[k + 1])
                site_reshaped = site.reshape(-1,site.shape[-1]) #Reshape to (pdE,Y)
                mps_reshaped = mps[k+1].reshape(mps[k+1].shape[0],-1) #reshape to (Y,lZ)
                temp = site_reshaped @ mps_reshaped #Resulting shape (pdE,lZ)
                site = temp.reshape(site.shape[0],site.shape[1],site.shape[2],mps[k+1].shape[1],mps[k+1].shape[2]) #Reshape to (p,d,E,l,Z)
                
                
                # site = np.einsum("pdElZ,EkFl->pdkFZ", site, mpo[k + 1])
                site_transposed = site.transpose(0,1,4,2,3) #Transpose to (p,d,Z,E,l)
                site_reshaped = site_transposed.reshape(-1,site_transposed.shape[3]*site_transposed.shape[4]) #Reshape to (pdZ,El)
                mpo_transposed = mpo[k+1].transpose(0,3,1,2) #Transpose to (E,l,k,F)
                mpo_reshaped = mpo_transposed.reshape(-1,mpo_transposed.shape[2]*mpo_transposed.shape[3]) #Reshape to (El,kF)
                temp = site_reshaped @ mpo_reshaped #Resulting shape (pdZ,kF)
                temp = temp.reshape(site.shape[0],site.shape[1],site.shape[4],mpo[k+1].shape[1],mpo[k+1].shape[2])#Reshape to (p,d,Z,k,F)
                site = temp.transpose(0,1,3,4,2) #Transpose to (p,d,k,F,Z)
                
                # site = np.einsum("pdkFZ,ZFW->pdkW", site, R[k])
                site_reshaped = site.reshape(-1,site.shape[3]*site.shape[4]) #Reshape to (pdk,FZ)
                R_transposed = R[k].transpose(1,0,2) # Transpose to (Z,F,W)
                R_reshaped  = R_transposed.reshape(-1,R_transposed.shape[-1]) #Reshape to (ZF,W)
                temp = site_reshaped @ R_reshaped #Resulting shape (pdk,W)
                site = temp.reshape(site.shape[0],site.shape[1],site.shape[2],R[k].shape[-1]) #Reshape to (p,d,k,W)
                
                U, _, _ = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2] * site.shape[3]), stop=stop)
                mps_out[k] = U.reshape(site.shape[0], mpo[k].shape[1], U.shape[1])

                # L[k] = np.einsum("pDX,pdq->qdDX", L[k - 1], mps_out[k])
                L_transposed = L[k-1].transpose(1,2,0) #Transpose to (D,X,l)
                L_reshaped = L_transposed.reshape(-1,L_transposed.shape[-1] )#Reshape to (Dx,p)
                mps_reshaped = mps_out[k].reshape(mps_out[k].shape[0],-1) #Reshape to (p,dq)
                temp = L_reshaped @ mps_reshaped #Resulting shape (DX,dq)
                temp = temp.reshape(L[k-1].shape[1],L[k-1].shape[2],mps_out[k].shape[1],mps_out[k].shape[-1]) #Reshape to (D,X,d,q)
                L[k] = temp.transpose(3,2,0,1) #Transpose to (q,d,D,X)
                
                # L[k] = np.einsum("qdDX,DdEl->qElX", L[k], mpo[k])
                L_transposed = L[k].transpose(0,3,2,1) #Transpose to (q,X,D,d)
                L_reshaped = L_transposed.reshape(-1,L_transposed.shape[2]*L_transposed.shape[3] )#Reshape to (qX,Dd)
                mpo_reshaped = mpo[k].reshape(-1,mpo[k].shape[2]*mpo[k].shape[3]) #Reshape to (Dd,El)
                temp = L_reshaped @  mpo_reshaped #resulting shape (qX,El)
                temp = temp.reshape(L[k].shape[0],L[k].shape[3],mpo[k].shape[2],mpo[k].shape[3]) #Reshape to (q,X,E,l)
                L[k] = temp.transpose(0,2,3,1) #Transpose to (q,E,l,X)
                
                # L[k] = np.einsum("qElX,XlY->qEY", L[k], mps[k])
                L_reshaped = L[k].reshape(-1,L[k].shape[2]*L[k].shape[3]) #Reshape to (qE,lX)
                mps_transposed = mps[k].transpose(1,0,2) # Transpose to (l,X,Y)
                mps_reshaped = mps_transposed.reshape(-1,mps_transposed.shape[-1]) #Reshape to (lX,Y)
                temp = L_reshaped @mps_reshaped #Resulting shape (qE,Y)
                L[k]= temp.reshape(L[k].shape[0],L[k].shape[1],mps[k].shape[2])

            # Final two sites
            
            # site = np.einsum("XlY,Yk->Xlk",mps[-2],mps[-1])
            mps_reshaped = mps[-2].reshape(-1,mps[-2].shape[-1]) #Reshape to (Xl,Y)
            temp = mps_reshaped  @ mps[-1] #Resulting shape (Xl,k)
            site = temp.reshape(mps[-2].shape[0],mps[-2].shape[1],mps[1].shape[1]) #Reshape to (X,l,k)
   
            
            # site = np.einsum("Xlk,Edk->XlEd",site,mpo[-1])
            site_reshaped = site.reshape(-1,site.shape[-1]) #Reshape to (Xl,k)
            mpo_transposed = mpo[-1].transpose(2,0,1) #Transpose to (k,E,d)
            mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0],-1) #Reshape to (k,Ed)
            temp = site_reshaped @ mpo_reshaped #Resulting shape (Xl,Ed)
            site = temp.reshape(site.shape[0],site.shape[1],mpo[-1].shape[0],mpo[-1].shape[1]) #Reshape to (Xl,Ed)
            
            # site = np.einsum("XlEd,DkEl->XDdk",site,mpo[-2])
            site_transposed = site.transpose(0,3,1,2) #Tranpose to (X,d,l,E)
            mpo_transposed = mpo[-2].transpose(3,2,0,1) #Transpose to (l,E,D,k)
            site_reshaped = site_transposed.reshape(site_transposed.shape[0]*site_transposed.shape[1],-1) #Reshape to (Xd,lE)
            mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1) #Reshape to (lE,Dk)
            temp = site_reshaped @mpo_reshaped #Resulting shape (Xd,Dk)
            temp = temp.reshape(site.shape[0],site.shape[3],mpo[-2].shape[0],mpo[-2].shape[3]) #Reshape to (X,d,D,k)
            site = temp.transpose(0,2,1,3) #Transpose to (X,D,d,k)
            
            # final_site = np.einsum("qDX,XDdk->qdk",L[k],site)
            L_reshaped = L[k].reshape(L[k].shape[0],-1) #Reshape to (q,DX)
            site_transposed = site.transpose(1,0,2,3) #Transpose to (D,X,d,k)
            site_reshaped = site_transposed.reshape(site_transposed.shape[0]*site_transposed.shape[1],-1) #Reshape to (DX,dk)
            temp = L_reshaped @ site_reshaped #Resulting shape (q,dk)
            final_site = temp.reshape(temp.shape[0],site.shape[2],site.shape[3])
            
            U, S, Vt = truncated_svd(final_site.reshape(final_site.shape[0] * final_site.shape[1], final_site.shape[2]), stop=stop)
            mps_out[-2] = U.reshape(final_site.shape[0], mps[-2].shape[1], final_site.shape[2])
            mps_out[-1] = (np.diag(S) @ Vt).reshape(final_site.shape[2], mps[-1].shape[1])

            return L, final_site, MPS(mps_out)

    def compute_left_envs(mps, mpo, guess): #TODO: Convert to einsums if needed ie starting from right
            L = [None] * (mps.N - 2)
            # -------- Left environments --------
            L[0] = np.einsum("dDl,dZ->ZDl", mpo[0], guess[0])
            L[0] = np.einsum("ZDl,lX->ZDX", L[0], mps[0])

            for i in range(1, mps.N - 2):
                L[i] = np.einsum("ZDX,XlY->ZDlY", L[i - 1], mps[i])
                L[i] = np.einsum("ZDlY,DdEl->ZdEY", L[i], mpo[i])
                L[i] = np.einsum("ZdEY,ZdW->WEY", L[i], guess[i])
            return L

    def compute_right_envs(mps, mpo, guess):
            # Only needed if starting from left
            R = [None] * (mps.N - 2)
            # -------- Right environments --------
            
            # R[-1] = np.einsum("Ddl,Zd->ZDl", mpo[-1], guess[-1])
            mpo_transposed = mpo[-1].transpose(0,2,1)                               # Transpose to (D,l,d)
            mpo_reshaped = mpo_transposed.reshape(-1,mpo_transposed.shape[-1])      # Reshape to (Dl,d)
            temp = mpo_reshaped @ guess[-1].T                                       # Resulting shape to (Dl,Z)
            temp = temp.reshape(mpo[-1].shape[0],mpo[-1].shape[1],temp.shape[-1])   # Reshape to (D,l,Z)
            R[-1] = temp.transpose(2,0,1)                                           # Transpose to (Z,D,L)
            
            
            #R[-1] = np.einsum("ZDl,Xl->XDZ", R[-1], mps[-1])
            R_reshaped = R[-1].reshape(-1,R[-1].shape[-1])                           # Reshape to (ZD,l)
            temp = R_reshaped @ mps[-1].T                                            # Resulting shape (ZD,X)
            temp = temp.reshape(R[-1].shape[0],R[-1].shape[1],mps[-1].shape[0])      # Reshape to (Z,D,X)
            R[-1]=temp.transpose(2,1,0)
            

            for i in range(mps.N - 2, 1, -1):
                # R[i - 2] = np.einsum("XDZ,YlX->YlDZ", R[i - 1], mps[i])
                R_transposed = R[i-1].transpose(1,2,0)                              # Tranpose to (D,Z,X)
                R_reshaped = R_transposed.reshape(-1,R_transposed.shape[-1])        #Reshape (DZ,X)
                mps_transposed = mps[i].transpose(2,0,1)                            #Transpose to (X,Y,l)
                mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0],-1)   # Reshape to (X,Yl)
                temp = R_reshaped @ mps_reshaped                                    #Resulting shape (DZ,Yl)
                temp = temp.reshape(R[i-1].shape[1],R[i-1].shape[2],mps[i].shape[0],mps[i].shape[1]) #Reshape to (D,Z,Y,l)
                R[i - 2] = temp.transpose(2,3,0,1)
                
                # R[i - 2] = np.einsum("YlDZ,EdDl->YEdZ", R[i - 2], mpo[i])
                R_transposed = R[i-2].transpose(0,3,1,2)                            # Transpose to (Y,Z,l,D)
                R_reshaped = R_transposed.reshape(-1,R_transposed.shape[2]*R_transposed.shape[3]) #Reshape to (YZ,lD)
                mpo_transposed = mpo[i].transpose(3,2,0,1)                          # Transpose to (l,D,E,d)
                mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1) # reshape to (ld,Ed)
                temp = R_reshaped @ mpo_reshaped                                # Resulting shape (YZ,Ed)
                temp = temp.reshape(R[i-2].shape[0],R[i-2].shape[3],mpo[i].shape[0],mpo[i].shape[1]) #reshape to (Y,Z,E,d)
                R[i-2] = temp.transpose(0,2,3,1)
                
                # R[i - 2] = np.einsum("YEdZ,WdZ->YEW", R[i - 2], guess[i])
                R_reshaped = R[i-2].reshape(-1,R[i-2].shape[2]*R[i-2].shape[3])         # reshape to (YE,dZ)
                guess_transposed = guess[i].transpose(1,2,0)                            #Transpose to (d,Z,W)
                guess_reshaped = guess_transposed.reshape(-1,guess_transposed.shape[-1])#Reshape to (dZ,W)
                temp = R_reshaped @ guess_reshaped                                      #Resulting shape (YE,W)
                R[i-2] = temp.reshape(R[i-2].shape[0],R[i-2].shape[1],guess[i].shape[0])#Reshape to (Y,E,W)
            return R
        
    # Form a random MPS |ψB> of bond dimensionm 
    # states= [np.random.randn(mps[1].shape[0]) for i in range(mps.N)]
    # guess=MPS([np.reshape(states[0],(len(states[0]),1))] + [np.reshape(states[i],(1,len(states[i]),1)) for i in range (1,len(states)-1)] + [np.reshape(states[-1],(1,len(states[-1])))])
    
    if guess is None:
        guess = MPS.rmps(n=mps.N, m=mps[0].shape[1], d=mps[0].shape[0], random_tensor=random_tensor)
    elif guess == "input":
        guess = mps.copy()

    # orthogonalize it to have any arbitrary orthogonality center.
    guess.orthR()

    R = compute_right_envs(mps, mpo, guess)


    final_site = None
    for sweep_count in range(max_sweeps):
        L, final_site, mps_approx = left_sweep(mps, mpo, R, stop=stop,final_site=final_site)
        R, final_site, mps_approx = right_sweep(mps, mpo, L, stop=stop,final_site=final_site)

    mps_approx.canform == "Left"
    return mps_approx


def fit_einsum(mpo, mps, max_sweeps=10, stop=Cutoff(1e-14), random_tensor=np.random.randn):
    """
    Einsum Indexing convention L->R:
     ______                 ______
    |      |    _______    |      |   
    |      |-Z-|mps_c_j|-W-|      |
    |      |   |_______|   |      |
    |      |    __|d___    |      |
    | L[j] |-D-| mpo_j |-E-|R[N-j]|
    |      |   |_______|   |      |
    |      |    __|l___    |      |
    |      |-X-| mps_j |-Y-|      |
    |______|   |_______|   |______|
    """
    
    def right_sweep(mps, mpo, L, final_site=None,stop=Cutoff(1e-14)):
        mps_out = [None] * mps.N
        R = [None] * (mps.N - 2)

        if final_site is None:
            base = np.einsum("XjY,Yl->Xjl", mps[-2], mps[-1])
            site = np.einsum("Xjl,Edl->XjEd", base, mpo[-1])
            site = np.einsum("XjEk,DdEj->XDkd", site, mpo[-2])
            site = np.einsum("pDX,XDkd->pdk", L[-1], site)
        else:
            site = final_site

        U, S, Vt = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2]), stop=stop)
        mps_out[-1] = (Vt).reshape(U.shape[1], mps[-1].shape[1])

        R[-1] = np.einsum("Ddl,Xl->XDd", mpo[-1], mps[-1])
        R[-1] = np.einsum("XDd,pd->XDp", R[-1], mps_out[-1])

        # Middle sites.
        for k in range(mps.N - 2, 1, -1):
            site = np.einsum("XDp,YlX->YlDp", R[k - 1], mps[k])
            site = np.einsum("YlDp,EdDl->YEdp", site, mpo[k])
            site = np.einsum("YEdp,ZlY->ZlEdp", site, mps[k - 1])
            site = np.einsum("ZlEdp,FkEl->ZFkdp", site, mpo[k - 1])  # five tensor?
            # print(site.shape,L[k-1].shape)
            # print(k)
            # print(site.shape,L[k-1].shape)
            site = np.einsum("ZFkdp,qFZ->qkdp", site, L[k - 2])

            U, S, Vt = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2] * site.shape[3]), stop=stop)
            mps_out[k] = (Vt).reshape(U.shape[1], mps[-1].shape[1], site.shape[3])

            
            R[k - 2] = np.einsum("XDp,qdp->XDdq", R[k - 1], mps_out[k])
            R[k - 2] = np.einsum("XDdq,EdDl->XlEq", R[k - 2], mpo[k])
            R[k - 2] = np.einsum("XlEq,YlX->YEq", R[k - 2], mps[k])

        
        site = np.einsum("lX,XkY->lkY",mps[0],mps[1])
        site = np.einsum("lkY,dDl->dDkY",site,mpo[0])
        site = np.einsum("kDlY,DdEl->kdEY",site,mpo[1])
        final_site = np.einsum("kdEY,YEq->kdq",site,R[0])

        U, S, Vt = truncated_svd(final_site.reshape(final_site.shape[0], final_site.shape[1] * final_site.shape[2]), stop=stop)
        mps_out[1] = Vt.reshape(Vt.shape[0], mps[0].shape[0], R[0].shape[2])
        mps_out[0] = U @ np.diag(S)

        return R, final_site, MPS(mps_out)
    
    def left_sweep(mps, mpo, R, final_site=None,stop=Cutoff(1e-14)):
            mps_out = [None] * mps.N
            L = [None] * (mps.N - 2)

            # First Local Site from the left (H_0ψ_0)(H_1ψ_1)R_0  of size (d,d,X)
            if final_site is None:
                site = np.einsum("lX,XjY->ljY", mps[0], mps[1])
                site = np.einsum("ljY,dDl->dDjY", site, mpo[0])
                site = np.einsum("dDjY,DkEj->dkEY", site, mpo[1])
                site = np.einsum("dkEY,YEW->dkW", site, R[0])
            else:
                site = final_site
            U, S, Vt = truncated_svd(site.reshape(site.shape[0], site.shape[1] * site.shape[2]), stop=stop)
            mps_out[0] = U

            L[0] = np.einsum("dp,dDl->pDl", U, mpo[0])
            L[0] = np.einsum("pDl,lX->pDX", L[0], mps[0])

            # Sweep through the remaining sites
            for k in range(1, len(mps) - 2):
                site = np.einsum("pDX,XlY->pDlY", L[k - 1], mps[k])
                site = np.einsum("pDlY,DdEl->pdEY", site, mpo[k])
                site = np.einsum("pdEY,YlZ->pdElZ", site, mps[k + 1])
                site = np.einsum("pdElZ,EkFl->pdkFZ", site, mpo[k + 1])
                site = np.einsum("pdkFZ,ZFW->pdkW", site, R[k])

                U, _, _ = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2] * site.shape[3]), stop=stop)
                mps_out[k] = U.reshape(site.shape[0], mpo[k].shape[1], U.shape[1])

                L[k] = np.einsum("pDX,pdq->qdDX", L[k - 1], mps_out[k])
                L[k] = np.einsum("qdDX,DdEl->qElX", L[k], mpo[k])
                L[k] = np.einsum("qElX,XlY->qEY", L[k], mps[k])

            # Final two sites
            site = np.einsum("XlY,Yk->Xlk",mps[-2],mps[-1])
            site = np.einsum("Xlk,Edk->XlEd",site,mpo[-1])
            site = np.einsum("XlEd,DkEl->XDdk",site,mpo[-2])
            final_site = np.einsum("qDX,XDdk->qdk",L[k],site)

            U, S, Vt = truncated_svd(final_site.reshape(final_site.shape[0] * final_site.shape[1], final_site.shape[2]), stop=stop)
            mps_out[-2] = U.reshape(final_site.shape[0], mps[-2].shape[1], final_site.shape[2])
            mps_out[-1] = (np.diag(S) @ Vt).reshape(final_site.shape[2], mps[-1].shape[1])

            return L, final_site, MPS(mps_out)

    def compute_left_envs(mps, mpo, guess):
            L = [None] * (mps.N - 2)
            # -------- Left environments --------
            L[0] = np.einsum("dDl,dZ->ZDl", mpo[0], guess[0])
            L[0] = np.einsum("ZDl,lX->ZDX", L[0], mps[0])

            for i in range(1, mps.N - 2):
                L[i] = np.einsum("ZDX,XlY->ZDlY", L[i - 1], mps[i])
                L[i] = np.einsum("ZDlY,DdEl->ZdEY", L[i], mpo[i])
                L[i] = np.einsum("ZdEY,ZdW->WEY", L[i], guess[i])
            return L

    def compute_right_envs(mps, mpo, guess):
            # Only needed if starting from left
            R = [None] * (mps.N - 2)
            # -------- Right environments --------
            R[-1] = np.einsum("Ddl,Zd->ZDl", mpo[-1], guess[-1])
            R[-1] = np.einsum("ZDl,Xl->XDZ", R[-1], mps[-1])

            for i in range(mps.N - 2, 1, -1):
                R[i - 2] = np.einsum("XDZ,YlX->YlDZ", R[i - 1], mps[i])
                R[i - 2] = np.einsum("YlDZ,EdDl->YEdZ", R[i - 2], mpo[i])
                R[i - 2] = np.einsum("YEdZ,WdZ->YEW", R[i - 2], guess[i])
            return R
        
    # Form a random MPS |ψB> of bond dimension m
    guess = MPS.rmps(n=mps.N, m=mps[0].shape[1], d=mps[0].shape[0], random_tensor=random_tensor)

    # orthogonalize it to have any arbitrary orthogonality center.
    guess.orthR()

    R = compute_right_envs(mps, mpo, guess)

    sweep_count = 0
    final_site = None
    for sweep_count in range(max_sweeps):
    # for sweep_count in tqdm(range(max_sweeps), desc="Sweeping Progress"):
        if sweep_count % 2 == 0:
            L, final_site, mps_approx = left_sweep(mps, mpo, R, stop=stop,final_site=final_site)
        else:
            R, final_site, mps_approx = right_sweep(mps, mpo, L,stop-stop, final_site=final_site)

    return mps_approx

