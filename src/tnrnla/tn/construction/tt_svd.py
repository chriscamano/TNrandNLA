from tn.mps import MPS 
from tn.stopping import Cutoff
import numpy as np 
from tnrnla.linalg.lra import truncated_svd

def TTSVD(tensor, epsilon):
    d = len(tensor.shape)
    norm_A = np.linalg.norm(tensor)
    delta = epsilon / np.sqrt(d-1) * norm_A
    
    cores = []
    tensor_shape = tensor.shape
    C = tensor.copy()
    r_prev = 1
    
    for k in range(d - 1):
        n_k = tensor_shape[k]
        C = C.reshape(r_prev * n_k, -1)
        U, S, Vh = truncated_svd(C, stop=Cutoff(delta))
        r_k = U.shape[1]
        
        G_k = U.reshape(r_prev, n_k, r_k)
        if k == 0: 
            cores.append(G_k.reshape(G_k.shape[1],G_k.shape[2]))
        else:
            cores.append(G_k)
        C = np.dot(np.diag(S), Vh)
        r_prev = r_k
    
    cores.append(C)  # The last core
    
    return MPS(cores)

def TTSVD(A,dimfactors,stop=Cutoff(1e-14)):
    
    d = len(dimfactors)

    if stop.cutoff is not None:
        norm_A = np.linalg.norm(A)
        delta = stop.cutoff / np.sqrt(d-1) * norm_A
        stop = Cutoff(delta**2)
    
    cores = []
    C = A.copy()
    r_prev = 1
    
    for i in range(d - 1):
        C = C.reshape(r_prev * dimfactors[i], -1)
        U, S, Vh = truncated_svd(C, stop=stop)
        r_k = U.shape[1]
        G_k = U.reshape(r_prev, dimfactors[i], r_k)
    
        if i == 0: 
            cores.append(G_k.squeeze())
        else:
            cores.append(G_k)
        C = np.diag(S) @ Vh
        r_prev = r_k
    
    cores.append(C)  # The last core
    
    return MPS(cores)

def reverse_TTSVD(A, dimfactors, stop=Cutoff(1e-14)):
    d = len(dimfactors)
    if stop.cutoff is not None:
        norm_A = np.linalg.norm(A)
        delta = stop.cutoff / np.sqrt(d-1) * norm_A
        stop = Cutoff(delta**2)
        
    cores = []
    C = A.copy()
    r_i = 1
    
    for i in range(d - 1, 0, -1):
        C = C.reshape(-1, dimfactors[i] * r_i)
        U, S, Vh = truncated_svd(C, stop=stop)
        new_r = Vh.shape[0]
        G_k = Vh.reshape(new_r, dimfactors[i], r_i)

        cores.insert(0, G_k.squeeze())  
        C = U @ np.diag(S)
        r_i = new_r
    
    cores.insert(0, C.squeeze())  

    return MPS(cores)


def TSQR_TTSVD(A,dimfactors,stop=Cutoff(1e-14)):
    
    d = len(dimfactors)
    cores = []
    r_i = 1    
    C = A.reshape(-1, dimfactors[-1])
    
    for i in range(d - 1, 0, -1):
        #QR trick
        R = np.linalg.qr(C, mode='r')
        _, S, Vh = np.linalg.svd(R)
        
        if stop.cutoff:
            delta = stop.cutoff / (np.sqrt(d - 1) * np.linalg.norm(S))
            sum_squared = 0
            new_r = len(S)  # Initially, consider all singular values
            for j in range(len(S) - 1, -1, -1):
                sum_squared += S[j] ** 2
                if sum_squared > delta ** 2:
                    new_r = j + 1
                    break
        else:
            # If stop.cutoff is not used, we simply truncate based on stop.output_dim
            new_r = min(len(S), stop.outputdim)
            
        G_k = Vh[:new_r].reshape(new_r, dimfactors[i], r_i)

        if i ==d-1:
            cores.insert(0, G_k.reshape(G_k.shape[0],G_k.shape[1]))  
        else:
            cores.insert(0, G_k)  

        C = (C @ Vh[:new_r].T).reshape(-1, dimfactors[i-1] * new_r)
        r_i = new_r
        
    # print("Shape of core before insertion: final ",C.shape)
    cores.insert(0,C.reshape(dimfactors[0],r_i))  # The last core
    
    return MPS(cores)