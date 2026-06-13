import numpy as np 
from tnrnla.linalg.lra import truncated_svd
from tnrnla.tn.stopping import Cutoff, FixedDimension
from tnrnla.tn.mps import MPS
from tnrnla.tn.mpo import MPO

def purity_from_cut(mps, cut):
        """
        Purity Tr(rho^2) for the bipartition (cut | cut+1), computed from Schmidt values.
        Assumes mps.move_pivot(cut) exists.
        """
        mps.move_pivot(cut)
        C = mps[cut]                             # shape (a, s, b) or boundary variant
        if C.ndim == 2:                          # coerce boundary to 3D
            if C.shape[0] <= C.shape[1]:
                s, b = C.shape; C = C.reshape(1, s, b)
            else:
                a, s = C.shape; C = C.reshape(a, s, 1)
        a, s, b = C.shape
        sing = np.linalg.svd(C.reshape(a * s, b), compute_uv=False)
        lam  = sing / np.linalg.norm(sing)
        return float(np.sum(lam**4))


def state_vector_to_mps(vector, num_spins, stop = FixedDimension(20)):
    N = int(np.log(len(vector)) / np.log(num_spins)) 
    mps_out = []
    
    # Initial reshape of the vector
    tensor = vector.reshape(num_spins, -1)
    
    U, S, Vh = truncated_svd(tensor, stop=stop)
    mps_out.append(U)
    tensor = np.diag(S) @ Vh
    for i in range(1, N - 1):
        tensor = tensor.reshape(tensor.shape[0]*num_spins,-1)
        U, S, Vh = truncated_svd(tensor, stop=stop)
        # Intermediate tensors: shape (previous bond_dim, d, bond_dim)
        U = U.reshape(mps_out[-1].shape[-1], num_spins, U.shape[1])
        mps_out.append(U)
        tensor = np.dot(np.diag(S), Vh)
    # Fina tensor: shape (bond_dim, d)
    tensor = tensor.reshape(mps_out[-1].shape[-1],num_spins)
    mps_out.append(tensor)
    
    return MPS(mps_out)

def flatten_mps(mps):
    """Flattens the MPS into a single vector of size d^n."""
    vector = mps.tensors[0]
    for tensor in mps.tensors[1:mps.N-1]:        
        vector = np.einsum("dX,XkY->dkY",vector,tensor)         # Absorb left site giving 
        vector = vector.reshape(-1,vector.shape[-1])            # Reshape to (d^j,X)
    vector = vector @ mps.tensors[-1]                                   # Resulting shape d^n-1 x d 
    return vector.reshape(mps.tensors[0].shape[0]**mps.N)               # Final reshape to a vector

#potentially unstable !
def hamiltonian_to_mpo(matrix, num_spins, stop=Cutoff(1e-14)):
    n_rows, n_cols = matrix.shape
    assert n_rows == n_cols and n_rows == num_spins ** int(np.log(n_rows) / np.log(num_spins)), "Matrix must be square and dimensions must be powers of d."
    
    N = int(np.log(n_rows) / np.log(num_spins))
    mpo_out = []

    # Initial reshape of the matrix
    tensor = matrix.reshape(num_spins**2, -1)

    U, S, Vh = truncated_svd(tensor, stop=stop)
    mpo_out.append(U.reshape(num_spins, U.shape[1],num_spins))
    tensor = np.diag(S) @ Vh
    for i in range(1, N - 1):
        tensor = tensor.reshape(tensor.shape[0] * num_spins**2, -1)
        U, S, Vh = truncated_svd(tensor, stop=stop)
        # Intermediate tensors: shape (previous bond_dim, d, bond_dim, d)
        U = U.reshape(mpo_out[-1].shape[-2], num_spins, U.shape[1],num_spins)
        mpo_out.append(U)
        tensor = np.dot(np.diag(S), Vh)
    # Final tensor: shape (bond_dim, d, d)
    
    tensor = tensor.reshape(mpo_out[-1].shape[-2], num_spins, num_spins)
    mpo_out.append(tensor)

    return MPO(mpo_out)

def flatten_mpo(mpo):
    tmp = np.einsum("arb,rARB->aAbBR",mpo[0],mpo[1])
    tmp = tmp.reshape((4,4,mpo[1].shape[2]))
    size = 4
    for i in range(2,mpo.N-1):
        tmp = np.einsum("abr,rARB->aAbBR",tmp,mpo[i])
        size *= 2
        tmp = tmp.reshape((size,size,mpo[i].shape[2]))
    H = np.einsum("abr,rAB->aAbB",tmp,mpo[-1])
    size *= 2
    H = H.reshape(size,size)
    return H 

def spin_up_state(N, d):
    spin_up_vector = np.array([1, 0])
    
    mps_matrices = []
    
    mps_matrices.append(spin_up_vector.reshape(d, 1))
    
    for _ in range(1, N-1):
        mps_matrices.append(spin_up_vector.reshape(1, d, 1))
    
    mps_matrices.append(spin_up_vector.reshape(1, d))
    
    return MPS(mps_matrices)

def measure_magnetizations(mps, op = np.array([[0.0,1.0],[1.0,0.0]])):
    mps.orthLR()
    magnetizations = np.zeros(mps.N)
    for i in range(mps.N):
        if i == 0:
            magnetizations[0] = np.trace(mps[0].conj().T @ op @ mps[0]).real
            mps[0], R = np.linalg.qr(mps[0],mode="reduced")
            
            # mps[1] = np.einsum("ab,bcd->acd",R,mps[1])
            prod =  R @ mps[1].reshape(mps[1].shape[0],-1) #resulting shape (a,cd)
            mps[1] = prod.reshape(R.shape[0],mps[1].shape[1],mps[1].shape[2])
        elif i < mps.N-1:
            # tmp = np.einsum("abc,bd->adc",mps[i], op)
            mps_transposed = mps[i].transpose(0,2,1) #Transpose to (a,c,b)
            product = mps_transposed.reshape(-1,mps_transposed.shape[-1]) @ op #Resulting shape (ac,d)
            product = product.reshape(mps[i].shape[0],mps[i].shape[2],op.shape[1]) #(a,c,d)
            tmp = product.transpose(0,2,1) #transpose to (a,d,c)
            
            #magnetizations[i] = np.einsum("adc,adc->",tmp,np.conj(mps[i]))
            magnetizations[i] = np.trace(np.dot(tmp.reshape(tmp.shape[0], -1), np.conj(mps[i]).reshape(mps[i].shape[0], -1).T)).real
            Q, R = np.linalg.qr(mps[i].reshape(-1,mps[i].shape[2]),mode="reduced")
            mps[i] = Q.reshape(mps[i].shape[0],mps[i].shape[1],-1)
            
            if i == mps.N-2:
                mps[i+1] =  R @ mps[i+1]
            else:
                product = R @ mps[i+1].reshape(mps[i+1].shape[0],-1) #resulting shape a,cd
                mps[i+1] =  product.reshape(R.shape[0],mps[i+1].shape[1],mps[i+1].shape[2])
                # mps[i+1] = np.einsum("ab,bcd->acd",R,mps[i+1])
        else:
            magnetizations[-1] = np.trace(mps[-1] @ op @ mps[-1].conj().T).real
    mps.canform = "Right"
    return magnetizations
