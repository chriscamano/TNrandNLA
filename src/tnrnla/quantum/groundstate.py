import numpy as np
from tnrnla.tn.mps import MPS
from tnrnla.tn.mpo import MPO
from tnrnla.tn.other.tensor_ops import contract_blas
from tnrnla.tn.stopping import Cutoff

class GroundState(MPS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _to_site_tensor(self, A):
        if A.ndim == 3:
            return A
        if A.ndim == 2:
            if A.shape[0] <= A.shape[1]:
                s, b = A.shape
                return A.reshape(1, s, b)
            else:
                a, s = A.shape
                return A.reshape(a, s, 1)
        raise ValueError("MPS site must be rank-2 or rank-3.")
    
    def reduced_density_mpo(self, cut_site: int, normalize: bool = False) -> MPO:
        tol = 1e-14

        psi = MPS(
            [np.asarray(T, copy=True) for T in self.tensors],
            canform=self.canform,
            rounded=self.rounded,
            dtype=self.dtype,
        )

        pdim = self[0].shape[0]
        bond = cut_site - 1

        psi.orthR()
        psi.move_pivot(bond)

        C = self._to_site_tensor(psi[bond])
        a, sdim, bdim = C.shape
        M = C.reshape(a * sdim, bdim)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        lam2 = S**2

        if normalize:
            tot = lam2.sum()
            if tot == 0.0:
                raise ValueError("Zero norm at the cut; cannot normalize λ^2.")
            lam2 /= tot

        chi = U.shape[1]
        lam2_diag = np.diag(lam2)

        mpo_cores = []
        for i in range(cut_site):
            if i == 0:
                W = contract_blas(psi[i], (), psi[i].conj(), ()).transpose(0, 2, 1, 3)
                site = W.reshape(pdim, pdim, -1)
                site = np.transpose(site, (0, 2, 1))
                mpo_cores.append(site.astype(self.dtype, copy=False))

            elif i == cut_site - 1:
                U_ten = U.reshape(a, sdim, chi)
                W = contract_blas(U_ten, (), U_ten.conj(), ()).transpose(0, 3, 1, 4, 2, 5)
                WS = contract_blas(W, (4, 5), lam2_diag, (0, 1))
                last_site = WS.reshape(-1, pdim, pdim)
                mpo_cores.append(last_site.astype(self.dtype, copy=False))

            else:
                W = contract_blas(psi[i], (), psi[i].conj(), ()).transpose(0, 3, 1, 4, 2, 5)
                site = W.reshape(
                    W.shape[0] * W.shape[1],
                    W.shape[2],
                    W.shape[3],
                    -1,
                )
                site = site.transpose(0, 1, 3, 2)
                mpo_cores.append(site.astype(self.dtype, copy=False))

        out_dtype = np.result_type(self.dtype, lam2.dtype)
        rho = MPO(mpo_cores, orthform="None", rounded=False, dtype=out_dtype)
        rho.round(stop=Cutoff(tol))
        return rho

    def density_matrix(self, normalize=True):
        psi = self.contract_all().reshape(-1)
        if normalize:
            norm_sq = np.vdot(psi, psi).real
            if norm_sq == 0:
                raise ValueError("Cannot normalize: the contracted state has zero norm.")
            psi = psi / np.sqrt(norm_sq)
        rho = np.outer(psi, psi.conj())
        return rho
