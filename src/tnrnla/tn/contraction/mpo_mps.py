import numpy as np
import numpy.linalg as la 
import random

from ..mps import MPS
from ..stopping import Cutoff
from ..compression.round_daas import daas_round_blas

"""
The Contract-then-Compress algorithm for the compressed MPO-MPS product: 
This file contains BLAS3 and einsum level implementations of the contract then compress algorithm 
which simply contracts then compresses the intermediate state via TT-SVD

NOTE: This method is the "analytic" baseline for MPO-MPS multiplication and is the slowest by far. 
Author: Chris Camaño Circa: 2023
"""

def mpo_mps(mpo, mps, round_type="standard", stop=Cutoff(1e-14),
                 r=None, num_iters=5, oversample=5, l=None, final_round=False):
    new_mps = []

    # site = np.einsum("dDl,lX->dDX",mpo[0],mps[0])
    mpo_reshaped = mpo.cache(
        ("mpo_mps_left", 0, tuple(mpo[0].shape), np.dtype(mpo[0].dtype).str),
        lambda: mpo[0].reshape(mpo[0].shape[0] * mpo[0].shape[1], mpo[0].shape[2]),
        namespace="mpo_mps",
    )                                                                                    # (dD,l)
    site = mpo_reshaped @ mps[0]                                                       # (dD,X)
    site = site.reshape(mpo[0].shape[0], mpo[0].shape[1]*mps[0].shape[1])              # (d,DX)
    new_mps.append(site)

    for i in range(1, mps.N-1):
        # site = np.einsum("DdEl,XlY->DXdEY",mpo[i],mps[i])
        mpo_reshaped = mpo.cache(
            ("mpo_mps_bulk", i, tuple(mpo[i].shape), np.dtype(mpo[i].dtype).str),
            lambda i=i: mpo[i].reshape(-1, mpo[i].shape[3]),
            namespace="mpo_mps",
        )                                                                               # (DdE,l)
        mps_transposed = mps[i].transpose(1, 0, 2)                                     # (l,X,Y)
        mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0], -1)             # (l,XY)
        site = mpo_reshaped @ mps_reshaped                                             # (DdE,XY)
        site = site.reshape(mpo[i].shape[0], mpo[i].shape[1],
                            mpo[i].shape[2], mps[i].shape[0],
                            mps[i].shape[2])                                           # (D,d,E,X,Y)
        site = site.transpose(0, 3, 1, 2, 4)                                           # (D,X,d,E,Y)
        site = site.reshape(site.shape[0]*site.shape[1],
                            site.shape[2],
                            site.shape[3]*site.shape[4])                               # (DE,d,XY)
        new_mps.append(site)

    # site = np.tensordot(mpo[-1], mps[-1], (2, 1))
    # site = np.einsum("Ddl,Xl->XDd",mpo[-1],mps[-1])
    mpo_reshaped = mpo.cache(
        ("mpo_mps_right", mps.N - 1, tuple(mpo[-1].shape), np.dtype(mpo[-1].dtype).str),
        lambda: mpo[-1].reshape(mpo[-1].shape[0] * mpo[-1].shape[1], -1),
        namespace="mpo_mps",
    )                                                                                    # (Dd,l)
    site = mpo_reshaped @ mps[-1].T                                                    # (Dd,X)
    site = site.reshape(mpo[-1].shape[0], mpo[-1].shape[1], mps[-1].shape[0])          # (D,d,X)
    site = site.transpose(0, 2, 1)                                                     # (D,X,d)
    site = site.reshape(site.shape[0]*site.shape[1], site.shape[2])                    # (DX,d)
    new_mps.append(site)

    new_mps = MPS(new_mps)
    if stop.is_truncation():
        if round_type == "standard":
            new_mps.round(stop=stop)
        elif round_type == "daas_blas":
            daas_round_blas(new_mps, stop=stop)
        # elif round_type == "dass":
        #     dass_round(new_mps, stop=stop)
        # elif round_type == "orth_then_rand_blas":
        #     orth_then_rand_blas(new_mps, r)
        # elif round_type == "rand_then_orth_blas":  # TODO: Strange runtime behavior for r=None
        #     rand_then_orth_blas(new_mps, r, finalround=final_round, stop=stop)
        # elif round_type == "nystrom_blas":
        #     nystrom_round_blas(new_mps, l, stop=stop)
        # elif round_type== "no_rounding":
        #     return new_mps
        # else:
    
            raise ValueError(f"Unknown rounding type: {round_type}")

    return new_mps


def mps_mpo_einsum(mps, mpo, round_type = "standard",stop=Cutoff(1e-14),r=None,num_iters=5,oversample=5,l=None,final_round=False):
    new_mps = []
 
    site = np.tensordot(mpo[0], mps[0], axes=(2, 0))
    new_mps.append(
        np.reshape(site, (site.shape[0], site.shape[1] * site.shape[2]))
    )

    for i in range(1, len(mps) - 1):
        site = np.tensordot(mpo[i], mps[i], axes=(3, 1))
        site = np.moveaxis(site, 3, 1)
        site = np.reshape(
            site,
            (
                site.shape[0] * site.shape[1],
                site.shape[2],
                site.shape[3] * site.shape[4],
            ),
        )

        new_mps.append(site)
    site = np.tensordot(mpo[-1], mps[-1], (2, 1))
    site = np.moveaxis(site, 2, 1)
    site = np.reshape(site, (site.shape[0] * site.shape[1], site.shape[2]))

    new_mps.append(site)

    new_mps = MPS(new_mps)
    if stop.is_truncation():
        if round_type == "standard":
            new_mps.round(stop=stop)
        elif round_type == "dass_blas":
            dass_round_blas(new_mps,stop=stop)
        elif round_type == "dass":
            dass_round(new_mps,stop=stop)
        elif round_type == "orth_then_rand_blas":
            orth_then_rand_blas(mps,r)
        elif round_type == "rand_then_orth_blas":
            rand_then_orth_blas(mps,r,finalround=final_round)
        elif round_type == "nystrom_blas":
            nystrom_round_blas(new_mps,l,stop=stop)
        else:
            raise ValueError(f"Unknown rounding type: {round_type}")
            
    return new_mps
