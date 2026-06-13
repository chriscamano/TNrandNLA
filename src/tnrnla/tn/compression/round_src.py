import copy
import math
from typing import Dict, Tuple

import numpy as np
import numpy.linalg as la
from tnrnla.linalg.orth import lq
from tnrnla.linalg.lra import truncated_svd
from ..mps import MPS
from ..contraction.src import SRC
from ..stopping import Cutoff, no_truncation


# ======================================================================
"""
Successive Randomized Compression (SRC) for MPS rounding: 
This file implements an application of SRC from the paper Successive randomized compression: A randomized algorithm for the compressed MPO-MPS product
https://arxiv.org/abs/2504.06475 for rounding a MPS and adaptivley determining the bond dimensions 

Author: Chris Camaño Circa: 2025
"""
# ======================================================================

# ============ SRC rounding from https://arxiv.org/abs/2504.06475 ===========
_IDENTITY_MPO_CACHE: Dict[Tuple[int, int, np.dtype], object] = {}


def _get_identity_mpo(n_sites, phys_dim, dtype):
    """
    Return a cached identity MPO keyed by (N, d, dtype) to avoid rebuilding it
    on every src_rounding call.
    """
    from ..mpo import MPO

    key = (int(n_sites), int(phys_dim), np.dtype(dtype))
    mpo = _IDENTITY_MPO_CACHE.get(key)
    if mpo is None:
        mpo = MPO.eye(int(n_sites), d=int(phys_dim), dtype=np.dtype(dtype))
        _IDENTITY_MPO_CACHE[key] = mpo
    return mpo


def src_rounding(mps, stop, **kwargs):
    """
    Compress an MPS by applying the identity MPO using SRC.
    This is a simple wrapper around SRC that uses an identity MPO of matching length.
    """
    mpo = _get_identity_mpo(mps.N, mps[0].shape[0], mps.dtype)  # assumes uniform phys dims
    return SRC(
        mpo,
        mps,
        stop=stop,
        finalround=True,
        sketchdim=32,
        sketchincrement=16,
        **kwargs,
    )
