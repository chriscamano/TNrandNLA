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
from ..other.tensor_ops import lq_left, qr_right, ttm3



# ======================================================================
"""
TTSVD-rounding
This file contains an implementation of TT-SVD rounding (Algorithm 2) from the original paper "Tensor Train Decomposition" 
by Ivan Oseledets. This method simply applies a sequence of SVDs to locally truncate each core of a MPS 

Author: Chris Camaño Circa: 2023 
"""
# ======================================================================


# ============ Exact MPS rounding of  ============
def round_left(mps, stop=Cutoff(1e-14)):
    """
    Left-to-right rounding that produces a right-canonical MPS.
    This updates tensors in-place and sets mps.rounded when complete.
    """
    if getattr(mps, "rounded", False):
        return

    canform = getattr(mps, "canform", "None")
    pivot = getattr(mps, "pivot_idx", None)
    if not (canform == "Left" and pivot == mps.N - 1):
        mps.orthR()

    normY = mps.norm()
    if stop.cutoff is None:
        stop_eff = stop
    else:
        tau = stop.cutoff * normY / math.sqrt(max(1, mps.N - 1))
        stop_eff = Cutoff(tau, maxdim=stop.maxdim)

    L, Q = lq(mps[-1])
    U, s, Vt = truncated_svd(L, stop=stop_eff)
    mps[-1] = Vt @ Q
    carry = (U @ np.diag(s)).T

    for i in range(mps.N - 2, 0, -1):
        mps[i] = ttm3(np.ascontiguousarray(mps[i]), carry, 2)
        L, Q = lq_left(mps[i])
        U, s, Vt = truncated_svd(L, stop=stop_eff)
        mps[i] = ttm3(np.ascontiguousarray(Q), Vt, 0)
        carry = (U @ np.diag(s)).T

    mps[0] = mps[0] @ carry.T
    mps.canform = "Right"
    mps.pivot_idx = 0
    mps.rounded = True


def round_right(mps, stop=Cutoff(1e-14)):
    """
    Right-to-left rounding that produces a left-canonical MPS.
    This updates tensors in-place and sets mps.rounded when complete.
    """
    if getattr(mps, "rounded", False):
        return

    canform = getattr(mps, "canform", "None")
    pivot = getattr(mps, "pivot_idx", None)
    if not (canform == "Right" and pivot == 0):
        mps.orthL()

    normY = mps.norm()
    if stop.cutoff is None:
        stop_eff = stop
    else:
        tau = stop.cutoff * normY / math.sqrt(max(1, mps.N - 1))
        stop_eff = Cutoff(tau, maxdim=stop.maxdim)

    Q, R = qr_right(mps[0])
    U, s, Vt = truncated_svd(R, stop=stop_eff)
    mps[0] = (Q @ U)
    carry = (np.diag(s) @ Vt)

    for i in range(1, mps.N - 1):
        mps[i] = ttm3(np.ascontiguousarray(mps[i]), carry, 0)
        Q, R = qr_right(mps[i])
        U, s, Vt = truncated_svd(R, stop=stop_eff)
        mps[i] = (Q @ U)
        carry = (np.diag(s) @ Vt)

    mps[-1] = carry @ mps[-1]
    mps.canform = "Left"
    mps.pivot_idx = mps.N - 1
    mps.rounded = True

