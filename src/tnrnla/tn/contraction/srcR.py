import numpy as np

from ..mps import MPS
from ..mpo import MPO
from ..stopping import Cutoff
from .src import SRC as _SRC_RTL


"""
SRC (left-to-right sweep) implemented as the reflected counterpart of src.SRC.

This preserves the same incremental-QR core and the same optimized contraction
sequences by running the existing right-to-left implementation on mirrored
MPO/MPS tensors, then mirroring the output back.
"""


def _reflect_mps_tensors(psi):
    n = int(psi.N)
    if n < 2:
        raise ValueError("SRC requires at least 2 sites")

    out = [None] * n
    out[0] = np.ascontiguousarray(np.asarray(psi[-1]).transpose(1, 0))
    for j in range(1, n - 1):
        out[j] = np.ascontiguousarray(np.asarray(psi[n - 1 - j]).transpose(2, 1, 0))
    out[-1] = np.ascontiguousarray(np.asarray(psi[0]).transpose(1, 0))
    return out


def _reflect_mpo_tensors(H):
    n = int(H.N)
    if n < 2:
        raise ValueError("SRC requires at least 2 sites")

    out = [None] * n
    out[0] = np.ascontiguousarray(np.asarray(H[-1]).transpose(1, 0, 2))
    for j in range(1, n - 1):
        out[j] = np.ascontiguousarray(np.asarray(H[n - 1 - j]).transpose(2, 1, 0, 3))
    out[-1] = np.ascontiguousarray(np.asarray(H[0]).transpose(1, 0, 2))
    return out


def _reflect_mps(psi):
    tensors = _reflect_mps_tensors(psi)
    return MPS(tensors, orthform="None", dtype=getattr(psi, "dtype", None))


def _reflect_mpo(H):
    tensors = _reflect_mpo_tensors(H)
    return MPO(
        tensors,
        orthform="None",
        rounded=getattr(H, "rounded", False),
        dtype=getattr(H, "dtype", None),
        mv_alg=getattr(H, "_mv_alg", "SRC"),
    )


def SRC_R(
    H,
    psi,
    stop=None,
    sketchdim: int = 32,
    sketchincrement: int = 32,
    finalround=None,
    accuracychecks: bool = False,
    cpp: bool = True,
    dtype=np.float64,
    rng=None,
    seed=None,
):
    """
    Left-to-right reflected SRC with identical API to tnrnla.tn.contraction.src.SRC.
    """
    if stop is None:
        stop = Cutoff(1e-8)

    H_ref = _reflect_mpo(H)
    psi_ref = _reflect_mps(psi)

    out_ref = _SRC_RTL(
        H_ref,
        psi_ref,
        stop=stop,
        sketchdim=sketchdim,
        sketchincrement=sketchincrement,
        finalround=finalround,
        accuracychecks=accuracychecks,
        cpp=cpp,
        dtype=dtype,
        rng=rng,
        seed=seed,
    )

    out = MPS(_reflect_mps_tensors(out_ref), orthform="Right", dtype=getattr(out_ref, "dtype", None))
    out.rounded = getattr(out_ref, "rounded", False)
    out.pivot_idx = out.N - 1
    return out

