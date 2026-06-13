import numpy as np

from tnrnla.tn.mps import MPS
from tnrnla.tn.trp import TRP


def apply_oracle_to_trp(oracle, trp_):
    try:
        out = oracle @ trp_
        if isinstance(out, TRP):
            return out
    except Exception:
        pass

    cols = getattr(trp_, "cols", None)
    if cols is None:
        raise TypeError("Expected a TRP with a .cols attribute")

    ycols = [oracle @ col for col in cols]
    return TRP._from_parent(trp_, ycols)


def sample_rmps_cols(*, n, probe_chi, count, rmps_d=2, dtype=np.complex128, rng=None):
    return [
        MPS.rmps(int(n), int(probe_chi), d=int(rmps_d), dtype=dtype, rng=rng)
        for _ in range(int(count))
    ]



def trp_row_inner_fast(trp_u, psi):
    cols = getattr(trp_u, "cols", None)
    if cols is None:
        raise TypeError("Expected a TRP with a .cols attribute")
    v = np.empty((len(cols),), dtype=complex)
    for j, uj in enumerate(cols):
        v[j] = uj.inner_product(psi)
    return v


def orth_trp(trp_):
    if hasattr(trp_, "orthLR"):
        trp_.orthLR()
    if hasattr(trp_, "orthRL"):
        trp_.orthRL()
    return trp_
