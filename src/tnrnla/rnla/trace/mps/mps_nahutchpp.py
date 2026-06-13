import numpy as np

from tnrnla.tn.trp import TRP
from ._helpers import apply_oracle_to_trp

def mps_nahutchpp(
    oracle,
    *,
    n,
    probe_chi,
    num_queries,
    rmps_d=2,
    base_seed=0,
    seed_offset=0,
    dtype=np.float64,
    trp_orth=False,
    trp_normalize=False,
    return_debug=False,
):
    m = int(num_queries)
    if m < 4:
        raise ValueError("num_queries must be >= 4 for mps_nahutchpp")

    n = int(n)
    probe_chi = int(probe_chi)
    rmps_d = int(rmps_d)

    s = int(round(m / 4))
    r = int(round(m / 2))
    g = int(round(m / 4))

    base = int(base_seed) + int(seed_offset)

    S = TRP.gaussian(
        n_sites=n,
        k=s,
        chi=probe_chi,
        d=rmps_d,
        seed=base + 1,
        dtype=dtype,
        orth=trp_orth,
        normalize=trp_normalize,
    )
    R = TRP.gaussian(
        n_sites=n,
        k=r,
        chi=probe_chi,
        d=rmps_d,
        seed=base + 2,
        dtype=dtype,
        orth=trp_orth,
        normalize=trp_normalize,
    )
    G = TRP.gaussian(
        n_sites=n,
        k=g,
        chi=probe_chi,
        d=rmps_d,
        seed=base + 3,
        dtype=dtype,
        orth=trp_orth,
        normalize=trp_normalize,
    )

    Z = apply_oracle_to_trp(oracle, R)
    W = apply_oracle_to_trp(oracle, S)

    M = np.asarray(S @ Z)
    Q, U = np.linalg.qr(M.T.conj(), mode="reduced")
    Uinv = np.linalg.inv(U)

    K = np.asarray(W @ Z)
    trest1 = float(np.real(np.trace(Uinv.conj().T @ K @ Q)))

    GG = apply_oracle_to_trp(oracle, G)
    termA = float(np.real(np.trace(np.asarray(G @ GG))))

    GZ = np.asarray(G @ Z)
    GX = GZ @ Q

    WG = np.asarray(W @ G)
    YG = Uinv.conj().T @ WG

    termB = float(np.real(np.trace(GX @ YG)))

    out = trest1 + 4.0 * (termA - termB) / float(m)

    if return_debug:
        dbg = {
            "m": int(m),
            "s": int(s),
            "r": int(r),
            "g": int(g),
            "trest1": float(trest1),
            "termA": float(termA),
            "termB": float(termB),
        }
        return out, dbg

    return out