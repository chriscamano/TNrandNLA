import numpy as np
from tnrnla.tn.trp import TRP
from tnrnla.tn.mpo import MPO


def mps_hutch(
    oracle,
    *,
    n,
    probe_chi,
    num_queries,
    rmps_d=2,
    base_seed=0,
    dtype=np.float64,
    return_samples=False,
):
    Omega = TRP.gaussian(
        n_sites=n,
        chi=probe_chi,
        k=num_queries,
        d=rmps_d,
        dtype=dtype,
        seed=base_seed,
        normalize=False,
        orth=False,
    )

    if isinstance(oracle, MPO):
        samples = Omega.quadform(oracle, dtype=dtype)
    else:
        samples = np.fromiter(
            (float(np.real_if_close(oracle.quadform(psi))) for psi in Omega.cols),
            dtype=np.float64,
            count=num_queries,
        )

    samples *= num_queries

    tr_hat = float(samples.mean())

    if num_queries == 1:
        stderr = 0.0
    else:
        scale = float(np.max(np.abs(samples)))
        if scale == 0.0:
            stderr = 0.0
        else:
            scaled = samples / scale
            stderr = float(scale * scaled.std(ddof=1) / np.sqrt(num_queries))

    if return_samples:
        return tr_hat, stderr, samples
    return tr_hat, stderr