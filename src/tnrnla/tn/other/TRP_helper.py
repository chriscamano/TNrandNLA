import numpy as np

from ..mps import MPS
from ..mpo import MPO

from ..stopping import Cutoff
from ..contraction.src_cacheview import SRCViewCache
from tnrnla.linalg.incrementalqr import *
from ..contraction.src import *


def hermitize_inplace(A):
    iu = np.triu_indices(A.shape[0], k=1)
    A[(iu[1], iu[0])] = np.conjugate(A[iu])
    A[np.diag_indices(A.shape[0])] = np.real(A[np.diag_indices(A.shape[0])]) 
    return A


def lincomb_exact_block(mps_list, coeffs):
    coeffs = np.asarray(coeffs)
    idx = [i for i, c in enumerate(coeffs) if c != 0]
    if len(idx) == 0:
        return mps_list[0] * 0

    ms = [mps_list[i] for i in idx]
    cs = coeffs[idx]

    out_dtype = np.result_type(
        *[np.dtype(getattr(m, "dtype", np.asarray(m[0]).dtype)) for m in ms],
        cs.dtype,
    )

    N = int(ms[0].N)
    left0 = np.asarray(ms[0][0])
    d = int(left0.shape[0])

    Dr_list = [int(np.asarray(m[0]).shape[1]) for m in ms]
    Dr_off = np.cumsum([0] + Dr_list)

    left = np.zeros((d, Dr_off[-1]), dtype=out_dtype)
    for j, (m, c) in enumerate(zip(ms, cs)):
        a, b = Dr_off[j], Dr_off[j + 1]
        left[:, a:b] = (np.asarray(m[0]) * c).astype(out_dtype, copy=False)

    blocks = [left]

    for site in range(1, N - 1):
        Aref = np.asarray(ms[0][site])
        dd = int(Aref.shape[1])

        Dl_list = [int(np.asarray(m[site]).shape[0]) for m in ms]
        Dr_list = [int(np.asarray(m[site]).shape[2]) for m in ms]
        Dl_off = np.cumsum([0] + Dl_list)
        Dr_off = np.cumsum([0] + Dr_list)

        T = np.zeros((Dl_off[-1], dd, Dr_off[-1]), dtype=out_dtype)
        for j, m in enumerate(ms):
            la, lb = Dl_off[j], Dl_off[j + 1]
            ra, rb = Dr_off[j], Dr_off[j + 1]
            T[la:lb, :, ra:rb] = np.asarray(m[site]).astype(out_dtype, copy=False)

        blocks.append(T)

    Dl_list = [int(np.asarray(m[-1]).shape[0]) for m in ms]
    Dl_off = np.cumsum([0] + Dl_list)

    right = np.zeros((Dl_off[-1], d), dtype=out_dtype)
    for j, m in enumerate(ms):
        a, b = Dl_off[j], Dl_off[j + 1]
        right[a:b, :] = np.asarray(m[-1]).astype(out_dtype, copy=False)

    blocks.append(right)

    return MPS(blocks, dtype=out_dtype)


def SRC_with_tv(
    tv,
    H,
    psi,
    stop=None,
    sketchdim=16,
    sketchincrement=16,
    finalround=None,
    accuracychecks=False,
    dtype=np.float64,
):
    from ..contraction.src_cacheview import SRCViewCache

    if stop is None:
        stop = Cutoff(1e-8)

    n = H.N
    if n != psi.N:
        raise ValueError("lengths of MPO and MPS do not match")
    if n == 1:
        raise NotImplementedError("n=1 not implemented")

    maxdim = stop.maxdim
    outputdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff

    if outputdim is None:
        if maxdim is None:
            maxdim = np.inf
        mindim = max(mindim, 1)
    else:
        maxdim = mindim = sketchdim = outputdim

    # Always recompute fresh psi views to avoid stale data when psi has been
    # modified after tv was originally constructed.  SRCViewCache.cache_psi
    # stores its result in psi.__dict__ and is not invalidated by the standard
    # tensorCache path, so we evict the stale entry before recomputing.
    V = int(tv.V)
    psi_dict = getattr(psi, "__dict__", None)
    if psi_dict is not None:
        psi_dict.pop(("src_psi", V), None)
    psi_cache = SRCViewCache.cache_psi(psi, V=V)
    from dataclasses import replace as _dc_replace
    tv = _dc_replace(
        tv,
        psi0=psi_cache["psi0"],
        psi_env=psi_cache["psi_env"],
        psi_sk=psi_cache["psi_sk"],
        psi_cap=psi_cache["psi_cap"],
    )

    base_dtype = np.dtype(dtype)
    sketch_dtype = np.result_type(np.asarray(tv.H_view[0]).dtype, np.asarray(psi[0]).dtype, base_dtype)

    envs = EnvStore(n)
    cap = None
    cap_dim = 1
    V = int(tv.V)
    psi_out = [None] * n

    for j in reversed(range(1, n)):
        is_boundary = (j == n - 1)
        Hj = tv.H_view[j]
        psij = psi[j]

        if is_boundary:
            prod_bonds = Hj.shape[0] * psij.shape[0]
        else:
            prod_bonds = max(Hj.shape[0] * psij.shape[0], Hj.shape[2] * psij.shape[2])

        maxdim_j = min(prod_bonds, maxdim, V * cap_dim)
        mindim_j = min(mindim, maxdim_j)
        sketchdim_j = max(min(sketchdim, maxdim_j), mindim_j)

        if is_boundary:
            sketch = np.empty((V, maxdim_j), dtype=sketch_dtype, order="C")
        else:
            sketch = np.empty((V, cap_dim, maxdim_j), dtype=sketch_dtype, order="C")

        sketch_sq = 0.0
        done_sketches = 0
        qr = None

        cap_flat = Hj_r = psi_r = None
        if not is_boundary:
            cap_flat = flatten_left(cap.transpose(1, 2, 0))
            Hj_r = tv.H_cap[j]
            psi_r = tv.psi_cap[j]

        while True:
            make_envs(tv, envs, j, sketchdim_j, block_env=64)
            make_sketch(sketch, tv, psi, envs, done_sketches, sketchdim_j, j, cap_flat)

            new_blk = sketch[..., done_sketches:sketchdim_j]
            sketch_sq += float(np.vdot(new_blk, new_blk).real)

            if is_boundary:
                qr = (
                    IncrementalQR(sketch[:, :sketchdim_j].copy(order="F"))
                    if qr is None
                    else (qr.append(new_blk) or qr)
                )
            else:
                flat0 = flatten_left(sketch[:, :, :sketchdim_j]).copy(order="F")
                qr = IncrementalQR(flat0) if qr is None else (qr.append(flatten_left(new_blk)) or qr)

            done_sketches = sketchdim_j

            if outputdim is not None or sketchdim_j == maxdim_j:
                done = True
            else:
                err_est = qr.error_estimate()
                norm_est = (sketch_sq**0.5) / (sketchdim_j**0.5)
                done = (err_est <= cutoff * norm_est) and (sketchdim_j >= mindim_j)

            if done:
                Q = qr.get_q()
                if is_boundary:
                    psi_out[j] = Q.transpose(1, 0)
                    cap = build_cap(tv, psi, j, Q, cap, is_boundary=True)
                else:
                    cap_dim_prev = cap_dim
                    Q3 = Q.reshape((V, cap_dim_prev, sketchdim_j))
                    psi_out[j] = Q3.transpose(2, 0, 1)
                    cap = build_cap(tv, psi, j, Q3, cap, is_boundary=False, Hj_r=Hj_r, psi_r=psi_r)

                cap_dim = sketchdim_j

                if accuracychecks and check_randomized_apply is not None:
                    check_randomized_apply(
                        H,
                        psi,
                        cap,
                        psi_out,
                        j + 1,
                        verbose=False,
                        cutoff=((100 * cutoff) if (maxdim is None) else np.inf),
                    )
                break

            sketchdim_j = min(maxdim_j, sketchdim_j + sketchincrement)

    temp = np.einsum("ijk,lk", cap, psi[0])
    psi_out[0] = np.einsum("ijk,ljk->il", tv.H_view[0], temp)

    out = MPS(psi_out, canform="Left")
    out.pivot_idx = 0
    if finalround:
        out.round(stop=stop)
    return out
