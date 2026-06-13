# tnrnla/tn/compression/krpsum.py

import math
import numpy as np
import numpy.linalg as la

from ..stopping import Cutoff
from ..other.tensor_ops import ttm3


def _normalize_maxdim(maxdim):
    """Normalize rank-cap input to either None (uncapped) or positive int."""
    if maxdim is None:
        return None
    arr = np.asarray(maxdim)
    if arr.ndim != 0:
        raise ValueError("maxdim must be a scalar or None")
    val = arr.item()
    if val is None:
        return None
    if not np.isfinite(val):
        return None
    return max(1, int(val))


def _as_rank3_cores(mps):
    N = int(mps.N)
    cores = []

    A = np.asarray(mps[0])
    cores.append(np.ascontiguousarray(A.reshape(1, A.shape[0], A.shape[1])))

    for k in range(1, N - 1):
        cores.append(np.ascontiguousarray(np.asarray(mps[k])))

    A = np.asarray(mps[-1])
    cores.append(np.ascontiguousarray(A.reshape(A.shape[0], A.shape[1], 1)))

    return cores


def _from_rank3_cores(cores):
    N = len(cores)
    out = [np.ascontiguousarray(cores[0].reshape(cores[0].shape[1], cores[0].shape[2]))]

    for k in range(1, N - 1):
        out.append(np.ascontiguousarray(cores[k]))

    out.append(np.ascontiguousarray(cores[-1].reshape(cores[-1].shape[0], cores[-1].shape[1])))
    return out


def _vertical(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def _horizontal(A):
    return A.reshape(A.shape[0], A.shape[1] * A.shape[2])


def _core_from_vertical(V, left_rank, phys_dim, right_rank):
    return np.ascontiguousarray(V.reshape(left_rank, phys_dim, right_rank))


def _core_from_horizontal(H, left_rank, phys_dim, right_rank):
    return np.ascontiguousarray(H.reshape(left_rank, phys_dim, right_rank))


def _real_dtype(dtype):
    dt = np.dtype(dtype)
    if dt == np.dtype(np.float32) or dt == np.dtype(np.complex64):
        return np.float32
    return np.float64


def _gaussian(rng, shape, dtype, complex_sketch):
    if complex_sketch and np.issubdtype(np.dtype(dtype), np.complexfloating):
        rdtype = _real_dtype(dtype)
        x = rng.standard_normal(shape).astype(rdtype, copy=False)
        y = rng.standard_normal(shape).astype(rdtype, copy=False)
        return ((x + 1j * y) / math.sqrt(2.0)).astype(dtype, copy=False)

    return rng.standard_normal(shape).astype(_real_dtype(dtype), copy=False)


def _last_contraction(A, Omega):
    # Boundary specialization: A has shape (Dl, d, 1), so this contraction is
    # just a dense matrix multiply (Dl x d) @ (d x width).
    A2 = np.ascontiguousarray(A[:, :, 0])
    return np.ascontiguousarray(A2 @ Omega)


def _krp_contraction_step(A, Omega, Wnext):
    T = ttm3(A, Wnext.T, 2)
    return np.ascontiguousarray(np.sum(T * Omega[None, :, :], axis=1))


def _right_partial_contractions(cores, start, width, rng, dtype, complex_sketch):
    N = len(cores)
    W = [None] * N
    Omegas = [None] * N

    for k in range(start, N):
        Omegas[k] = _gaussian(rng, (cores[k].shape[1], width), dtype, complex_sketch)

    W[-1] = _last_contraction(cores[-1], Omegas[-1])

    for k in range(N - 2, start - 1, -1):
        W[k] = _krp_contraction_step(cores[k], Omegas[k], W[k + 1])

    return W


def _append_partial_contractions(W, cores, start, width, rng, dtype, complex_sketch):
    Wnew = _right_partial_contractions(cores, start, width, rng, dtype, complex_sketch)

    for k in range(start, len(cores)):
        if W[k] is None:
            W[k] = Wnew[k]
        else:
            W[k] = np.ascontiguousarray(np.concatenate((W[k], Wnew[k]), axis=1))

    return W


def _ensure_width(W, cores, start, width, rng, dtype, complex_sketch):
    current = 0 if W[start] is None else W[start].shape[1]

    if current >= width:
        return W

    return _append_partial_contractions(
        W,
        cores,
        start,
        width - current,
        rng,
        dtype,
        complex_sketch,
    )


def _thin_qr(A):
    Q, _ = la.qr(A, mode="reduced")
    return np.ascontiguousarray(Q)


def _reorth(Q, Z):
    Z = Z - Q @ (Q.conj().T @ Z)
    Z = _thin_qr(Z)
    Z = Z - Q @ (Q.conj().T @ Z)
    return _thin_qr(Z)


def _residual_sketch(W, cores, k, Z, Q, b, rng, dtype, complex_sketch):
    start = k + 1
    need = Q.shape[1] + b

    W = _ensure_width(W, cores, start, need, rng, dtype, complex_sketch)

    S = Z @ W[start][:, Q.shape[1]:need]
    S = S - Q @ (Q.conj().T @ S)

    return np.ascontiguousarray(S), W


def _estimate_norm(cores, W):
    S = _vertical(cores[0]) @ W[1]
    return float(la.norm(S, "fro") / math.sqrt(W[1].shape[1]))


def _cache_cell():
    return {"data": None, "ncols": 0}


def _cache_append(cell, block):
    if block is None:
        return
    block = np.ascontiguousarray(block)
    nb = int(block.shape[1])
    if nb == 0:
        return
    if cell["data"] is None:
        cap = max(8, nb)
        arr = np.empty((int(block.shape[0]), cap), dtype=block.dtype)
        arr[:, :nb] = block
        cell["data"] = arr
        cell["ncols"] = nb
        return
    nold = int(cell["ncols"])
    need = nold + nb
    data = cell["data"]
    if need > data.shape[1]:
        new_cap = max(need, int(math.ceil(1.6 * data.shape[1])) + 8)
        arr = np.empty((data.shape[0], new_cap), dtype=data.dtype)
        arr[:, :nold] = data[:, :nold]
        cell["data"] = arr
        data = arr
    data[:, nold:need] = block
    cell["ncols"] = need


def _cache_ncols(cell):
    return int(cell["ncols"])


def _cache_view(cell, start=0, stop=None):
    if stop is None:
        stop = int(cell["ncols"])
    return cell["data"][:, int(start):int(stop)]


def _append_shared_omega_block_cached(omega_cache, phys_dims, start, width, rng, dtype, complex_sketch):
    """
    Append `width` new shared sketch columns Ω_k(:,new) for k >= start to cache.
    Returns only the newly-generated Ω blocks for this append.
    """
    N = len(phys_dims)
    new_block = [None] * N
    if width <= 0:
        return new_block
    for k in range(start, N):
        Om_new = _gaussian(rng, (int(phys_dims[k]), int(width)), dtype, complex_sketch)
        _cache_append(omega_cache[k], Om_new)
        new_block[k] = Om_new
    return new_block


def _append_right_partials_many_cached(Ws_cache, cores_list, omega_block, start):
    """
    For each summand j, compute only the *new* right-partial contractions induced
    by `omega_block` and append them to cached Ws_cache[j][k], k >= start.
    """
    s = len(cores_list)
    N = len(cores_list[0])
    for j in range(s):
        cores = cores_list[j]
        Wnew = [None] * N
        Wnew[N - 1] = _last_contraction(cores[N - 1], omega_block[N - 1])
        for k in range(N - 2, start - 1, -1):
            Wnew[k] = _krp_contraction_step(cores[k], omega_block[k], Wnew[k + 1])
        for k in range(start, N):
            _cache_append(Ws_cache[j][k], Wnew[k])


def _accumulate_weighted_sketch(out, coeffs, Z_list, Ws_cache_col, start, stop):
    out.fill(0)
    for j in range(len(coeffs)):
        out += coeffs[j] * (Z_list[j] @ _cache_view(Ws_cache_col[j], start, stop))
    return out


def _resolve_fixed_ranks(rank_spec, N, maxdim=None):
    """
    Resolve user-provided fixed-rank specification to internal bond ranks.

    Returns a length-(N-1) integer array with one target rank per internal bond.
    Accepted inputs:
      - scalar r: use r on all bonds
      - length N-1 array: direct bond ranks
      - length N array: MATLAB-style indexing, use entries [1:N]
      - length N+1 array: TT-rank vector with boundary ranks, use [1:N]
    """
    maxdim_eff = _normalize_maxdim(maxdim)

    if np.isscalar(rank_spec):
        r = max(1, int(rank_spec))
        ranks = np.full(N - 1, r, dtype=np.int64)
    else:
        arr = np.asarray(rank_spec).ravel()
        if arr.size == (N - 1):
            ranks = arr.astype(np.int64, copy=False)
        elif arr.size == N:
            ranks = arr[1:].astype(np.int64, copy=False)
        elif arr.size == (N + 1):
            ranks = arr[1:N].astype(np.int64, copy=False)
        else:
            raise ValueError(
                "rank_spec must be scalar, length N-1, length N, or length N+1."
            )
        ranks = np.maximum(ranks, 1)

    if maxdim_eff is not None:
        ranks = np.minimum(ranks, int(maxdim_eff))
        ranks = np.maximum(ranks, 1)

    return ranks


def _append_shared_omega_block(omegas, phys_dims, start, width, rng, dtype, complex_sketch):
    """
    Append `width` new shared sketch columns Ω_k(:,new) for k >= start.
    Returns a block list containing only the newly generated columns.
    """
    N = len(phys_dims)
    new_block = [None] * N
    if width <= 0:
        return new_block
    for k in range(start, N):
        Om_new = _gaussian(rng, (int(phys_dims[k]), int(width)), dtype, complex_sketch)
        if omegas[k] is None:
            omegas[k] = np.ascontiguousarray(Om_new)
        else:
            omegas[k] = np.ascontiguousarray(np.concatenate((omegas[k], Om_new), axis=1))
        new_block[k] = Om_new
    return new_block


def _append_right_partials_many(Ws_list, cores_list, omega_block, start):
    """
    For each summand j, compute only the *new* right-partial contractions induced
    by `omega_block` and append them to cached Ws_list[j][k], k >= start.
    """
    s = len(cores_list)
    N = len(cores_list[0])
    for j in range(s):
        cores = cores_list[j]
        Wnew = [None] * N
        Wnew[N - 1] = _last_contraction(cores[N - 1], omega_block[N - 1])
        for k in range(N - 2, start - 1, -1):
            Wnew[k] = _krp_contraction_step(cores[k], omega_block[k], Wnew[k + 1])
        for k in range(start, N):
            if Ws_list[j][k] is None:
                Ws_list[j][k] = Wnew[k]
            else:
                Ws_list[j][k] = np.ascontiguousarray(
                    np.concatenate((Ws_list[j][k], Wnew[k]), axis=1)
                )


def krp_adaptive_sum_many(
    mpss,
    coeffs,
    stop=Cutoff(1e-14),
    *,
    init_f=0.10,
    incr_f=0.05,
    min_samples=20,
    tol_scale=1.0,
    rng=None,
    seed=None,
    nrmx=None,
    maxdim=None,
    final_round=True,
    final_round_factor=1.0,
    complex_sketch=False,
    max_extra_blocks=None,
    verbose=False,
):
    """
    Fused adaptive KRP rounding of a linear combination without forming TT-sum:
        X ~= sum_j coeffs[j] * mpss[j].
    """
    if len(mpss) == 0:
        raise ValueError("mpss must be non-empty")
    coeffs = np.asarray(coeffs).reshape(-1)
    if coeffs.size != len(mpss):
        raise ValueError("coeffs length must match number of summands")

    if rng is None:
        rng = np.random.default_rng(seed)

    # Symmetry-safe fallback: dense KRP sketch is not charge-block aware.
    if hasattr(mpss[0], "bond_charges") and hasattr(mpss[0], "phys_charges"):
        out = mpss[0] * float(coeffs[0])
        for j in range(1, len(mpss)):
            out = out.add(mpss[j] * float(coeffs[j]), subtract=False, compress=False)
        out.round(stop=stop)
        return out

    N = int(mpss[0].N)
    if N < 2:
        out = mpss[0] * float(coeffs[0])
        for j in range(1, len(mpss)):
            out = out.add(mpss[j] * float(coeffs[j]), subtract=False, compress=False)
        if final_round:
            out.round(stop=stop)
        return out

    for x in mpss[1:]:
        if int(x.N) != N:
            raise ValueError("All summands must have the same number of sites")

    cores_list = [_as_rank3_cores(x) for x in mpss]
    phys_dims = [int(cores_list[0][k].shape[1]) for k in range(N)]
    for cores in cores_list[1:]:
        for k in range(N):
            if int(cores[k].shape[1]) != phys_dims[k]:
                raise ValueError("All summands must share identical physical dimensions")

    eps = float(stop.cutoff) if getattr(stop, "cutoff", None) is not None else 0.0
    if maxdim is None:
        maxdim = getattr(stop, "maxdim", None)
    maxdim = _normalize_maxdim(maxdim)

    out_dtype = np.result_type(
        coeffs.dtype,
        *[np.result_type(*[A.dtype for A in cores]) for cores in cores_list],
    )
    coeffs = coeffs.astype(out_dtype, copy=False)

    s = len(mpss)

    # Cached shared Omegas and per-summand right partial contractions.
    omega_cache = [_cache_cell() for _ in range(N)]
    Ws_cache = [[_cache_cell() for _ in range(N)] for _ in range(s)]

    max_input_rank = max(int(cores[k].shape[2]) for cores in cores_list for k in range(N - 1))
    init_cols = max(int(min_samples), max(1, int(math.ceil(float(init_f) * max_input_rank))))
    Om_block = _append_shared_omega_block_cached(
        omega_cache, phys_dims, start=1, width=init_cols, rng=rng, dtype=out_dtype, complex_sketch=complex_sketch
    )
    _append_right_partials_many_cached(Ws_cache, cores_list, Om_block, start=1)

    # Cache horizontal unfoldings once; these cores are immutable.
    hcores_list = [[None] * N for _ in range(s)]
    for j in range(s):
        for k in range(1, N):
            hcores_list[j][k] = _horizontal(cores_list[j][k])

    active = [np.ascontiguousarray(cores[0].copy()) for cores in cores_list]
    Y = [None] * N

    if nrmx is None:
        S0 = None
        for j in range(s):
            Zj = _vertical(active[j])
            Sj = Zj @ _cache_view(Ws_cache[j][1], 0, init_cols)
            Sj = coeffs[j] * Sj
            S0 = Sj if S0 is None else (S0 + Sj)
        nrmx = float(la.norm(S0, "fro") / math.sqrt(init_cols))
    else:
        nrmx = float(abs(nrmx))

    if nrmx == 0.0:
        out = mpss[0] * 0.0
        return out

    tau = eps * nrmx / math.sqrt(max(1, N - 1))
    rank_trace = []

    for k in range(N - 1):
        # Current core vertical unfolding from projected active cores.
        Z_list = [_vertical(a) for a in active]
        Ws_col = [Ws_cache[j][k + 1] for j in range(s)]
        z_rows = int(Z_list[0].shape[0])
        for Zj in Z_list[1:]:
            # Summands can have different right-link sizes (column counts) after
            # independent MPO/KRP updates. Only row-space alignment is required
            # for the fused sketch accumulation.
            if int(Zj.shape[0]) != z_rows:
                raise RuntimeError("Projected summand row spaces lost alignment.")

        rbar = z_rows
        if maxdim is not None:
            rbar = min(rbar, maxdim)
        rbar = int(max(1, rbar))

        b_init = max(int(min_samples), max(1, int(math.ceil(float(init_f) * rbar))))
        b_init = min(b_init, rbar)
        b_inc = max(int(min_samples), max(1, int(math.ceil(float(incr_f) * rbar))))
        b_inc = min(b_inc, rbar)

        cur_cols = _cache_ncols(Ws_cache[0][k + 1])
        if cur_cols < b_init:
            Om_block = _append_shared_omega_block_cached(
                omega_cache,
                phys_dims,
                start=k + 1,
                width=b_init - cur_cols,
                rng=rng,
                dtype=out_dtype,
                complex_sketch=complex_sketch,
            )
            _append_right_partials_many_cached(Ws_cache, cores_list, Om_block, start=k + 1)

        S = np.zeros((z_rows, b_init), dtype=out_dtype)
        _accumulate_weighted_sketch(
            S,
            coeffs,
            Z_list,
            Ws_col,
            0,
            b_init,
        )
        Q = _thin_qr(S)
        if Q.shape[1] > rbar:
            Q = np.ascontiguousarray(Q[:, :rbar])

        q_rows = int(Q.shape[0])
        q_cols = int(Q.shape[1])
        next_phys = int(cores_list[0][k + 1].shape[1])
        next_rights = [int(cores_list[j][k + 1].shape[2]) for j in range(s)]

        # Preallocate to avoid repeated concatenate in adaptive enrichment.
        Q_storage = np.empty((q_rows, rbar), dtype=Q.dtype)
        Q_storage[:, :q_cols] = Q
        Q_view = Q_storage[:, :q_cols]
        Qh_view = Q_view.conj().T

        active_next_blocks = []
        for j in range(s):
            Mj = Qh_view @ Z_list[j]
            Hj = hcores_list[j][k + 1]
            h_cols_j = int(Hj.shape[1])
            Hstore = np.empty((rbar, h_cols_j), dtype=np.result_type(Mj.dtype, Hj.dtype))
            Hstore[:q_cols, :] = Mj @ Hj
            active_next_blocks.append((Hstore, Hj))

        extra_blocks = 0
        res_est = 0.0
        Sres = np.empty((z_rows, max(1, b_inc)), dtype=out_dtype)
        if q_cols < rbar:
            b = min(b_inc, rbar - q_cols)
            need = q_cols + b
            cur_cols = _cache_ncols(Ws_cache[0][k + 1])
            if cur_cols < need:
                Om_block = _append_shared_omega_block_cached(
                    omega_cache,
                    phys_dims,
                    start=k + 1,
                    width=need - cur_cols,
                    rng=rng,
                    dtype=out_dtype,
                    complex_sketch=complex_sketch,
                )
                _append_right_partials_many_cached(Ws_cache, cores_list, Om_block, start=k + 1)
            Sres_use = Sres[:, :b]
            _accumulate_weighted_sketch(
                Sres_use,
                coeffs,
                Z_list,
                Ws_col,
                q_cols,
                need,
            )
            Sres_use -= Q_view @ (Qh_view @ Sres_use)
            res_est = float(la.norm(Sres_use, "fro") / math.sqrt(b))

        while res_est > (tau / max(float(tol_scale), 1e-16)) and q_cols < rbar:
            if max_extra_blocks is not None and extra_blocks >= int(max_extra_blocks):
                break
            Qnew = _thin_qr(Sres_use)
            Qnew = _reorth(Q_view, Qnew)

            remaining = rbar - q_cols
            if Qnew.shape[1] > remaining:
                Qnew = np.ascontiguousarray(Qnew[:, :remaining])
            if Qnew.shape[1] == 0:
                break

            nnew = int(Qnew.shape[1])
            Q_storage[:, q_cols:q_cols + nnew] = Qnew
            for j in range(s):
                Hstore, Hj = active_next_blocks[j]
                Mnew = Qnew.conj().T @ Z_list[j]
                Hstore[q_cols:q_cols + nnew, :] = Mnew @ Hj
                active_next_blocks[j] = (Hstore, Hj)

            q_cols += nnew
            Q_view = Q_storage[:, :q_cols]
            Qh_view = Q_view.conj().T
            extra_blocks += 1

            if q_cols >= rbar:
                break

            b = min(b_inc, rbar - q_cols)
            need = q_cols + b
            cur_cols = _cache_ncols(Ws_cache[0][k + 1])
            if cur_cols < need:
                Om_block = _append_shared_omega_block_cached(
                    omega_cache,
                    phys_dims,
                    start=k + 1,
                    width=need - cur_cols,
                    rng=rng,
                    dtype=out_dtype,
                    complex_sketch=complex_sketch,
                )
                _append_right_partials_many_cached(Ws_cache, cores_list, Om_block, start=k + 1)
            Sres_use = Sres[:, :b]
            _accumulate_weighted_sketch(
                Sres_use,
                coeffs,
                Z_list,
                Ws_col,
                q_cols,
                need,
            )
            Sres_use -= Q_view @ (Qh_view @ Sres_use)
            res_est = float(la.norm(Sres_use, "fro") / math.sqrt(b))

        new_rank = q_cols
        Q_final = np.ascontiguousarray(Q_storage[:, :new_rank])
        Y[k] = _core_from_vertical(Q_final, int(active[0].shape[0]), int(active[0].shape[1]), new_rank)

        active_new = []
        for j in range(s):
            Hstore, _ = active_next_blocks[j]
            Hnext = np.ascontiguousarray(Hstore[:new_rank, :])
            active_new.append(_core_from_horizontal(Hnext, new_rank, next_phys, next_rights[j]))
        active = active_new
        rank_trace.append(new_rank)

        if verbose:
            print(
                f"KRP fused sum site {k}: rank={new_rank}, rbar={rbar}, "
                f"residual_est={res_est:.3e}, tau={tau:.3e}"
            )

    # Last core is the weighted sum of projected active last cores.
    Y_last = np.zeros_like(active[0], dtype=out_dtype)
    for j in range(s):
        Y_last += coeffs[j] * active[j]
    Y[-1] = np.ascontiguousarray(Y_last)

    out = mpss[0].copy()
    out.tensors = [np.ascontiguousarray(A.astype(out_dtype, copy=False)) for A in _from_rank3_cores(Y)]
    out.dtype = np.dtype(out_dtype)
    out.orthform = "Right"
    out.pivot_idx = N - 1
    out.rounded = False
    out.krp_adaptive_rank_trace = tuple(rank_trace)
    out.krp_adaptive_norm_estimate = nrmx
    out.krp_adaptive_tau = tau
    if hasattr(out, "_invalidate_cache"):
        out._invalidate_cache()

    if final_round:
        final_stop = Cutoff(float(final_round_factor) * eps, maxdim=(np.inf if maxdim is None else maxdim))
        out.roundRL(stop=final_stop)
    else:
        out.rounded = True
    return out


def krp_fixed_sum_many(
    mpss,
    coeffs,
    ranks,
    *,
    rng=None,
    seed=None,
    maxdim=None,
    oversample=0,
    complex_sketch=False,
    final_round=False,
    stop=None,
    verbose=False,
):
    """
    Fast non-adaptive fused KRP sum+round with fixed per-bond target ranks.

    Computes:
        X ~= sum_j coeffs[j] * mpss[j]
    without forming the explicit TT/MPS sum, using a single right-partial KRP
    sketch sized to max(target_rank) and a left-to-right fixed-rank projection.
    """
    if len(mpss) == 0:
        raise ValueError("mpss must be non-empty")
    coeffs = np.asarray(coeffs).reshape(-1)
    if coeffs.size != len(mpss):
        raise ValueError("coeffs length must match number of summands")

    if rng is None:
        rng = np.random.default_rng(seed)

    # Symmetry-safe fallback: dense KRP sketch is not charge-block aware.
    if hasattr(mpss[0], "bond_charges") and hasattr(mpss[0], "phys_charges"):
        out = mpss[0] * float(coeffs[0])
        for j in range(1, len(mpss)):
            out = out.add(mpss[j] * float(coeffs[j]), subtract=False, compress=False)
        if final_round:
            if stop is None:
                md = _normalize_maxdim(maxdim)
                out.round(stop=Cutoff(1e-14, maxdim=(np.inf if md is None else md)))
            else:
                out.round(stop=stop)
        return out

    N = int(mpss[0].N)
    if N < 2:
        out = mpss[0] * float(coeffs[0])
        for j in range(1, len(mpss)):
            out = out.add(mpss[j] * float(coeffs[j]), subtract=False, compress=False)
        if final_round:
            if stop is None:
                md = _normalize_maxdim(maxdim)
                out.round(stop=Cutoff(1e-14, maxdim=(np.inf if md is None else md)))
            else:
                out.round(stop=stop)
        return out

    for x in mpss[1:]:
        if int(x.N) != N:
            raise ValueError("All summands must have the same number of sites")

    target_ranks = _resolve_fixed_ranks(ranks, N, maxdim=maxdim)
    p = max(0, int(oversample))
    sketch_width = int(np.max(target_ranks)) + p
    if sketch_width <= 0:
        raise ValueError("Resolved sketch width must be positive.")

    cores_list = [_as_rank3_cores(x) for x in mpss]
    phys_dims = [int(cores_list[0][k].shape[1]) for k in range(N)]
    for cores in cores_list[1:]:
        for k in range(N):
            if int(cores[k].shape[1]) != phys_dims[k]:
                raise ValueError("All summands must share identical physical dimensions")

    out_dtype = np.result_type(
        coeffs.dtype,
        *[np.result_type(*[A.dtype for A in cores]) for cores in cores_list],
    )
    coeffs = coeffs.astype(out_dtype, copy=False)

    s = len(mpss)
    omega_cache = [_cache_cell() for _ in range(N)]
    Ws_cache = [[_cache_cell() for _ in range(N)] for _ in range(s)]

    Om_block = _append_shared_omega_block_cached(
        omega_cache,
        phys_dims,
        start=1,
        width=sketch_width,
        rng=rng,
        dtype=out_dtype,
        complex_sketch=complex_sketch,
    )
    _append_right_partials_many_cached(Ws_cache, cores_list, Om_block, start=1)

    hcores_list = [[None] * N for _ in range(s)]
    for j in range(s):
        for k in range(1, N):
            hcores_list[j][k] = _horizontal(cores_list[j][k])

    active = [np.ascontiguousarray(cores[0].copy()) for cores in cores_list]
    Y = [None] * N
    rank_trace = []

    for k in range(N - 1):
        Z_list = [_vertical(a) for a in active]
        Ws_col = [Ws_cache[j][k + 1] for j in range(s)]
        z_rows = int(Z_list[0].shape[0])
        for Zj in Z_list[1:]:
            # Right-link sizes may differ across summands; only row-space
            # alignment is necessary for forming the fused sketch.
            if int(Zj.shape[0]) != z_rows:
                raise RuntimeError("Projected summand row spaces lost alignment.")

        rk = int(target_ranks[k])
        rk = min(rk, z_rows)
        rk = max(rk, 1)
        rsk = min(z_rows, rk + p)
        rsk = max(rsk, rk)

        S = np.zeros((z_rows, rsk), dtype=out_dtype)
        _accumulate_weighted_sketch(
            S,
            coeffs,
            Z_list,
            Ws_col,
            0,
            rsk,
        )
        Q = _thin_qr(S)
        q_cols = min(int(Q.shape[1]), rk)
        if q_cols < int(Q.shape[1]):
            Q = np.ascontiguousarray(Q[:, :q_cols])
        else:
            Q = np.ascontiguousarray(Q)
        if q_cols <= 0:
            raise RuntimeError("Fixed-rank KRP produced empty basis block.")

        left_rank = int(active[0].shape[0])
        phys_dim = int(active[0].shape[1])
        Y[k] = _core_from_vertical(Q, left_rank, phys_dim, q_cols)

        next_phys = int(cores_list[0][k + 1].shape[1])
        next_rights = [int(cores_list[j][k + 1].shape[2]) for j in range(s)]
        Qh = Q.conj().T
        active_new = []
        for j in range(s):
            Mj = Qh @ Z_list[j]
            Hj = hcores_list[j][k + 1]
            Hnext = np.ascontiguousarray(Mj @ Hj)
            active_new.append(_core_from_horizontal(Hnext, q_cols, next_phys, next_rights[j]))
        active = active_new
        rank_trace.append(q_cols)

        if verbose:
            print(f"KRP fixed sum site {k}: rank={q_cols}, target={rk}, sketch={rsk}")

    Y_last = np.zeros_like(active[0], dtype=out_dtype)
    for j in range(s):
        Y_last += coeffs[j] * active[j]
    Y[-1] = np.ascontiguousarray(Y_last)

    out = mpss[0].copy()
    out.tensors = [np.ascontiguousarray(A.astype(out_dtype, copy=False)) for A in _from_rank3_cores(Y)]
    out.dtype = np.dtype(out_dtype)
    out.orthform = "Right"
    out.pivot_idx = N - 1
    out.rounded = False
    out.krp_fixed_rank_trace = tuple(rank_trace)
    if hasattr(out, "_invalidate_cache"):
        out._invalidate_cache()

    if final_round:
        if stop is not None:
            out.round(stop=stop)
        else:
            md = _normalize_maxdim(maxdim)
            out.round(stop=Cutoff(1e-14, maxdim=(np.inf if md is None else md)))
    else:
        out.rounded = True

    return out


def krp_fixed_round(
    mps,
    ranks,
    *,
    rng=None,
    seed=None,
    maxdim=None,
    oversample=0,
    complex_sketch=False,
    final_round=False,
    stop=None,
    verbose=False,
):
    """
    Fixed-rank KRP TT/MPS rounding for a single tensor, analogous to TTroundingKRP.
    """
    return krp_fixed_sum_many(
        [mps],
        np.array([1.0], dtype=np.float64),
        ranks,
        rng=rng,
        seed=seed,
        maxdim=maxdim,
        oversample=oversample,
        complex_sketch=complex_sketch,
        final_round=final_round,
        stop=stop,
        verbose=verbose,
    )


def krp_adaptive_sum(
    mps,
    stop=Cutoff(1e-14),
    *,
    finit=0.10,
    finc=0.05,
    rng=None,
    seed=None,
    nrmx=None,
    norm_samples=None,
    maxdim=None,
    final_round=True,
    final_round_factor=1.0,
    complex_sketch=False,
    max_extra_blocks=None,
    verbose=False,
):
    if rng is None:
        rng = np.random.default_rng(seed)

    eps = float(stop.cutoff) if getattr(stop, "cutoff", None) is not None else 0.0
    cores = _as_rank3_cores(mps)
    N = len(cores)
    dtype = np.result_type(*[A.dtype for A in cores])

    input_ranks = [cores[k].shape[2] for k in range(N - 1)]
    max_input_rank = max(input_ranks)

    if maxdim is None:
        maxdim = getattr(stop, "maxdim", None)
    maxdim = _normalize_maxdim(maxdim)

    if norm_samples is None:
        initial_width = max(1, int(math.ceil(float(finit) * max_input_rank)))
    else:
        initial_width = max(1, int(norm_samples))

    W = [None] * N
    W = _append_partial_contractions(
        W,
        cores,
        1,
        initial_width,
        rng,
        dtype,
        complex_sketch,
    )

    if nrmx is None:
        nrmx = _estimate_norm(cores, W)
    else:
        nrmx = float(abs(nrmx))

    if nrmx == 0.0:
        mps.tensors = [np.zeros_like(t) for t in mps.tensors]
        mps.orthform = "None"
        mps.pivot_idx = None
        mps.rounded = True
        if hasattr(mps, "_invalidate_cache"):
            mps._invalidate_cache()
        return mps

    tau = eps * nrmx / math.sqrt(max(1, N - 1))

    Y = [None] * N
    Y[0] = np.array(cores[0], copy=True, order="C")
    rank_trace = []

    for k in range(N - 1):
        Zcore = Y[k]
        left_rank, phys_dim, old_right_rank = Zcore.shape
        Z = _vertical(Zcore)

        rbar = min(Z.shape[0], Z.shape[1])
        if maxdim is not None:
            rbar = min(rbar, maxdim)

        b_init = max(1, int(math.ceil(float(finit) * rbar)))
        b_inc = max(1, int(math.ceil(float(finc) * rbar)))

        b_init = min(b_init, rbar)
        b_inc = min(b_inc, rbar)

        W = _ensure_width(W, cores, k + 1, b_init, rng, dtype, complex_sketch)

        S = Z @ W[k + 1][:, :b_init]
        Q0 = _thin_qr(S)

        if Q0.shape[1] > rbar:
            Q0 = np.ascontiguousarray(Q0[:, :rbar])

        Hcore_h = _horizontal(cores[k + 1])
        h_cols = int(Hcore_h.shape[1])
        q_rows = int(Q0.shape[0])

        # Avoid repeated reallocation from np.concatenate in the adaptive loop.
        Q_storage = np.empty((q_rows, rbar), dtype=Q0.dtype)
        H_storage = np.empty((rbar, h_cols), dtype=np.result_type(Q0.dtype, Hcore_h.dtype))
        q_cols = int(Q0.shape[1])
        Q_storage[:, :q_cols] = Q0

        Q_view = Q_storage[:, :q_cols]
        M = Q_view.conj().T @ Z
        H_storage[:q_cols, :] = M @ Hcore_h

        extra_blocks = 0
        res_est = 0.0

        if q_cols < rbar:
            b = min(b_inc, rbar - q_cols)
            Sres, W = _residual_sketch(W, cores, k, Z, Q_view, b, rng, dtype, complex_sketch)
            res_est = float(la.norm(Sres, "fro") / math.sqrt(b))

        while res_est > tau and q_cols < rbar:
            if max_extra_blocks is not None and extra_blocks >= int(max_extra_blocks):
                break

            Qnew = _thin_qr(Sres)
            Qnew = _reorth(Q_view, Qnew)

            remaining = rbar - q_cols
            if Qnew.shape[1] > remaining:
                Qnew = np.ascontiguousarray(Qnew[:, :remaining])

            if Qnew.shape[1] == 0:
                break

            nnew = int(Qnew.shape[1])
            Q_storage[:, q_cols:q_cols + nnew] = Qnew
            Mnew = Qnew.conj().T @ Z
            H_storage[q_cols:q_cols + nnew, :] = Mnew @ Hcore_h
            q_cols += nnew
            Q_view = Q_storage[:, :q_cols]

            extra_blocks += 1

            if q_cols >= rbar:
                res_est = 0.0
                break

            b = min(b_inc, rbar - q_cols)
            Sres, W = _residual_sketch(W, cores, k, Z, Q_view, b, rng, dtype, complex_sketch)
            res_est = float(la.norm(Sres, "fro") / math.sqrt(b))

        new_rank = q_cols
        Q = np.ascontiguousarray(Q_storage[:, :new_rank])
        H_next = np.ascontiguousarray(H_storage[:new_rank, :])

        Y[k] = _core_from_vertical(Q, left_rank, phys_dim, new_rank)

        next_phys = cores[k + 1].shape[1]
        next_right = cores[k + 1].shape[2]
        Y[k + 1] = _core_from_horizontal(H_next, new_rank, next_phys, next_right)

        rank_trace.append(new_rank)

        if verbose:
            print(
                f"KRP adaptive sum site {k}: "
                f"rank={new_rank}, rbar={rbar}, residual_est={res_est:.3e}, tau={tau:.3e}"
            )

    mps.tensors = [np.ascontiguousarray(A.astype(dtype, copy=False)) for A in _from_rank3_cores(Y)]
    mps.dtype = np.dtype(dtype)
    mps.orthform = "Right"
    mps.pivot_idx = N - 1
    mps.rounded = False

    mps.krp_adaptive_rank_trace = tuple(rank_trace)
    mps.krp_adaptive_norm_estimate = nrmx
    mps.krp_adaptive_tau = tau

    if hasattr(mps, "_invalidate_cache"):
        mps._invalidate_cache()

    if final_round:
        final_stop = Cutoff(float(final_round_factor) * eps, maxdim=(np.inf if maxdim is None else maxdim))
        mps.roundRL(stop=final_stop)
    else:
        mps.rounded = True

    return mps


def krp_adaptive_round(*args, **kwargs):
    """
    Backward-compatible alias for `krp_adaptive_sum`.
    """
    return krp_adaptive_sum(*args, **kwargs)
