from __future__ import annotations

import numpy as np

from tnrnla.tn.contraction.src import (
    EnvStore,
    SRCViewCache,
    IncrementalQR,
    flatten_left,
    batch_rightmul,
)
from tnrnla import Cutoff, MPS


def _make_term_view(H, psi):
    tv = SRCViewCache.term_views(H, psi, assume_identity=False)

    if tv.identity_simple:
        raise ValueError("This implementation does not support identity_simple=True")

    if tv.H0_flat is not None and not tv.H0_flat.flags["C_CONTIGUOUS"]:
        tv.H0_flat = np.ascontiguousarray(tv.H0_flat)

    if tv.H_env_T_erl is not None:
        for k in range(len(tv.H_env_T_erl)):
            a = tv.H_env_T_erl[k]
            if not a.flags["C_CONTIGUOUS"]:
                tv.H_env_T_erl[k] = np.ascontiguousarray(a)

    if tv.H_sk is not None:
        for k in range(len(tv.H_sk)):
            a = tv.H_sk[k]
            if not a.flags["C_CONTIGUOUS"]:
                tv.H_sk[k] = np.ascontiguousarray(a)

    if tv.H_cap is not None:
        for k in range(1, psi.N - 1):
            a = tv.H_cap[k]
            if a is not None and not a.flags["C_CONTIGUOUS"]:
                tv.H_cap[k] = np.ascontiguousarray(a)

    return tv


def _reserve_env_capacity(tv, envs: EnvStore, j: int, need: int):
    if need <= 0:
        return

    H0 = tv.H_view[0]
    psi0 = tv.psi0
    dt0 = np.result_type(H0.dtype, psi0.dtype, np.float64)
    tail0 = (int(H0.shape[1]), int(psi0.shape[1]))
    envs._ensure(0, need, tail0, dt0, filled=len(envs))

    psi_env = tv.psi_env
    env_shapes = tv.env_shapes
    for k in range(1, j):
        _, E, _, _ = env_shapes[k - 1]
        Rp_next = int(psi_env[k - 1].shape[1])
        dtk = np.result_type(psi_env[k - 1].dtype, np.float64)
        tailk = (int(E), Rp_next)
        envs._ensure(k, need, tailk, dtk, filled=envs.pstart)


def _draw_shared_block_list(j: int, Z: int, V: int, rng):
    return [rng.standard_normal((Z, V)) for _ in range(j)]


def _append_shared_env_block(tv, envs: EnvStore, j: int, g_list):
    if j <= 0:
        raise ValueError("j must be at least 1")

    V = int(tv.V)
    H0 = tv.H_view[0]
    H0_s1 = int(H0.shape[1])
    H0_s2 = int(H0.shape[2])

    H0_flat = tv.H0_flat
    if H0_flat is None:
        raise ValueError("missing H0_flat")

    H_env = tv.H_env_T_erl
    if H_env is None:
        raise ValueError("missing H_env_T_erl")

    psi0 = tv.psi0
    psi_env = tv.psi_env
    env_shapes = tv.env_shapes

    g0 = np.asarray(g_list[0])
    if g0.ndim != 2 or g0.shape[1] != V:
        raise ValueError("bad shared block shape")
    if not g0.flags["C_CONTIGUOUS"]:
        g0 = np.ascontiguousarray(g0)

    Z = int(g0.shape[0])

    r0 = g0 @ H0_flat
    prev = (r0.reshape(Z, H0_s1, H0_s2) @ psi0)
    if not prev.flags["C_CONTIGUOUS"]:
        prev = np.ascontiguousarray(prev)
    envs.append_batch(0, prev)

    for k in range(1, j):
        L, E, R, ER = env_shapes[k - 1]

        gk = np.asarray(g_list[k])
        if not gk.flags["C_CONTIGUOUS"]:
            gk = np.ascontiguousarray(gk)

        vec = gk @ H_env[k - 1]
        t2T = vec.reshape(Z, ER, L)
        t3 = (t2T @ prev).reshape(Z, E, R, prev.shape[2])

        prev = batch_rightmul(
            t3.reshape(Z, E, R * prev.shape[2]),
            psi_env[k - 1],
        )
        if not prev.flags["C_CONTIGUOUS"]:
            prev = np.ascontiguousarray(prev)

        envs.append_batch(k, prev)


def _accumulate_local_sketch_from_env(
    out_block,
    coeff,
    tv,
    psi,
    env_block,
    j: int,
    cap_flat,
):
    if coeff == 0:
        return

    Z = int(env_block.shape[0])
    Hj = tv.H_view[j]
    psij = psi[j]

    if j == psi.N - 1:
        temp = env_block @ psij
        Hmat = Hj.reshape(Hj.shape[1], -1)
        out_block += coeff * (Hmat @ temp.reshape(Z, -1).T)
        return

    H_sk = tv.H_sk
    if H_sk is None:
        raise ValueError("missing H_sk")

    psi_sk = tv.psi_sk

    temp = env_block @ psi_sk[j - 1]
    temp = temp.reshape(Z, -1, psij.shape[1], psij.shape[2])
    temp_DdY = temp.reshape(Z, temp.shape[1] * temp.shape[2], temp.shape[3])

    C = H_sk[j - 1][None, :, :] @ temp_DdY
    t2 = C.reshape(Z, Hj.shape[1], Hj.shape[2], temp_DdY.shape[2])
    t2d = t2.reshape(Z * Hj.shape[1], Hj.shape[2] * temp_DdY.shape[2])

    if cap_flat is None:
        raise ValueError("cap_flat required on interior bonds")

    blk2d = t2d @ cap_flat
    out_block += coeff * blk2d.reshape(Z, Hj.shape[1], cap_flat.shape[1]).transpose(1, 2, 0)


def _build_cap_boundary_fast(tv, psi, j: int, Q: np.ndarray):
    Hj = tv.H_view[j]
    psij = psi[j]

    Hj_mat = Hj.transpose(1, 0, 2).reshape(Hj.shape[1], -1)
    temp = (np.conj(Q).T @ Hj_mat).reshape(Q.shape[1], Hj.shape[0], Hj.shape[2])
    return (temp.reshape(-1, temp.shape[-1]) @ psij.reshape(psij.shape[0], -1).T).reshape(
        temp.shape[0], temp.shape[1], psij.shape[0]
    )


def _build_cap_interior_fast(tv, psi, j: int, Q3: np.ndarray, Qr_flat: np.ndarray, cap, Hj_r, psi_r):
    if cap is None or Hj_r is None or psi_r is None:
        raise ValueError("cap/Hj_r/psi_r required")

    Hj = tv.H_view[j]
    psij = psi[j]

    cap_r = cap.reshape(cap.shape[0], Hj.shape[2] * psij.shape[2])
    tmp = (Qr_flat @ cap_r).reshape(Q3.shape[0], Q3.shape[2], Hj.shape[2], psij.shape[2])

    left1 = np.ascontiguousarray(
        tmp.transpose(1, 3, 0, 2)
    ).reshape(Q3.shape[2] * psij.shape[2], Q3.shape[0] * Hj.shape[2])

    mid = (left1 @ Hj_r).reshape(Q3.shape[2], psij.shape[2], Hj.shape[0], Hj.shape[3])

    left2 = np.ascontiguousarray(
        mid.transpose(0, 2, 1, 3)
    ).reshape(Q3.shape[2] * Hj.shape[0], psij.shape[2] * Hj.shape[3])

    return (left2 @ psi_r).reshape(Q3.shape[2], Hj.shape[0], psij.shape[0])


def SRCsum(
    H_list,
    psi_list,
    *,
    alpha=None,
    stop=None,
    sketchdim: int = 16,
    sketchincrement: int = 16,
    finalround: bool = True,
    dtype=np.float64,
    rng=None,
    seed=None,
    block_env: int | None = 64,
):
    """
    Randomized approximation to

        eta ~= sum_i alpha_i H_i psi_i

    Optimized version with
    - shared random blocks across all terms
    - blocked environment growth for better cache behavior
    - no redundant post-round orthogonalization passes
    """
    if seed is not None:
        rng = np.random.default_rng(int(seed))
    elif rng is None:
        rng = np.random.default_rng()

    if stop is None:
        stop = Cutoff(1e-8)

    t = int(len(H_list))
    if t == 0:
        raise ValueError("H_list must be non-empty")
    if len(psi_list) != t:
        raise ValueError("H_list and psi_list must have the same length")

    n = int(H_list[0].N)
    for H, psi in zip(H_list, psi_list):
        if int(H.N) != n or int(psi.N) != n:
            raise ValueError("All MPOs and MPSs must have the same number of sites")

    if n == 1:
        raise NotImplementedError("n=1 not implemented")

    if alpha is None:
        alpha = np.ones(t, dtype=np.dtype(dtype))
    alpha = np.asarray(alpha).reshape(-1)
    if alpha.size != t:
        raise ValueError("alpha must have length len(H_list)")

    keep = np.abs(alpha) > 0
    if not np.all(keep):
        H_list = [H for H, k in zip(H_list, keep) if k]
        psi_list = [psi for psi, k in zip(psi_list, keep) if k]
        alpha = alpha[keep]
        t = int(len(H_list))
        if t == 0:
            raise ValueError("All terms were pruned because alpha was zero")

    maxdim = stop.maxdim
    outputdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff

    if outputdim is None:
        if maxdim is None:
            raise ValueError("stop.maxdim must be set, or stop.outputdim must be set")
        mindim = max(mindim, 1)
    else:
        maxdim = outputdim
        mindim = outputdim
        sketchdim = outputdim

    tv_list = [_make_term_view(H, psi) for H, psi in zip(H_list, psi_list)]

    init_cap = max(64, int(sketchdim))
    envs_list = [EnvStore(n, init_cap=init_cap) for _ in range(t)]

    V = int(tv_list[0].V)
    for tv in tv_list[1:]:
        if int(tv.V) != V:
            raise ValueError("All term views must have the same V")

    base_dtype = np.dtype(dtype)
    sketch_dtype = np.result_type(
        base_dtype,
        alpha.dtype,
        *[np.asarray(tv.H_view[0]).dtype for tv in tv_list],
        *[np.asarray(psi[0]).dtype for psi in psi_list],
    )
    alpha = alpha.astype(sketch_dtype, copy=False)

    psi_out = [None] * n
    caps = [None] * t
    cap_dim = 1

    for j in reversed(range(1, n)):
        is_boundary = (j == n - 1)

        prod_bonds_sum = 0
        for tv, psi in zip(tv_list, psi_list):
            Hj = tv.H_view[j]
            psij = psi[j]
            if is_boundary:
                prod_bonds_sum += int(Hj.shape[0]) * int(psij.shape[0])
            else:
                prod_bonds_sum += max(
                    int(Hj.shape[0]) * int(psij.shape[0]),
                    int(Hj.shape[2]) * int(psij.shape[2]),
                )

        maxdim_j = min(prod_bonds_sum, maxdim, V * cap_dim)
        mindim_j = min(mindim, maxdim_j)
        sketchdim_j = max(min(sketchdim, maxdim_j), mindim_j)

        sketch_sq = 0.0
        done_sketches = 0
        qr = None
        reserved = -1

        cap_flat_list = [None] * t
        Hj_r_list = [None] * t
        psi_r_list = [None] * t

        if not is_boundary:
            for i in range(t):
                if caps[i] is None:
                    raise ValueError("Internal error. cap missing on non-boundary step")
                cap_flat_list[i] = flatten_left(caps[i].transpose(1, 2, 0))
                Hj_r_list[i] = tv_list[i].H_cap[j]
                psi_r_list[i] = tv_list[i].psi_cap[j]

        while True:
            if sketchdim_j != reserved:
                for i in range(t):
                    _reserve_env_capacity(tv_list[i], envs_list[i], j, sketchdim_j)
                reserved = sketchdim_j

            Z = int(sketchdim_j - done_sketches)
            if Z <= 0:
                raise RuntimeError("Non-positive sketch block width encountered")

            if block_env is None:
                g_list = _draw_shared_block_list(j, Z, V, rng)
                for i in range(t):
                    _append_shared_env_block(tv_list[i], envs_list[i], j, g_list)
            else:
                remaining = Z
                while remaining > 0:
                    Zblk = min(int(block_env), remaining)
                    g_list = _draw_shared_block_list(j, Zblk, V, rng)
                    for i in range(t):
                        _append_shared_env_block(tv_list[i], envs_list[i], j, g_list)
                    remaining -= Zblk

            a = done_sketches
            b = sketchdim_j

            if is_boundary:
                global_blk = np.zeros((V, b - a), dtype=sketch_dtype, order="C")
            else:
                global_blk = np.zeros((V, cap_dim, b - a), dtype=sketch_dtype, order="C")

            for i in range(t):
                env_block = envs_list[i].get(j - 1, a, b)
                _accumulate_local_sketch_from_env(
                    global_blk,
                    alpha[i],
                    tv_list[i],
                    psi_list[i],
                    env_block,
                    j,
                    cap_flat_list[i],
                )

            sketch_sq += float(np.vdot(global_blk, global_blk).real)

            if is_boundary:
                if qr is None:
                    qr = IncrementalQR(global_blk.copy(order="F"))
                else:
                    qr.append(global_blk)
            else:
                flat_blk = flatten_left(global_blk)
                if qr is None:
                    qr = IncrementalQR(flat_blk.copy(order="F"))
                else:
                    qr.append(flat_blk)

            done_sketches = sketchdim_j

            if outputdim is not None or sketchdim_j == maxdim_j:
                done = True
            else:
                err_est = qr.error_estimate()
                norm_est = (sketch_sq ** 0.5) / (sketchdim_j ** 0.5)
                done = (err_est <= cutoff * norm_est) and (sketchdim_j >= mindim_j)

            if done:
                Q = qr.get_q()

                if is_boundary:
                    psi_out[j] = Q.transpose(1, 0)
                    for i in range(t):
                        caps[i] = _build_cap_boundary_fast(tv_list[i], psi_list[i], j, Q)
                else:
                    old_cap_dim = cap_dim
                    Q3 = Q.reshape((V, old_cap_dim, sketchdim_j))
                    psi_out[j] = Q3.transpose(2, 0, 1)

                    Qr_flat = np.conj(Q3).transpose(0, 2, 1).reshape(
                        V * sketchdim_j, old_cap_dim
                    )

                    for i in range(t):
                        caps[i] = _build_cap_interior_fast(
                            tv_list[i],
                            psi_list[i],
                            j,
                            Q3,
                            Qr_flat,
                            caps[i],
                            Hj_r_list[i],
                            psi_r_list[i],
                        )

                cap_dim = sketchdim_j
                break

            sketchdim_j = min(maxdim_j, sketchdim_j + sketchincrement)

        cap_flat_list = None
        Hj_r_list = None
        psi_r_list = None

    psi0_sum = None
    for i in range(t):
        temp = np.einsum("ijk,lk", caps[i], psi_list[i][0])
        term0 = np.einsum("ijk,ljk->il", tv_list[i].H_view[0], temp)
        psi0_sum = alpha[i] * term0 if psi0_sum is None else psi0_sum + alpha[i] * term0

    psi_out[0] = psi0_sum

    out = MPS(psi_out, orthform="Left")

    if finalround:
        out.round(stop=stop)

    return out