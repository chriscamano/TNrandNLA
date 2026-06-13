from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from tnrnla.linalg.incrementalqr.incrementalqr import IncrementalQR
from ..mps import MPS
from ..stopping import Cutoff


def _term_dtype(psi):
    try:
        return np.dtype(getattr(psi, "dtype"))
    except Exception:
        return np.asarray(psi[0]).dtype


def _normalize_bound(x):
    if x is None:
        return None
    x_arr = np.asarray(x)
    if x_arr.ndim != 0:
        raise ValueError("bond-dimension bound must be scalar or None")
    x_val = x_arr.item()
    if x_val is None:
        return None
    if not np.isfinite(x_val):
        return None
    return int(x_val)


def _draw_gaussian(shape, rng, dtype):
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.complexfloating):
        return (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ).astype(dt, copy=False)
    return rng.standard_normal(shape).astype(dt, copy=False)


@dataclass
class _SiteCache:
    A: np.ndarray
    Dl: int
    d: int
    Dr: int

    @property
    def A2(self):
        return self.A.reshape(self.Dl * self.d, self.Dr)


@dataclass
class _BoundaryCache:
    M: np.ndarray
    Dl: int
    d: int


def _make_site_cache(Aj, out_dtype):
    A = np.asarray(Aj, dtype=out_dtype)
    Dl, d, Dr = A.shape
    return _SiteCache(A=A, Dl=int(Dl), d=int(d), Dr=int(Dr))


def _make_boundary_cache(Aj, coeff, out_dtype):
    M = np.asarray(Aj, dtype=out_dtype)
    if coeff != 1:
        M = coeff * M
    Dl, d = M.shape
    return _BoundaryCache(M=np.ascontiguousarray(M), Dl=int(Dl), d=int(d))


def _effective_rowblock_boundary(Aj, coeff, out_dtype):
    bc = _make_boundary_cache(Aj, coeff, out_dtype)
    return bc.M


def _effective_rowblock_interior(Aj, cap_next, out_dtype):
    sc = _make_site_cache(Aj, out_dtype)
    C = np.asarray(cap_next, dtype=out_dtype)

    if C.shape[0] != sc.Dr:
        raise ValueError("cap shape mismatch on interior site")

    AC = sc.A2 @ C
    return np.ascontiguousarray(AC.reshape(sc.Dl, sc.d * C.shape[1]))


def _build_local_sketch_block(
    mps_list,
    coeffs,
    caps,
    j,
    Z,
    rng,
    out_dtype,
):
    if j == mps_list[0].N - 1:
        d = int(np.asarray(mps_list[0][j]).shape[1])
        Y = np.zeros((d, Z), dtype=out_dtype, order="F")

        for psi, c in zip(mps_list, coeffs):
            M = _effective_rowblock_boundary(psi[j], c, out_dtype)
            G = _draw_gaussian((M.shape[0], Z), rng, out_dtype)
            Y += np.conj(M).T @ G

        return np.asfortranarray(Y)

    if caps is None:
        raise ValueError("caps must be present on interior sites")

    chi_next = int(np.asarray(caps[0]).shape[1])
    d = int(np.asarray(mps_list[0][j]).shape[1])
    Y = np.zeros((d * chi_next, Z), dtype=out_dtype, order="F")

    for psi, cap_next in zip(mps_list, caps):
        M = _effective_rowblock_interior(psi[j], cap_next, out_dtype)
        G = _draw_gaussian((M.shape[0], Z), rng, out_dtype)
        Y += np.conj(M).T @ G

    return np.asfortranarray(Y)


def _project_cap_boundary(Aj, coeff, Q, out_dtype):
    M = _effective_rowblock_boundary(Aj, coeff, out_dtype)
    if Q.shape[0] != M.shape[1]:
        raise ValueError("Q has incompatible leading dimension")
    return np.ascontiguousarray(M @ Q)


def _project_cap_interior(Aj, cap_next, Q, out_dtype):
    sc = _make_site_cache(Aj, out_dtype)
    C = np.asarray(cap_next, dtype=out_dtype)
    Q = np.asarray(Q, dtype=out_dtype)

    if C.shape[0] != sc.Dr:
        raise ValueError("cap shape mismatch on interior site")
    if Q.shape[0] != sc.d * C.shape[1]:
        raise ValueError("Q has incompatible leading dimension")

    AC = sc.A2 @ C
    M = AC.reshape(sc.Dl, sc.d * C.shape[1])
    return np.ascontiguousarray(M @ Q)


def _build_caps_boundary(mps_list, coeffs, Q, out_dtype):
    new_caps = []
    for psi, c in zip(mps_list, coeffs):
        new_caps.append(_project_cap_boundary(psi[-1], c, Q, out_dtype))
    return new_caps


def _build_caps_interior(mps_list, caps_next, j, Q, out_dtype):
    new_caps = []
    for psi, cap_next in zip(mps_list, caps_next):
        new_caps.append(_project_cap_interior(psi[j], cap_next, Q, out_dtype))
    return new_caps


def _site_rank_upper_bound(mps_list, j, caps_next):
    if j == mps_list[0].N - 1:
        row_sum = sum(int(np.asarray(psi[j]).shape[0]) for psi in mps_list)
        d = int(np.asarray(mps_list[0][j]).shape[1])
        return min(row_sum, d)

    if caps_next is None:
        raise ValueError("caps_next must be present on interior sites")

    row_sum = sum(int(np.asarray(psi[j]).shape[0]) for psi in mps_list)
    d = int(np.asarray(mps_list[0][j]).shape[1])
    chi_next = int(np.asarray(caps_next[0]).shape[1])
    return min(row_sum, d * chi_next)


def lincomb_SRC(
    mps_list,
    coeffs,
    *,
    stop=None,
    sketchdim: int = 16,
    sketchincrement: int = 16,
    finalround: bool = False,
    dtype=None,
    rng=None,
    seed=None,
):
    if seed is not None:
        rng = np.random.default_rng(int(seed))
    elif rng is None:
        rng = np.random.default_rng()

    if stop is None:
        stop = Cutoff(1e-8)

    coeffs = np.asarray(coeffs)
    if coeffs.ndim != 1:
        coeffs = coeffs.reshape(-1)

    if len(mps_list) == 0:
        raise ValueError("mps_list must be non-empty")
    if len(mps_list) != coeffs.size:
        raise ValueError("mps_list and coeffs must have the same length")

    keep = np.abs(coeffs) > 0
    if not np.all(keep):
        mps_list = [psi for psi, k in zip(mps_list, keep) if k]
        coeffs = coeffs[keep]

    if len(mps_list) == 0:
        raise ValueError("all coefficients were zero")

    N = int(mps_list[0].N)
    for psi in mps_list[1:]:
        if int(psi.N) != N:
            raise ValueError("all MPS must have the same number of sites")

    if N == 1:
        out0 = None
        for psi, c in zip(mps_list, coeffs):
            v = c * np.asarray(psi[0])
            out0 = v if out0 is None else out0 + v
        return MPS([out0])

    if dtype is None:
        out_dtype = np.result_type(
            coeffs.dtype,
            *[_term_dtype(psi) for psi in mps_list],
            np.float64,
        )
    else:
        out_dtype = np.dtype(dtype)

    coeffs = coeffs.astype(out_dtype, copy=False)

    outputdim_eff = _normalize_bound(stop.outputdim)
    maxdim_eff = _normalize_bound(stop.maxdim)
    cutoff = stop.cutoff

    mindim_raw = stop.mindim
    if mindim_raw is None:
        mindim_eff = 1
    else:
        mindim_eff = max(int(mindim_raw), 1)

    if outputdim_eff is not None:
        maxdim_eff = outputdim_eff
        mindim_eff = outputdim_eff
        sketchdim = outputdim_eff

    psi_out = [None] * N
    caps = None
    chi_next = 1

    for j in reversed(range(1, N)):
        rank_cap = int(_site_rank_upper_bound(mps_list, j, caps))

        if maxdim_eff is None:
            maxdim_j = rank_cap
        else:
            maxdim_j = min(maxdim_eff, rank_cap)

        mindim_j = min(mindim_eff, maxdim_j)
        sketchdim_j = max(min(int(sketchdim), maxdim_j), mindim_j)

        if maxdim_j <= 0:
            raise ValueError("encountered a non-positive local rank bound")

        # Pre-compute row-block matrices once per site (caps from the right
        # do not change during the sketch while-loop for this site).
        is_boundary = (j == N - 1)
        if not is_boundary:
            Ms = [
                _effective_rowblock_interior(psi[j], cap_next, out_dtype)
                for psi, cap_next in zip(mps_list, caps)
            ]

        qr = None
        sketch_sq = 0.0
        done_sketches = 0

        while True:
            Z = int(sketchdim_j - done_sketches)
            if Z <= 0:
                raise RuntimeError("non-positive sketch width encountered")

            if is_boundary:
                Y = _build_local_sketch_block(
                    mps_list=mps_list,
                    coeffs=coeffs,
                    caps=caps,
                    j=j,
                    Z=Z,
                    rng=rng,
                    out_dtype=out_dtype,
                )
            else:
                Y = _sketch_increment_from_Ms(Ms, Z, rng, out_dtype)

            sketch_sq += float(np.vdot(Y, Y).real)

            if qr is None:
                qr = IncrementalQR(np.asfortranarray(Y))
            else:
                qr.append(np.asfortranarray(Y))

            done_sketches = sketchdim_j

            if outputdim_eff is not None or sketchdim_j == maxdim_j:
                done = True
            else:
                err_est = qr.error_estimate()
                norm_est = (sketch_sq ** 0.5) / (sketchdim_j ** 0.5)
                done = (err_est <= cutoff * norm_est) and (sketchdim_j >= mindim_j)

            if done:
                Q = qr.get_q()
                chi_j = int(Q.shape[1])

                if is_boundary:
                    d = int(np.asarray(mps_list[0][j]).shape[1])
                    psi_out[j] = np.ascontiguousarray(np.conj(Q.T).reshape(chi_j, d))
                    caps = _build_caps_boundary(mps_list, coeffs, Q, out_dtype)
                else:
                    d = int(np.asarray(mps_list[0][j]).shape[1])
                    psi_out[j] = np.ascontiguousarray(
                        np.conj(Q.T).reshape(chi_j, d, chi_next)
                    )
                    caps = _caps_from_Ms_interior(Ms, Q, out_dtype)

                chi_next = chi_j
                break

            sketchdim_j = min(maxdim_j, sketchdim_j + int(sketchincrement))

    left = None
    for psi, cap in zip(mps_list, caps):
        term = np.asarray(psi[0], dtype=out_dtype) @ np.asarray(cap, dtype=out_dtype)
        left = term if left is None else left + term

    psi_out[0] = np.ascontiguousarray(left)

    out = MPS(psi_out, dtype=out_dtype, orthform="Left")

    if finalround:
        out.round(stop=stop)

    return out



def _site_rank_upper_bound_pair(psi1, psi2, j, cap1_next, cap2_next):
    if j == psi1.N - 1:
        Dl1 = int(np.asarray(psi1[j]).shape[0])
        Dl2 = int(np.asarray(psi2[j]).shape[0])
        d = int(np.asarray(psi1[j]).shape[1])
        return min(Dl1 + Dl2, d)

    if cap1_next is None or cap2_next is None:
        raise ValueError("cap1_next and cap2_next must be present on interior sites")

    chi1_next = int(np.asarray(cap1_next).shape[1])
    chi2_next = int(np.asarray(cap2_next).shape[1])
    if chi1_next != chi2_next:
        raise ValueError("right cap width mismatch between terms")

    Dl1 = int(np.asarray(psi1[j]).shape[0])
    Dl2 = int(np.asarray(psi2[j]).shape[0])
    d = int(np.asarray(psi1[j]).shape[1])

    return min(Dl1 + Dl2, d * chi1_next)


def _build_local_sketch_block_pair(
    psi1,
    psi2,
    alpha,
    beta,
    cap1_next,
    cap2_next,
    j,
    Z,
    rng,
    out_dtype,
):
    if j == psi1.N - 1:
        d = int(np.asarray(psi1[j]).shape[1])
        Y = np.zeros((d, Z), dtype=out_dtype, order="F")

        M1 = _effective_rowblock_boundary(psi1[j], alpha, out_dtype)
        G1 = _draw_gaussian((M1.shape[0], Z), rng, out_dtype)
        Y += np.conj(M1).T @ G1

        M2 = _effective_rowblock_boundary(psi2[j], beta, out_dtype)
        G2 = _draw_gaussian((M2.shape[0], Z), rng, out_dtype)
        Y += np.conj(M2).T @ G2

        return np.asfortranarray(Y)

    if cap1_next is None or cap2_next is None:
        raise ValueError("cap1_next and cap2_next must be present on interior sites")

    chi1_next = int(np.asarray(cap1_next).shape[1])
    chi2_next = int(np.asarray(cap2_next).shape[1])
    if chi1_next != chi2_next:
        raise ValueError("right cap width mismatch between terms")

    d = int(np.asarray(psi1[j]).shape[1])
    Y = np.zeros((d * chi1_next, Z), dtype=out_dtype, order="F")

    M1 = _effective_rowblock_interior(psi1[j], cap1_next, out_dtype)
    G1 = _draw_gaussian((M1.shape[0], Z), rng, out_dtype)
    Y += np.conj(M1).T @ G1

    M2 = _effective_rowblock_interior(psi2[j], cap2_next, out_dtype)
    G2 = _draw_gaussian((M2.shape[0], Z), rng, out_dtype)
    Y += np.conj(M2).T @ G2

    return np.asfortranarray(Y)


def _build_caps_boundary_pair(psi1, psi2, alpha, beta, Q, out_dtype):
    cap1 = _project_cap_boundary(psi1[-1], alpha, Q, out_dtype)
    cap2 = _project_cap_boundary(psi2[-1], beta, Q, out_dtype)
    return cap1, cap2


def _build_caps_interior_pair(psi1, psi2, cap1_next, cap2_next, j, Q, out_dtype):
    cap1 = _project_cap_interior(psi1[j], cap1_next, Q, out_dtype)
    cap2 = _project_cap_interior(psi2[j], cap2_next, Q, out_dtype)
    return cap1, cap2


def _sketch_increment_from_Ms(Ms, Z, rng, out_dtype):
    """Build Z sketch columns from precomputed row-block matrices."""
    rows = Ms[0].shape[1]
    Y = np.zeros((rows, Z), dtype=out_dtype, order="F")
    for M in Ms:
        G = _draw_gaussian((M.shape[0], Z), rng, out_dtype)
        Y += np.conj(M).T @ G
    return np.asfortranarray(Y)


def _caps_from_Ms_interior(Ms, Q, out_dtype):
    """Build new caps from precomputed row-block matrices and Q, avoiding redundant A2@cap."""
    Q = np.asarray(Q, dtype=out_dtype)
    return [np.ascontiguousarray(M @ Q) for M in Ms]


def lincomb2_SRC(
    psi1,
    psi2,
    alpha=1.0,
    beta=1.0,
    *,
    stop=None,
    sketchdim=16,
    sketchincrement=16,
    finalround=False,
    dtype=None,
    rng=None,
    seed=None,
):
    if seed is not None:
        rng = np.random.default_rng(int(seed))
    elif rng is None:
        rng = np.random.default_rng()

    if stop is None:
        stop = Cutoff(1e-8)

    alpha = np.asarray(alpha).reshape(()).item()
    beta = np.asarray(beta).reshape(()).item()

    if int(psi1.N) != int(psi2.N):
        raise ValueError("psi1 and psi2 must have the same number of sites")

    if abs(alpha) == 0 and abs(beta) == 0:
        raise ValueError("at least one coefficient must be nonzero")

    if abs(alpha) == 0:
        out = psi2 * beta
        if finalround:
            out.round(stop=stop)
        return out

    if abs(beta) == 0:
        out = psi1 * alpha
        if finalround:
            out.round(stop=stop)
        return out

    N = int(psi1.N)

    if dtype is None:
        out_dtype = np.result_type(
            np.asarray(alpha).dtype,
            np.asarray(beta).dtype,
            _term_dtype(psi1),
            _term_dtype(psi2),
            np.float64,
        )
    else:
        out_dtype = np.dtype(dtype)

    alpha = np.asarray(alpha, dtype=out_dtype).reshape(()).item()
    beta = np.asarray(beta, dtype=out_dtype).reshape(()).item()

    if N == 1:
        out0 = (
            alpha * np.asarray(psi1[0], dtype=out_dtype)
            + beta * np.asarray(psi2[0], dtype=out_dtype)
        )
        return MPS([np.ascontiguousarray(out0)], dtype=out_dtype)

    outputdim_eff = _normalize_bound(stop.outputdim)
    maxdim_eff = _normalize_bound(stop.maxdim)
    cutoff = stop.cutoff

    mindim_raw = stop.mindim
    if mindim_raw is None:
        mindim_eff = 1
    else:
        mindim_eff = max(int(mindim_raw), 1)

    if outputdim_eff is not None:
        maxdim_eff = outputdim_eff
        mindim_eff = outputdim_eff
        sketchdim = outputdim_eff

    psi_out = [None] * N
    cap1 = None
    cap2 = None
    chi_next = 1

    for j in reversed(range(1, N)):
        rank_cap = int(_site_rank_upper_bound_pair(psi1, psi2, j, cap1, cap2))

        if maxdim_eff is None:
            maxdim_j = rank_cap
        else:
            maxdim_j = min(maxdim_eff, rank_cap)

        mindim_j = min(mindim_eff, maxdim_j)
        sketchdim_j = max(min(int(sketchdim), maxdim_j), mindim_j)

        if maxdim_j <= 0:
            raise ValueError("encountered a non-positive local rank bound")

        # Pre-compute row-block matrices once per site (cap1/cap2 from the right
        # do not change during the sketch while-loop for this site).
        is_boundary = (j == N - 1)
        if not is_boundary:
            M1 = _effective_rowblock_interior(psi1[j], cap1, out_dtype)
            M2 = _effective_rowblock_interior(psi2[j], cap2, out_dtype)

        qr = None
        sketch_sq = 0.0
        done_sketches = 0

        while True:
            Z = int(sketchdim_j - done_sketches)
            if Z <= 0:
                raise RuntimeError("non-positive sketch width encountered")

            if is_boundary:
                Y = _build_local_sketch_block_pair(
                    psi1=psi1,
                    psi2=psi2,
                    alpha=alpha,
                    beta=beta,
                    cap1_next=cap1,
                    cap2_next=cap2,
                    j=j,
                    Z=Z,
                    rng=rng,
                    out_dtype=out_dtype,
                )
            else:
                Y = _sketch_increment_from_Ms([M1, M2], Z, rng, out_dtype)

            sketch_sq += float(np.vdot(Y, Y).real)

            if qr is None:
                qr = IncrementalQR(np.asfortranarray(Y))
            else:
                qr.append(np.asfortranarray(Y))

            done_sketches = sketchdim_j

            if outputdim_eff is not None or sketchdim_j == maxdim_j:
                done = True
            elif not qr.open:
                done = True
            else:
                err_est = qr.error_estimate()
                norm_est = (sketch_sq ** 0.5) / max(float(sketchdim_j), 1.0) ** 0.5
                done = (err_est <= cutoff * norm_est) and (sketchdim_j >= mindim_j)

            if done:
                Q = qr.get_q()
                chi_j = int(Q.shape[1])

                if is_boundary:
                    d = int(np.asarray(psi1[j]).shape[1])
                    psi_out[j] = np.ascontiguousarray(np.conj(Q.T).reshape(chi_j, d))
                    cap1, cap2 = _build_caps_boundary_pair(
                        psi1, psi2, alpha, beta, Q, out_dtype
                    )
                else:
                    d = int(np.asarray(psi1[j]).shape[1])
                    psi_out[j] = np.ascontiguousarray(
                        np.conj(Q.T).reshape(chi_j, d, chi_next)
                    )
                    cap1, cap2 = _caps_from_Ms_interior([M1, M2], Q, out_dtype)

                chi_next = chi_j
                break

            sketchdim_j = min(maxdim_j, sketchdim_j + int(sketchincrement))

    left = (
        np.asarray(psi1[0], dtype=out_dtype) @ np.asarray(cap1, dtype=out_dtype)
        + np.asarray(psi2[0], dtype=out_dtype) @ np.asarray(cap2, dtype=out_dtype)
    )

    psi_out[0] = np.ascontiguousarray(left)

    out = MPS(psi_out, dtype=out_dtype, orthform="Left")

    if finalround:
        out.round(stop=stop)

    return out

def trp_add_SRC_pairwise(
        A,
        B,
        *,
        alpha=1.0,
        beta=1.0,
        stop=None,
        sketchdim=16,
        sketchincrement=16,
        finalround=True,
        dtype=None,
        rng=None,
        seed=None,
    ):
        if seed is not None:
            rng = np.random.default_rng(int(seed))
        elif rng is None:
            rng = np.random.default_rng()

        if not hasattr(A, "cols") or not hasattr(B, "cols"):
            raise TypeError("A and B must be TRP-like objects with .cols")

        if A.n != B.n or A.d != B.d or A.k != B.k:
            raise ValueError("TRP dimension mismatch in pairwise SRC addition")

        if dtype is None:
            out_dtype = np.result_type(
                A.dtype,
                B.dtype,
                np.asarray(alpha).dtype,
                np.asarray(beta).dtype,
                np.float64,
            )
        else:
            out_dtype = np.dtype(dtype)

        out_cols = [
            lincomb2_SRC(
                A.cols[j],
                B.cols[j],
                alpha=alpha,
                beta=beta,
                stop=stop,
                sketchdim=sketchdim,
                sketchincrement=sketchincrement,
                finalround=finalround,
                dtype=out_dtype,
                rng=rng,
            )
            for j in range(A.k)
        ]

        out = A._from_parent(A, out_cols)
        out.dtype = out_dtype
        out.orthform = None
        out.invalidate_ip_cache()
        out.invalidate_rhs_cache()
        return out