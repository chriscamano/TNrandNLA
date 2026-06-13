"""experiment.py — First-to-1% adaptive sketch sweep for tr[exp(-beta H)].

Incremental XNysTrace with MPS probes; oracle applies exp(-beta(H+bI)) via
TDVP-GSE. Live JSON is written atomically after every (n, chi, trial).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import math

import numpy as np
import scipy.linalg as sla

from oracle import (
    TDVPGSEExpOracle,
    build_oracle,
    default_shift_b,
    shifted_reference_trace,
)
from tfim import periodic_tfim_zz_x, tfim_periodic_trace_exp_analytic
from tnrnla.tn.trp import MPS


# ── ANSI colours ──────────────────────────────────────────────────────────────

_RST  = "\033[0m"
_BOLD = "\033[1m"
_RED  = "\033[38;5;196m"
_YEL  = "\033[38;5;220m"
_GRN  = "\033[38;5;46m"
_CYN  = "\033[38;5;51m"
_MAG  = "\033[38;5;201m"


def _color_relerr(rel, target):
    r = float(rel) / max(float(target), 1e-300)
    if r <= 1.0:
        return _GRN
    if r <= 2.0:
        return _YEL
    return _RED


def _fmt_relerr(rel, target):
    c = _color_relerr(rel, target)
    return f"{c}{rel:.3e}{_RST}"


def _banner(msg):
    bar = "─" * min(len(msg) + 4, 88)
    return f"{_BOLD}{_MAG}┌{bar}┐\n│  {msg}  │\n└{bar}┘{_RST}"


# ── time step helpers ─────────────────────────────────────────────────────────

def steps_from_divisor(dt_divisor):
    """Number of imaginary-time steps: equal to the divisor (dt = beta/dt_divisor), minimum 1."""
    return max(1, int(dt_divisor))


def auto_dt_divisor(n, J, h, beta, *, factor=2.0):
    """Default heuristic: n_steps = ceil(factor * sqrt(n)).

    The default factor=2.0 gives:
      n=8 -> 6, n=16 -> 8, n=36 -> 12, n=64 -> 16.

    Args J/h/beta are accepted for API compatibility with other heuristics.
    """
    _ = (J, h, beta)  # reserved for alternate heuristics; unused in sqrt mode
    return max(1, math.ceil(float(factor) * math.sqrt(max(float(n), 1.0))))


def resolve_dt_divisor(dt_divisor, n, J, h, beta, *, factor=2.0):
    """Return integer n_steps for this n.

    dt_divisor=None  → auto_dt_divisor heuristic (default: ceil(2*sqrt(n)))
    dt_divisor=int   → that fixed value
    dt_divisor=callable → dt_divisor(n) (user-supplied function of n)
    """
    if dt_divisor is None:
        return auto_dt_divisor(n, J, h, beta, factor=factor)
    if callable(dt_divisor):
        return max(1, int(dt_divisor(n)))
    return steps_from_divisor(dt_divisor)


def _serialize_dt_divisor_cfg(dt_divisor):
    """JSON-safe representation of dt_divisor config."""
    if callable(dt_divisor):
        nm = getattr(dt_divisor, "__name__", "<callable>")
        return f"callable:{nm}"
    if dt_divisor is None:
        return None
    return int(dt_divisor)


def resolve_b_shift(b_shift, n, J, h):
    """Return the spectral shift for this n.

    b_shift=None        -> default_shift_b(n, J, h)
    b_shift=callable    -> float(b_shift(n, J, h))
    b_shift=float/int   -> that fixed value
    """
    if b_shift is None:
        return default_shift_b(n, J, h)
    if callable(b_shift):
        return float(b_shift(n, J, h))
    return float(b_shift)


def _serialize_b_shift_cfg(b_shift):
    """JSON-safe representation of b_shift config."""
    if callable(b_shift):
        nm = getattr(b_shift, "__name__", "<callable>")
        return f"callable:{nm}"
    if b_shift is None:
        return None
    return float(b_shift)


# ── memory helpers (no external imports) ──────────────────────────────────────

def _dense_memory_gb(n):
    dim = 1 << int(n)
    return float(dim * dim * 8) / (1024.0 ** 3)


def _sparse_memory_gb(n):
    n = int(n)
    dim = 1 << n
    nnz = dim * (n + 1)
    return float(nnz * (8 + 4) + (dim + 1) * 8) / (1024.0 ** 3)


def _mps_col_memory_gb(n, bond, *, d=2):
    n, bond, d = int(n), max(1, int(bond)), int(d)
    if n <= 1:
        elems = d
    elif n == 2:
        elems = 2 * d * bond
    else:
        elems = d * ((n - 2) * bond * bond + 2 * bond)
    return float(elems * 8) / (1024.0 ** 3)


def _sketch_memory_gb(n, chi, s, *, d=2):
    col_omega = _mps_col_memory_gb(n, chi, d=d)
    return float(s) * 2.0 * col_omega   # Omega + Y columns (Y bond ≈ chi)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


# ── incremental gram helpers ──────────────────────────────────────────────────

def _sym(M):
    return 0.5 * (M + M.T)


def _gram_extend_sym(G_prev, cols, k_prev):
    """Extend symmetric Gram matrix G[i,j]=<cols[i],cols[j]> to include cols[k_prev:].

    Only computes inner products involving at least one new column; the
    k_prev×k_prev block is copied from G_prev unchanged.
    """
    k = len(cols)
    if k == k_prev:
        return G_prev
    dtype = G_prev.dtype if k_prev > 0 else np.float64
    G = np.empty((k, k), dtype=dtype)
    if k_prev > 0:
        G[:k_prev, :k_prev] = G_prev
    for j in range(k_prev, k):
        for i in range(j):
            v = float(np.real(cols[i].inner_product(cols[j])))
            G[i, j] = v
            G[j, i] = v
        G[j, j] = float(np.real(cols[j].inner_product(cols[j])))
    return _sym(G)


def _gram_extend_cross(C_prev, lhs_cols, rhs_cols, k_prev):
    """Extend cross-Gram matrix C[i,j]=<lhs[i],rhs[j]> to include index k_prev onward.

    A = exp(-βH) is symmetric PSD, so C[i,j] = C[j,i] exactly. We exploit this:
    - Block 1 (old×new): compute k_prev × n_new inner products, copy transpose for Block 2.
    - Block 3 (new×new): upper triangle only, reflect for lower triangle.
    This halves the inner product count versus computing both off-diagonal blocks.
    """
    k = len(lhs_cols)
    if k == k_prev:
        return C_prev
    dtype = C_prev.dtype if k_prev > 0 else np.float64
    C = np.empty((k, k), dtype=dtype)
    if k_prev > 0:
        C[:k_prev, :k_prev] = C_prev

    # Block 1: old lhs × new rhs
    for j in range(k_prev, k):
        for i in range(k_prev):
            C[i, j] = float(np.real(lhs_cols[i].inner_product(rhs_cols[j])))

    # Block 2: new lhs × old rhs — free via symmetry of A
    if k_prev > 0:
        C[k_prev:k, :k_prev] = C[:k_prev, k_prev:k].T

    # Block 3: new lhs × new rhs — upper triangle + reflect
    for i in range(k_prev, k):
        C[i, i] = float(np.real(lhs_cols[i].inner_product(rhs_cols[i])))
        for j in range(i + 1, k):
            v = float(np.real(lhs_cols[i].inner_product(rhs_cols[j])))
            C[i, j] = v
            C[j, i] = v

    return _sym(C)


def _xnystrace_from_grams(G, C, T, *, k, n, denom_eps=1e-300):
    """XNysTrace estimator from pre-computed symmetrised gram matrices.

    Matches mps_xnystrace_chol_gram(..., resphere=False, trp_scaled=False).
    Probes are MPS.rmps (unscaled, isotropic), so corr = 1/d_i.
    G = Ω^T Ω,  C = sym(Ω^T Y),  T = Y^T Y,  all float64 symmetric k×k.
    """
    eps = np.finfo(G.dtype).eps

    R    = sla.cholesky(G, lower=False, check_finite=False)
    Rinv = sla.solve_triangular(R, np.eye(k, dtype=R.dtype),
                                lower=False, check_finite=False)

    B   = _sym(Rinv.T @ C @ Rinv)
    lam = np.linalg.eigvalsh(B)
    mu  = max(0.0, np.sqrt(eps) * abs(float(lam[-1])) - float(lam[0]))

    H = _sym(C + mu * G)
    S = _sym(T + mu * (2.0 * C + mu * G))

    L = sla.cholesky(H, lower=True, check_finite=False)
    M = _sym(sla.cho_solve((L, True), np.eye(k, dtype=H.dtype), check_finite=False))

    d = np.maximum(np.diag(M), denom_eps)
    u = np.diag(M @ S @ M)
    t = float(np.trace(S @ M))
    N = float(2 ** n)

    corr = 1.0 / d

    samples    = t - u / d + corr - mu * N
    trace_est  = float(np.mean(samples))
    stderr     = 0.0 if k == 1 else float(np.std(samples, ddof=1) / np.sqrt(k))
    return trace_est, stderr


# ── inner k-sweep loop ────────────────────────────────────────────────────────

def _run_k_sweep(
    oracle,
    *,
    n,
    chi,
    ref_shifted,
    target,
    k_init,
    k_step,
    kmax,
    seed,
    matvec_time_offset_sec=0.0,
    verbose=True,
    trial_label="",
    relerr_abort_threshold=None,
    unshift_scale=1.0,
    ref_unshifted=None,
):
    """Sweep k from k_init to kmax, accumulating oracle calls across checkpoints.

    Probes and oracle outputs are accumulated incrementally: at each checkpoint
    only k_step new probes are drawn and the oracle is applied only to those.
    The estimator is then re-run on the full accumulated (Omega, Y) sketch.

    Total oracle calls = k* (the hitting k), never more — no redundant recomputation.
    """
    rows      = []
    matvec_time_running = 0.0
    kmax_i    = int(kmax)
    k_step_i  = int(k_step)
    k_init_i  = int(k_init)
    base_seed = int(seed)
    matvec_time_offset_sec = float(matvec_time_offset_sec)

    omega_cols: list = []
    y_cols: list     = []
    G = np.zeros((0, 0), dtype=np.float64)   # Ω^T Ω  — grown incrementally
    C = np.zeros((0, 0), dtype=np.float64)   # sym(Ω^T Y) — grown incrementally
    T = np.zeros((0, 0), dtype=np.float64)   # Y^T Y  — grown incrementally
    post_gse_bond_running = None
    post_first_sweep_bond_running = None
    max_bond_running      = None
    oracle_mem_gb_running = None

    # Build checkpoint targets. If the regular stride lands near (but not on)
    # kmax, include a final kmax checkpoint to avoid skipping the budget edge.
    targets = []
    next_target = k_init_i
    while next_target <= kmax_i:
        targets.append(int(next_target))
        next_target += k_step_i
    if targets and targets[-1] != kmax_i and (kmax_i - targets[-1]) <= k_step_i:
        targets.append(int(kmax_i))

    for next_target in targets:
        # ── draw k_step new probes and apply oracle only to them ──────────────
        # Use MPS.rmps (unscaled) so accumulated columns have consistent norms.
        # trp_scaled=False in the estimator matches this convention.
        # Fix 3: use col_idx directly as loop variable; seed each probe
        # deterministically by index so retries with different sweeps counts
        # still draw the same probes.
        k_prev = len(omega_cols)
        for col_idx in range(k_prev, next_target):
            mv_t0    = time.perf_counter()                          # includes probe gen
            col      = MPS.rmps(int(n), int(chi), d=2,
                                rng=np.random.default_rng(base_seed + col_idx))
            col_seed = base_seed + col_idx + 7919
            y_col    = oracle.apply(col, col_seed=col_seed)
            matvec_time_running += float(time.perf_counter() - mv_t0)
            omega_cols.append(col)
            y_cols.append(y_col)
            pgb = oracle.last_post_gse_bond
            pfs = getattr(oracle, "last_post_first_sweep_bond", None)
            mb  = oracle.last_max_bond
            omg = getattr(oracle, "last_output_memory_gb", None)
            if pgb is not None:
                post_gse_bond_running = (
                    pgb if post_gse_bond_running is None
                    else max(post_gse_bond_running, pgb)
                )
            if pfs is not None:
                post_first_sweep_bond_running = (
                    pfs if post_first_sweep_bond_running is None
                    else max(post_first_sweep_bond_running, pfs)
                )
            if mb is not None:
                max_bond_running = (
                    mb if max_bond_running is None
                    else max(max_bond_running, mb)
                )
            if omg is not None:
                oracle_mem_gb_running = omg

        k = len(omega_cols)

        # ── extend gram matrices with only new-column inner products ──────────
        gram_t0 = time.perf_counter()
        G = _gram_extend_sym(G, omega_cols, k_prev)
        C = _gram_extend_cross(C, omega_cols, y_cols, k_prev)
        T = _gram_extend_sym(T, y_cols, k_prev)
        matvec_time_running += float(time.perf_counter() - gram_t0)

        # ── re-estimate from pre-computed gram matrices ───────────────────────
        post_t0 = time.perf_counter()
        est, stderr = _xnystrace_from_grams(G, C, T, k=k, n=int(n))
        post_time = float(time.perf_counter() - post_t0)
        total_matvec_time = matvec_time_offset_sec + matvec_time_running
        # Streaming lets us inspect several checkpoints, but the reported time
        # should model running directly to this k: all oracle applications plus
        # this checkpoint's final estimator, not earlier failed estimators.
        wall  = float(total_matvec_time + post_time)
        est_r = float(np.real(est))
        if ref_unshifted is not None:
            rel = float(abs(est_r * float(unshift_scale) - ref_unshifted) / (abs(ref_unshifted) + 1e-300))
        else:
            rel = float(abs(est_r - ref_shifted) / (abs(ref_shifted) + 1e-300))
        mem   = _sketch_memory_gb(int(n), int(chi), int(k))

        post_gse_bond = (None if post_gse_bond_running is None
                         else int(post_gse_bond_running))
        post_first_sweep_bond = (
            None if post_first_sweep_bond_running is None
            else int(post_first_sweep_bond_running)
        )
        max_bond      = (None if max_bond_running is None
                         else int(max_bond_running))

        row = {
            "k":                        int(k),
            "trace_hat":                est_r,
            "stderr_hat":               float(np.real(stderr)),
            "relerr":                   rel,
            "wall_time_sec":            wall,
            "total_matvec_time_sec":     float(total_matvec_time),
            "attempt_matvec_time_sec":   float(matvec_time_running),
            "postprocess_time_sec":      float(post_time),
            "sketch_memory_gb":         mem,
            "omega_memory_gb":          float(mem / 2.0),
            "y_memory_gb":              float(mem / 2.0),
            "oracle_post_gse_bond":     post_gse_bond,
            "oracle_post_first_sweep_bond": post_first_sweep_bond,
            "oracle_max_bond":          max_bond,
            "oracle_output_memory_gb":  oracle_mem_gb_running,
            "stop_reason":              None,
        }
        rows.append(row)

        enriched_txt = "-" if post_gse_bond is None else str(post_gse_bond)
        compressed_txt = "-" if post_first_sweep_bond is None else str(post_first_sweep_bond)
        if verbose:
            print(
                f"{_CYN}n={int(n):3d} chi={int(chi):2d}{trial_label} "
                f"k={k:4d}{_RST} | "
                f"relerr={_fmt_relerr(rel, target)} | "
                f"enriched_bond={enriched_txt} | "
                f"compressed_bond={compressed_txt} | "
                f"final_time={wall:.2f}s",
                flush=True,
            )

        if relerr_abort_threshold is not None and rel > float(relerr_abort_threshold):
            rows[-1]["stop_reason"] = "aborted"
            if verbose:
                print(
                    f"{_BOLD}{_RED}  ✗ ABORT{_RST} "
                    f"n={int(n):3d} chi={int(chi):2d}{trial_label} "
                    f"k={k:4d} | "
                    f"relerr={_fmt_relerr(rel, target)} > abort_threshold={relerr_abort_threshold:.3g}",
                    flush=True,
                )
            break

        if rel <= float(target):
            row["stop_reason"] = "hit"
            if verbose:
                print(
                    f"{_BOLD}{_GRN}  ✓ HIT{_RST} "
                    f"n={int(n):3d} chi={int(chi):2d}{trial_label} "
                    f"k={k:4d} | "
                    f"relerr={_fmt_relerr(rel, target)} ≤ target={target:.3e} | "
                    f"final_time={wall:.2f}s",
                    flush=True,
                )
            break

    # If we exhausted the k-budget without hitting target, mark the terminal row.
    # This covers both exact hits (k == kmax) and stepped sweeps that stop below
    # kmax because the next checkpoint would exceed the budget.
    if rows and rows[-1].get("stop_reason") is None:
        rows[-1]["stop_reason"] = "kmax"

    return rows


# ── single (n, chi, trial) adaptive experiment ────────────────────────────────
def _run_n_chi_trial(
    *,
    n,
    chi,
    J,
    h,
    beta,
    target,
    kmax,
    k_init,
    k_step,
    k_step_chi1=None,
    dt_divisor,
    dt_divisor_factor,
    gse_k,
    tdvp_krylov_depth,
    epsEnrich,
    epsSRC,
    oracle_seed,
    sample_seed,
    trial_idx,
    n_trials,
    tdvp_cutoff,
    gse_max_enriched_bond,
    b_shift,
    enrich_every_step,
    verbose,
    relerr_abort_threshold=None,
    unshift_error=False,
):
    n   = int(n)
    chi = int(chi)

    k_step_eff = int(k_step_chi1) if (chi == 1 and k_step_chi1 is not None) else int(k_step)

    b_shift_now = resolve_b_shift(b_shift, n, J, h)
    ref_shifted = shifted_reference_trace(n, h, J, beta, b_shift=b_shift_now)

    if unshift_error:
        _ref_unshifted = float(tfim_periodic_trace_exp_analytic(n, float(h), float(J), float(beta)))
        _unshift_scale = float(np.exp(float(beta) * float(b_shift_now)))
    else:
        _ref_unshifted = None
        _unshift_scale = 1.0

    sweeps_now = resolve_dt_divisor(
        dt_divisor, n, J, h, beta, factor=float(dt_divisor_factor)
    )

    n_tot    = int(n_trials)
    trial_id = int(trial_idx) + 1
    tr_lbl   = f" tr={trial_id}/{n_tot}" if n_tot > 1 else ""

    dense_mem  = _dense_memory_gb(n)
    sparse_mem = _sparse_memory_gb(n)

    oracle = build_oracle(
        n, J, h, beta,
        sweeps=sweeps_now,
        gse_k=int(gse_k),
        seed=int(oracle_seed),
        tdvp_krylov_depth=int(tdvp_krylov_depth),
        epsEnrich=epsEnrich,
        epsSRC=float(epsSRC),
        tdvp_cutoff=float(tdvp_cutoff),
        gse_max_enriched_bond=gse_max_enriched_bond,
        b_shift=float(b_shift_now),
        enrich_every_step=bool(enrich_every_step),
    )

    all_rows = _run_k_sweep(
        oracle,
        n=n,
        chi=chi,
        ref_shifted=ref_shifted,
        target=float(target),
        k_init=int(k_init),
        k_step=k_step_eff,
        kmax=int(kmax),
        seed=int(sample_seed),
        verbose=verbose,
        trial_label=tr_lbl,
        relerr_abort_threshold=relerr_abort_threshold,
        unshift_scale=_unshift_scale,
        ref_unshifted=_ref_unshifted,
    )

    hit_candidates = [r for r in all_rows if r.get("stop_reason") == "hit"]
    hit_row   = hit_candidates[0] if hit_candidates else None
    final_row = all_rows[-1] if all_rows else None
    hit       = hit_row is not None
    aborted   = (final_row is not None and final_row.get("stop_reason") == "aborted")

    record = {
        "n": n,
        "chi": chi,
        "trial_index": int(trial_idx),
        "trial": trial_id,
        "ntrials": n_tot,
        "target_accuracy": float(target),
        "target_relerr": float(target),
        "tdvp_sweeps": sweeps_now,
        "tdvp_divisor": sweeps_now,
        "tdvp_dt": float(beta) / float(sweeps_now) if sweeps_now > 0 else None,
        "enrich_every_step": bool(enrich_every_step),
        "gse_k": int(gse_k),
        "gse_max_enriched_bond": (
            None if gse_max_enriched_bond is None else int(gse_max_enriched_bond)
        ),
        "b_shift": float(b_shift_now),
        "kmax": int(kmax),
        "k_step_used": int(k_step_eff),
        "hit": hit,
        "aborted": aborted,
        "hit_k": int(hit_row["k"]) if hit else None,
        "hit_relerr": float(hit_row["relerr"]) if hit else None,
        "matvecs_to_target": int(hit_row["k"]) if hit else (int(final_row["k"]) if final_row else None),
        "time_to_target_sec": float(hit_row["wall_time_sec"]) if hit else (float(final_row["wall_time_sec"]) if final_row else None),
        "time_to_target_matvec_sec":
            float(hit_row["total_matvec_time_sec"])
            if (hit and hit_row.get("total_matvec_time_sec") is not None) else None,
        "time_to_target_postprocess_sec":
            float(hit_row["postprocess_time_sec"])
            if (hit and hit_row.get("postprocess_time_sec") is not None) else None,
        "time_to_target_sketch_memory_gb":
            float(hit_row["sketch_memory_gb"]) if hit else None,
        "time_to_target_omega_memory_gb":
            float(hit_row["omega_memory_gb"]) if hit else None,
        "time_to_target_oracle_memory_gb":
            float(hit_row["oracle_output_memory_gb"])
            if (hit and hit_row.get("oracle_output_memory_gb") is not None) else None,
        "final_relerr": float(final_row["relerr"]) if final_row else None,
        "final_k": int(final_row["k"]) if final_row else None,
        "final_wall_sec": float(final_row["wall_time_sec"]) if final_row else None,
        "final_matvec_time_sec":
            float(final_row["total_matvec_time_sec"])
            if (final_row and final_row.get("total_matvec_time_sec") is not None) else None,
        "final_postprocess_time_sec":
            float(final_row["postprocess_time_sec"])
            if (final_row and final_row.get("postprocess_time_sec") is not None) else None,
        "final_sketch_memory_gb":
            float(final_row["sketch_memory_gb"]) if final_row else None,
        "final_oracle_max_bond":
            final_row.get("oracle_max_bond") if final_row else None,
        "final_oracle_post_gse_bond":
            final_row.get("oracle_post_gse_bond") if final_row else None,
        "final_oracle_post_first_sweep_bond":
            final_row.get("oracle_post_first_sweep_bond") if final_row else None,
        "final_oracle_memory_gb":
            float(final_row["oracle_output_memory_gb"])
            if (final_row and final_row.get("oracle_output_memory_gb") is not None) else None,
        "dense_memory_gb_est": dense_mem,
        "sparse_memory_gb_est": sparse_mem,
        "oracle_seed": int(oracle_seed),
        "sample_seed": int(sample_seed),
    }
    return record


# ── top-level experiment driver ───────────────────────────────────────────────
def run_experiment(
    *,
    n_values,
    chi_values=None,
    J=1.0,
    h=10.0,
    beta=0.5,
    target=1e-2,
    kmax=128,
    k_init=2,
    k_step=1,
    k_step_chi1=None,
    n_trials=1,
    gse_k_kr=3,
    gse_k_rmps=3,
    tdvp_krylov_depth=32,
    epsEnrich=1e-2,
    epsSRC=1e-8,
    tdvp_cutoff=1e-10,
    b_shift=None,
    sample_seed=3234,
    oracle_seed=901234,
    trial_seed_stride=1_000_003,
    dt_divisor=None,
    dt_divisor_factor=2.0,
    enrich_every_step=False,
    verbose=True,
    live_json_path=None,
    relerr_abort_threshold=None,
    max_retries=3,
    unshift_error=True,
):
    n_values   = [int(nv) for nv in n_values]
    chi_values = [1] if chi_values is None else [int(c) for c in chi_values]
    n_trials   = int(n_trials)

    base_cfg = dict(
        n_values=n_values,
        chi_values=chi_values,
        J=float(J), h=float(h), beta=float(beta),
        target=float(target),
        kmax=int(kmax),
        k_init=int(k_init),
        k_step=int(k_step),
        k_step_chi1=None if k_step_chi1 is None else int(k_step_chi1),
        n_trials=n_trials,
        gse_k_kr=int(gse_k_kr),
        gse_k_rmps=int(gse_k_rmps),
        tdvp_krylov_depth=int(tdvp_krylov_depth),
        epsEnrich=epsEnrich if isinstance(epsEnrich, int) else float(epsEnrich),
        epsSRC=float(epsSRC),
        tdvp_cutoff=float(tdvp_cutoff),
        b_shift=_serialize_b_shift_cfg(b_shift),
        sample_seed=int(sample_seed),
        oracle_seed=int(oracle_seed),
        trial_seed_stride=int(trial_seed_stride),
        dt_divisor=_serialize_dt_divisor_cfg(dt_divisor),
        dt_divisor_factor=float(dt_divisor_factor),
        enrich_every_step=bool(enrich_every_step),
        relerr_abort_threshold=(
            None if relerr_abort_threshold is None else float(relerr_abort_threshold)
        ),
        max_retries=int(max_retries),
        unshift_error=bool(unshift_error),
        runtime_accounting="oracle_matvecs_plus_current_postprocess",
        sweeps_mode=(
            "dt_divisor_auto"
            if dt_divisor is None
            else ("dt_divisor_callable" if callable(dt_divisor) else "dt_divisor")
        ),
    )

    records = []
    total_cases = len(n_values) * len(chi_values) * n_trials
    case_num = [0]

    def _flush(status="running"):
        if live_json_path is None:
            return
        _write_json_atomic(
            live_json_path,
            {
                "status": status,
                "base_cfg": base_cfg,
                "records": records,
                "cases_done": case_num[0],
                "cases_total": total_cases,
                "latest_record": records[-1] if records else None,
            },
        )

    _flush("running")

    if verbose:
        print(
            _banner(
                f"PostBerkeley experiment | n={n_values} | "
                f"chi={chi_values} | kmax={kmax} | "
                f"n_trials={n_trials} | target={target:.2%}"
            ),
            flush=True,
        )

    try:
        for n in n_values:
            if verbose:
                _n_steps_init = resolve_dt_divisor(
                    dt_divisor, n, J, h, beta, factor=float(dt_divisor_factor)
                )
                _dt_init = float(beta) / float(_n_steps_init)
                print(
                    _banner(
                        f"n = {n} | n_steps = {_n_steps_init} (dt=β/{_n_steps_init}={_dt_init:.4g}) | "
                        f"active chi = {chi_values}"
                    ),
                    flush=True,
                )

            for chi in chi_values:
                # ── select gse_k based on probe type ──────────────────────────
                gse_k_eff = int(gse_k_kr) if chi == 1 else int(gse_k_rmps)
                gse_bond_cap_eff = None

                for trial_idx in range(n_trials):
                    t_samp = int(sample_seed) + trial_idx * int(trial_seed_stride)
                    t_orac = int(oracle_seed) + trial_idx * int(trial_seed_stride)

                    for retry in range(int(max_retries) + 1):
                        if retry > 0:
                            # shift seeds so each retry draws different probes/oracle
                            t_samp += 1
                            t_orac += 1
                            if verbose:
                                print(
                                    f"{_YEL}  [retry {retry}/{max_retries}] "
                                    f"n={n} chi={chi} trial={trial_idx+1} "
                                    f"— re-seeding and re-running{_RST}",
                                    flush=True,
                                )

                        record = _run_n_chi_trial(
                            n=n,
                            chi=chi,
                            J=float(J),
                            h=float(h),
                            beta=float(beta),
                            target=float(target),
                            kmax=int(kmax),
                            k_init=int(k_init),
                            k_step=int(k_step),
                            k_step_chi1=k_step_chi1,
                            dt_divisor=dt_divisor,
                            dt_divisor_factor=float(dt_divisor_factor),
                            gse_k=gse_k_eff,
                            tdvp_krylov_depth=int(tdvp_krylov_depth),
                            epsEnrich=epsEnrich,
                            epsSRC=float(epsSRC),
                            tdvp_cutoff=float(tdvp_cutoff),
                            gse_max_enriched_bond=gse_bond_cap_eff,
                            b_shift=b_shift,
                            oracle_seed=t_orac,
                            sample_seed=t_samp,
                            trial_idx=trial_idx,
                            n_trials=n_trials,
                            enrich_every_step=bool(enrich_every_step),
                            verbose=verbose,
                            relerr_abort_threshold=relerr_abort_threshold,
                            unshift_error=bool(unshift_error),
                        )

                        if not record.get("aborted", False):
                            break

                        if retry == int(max_retries):
                            if verbose:
                                print(
                                    f"{_RED}  [abort] n={n} chi={chi} trial={trial_idx+1} "
                                    f"exhausted {max_retries} retries — keeping last result{_RST}",
                                    flush=True,
                                )

                    records.append(record)
                    case_num[0] += 1
                    _flush("running")

        _flush("completed")

    except Exception:
        _flush("failed")
        raise

    return {"records": records, "base_cfg": base_cfg}

# ── convenience: load / save results ─────────────────────────────────────────

def save_results(out, path):
    """Save run_experiment output to JSON."""
    _write_json_atomic(path, out)


def load_results(path):
    """Load run_experiment output from JSON."""
    with open(path) as fh:
        return json.load(fh)
