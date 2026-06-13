"""
fig5_driver.py
--------------
Experiment driver for Figure 5.

Compares MPS trace estimators (xnystrace, hutch, nystrom++)
across three synthetic benchmark families as a function of matvec count.

Benchmark kinds
---------------
    "exp"                 diagonal exponential spectrum
    "inverse_laplacian"   exact inverse 1D Dirichlet Laplacian MPO
    "staircase"           multi-level staircase spectrum (auto breakpoints)

`run_experiment_multi` saves incremental JSON snapshots during execution,
including relative trace error and per-evaluation runtime.
"""

import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from tnrnla.rnla.trace.mps.mps_hutch import mps_hutch
from tnrnla.rnla.trace.mps.mps_nystrompp import mps_npp_gram
from tnrnla.rnla.trace.mps.mps_xnystrace import mps_xnystrace_chol_gram

from example_mpos import BENCHMARK_DISPLAY_NAMES, BENCHMARK_KINDS, build_mpo


# ── estimator wrappers (uniform interface) ─────────────────────────────

def _run_xnystrace(oracle, *, n, k, probe_chi, seed, d, dtype):
    trace_hat, _ = mps_xnystrace_chol_gram(
        oracle,
        n=n,
        k=k,
        chi=probe_chi,
        seed=seed,
        dtype=dtype,
        resphere=False,
    )
    return float(np.real(trace_hat))


def _run_xnystrace_resphere(oracle, *, n, k, probe_chi, seed, d, dtype):
    trace_hat, _ = mps_xnystrace_chol_gram(
        oracle,
        n=n,
        k=k,
        chi=probe_chi,
        seed=seed,
        dtype=dtype,
        resphere=True,
    )
    return float(np.real(trace_hat))


def _run_hutch(oracle, *, n, k, probe_chi, seed, d, dtype):
    trace_hat, _ = mps_hutch(
        oracle,
        n=n,
        probe_chi=probe_chi,
        num_queries=k,
        rmps_d=d,
        base_seed=seed,
        dtype=dtype,
    )
    return float(np.real(trace_hat))


def _run_npp(oracle, *, n, k, probe_chi, seed, d, dtype):
    trace_hat, _ = mps_npp_gram(
        oracle,
        k,
        n=n,
        probe_chi=probe_chi,
        seed=seed,
        dtype=dtype,
        resphere=False,
    )
    return float(np.real(trace_hat))


def _run_npp_resphere(oracle, *, n, k, probe_chi, seed, d, dtype):
    trace_hat, _ = mps_npp_gram(
        oracle,
        k,
        n=n,
        probe_chi=probe_chi,
        seed=seed,
        dtype=dtype,
        resphere=True,
    )
    return float(np.real(trace_hat))


ESTIMATORS = {
    "xnystrace": _run_xnystrace,
    "hutch": _run_hutch,
    "nystrom++": _run_npp,
}

# Additional resphere variants — included when resphere_methods=True
RESPHERE_ESTIMATORS = {
    "xnystrace_resph": _run_xnystrace_resphere,
    "nystrom++_resph": _run_npp_resphere,
}


# ── JSON helpers ───────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_experiment_multi_snapshot(
    json_path,
    *,
    kind,
    N,
    d,
    chi_list,
    n_runs,
    base_seed,
    dtype,
    estimators,
    k_vals,
    trace_true,
    results,
    runtimes,
    completed,
    total,
):
    if json_path is None:
        return

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")

    payload = {
        "kind": kind,
        "config": {
            "N": int(N),
            "d": int(d),
            "chi_list": [int(x) for x in chi_list],
            "n_runs": int(n_runs),
            "base_seed": int(base_seed),
            "dtype": str(np.dtype(dtype)),
            "estimators": list(estimators),
            "k_vals": [int(x) for x in np.asarray(k_vals).ravel()],
        },
        "trace_true": float(trace_true),
        "progress": {
            "completed": int(completed),
            "total": int(total),
        },
        "relative_trace_error": {
            str(chi): {name: arr.tolist() for name, arr in method_dict.items()}
            for chi, method_dict in results.items()
        },
        "runtime_s": {
            str(chi): {name: arr.tolist() for name, arr in method_dict.items()}
            for chi, method_dict in runtimes.items()
        },
    }

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    tmp_path.replace(json_path)


# ── helpers ────────────────────────────────────────────────────────────

def int_linspace(start: int, stop: int, num: int, *, unique: bool = True):
    vals = np.linspace(start, stop, num=num)
    vals = np.rint(vals).astype(int)
    vals = np.clip(vals, min(start, stop), max(start, stop))
    if unique:
        vals = np.unique(vals)
    return tuple(int(x) for x in vals)


# ── single experiment (one chi, one run) ──────────────────────────────

def run_experiment(
    kind,
    *,
    N=10,
    d=2,
    probe_chi=8,
    matvec_list=None,
    seed=0,
    dtype=np.float64,
    estimators=None,
    progress=True,
):
    """
    Build the MPO for `kind` and sweep all estimators over `matvec_list`.

    Returns
    -------
    k_vals
        ndarray[int]
    results
        dict[str, ndarray]
        Relative trace error, shape (n_k_vals,)
    trace_true
        float
    """
    if matvec_list is None:
        matvec_list = int_linspace(16, 300, 16)
    if estimators is None:
        estimators = list(ESTIMATORS.keys())

    mpo, trace_true = build_mpo(kind, N, d, seed=seed, dtype=dtype)
    k_vals = np.asarray(matvec_list, dtype=int)
    results = {}

    for name in estimators:
        fn = ESTIMATORS[name]
        rel_err = np.empty_like(k_vals, dtype=float)

        it = enumerate(k_vals)
        if progress:
            it = tqdm(it, total=len(k_vals), desc=f"{kind}/{name}")

        for i, k in it:
            trace_hat = fn(
                mpo,
                n=N,
                k=int(k),
                probe_chi=int(probe_chi),
                seed=int(seed + 10_000 + i),
                d=int(d),
                dtype=dtype,
            )
            rel_err[i] = (
                np.nan
                if np.isnan(trace_hat)
                else abs(trace_hat - trace_true) / (abs(trace_true) + 1e-300)
            )
            if progress and hasattr(it, "set_postfix"):
                it.set_postfix(k=int(k), rel_err=float(rel_err[i]))

        results[name] = rel_err

    return k_vals, results, trace_true


# ── multi-chi / multi-run experiment ──────────────────────────────────

def run_experiment_multi(
    kind,
    *,
    N=10,
    d=2,
    chi_list=(1, 2, 4, 8, 16),
    n_runs=5,
    matvec_list=None,
    base_seed=0,
    dtype=np.float64,
    estimators=None,
    resphere_methods=False,
    progress=True,
    json_path=None,
    flush_every=1,
    return_runtimes=False,
):
    """
    Sweep over a list of chi values and multiple independent runs.

    Parameters
    ----------
    resphere_methods
        bool
        If True, also run the xnystrace and nystrom++ estimators with
        resphere=True (registered as 'xnystrace_resph' and 'nystrom++_resph').
    json_path
        str | Path | None
        If provided, write a JSON snapshot during the run.
    flush_every
        int
        Write the JSON snapshot every `flush_every` completed evaluations.
    return_runtimes
        bool
        If True, return the in-memory runtime arrays as a fourth output.

    Returns
    -------
    k_vals
        ndarray[int], shape (n_k,)
    results
        dict[chi, dict[method, ndarray(n_runs, n_k)]]
    trace_true
        float
    runtimes
        dict[chi, dict[method, ndarray]], optional
        runtimes[chi][method] has shape (n_runs, n_k)
    """
    if matvec_list is None:
        matvec_list = int_linspace(16, 300, 16)
    if estimators is None:
        estimators = list(ESTIMATORS.keys())
        if resphere_methods:
            estimators = estimators + list(RESPHERE_ESTIMATORS.keys())

    mpo, trace_true = build_mpo(kind, N, d, seed=base_seed, dtype=dtype)
    k_vals = np.asarray(matvec_list, dtype=int)
    n_k = len(k_vals)

    results = {}
    runtimes = {}
    for chi in chi_list:
        results[chi] = {
            name: np.full((n_runs, n_k), np.nan, dtype=float)
            for name in estimators
        }
        runtimes[chi] = {
            name: np.full((n_runs, n_k), np.nan, dtype=float)
            for name in estimators
        }

    total = len(chi_list) * n_runs * len(estimators) * n_k
    completed = 0

    _write_experiment_multi_snapshot(
        json_path,
        kind=kind,
        N=N,
        d=d,
        chi_list=chi_list,
        n_runs=n_runs,
        base_seed=base_seed,
        dtype=dtype,
        estimators=estimators,
        k_vals=k_vals,
        trace_true=trace_true,
        results=results,
        runtimes=runtimes,
        completed=completed,
        total=total,
    )

    for chi in chi_list:
        for run_idx in range(n_runs):
            seed = base_seed + run_idx * 100_000
            for name in estimators:
                fn = ESTIMATORS.get(name) or RESPHERE_ESTIMATORS[name]
                desc = f"{kind}/chi={chi}/run{run_idx}/{name}"
                it = enumerate(k_vals)
                if progress:
                    it = tqdm(it, total=n_k, desc=desc)

                for i, k in it:
                    eval_seed = int(seed + 10_000 + i)

                    t0 = time.perf_counter()
                    trace_hat = fn(
                        mpo,
                        n=N,
                        k=int(k),
                        probe_chi=int(chi),
                        seed=eval_seed,
                        d=int(d),
                        dtype=dtype,
                    )
                    dt = time.perf_counter() - t0

                    val = (
                        np.nan
                        if np.isnan(trace_hat)
                        else abs(trace_hat - trace_true) / (abs(trace_true) + 1e-300)
                    )

                    results[chi][name][run_idx, i] = val
                    runtimes[chi][name][run_idx, i] = dt
                    completed += 1

                    if progress and hasattr(it, "set_postfix"):
                        it.set_postfix(
                            k=int(k),
                            rel_err=float(val),
                            t_s=float(dt),
                        )

                    if json_path is not None and (
                        flush_every <= 1
                        or completed % int(flush_every) == 0
                        or completed == total
                    ):
                        _write_experiment_multi_snapshot(
                            json_path,
                            kind=kind,
                            N=N,
                            d=d,
                            chi_list=chi_list,
                            n_runs=n_runs,
                            base_seed=base_seed,
                            dtype=dtype,
                            estimators=estimators,
                            k_vals=k_vals,
                            trace_true=trace_true,
                            results=results,
                            runtimes=runtimes,
                            completed=completed,
                            total=total,
                        )

    if return_runtimes:
        return k_vals, results, trace_true, runtimes
    return k_vals, results, trace_true


# ── top-level experiment entry point ──────────────────────────────────

def experiment(
    *,
    N=50,
    d=2,
    chi_list=(1, 2, 4, 8, 16),
    n_runs=60,
    matvec_list=None,
    base_seed=0,
    dtype=np.float64,
    estimators=None,
    kinds=("exp", "inverse_laplacian", "staircase"),
    out_dir="data",
    progress=True,
):
    """Run the full Figure 5 benchmark suite and save JSON results to out_dir.

    For each kind in `kinds`, calls `run_experiment_multi` and writes the
    result to ``out_dir/<kind>_<timestamp>.json`` using atomic .tmp-then-replace
    checkpointing.

    Parameters
    ----------
    N : int
        Number of sites (length of MPS/MPO).
    d : int
        Local physical dimension.
    chi_list : sequence of int
        Probe bond dimensions to sweep.
    n_runs : int
        Number of independent random runs per (chi, k) point.
    matvec_list : sequence of int or None
        Query counts to evaluate.  Defaults to ``int_linspace(16, 300, 16)``.
    base_seed : int
        Base random seed.
    dtype : numpy dtype
        Floating-point precision.
    estimators : list of str or None
        Estimator names from ESTIMATORS.  Defaults to all three.
    kinds : sequence of str
        Benchmark kinds to run.
    out_dir : str or Path
        Directory in which JSON snapshots are written.
    progress : bool
        Whether to show tqdm progress bars.

    Returns
    -------
    all_results : dict[str kind, tuple(k_vals, results, trace_true)]
    """
    import datetime

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for kind in kinds:
        json_path = out_dir / f"{kind}_{timestamp}.json"
        k_vals, results, trace_true = run_experiment_multi(
            kind,
            N=N,
            d=d,
            chi_list=chi_list,
            n_runs=n_runs,
            matvec_list=matvec_list,
            base_seed=base_seed,
            dtype=dtype,
            estimators=estimators,
            progress=progress,
            json_path=json_path,
        )
        all_results[kind] = (k_vals, results, trace_true)

    return all_results
