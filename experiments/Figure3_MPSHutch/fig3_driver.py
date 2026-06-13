import copy
import json
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from tnrnla.quantum.hamiltonians import classical_ising_exp
from tnrnla.rnla.trace.mps.mps_hutch import mps_hutch





# ── Exact trace formulas ──────────────────────────────────────────────────────
def _log_trace(n, beta):
    log2 = np.log(2.0)
    if beta >= 0.0:
        log_cosh = beta  - log2 + np.log1p(np.exp(-2.0 * beta))
    else:
        log_cosh = -beta - log2 + np.log1p(np.exp( 2.0 * beta))
    return log2 + (n - 1) * (log2 + log_cosh)


def _exact_trace(n, beta):
    log_tr = _log_trace(n, beta)
    return np.inf if log_tr > np.log(np.finfo(np.float64).max) else np.exp(log_tr)


# ── MPO helpers ───────────────────────────────────────────────────────────────
def _normalize_mpo(mpo, log_tr):
    """Scale MPO by exp(-log_tr) distributed evenly across sites."""
    mpo = copy.deepcopy(mpo)
    site_scale = np.exp(-log_tr / mpo.N)
    for i in range(mpo.N):
        mpo.tensors[i] = np.asarray(mpo.tensors[i]) * site_scale
    return mpo


# ── JSON helpers ──────────────────────────────────────────────────────────────
def _to_json_scalar(x):
    x = np.asarray(x).item()
    if isinstance(x, complex):
        return {"real": x.real, "imag": x.imag}
    return x  # float or int, already JSON-serializable


def _atomic_dump(payload, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


# ── Payload builder ───────────────────────────────────────────────────────────
def _build_payload(cfg, arrays, completed):
    rte, eps = arrays["rel_trace_err"], 1e-300
    return {
        **cfg,
        "est_trace_norm": arrays["est_trace_norm"].tolist(),
        "rel_trace_err":  rte.tolist(),
        "median_rte":     np.maximum(np.median(rte,            axis=2), eps).tolist(),
        "q10_rte":        np.maximum(np.quantile(rte, 0.10, axis=2), eps).tolist(),
        "q90_rte":        np.maximum(np.quantile(rte, 0.90, axis=2), eps).tolist(),
        "completed":      completed,
    }


# ── Main experiment ───────────────────────────────────────────────────────────
def run_experiment(
    n         = 50,
    beta_list = (0.1, 1.0, 10.0),
    chi_list  = (1, 25, 50, 75, 100),
    k_vals    = None,
    n_trials  = 60,
    base_seed = 123,
    out_dir   = "data",
):
    if k_vals is None:
        k_vals = np.unique(np.rint(np.geomspace(1, 200, 12)).astype(int))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(out_dir, f"fig3_results_{timestamp}.json")
    os.makedirs(out_dir, exist_ok=True)

    log_tr_list = [_log_trace(n, b)  for b in beta_list]
    tr_list     = [_exact_trace(n, b) for b in beta_list]
    A_list      = [_normalize_mpo(classical_ising_exp(n=n, beta=b, pauli="X"), lt)
                   for b, lt in zip(beta_list, log_tr_list)]

    for b, tr in zip(beta_list, tr_list):
        tr_str = f"{tr:.6e}" if np.isfinite(tr) else "overflow in float64"
        print(f"beta={b:>4}, exact_trace={tr_str}, normalized_target=1")

    cfg = {
        "n": n, "n_sites": A_list[0].N,
        "beta_list": list(beta_list), "chi_list": list(chi_list), "k_vals": [int(k) for k in k_vals],
        "n_trials": n_trials, "base_seed": base_seed,
        "log_tr_exact_list":      log_tr_list,
        "tr_exact_list":          [_to_json_scalar(x) for x in tr_list],
        "normalized_trace_target": 1.0,
    }

    shape  = (len(beta_list), len(chi_list), n_trials, len(k_vals))
    arrays = {"rel_trace_err": np.full(shape, np.nan), "est_trace_norm": np.full(shape, np.nan)}

    completed = {
        "beta_index": -1, "chi_index": -1, "trial_index": -1, "k_index": -1,
        "num_completed_points": 0, "filename": filename, "status": "running",
    }
    _atomic_dump(_build_payload(cfg, arrays, completed), filename)

    for bi, (beta, A) in enumerate(zip(beta_list, A_list)):
        for ci, chi in enumerate(chi_list):
            for trial in tqdm(range(n_trials), desc=f"beta={beta}, chi={chi}"):
                seed = base_seed + 100000 * bi + trial * 997 + chi * 37
                for ki, k in enumerate(k_vals):
                    est, _ = mps_hutch(A, n=A.N, probe_chi=chi, num_queries=k,
                                       base_seed=seed, dtype=np.float64, return_samples=False)
                    arrays["est_trace_norm"][bi, ci, trial, ki] = est
                    arrays["rel_trace_err"][bi, ci, trial, ki]  = abs(est - 1.0)
                    completed.update(beta_index=bi, chi_index=ci, trial_index=trial, k_index=ki)
                    completed["num_completed_points"] += 1
                    _atomic_dump(_build_payload(cfg, arrays, completed), filename)

    completed["status"] = "finished"
    payload = _build_payload(cfg, arrays, completed)
    _atomic_dump(payload, filename)
    print(f"Saved to {filename}")
    return payload


if __name__ == "__main__":
    run_experiment()
