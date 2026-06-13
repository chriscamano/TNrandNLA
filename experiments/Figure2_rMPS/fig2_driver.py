import os
N_THREADS = 8 
os.environ.update({
    "MKL_DEBUG_CPU_TYPE":   "5",
    "MKL_THREADING_LAYER":  "INTEL",
    "MKL_DYNAMIC":          "FALSE",
    "OMP_DYNAMIC":          "FALSE",
    "MKL_NUM_THREADS":      str(N_THREADS),
    "OMP_NUM_THREADS":      str(N_THREADS),
    "KMP_AFFINITY":         "granularity=fine,compact,1,0",
    "KMP_BLOCKTIME":        "1",
    "OPENBLAS_NUM_THREADS": "1",
    "BLIS_NUM_THREADS":     "1",
    "NUMEXPR_NUM_THREADS":  "1",
})
import numpy as np
import numpy.linalg as la
from tqdm.auto import tqdm
from tnrnla import MPS, MPO
from tnrnla.rnla.lra import nystrom
from tnrnla.rnla.lra.mps_nystrom import mps_nystrom


# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#
def nuc_sym(M):
    M = M.to_dense()
    M = 0.5 * (M + M.T)
    return float(np.sum(np.abs(la.eigvalsh(M))))


# -----------------------------------------------------------------------------#
# Variance estimation for quadratic forms
# -----------------------------------------------------------------------------#
def sample_quadforms_rmps(A_mpo, n, chi, nsamples=200, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    q = np.empty(nsamples, dtype=float)
    for t in range(nsamples):
        x_mps = MPS.rmps(n, chi, d=2, dtype=dtype, rng=rng)
        q[t] = float(np.real(A_mpo.quadform(x_mps)))
    return q


def sample_quadforms_gaussian(A, nsamples=2000, seed=0, dtype=np.float64):
    """Draw raw quadratic-form samples q = x^T A x from the Gaussian ensemble."""
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    q = np.empty(nsamples, dtype=float)
    for t in range(nsamples):
        x = rng.standard_normal(N).astype(dtype, copy=False)
        q[t] = float(x @ (A @ x))
    return q


def normalized_sample_variance(q, trace_value):
    """Return sample variance(q) / trace_value^2 with ddof=1."""
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size <= 1:
        return 0.0
    scale = float(trace_value) ** 2
    var_q = float(np.var(q, ddof=1))
    return var_q / scale if scale != 0.0 else var_q


# -----------------------------------------------------------------------------#
# Walsh-Hadamard helpers
# -----------------------------------------------------------------------------#
def walsh_bits(n, K):
    return [[(s >> j) & 1 for j in range(n)] for s in range(K)]


def local_projectors(bits, dtype=np.float64):
    inv = dtype(1.0 / np.sqrt(2.0))
    v0 = np.array([inv,  inv], dtype=dtype)
    v1 = np.array([inv, -inv], dtype=dtype)
    return [np.outer(v1 if b else v0, v1 if b else v0) for b in bits]


# -----------------------------------------------------------------------------#
# MPO construction  (A = hadamard_truncated_projector + eps * I)
# -----------------------------------------------------------------------------#
def hadamard_trunc_projector_mpo(n, K=50, eps=0.0, dtype=np.float64):
    eps = float(eps)
    if K <= 0 or K > 2**n:
        raise ValueError("need 1 <= K <= 2^n")

    P = [local_projectors(sb, dtype=dtype) for sb in walsh_bits(n, K)]
    I2 = np.eye(2, dtype=dtype)
    chi = K + (1 if eps != 0.0 else 0)

    if n == 1:
        A1 = np.zeros((1, 2, 2), dtype=dtype)
        for r in range(K):
            A1[0] += P[r][0]
        if eps != 0.0:
            A1[0] += eps * I2
        return MPO([A1], dtype=dtype)

    W = []

    # Left boundary
    W0 = np.zeros((2, chi, 2), dtype=dtype)
    for r in range(K):
        W0[:, r, :] = P[r][0]
    if eps != 0.0:
        W0[:, K, :] = eps * I2
    W.append(W0)

    # Bulk
    for site in range(1, n - 1):
        Wi = np.zeros((chi, 2, chi, 2), dtype=dtype)
        for r in range(K):
            Wi[r, :, r, :] = P[r][site]
        if eps != 0.0:
            Wi[K, :, K, :] = I2
        W.append(Wi)

    # Right boundary
    WN = np.zeros((chi, 2, 2), dtype=dtype)
    for r in range(K):
        WN[r, :, :] = P[r][n - 1]
    if eps != 0.0:
        WN[K, :, :] = I2
    W.append(WN)

    return MPO(W, dtype=dtype)


# -----------------------------------------------------------------------------#
# Dense Nystrom
# -----------------------------------------------------------------------------#
def nystrom_dense(A, k, rng, ridge, dtype):
    A_arr = np.asarray(A, dtype=dtype)
    U, d, _nu = nystrom(A_arr, n=A_arr.shape[0], k=k, rng=rng, form="svd", ridge=ridge)
    return (U * d[None, :]) @ U.T

# -----------------------------------------------------------------------------#
# MPS Nystrom
# -----------------------------------------------------------------------------#
def nystrom_trp_Ahat(A_mpo, n, k, chi, seed, ridge, dtype):
    U_trp, d = mps_nystrom(A_mpo, n=n, k=k, chi=chi, seed=seed)
    U_dense = np.empty((1 << n, len(d)), dtype=dtype)
    for j, col in enumerate(U_trp.cols):
        U_dense[:, j] = np.asarray(col.to_dense(), dtype=dtype).reshape(-1)
    return (U_dense * d[None, :]) @ U_dense.T


# -----------------------------------------------------------------------------#
# Main experiment
# -----------------------------------------------------------------------------#
def experiment(
    n,
    K=100,
    eps=1e-4,
    k_list=(4, 8, 16, 32, 64),
    chi_list=(1, 6, 12, 24),
    trials=5,
    seed=0,
    ridge=1e-12,
    dtype=np.float64,
    var_nsamples=200,
    gauss_nsamples=2000,
    chi_var_list=None,
    var_trials=None,
    run_nystrom=True,
    run_gauss=True,
):
    var_trials = trials if var_trials is None else var_trials

    print(f"[build] n={n} K={K} eps={eps:.2e} dtype={dtype}")
    print("[build] constructing MPO (A + eps*I)")

    A_mpo = hadamard_trunc_projector_mpo(n, K=K, eps=eps, dtype=dtype)
    A_mpo.show()
    A_mpo.round()
    A_mpo.show()

    trA = float(A_mpo.trace())
    nucA = trA
    print(f"[check] trace(A)={trA:.6e}  expected={K + eps * (1 << n):.6e}")

    if chi_var_list is None:
        chi_max = max(chi_list) if len(chi_list) else 1
        chi_var_list = list(range(1, chi_max + 1, 2))
    chi_var_list = sorted({c for c in chi_var_list if c >= 1})

    print("[stats] estimating Var(w^T A w) / tr(A)^2 with pooled rmps samples")

    total_var_samples = var_trials * var_nsamples
    quad_var_samples = {chi: np.empty(total_var_samples, dtype=float) for chi in chi_var_list}
    quad_var_runs = {chi: np.empty(var_trials, dtype=float) for chi in chi_var_list}

    for chi in tqdm(chi_var_list, desc="rmps var sweep", leave=False, dynamic_ncols=True):
        for r in range(var_trials):
            start = r * var_nsamples
            q_block = sample_quadforms_rmps(
                A_mpo, n=n, chi=chi, nsamples=var_nsamples,
                seed=seed + 271 * chi + 1009 * K + 10007 * n + 1000003 * (r + 1),
                dtype=dtype,
            )
            quad_var_samples[chi][start:start + var_nsamples] = q_block
            quad_var_runs[chi][r] = normalized_sample_variance(q_block, trA)

    quad_var_median = {chi: normalized_sample_variance(quad_var_samples[chi], trA)
                       for chi in quad_var_samples}

    if run_gauss or run_nystrom:
        print("[build] densifying A")
        A = A_mpo.to_dense()
    else:
        A = None

    if run_gauss:
        print("[stats] estimating Gaussian baseline Var(w^T A w) / tr(A)^2 with pooled samples")

        total_gauss_samples = var_trials * gauss_nsamples
        gauss_var_samples = np.empty(total_gauss_samples, dtype=float)
        gauss_var_runs = np.empty(var_trials, dtype=float)

        for r in tqdm(range(var_trials), desc="gaussian baseline", leave=False, dynamic_ncols=True):
            start = r * gauss_nsamples
            q_block = sample_quadforms_gaussian(
                A, nsamples=gauss_nsamples,
                seed=seed + 424242 + 1009 * K + 10007 * n + 2000003 * (r + 1),
                dtype=dtype,
            )
            gauss_var_samples[start:start + gauss_nsamples] = q_block
            gauss_var_runs[r] = normalized_sample_variance(q_block, trA)

        gauss_var_median = normalized_sample_variance(gauss_var_samples, trA)
    else:
        gauss_var_runs = np.empty(0, dtype=float)
        gauss_var_samples = np.empty(0, dtype=float)
        gauss_var_median = None
        total_gauss_samples = 0

    rng0 = np.random.default_rng(seed + 10007 * n + 9973 * K)
    res = {
        "n": n,
        "K": K,
        "eps": eps,
        "k_list": list(k_list),
        "chi_list": list(chi_list),
        "chi_var_list": chi_var_list,
        "trace_true": trA,
        "dense_median": [],
        "dense_p10": [],
        "dense_p90": [],
        "trp_median": {chi: [] for chi in chi_list},
        "trp_p10": {chi: [] for chi in chi_list},
        "trp_p90": {chi: [] for chi in chi_list},
        "quad_var_runs": quad_var_runs,
        "quad_var_samples": quad_var_samples,
        "quad_var_median": quad_var_median,
        "gauss_var_runs": gauss_var_runs,
        "gauss_var_samples": gauss_var_samples,
        "gauss_var_median": gauss_var_median,
        "var_nsamples": var_nsamples,
        "gauss_nsamples": gauss_nsamples,
        "var_trials": var_trials,
        "total_var_samples": total_var_samples,
        "total_gauss_samples": total_gauss_samples,
    }

    if run_nystrom:
        total = trials * (len(k_list) * (1 + len(chi_list)))
        with tqdm(total=total, desc="nystrom runs", leave=False, dynamic_ncols=True) as pbar:
            for k in k_list:
                print(f"[run] dense Nystrom k={k} trials={trials}")
                e = np.empty(trials, dtype=float)
                for t in range(trials):
                    rng = np.random.default_rng(rng0.integers(0, 2**63 - 1))
                    Ahat = nystrom_dense(A, k=k, rng=rng, ridge=ridge, dtype=dtype)
                    e[t] = nuc_sym(A - Ahat) / nucA
                    pbar.update(1)

                res["dense_median"].append(float(np.median(e)))
                res["dense_p10"].append(float(np.percentile(e, 10)))
                res["dense_p90"].append(float(np.percentile(e, 90)))

                for chi in chi_list:
                    print(f"[run] TRP Nystrom k={k} chi={chi} trials={trials}")
                    e2 = np.empty(trials, dtype=float)
                    for t in range(trials):
                        s = seed + 1000003 * (t + 1) + 97 * k + 13 * chi + 1009 * K
                        Ahat = nystrom_trp_Ahat(A_mpo, n=n, k=k, chi=chi, seed=s, ridge=ridge, dtype=dtype)
                        e2[t] = nuc_sym(A - Ahat) / nucA
                        pbar.update(1)

                    res["trp_median"][chi].append(float(np.median(e2)))
                    res["trp_p10"][chi].append(float(np.percentile(e2, 10)))
                    res["trp_p90"][chi].append(float(np.percentile(e2, 90)))

    print("[done] experiment finished")
    return res
