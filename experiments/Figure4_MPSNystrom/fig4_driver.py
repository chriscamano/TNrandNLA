import numpy as np
import scipy.linalg as sla

from tnrnla import Cutoff, MPS
from tnrnla.linalg.utils import sym
from tnrnla.tn.trp import TRP

from densityoracle import (
    make_halfchain_oracle,
    prep_halfchain,
    trp_round,
)


# ── Entropy / spectrum utilities ──────────────────────────────────────────────

def vn_entropy_from_eigs(lam, tol=1e-15):
    lam = np.real(np.asarray(lam, dtype=np.float64))
    lam = lam[np.isfinite(lam)]
    lam = np.clip(lam, 0.0, None)
    lam = lam[lam > tol]
    if lam.size == 0:
        return 0.0
    return float(-np.sum(lam * np.log(lam)))


def mpsCutEigsEntropy(psi, ell=None, tol=1e-14):
    N = int(psi.N)
    if ell is None:
        ell = N // 2
    ell = int(ell)
    if not (1 <= ell <= N - 1):
        raise ValueError("ell must satisfy 1 <= ell <= N - 1.")
    phi = psi.copy()
    phi.orthL()
    phi.move_pivot(ell - 1)
    A = np.asarray(phi[ell - 1])
    if A.ndim == 2:
        M = A
    elif A.ndim == 3:
        M = A.reshape(A.shape[0] * A.shape[1], A.shape[2])
    else:
        raise ValueError("Unexpected MPS tensor rank at the cut.")
    s = np.linalg.svd(M, compute_uv=False)
    eigs = s ** 2
    total = np.sum(eigs)
    if total <= 0:
        raise ValueError("MPS has zero norm at this cut.")
    eigs = eigs / total
    eigs = np.sort(eigs)[::-1]
    nz = eigs[eigs > tol]
    entropy = float(np.real(-np.sum(nz * np.log(nz))))
    return eigs, entropy


def exact_halfchain_spectrum(groundstate, ell):
    psi = prep_halfchain(groundstate, ell)
    A = np.asarray(psi[ell - 1])
    W = A.reshape(-1, A.shape[-1])
    s = np.linalg.svd(W, compute_uv=False)
    root_eps = np.sqrt(np.finfo(np.real(W).dtype).eps)
    cutoff = root_eps * float(np.max(np.abs(s))) if s.size else 0.0
    keep = s > cutoff
    if s.size and not np.any(keep):
        keep[0] = True
    lam = np.clip(np.real(s[keep]) ** 2, 0.0, None)
    return np.sort(np.asarray(lam, dtype=np.float64))[::-1]


def exact_halfchain_spectrum_entropy(groundstate, ell, tol=1e-15):
    lam = exact_halfchain_spectrum(groundstate, ell)
    S = vn_entropy_from_eigs(lam, tol=tol)
    return lam, float(S)


# ── Free-fermion theory ───────────────────────────────────────────────────────

def ptfimCutTheory(
    n,
    h,
    J=1.0,
    ell=None,
    start=0,
    numEigs=128,
    fullSpectrum=False,
    maxFullEll=24,
    tol=1e-14,
    compatible_with_mps_cut=False,
):
    n = int(n)
    h = float(h)
    J = float(J)
    start = int(start)
    if ell is None:
        ell = n // 2
    ell = int(ell)
    if n < 2:
        raise ValueError("n must be at least 2.")
    if not (1 <= ell <= n):
        raise ValueError("ell must satisfy 1 <= ell <= n.")
    if not (0 <= start < n):
        raise ValueError("start must satisfy 0 <= start < n.")

    A = np.zeros((2 * n, 2 * n), dtype=np.float64)
    for j in range(n):
        a, b = 2 * j, 2 * j + 1
        A[a, b] = 2.0 * h
        A[b, a] = -2.0 * h
    for j in range(n - 1):
        b = 2 * j + 1
        aNext = 2 * (j + 1)
        A[b, aNext] = 2.0 * J
        A[aNext, b] = -2.0 * J
    A[2 * n - 1, 0] = -2.0 * J
    A[0, 2 * n - 1] = 2.0 * J

    eigvals, U = np.linalg.eigh(1j * A)
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    signs = np.sign(np.real(eigvals))
    signs[np.abs(eigvals) <= tol * scale] = 0.0
    Gamma = 1j * ((U * signs[None, :]) @ U.conj().T)
    Gamma = np.real_if_close(Gamma, tol=1000)
    Gamma = np.asarray(np.real(Gamma), dtype=np.float64)
    Gamma = 0.5 * (Gamma - Gamma.T)

    sites = (start + np.arange(ell)) % n
    idx = np.empty(2 * ell, dtype=np.int64)
    idx[0::2] = 2 * sites
    idx[1::2] = 2 * sites + 1
    GammaA = Gamma[np.ix_(idx, idx)]

    vals = np.linalg.eigvalsh(1j * GammaA)
    vals = np.asarray(np.real(np.real_if_close(vals, tol=1000)), dtype=np.float64)
    vals = np.sort(vals)
    nu = np.clip(vals[ell:], 0.0, 1.0)

    p0 = 0.5 * (1.0 + nu)
    p1 = 0.5 * (1.0 - nu)
    entropy = 0.0
    for p in (p0, p1):
        q = p[p > tol]
        entropy -= np.sum(q * np.log(q))
    entropy = float(entropy)

    if fullSpectrum:
        if ell > maxFullEll:
            raise ValueError(
                f"fullSpectrum=True would form 2^{ell} eigenvalues. "
                f"Increase maxFullEll only if you really want this."
            )
        rdmEigs = np.array([1.0], dtype=np.float64)
        for x in nu:
            rdmEigs = np.kron(rdmEigs, np.array([0.5 * (1 + x), 0.5 * (1 - x)]))
        rdmEigs = np.sort(rdmEigs)[::-1]
    else:
        numEigs = int(numEigs)
        if numEigs <= 0:
            raise ValueError("numEigs must be positive.")
        rdmEigs = np.array([1.0], dtype=np.float64)
        for x in nu:
            q0, q1 = 0.5 * (1.0 + x), 0.5 * (1.0 - x)
            candidates = np.concatenate((rdmEigs * q0, rdmEigs * q1))
            if candidates.size > numEigs:
                keep = np.argpartition(candidates, -numEigs)[-numEigs:]
                rdmEigs = np.sort(candidates[keep])[::-1]
            else:
                rdmEigs = np.sort(candidates)[::-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        energies = np.log((1.0 + nu) / (1.0 - nu))

    entropy_out = 0.5 * entropy if bool(compatible_with_mps_cut) else entropy
    compatibility_mode = "mps_single_cut_entropy" if bool(compatible_with_mps_cut) else "periodic_block_rdm"

    return {
        "rdmEigs": rdmEigs,
        "entropy": float(entropy_out),
        "blockEntropy": float(entropy),
        "nu": nu,
        "energies": energies,
        "compatibilityMode": compatibility_mode,
    }


# ── Core Nyström solver ───────────────────────────────────────────────────────

def mps_nystrom(A, n, k, chi, seed=0, Om=None, *, tol=None, just_eigs=False):
    Omega = Om if Om is not None else TRP.gaussian(n_sites=n, k=k, chi=chi, seed=int(seed))
    Y = A(Omega)
    if tol is not None:
        trp_round(Y, tol)

    G = sym(np.asarray(Omega.gram(assume_hermitian=True)))
    C = sym(np.asarray(Omega.gram(Y, assume_hermitian=False)))
    Q = sym(np.asarray(Y.gram(Y, assume_hermitian=True)))

    root_eps = np.sqrt(np.finfo(np.real(G).dtype).eps)
    Ik = np.eye(k, dtype=G.dtype)
    T = sla.cholesky(G, lower=False, check_finite=False)
    Tinv = sla.solve_triangular(T, Ik, lower=False, check_finite=False)
    B = sym(Tinv.conj().T @ C @ Tinv)
    evals = np.real(np.linalg.eigvalsh(B))
    nu = max(0.0, root_eps * evals[-1] - evals[0])

    R = sla.cholesky(sym(C + nu * G), lower=False, check_finite=False)
    S = sym(Q + 2.0 * nu * C + nu * nu * G)
    Rinv = sla.solve_triangular(R, Ik, lower=False, check_finite=False)
    E = sym(Rinv.conj().T @ S @ Rinv)

    Lambda_nu, V = np.linalg.eigh(E)
    Lambda_nu = np.real(Lambda_nu[::-1])
    V = V[:, ::-1]
    Lambda = np.maximum(Lambda_nu - nu, 0.0)

    if just_eigs:
        return Lambda

    Y_nu = Y + nu * Omega
    W = sla.solve_triangular(R, V, lower=False, check_finite=False)
    W = W / np.sqrt(Lambda_nu)[None, :]
    U = Y_nu @ W
    if tol is not None:
        trp_round(U, tol)
    return U, Lambda


def mps_nystrom_halfchain(groundstate, ell, k, chi, seed=0, Om=None, *,
                          tol=1e-8, just_eigs=False):
    A = make_halfchain_oracle(groundstate, ell, tol=tol)
    return mps_nystrom(A, n=ell, k=k, chi=chi, seed=seed, Om=Om,
                       tol=tol, just_eigs=just_eigs)


# ── Streaming Nyström (standard) ─────────────────────────────────────────────

def mps_nystrom_streaming(
    A, n, k_values, chi, seed=0, *,
    tol=None, just_eigs=False, d=2,
    progress_bar=None, progress_desc=None, leave_progress=False,
):
    """
    Sweeps over sorted k_values, reusing all oracle evaluations and Gram entries
    accumulated at smaller k. Sketch columns are a deterministic prefix of the
    k_max sketch, so results are directly comparable across k.
    """
    k_values_sorted = sorted(set(int(kv) for kv in k_values))
    k_max = k_values_sorted[-1]
    ss = np.random.SeedSequence(int(seed))
    child_ss = ss.spawn(k_max)

    om_cols, y_cols = [], []
    G_raw = np.empty((0, 0), dtype=np.float64)
    C_raw = np.empty((0, 0), dtype=np.float64)
    Q_raw = np.empty((0, 0), dtype=np.float64)
    results = {}
    k_prev = 0

    k_iter = k_values_sorted
    internal_pbar = None
    if progress_bar is None and progress_desc is not None:
        try:
            from tqdm.auto import tqdm
            internal_pbar = tqdm(k_values_sorted, desc=str(progress_desc),
                                 leave=bool(leave_progress))
            k_iter = internal_pbar
        except Exception:
            pass

    for k in k_iter:
        new_om = [MPS.rmps(n, chi, d=d, rng=np.random.default_rng(child_ss[j]))
                  for j in range(k_prev, k)]
        Om_new = TRP(new_om, orthform=None)
        if tol is not None:
            trp_round(Om_new, tol)

        Y_new = A(Om_new)
        if tol is not None:
            trp_round(Y_new, tol)

        G_nn = sym(np.asarray(Om_new.gram(assume_hermitian=True)))
        C_nn = sym(np.asarray(Om_new.gram(Y_new, assume_hermitian=False)))
        Q_nn = sym(np.asarray(Y_new.gram(assume_hermitian=True)))

        if k_prev > 0:
            Om_old = TRP(list(om_cols), orthform="up")
            Y_old  = TRP(list(y_cols),  orthform="up")
            G_on = np.asarray(Om_old.gram(Om_new, assume_hermitian=False))
            C_on = np.asarray(Om_old.gram(Y_new,  assume_hermitian=False))
            Q_on = np.asarray(Y_old.gram(Y_new,   assume_hermitian=False))

            def _embed(old, cross, nn):
                out = np.empty((k, k), dtype=nn.dtype)
                out[:k_prev, :k_prev] = old
                out[:k_prev, k_prev:] = cross
                out[k_prev:, :k_prev] = cross.conj().T
                out[k_prev:, k_prev:] = nn
                return sym(out)

            G_raw = _embed(G_raw, G_on, G_nn)
            C_raw = _embed(C_raw, C_on, C_nn)
            Q_raw = _embed(Q_raw, Q_on, Q_nn)
        else:
            G_raw, C_raw, Q_raw = sym(G_nn), sym(C_nn), sym(Q_nn)

        om_cols.extend(Om_new.cols)
        y_cols.extend(Y_new.cols)

        scale_sq = 1.0 / float(k)
        G_eff = sym(scale_sq * G_raw)
        C_eff = sym(scale_sq * C_raw)
        Q_eff = sym(scale_sq * Q_raw)

        root_eps = np.sqrt(np.finfo(np.float64).eps)
        Ik = np.eye(k, dtype=np.float64)
        T_nu = sla.cholesky(G_eff, lower=False, check_finite=False)
        Tinv = sla.solve_triangular(T_nu, Ik, lower=False, check_finite=False)
        B = sym(Tinv.conj().T @ C_eff @ Tinv)
        evals = np.real(np.linalg.eigvalsh(B))
        nu = max(0.0, root_eps * evals[-1] - evals[0])

        R = sla.cholesky(sym(C_eff + nu * G_eff), lower=False, check_finite=False)
        S = sym(Q_eff + 2.0 * nu * C_eff + nu * nu * G_eff)
        Rinv = sla.solve_triangular(R, Ik, lower=False, check_finite=False)
        E = sym(Rinv.conj().T @ S @ Rinv)
        Lambda, V = np.linalg.eigh(E)
        Lambda = np.maximum(np.real(Lambda[::-1]) - nu, 0.0)
        V = V[:, ::-1]

        if just_eigs:
            results[k] = Lambda
        else:
            Om_k = TRP(list(om_cols), orthform="up")
            Y_k  = TRP(list(y_cols),  orthform="up")
            Y_nu = Y_k + nu * Om_k
            W = sla.solve_triangular(R, V, lower=False, check_finite=False)
            W = W / np.sqrt(Lambda)[None, :]
            U = Y_nu @ W
            if tol is not None:
                trp_round(U, tol)
            results[k] = (U, Lambda)

        k_prev = k
        if progress_bar is not None:
            progress_bar.update(1)

    if internal_pbar is not None:
        internal_pbar.close()
    return results


def mps_nystrom_halfchain_streaming(
    groundstate, ell, k_values, chi, seed=0, *,
    tol=None, just_eigs=False,
    progress_bar=None, progress_desc=None, leave_progress=False,
):
    d_phys = int(np.asarray(groundstate[0]).shape[0])
    A = make_halfchain_oracle(groundstate, ell, tol=tol)
    return mps_nystrom_streaming(
        A, n=ell, k_values=k_values, chi=chi, seed=seed,
        tol=tol, just_eigs=just_eigs, d=d_phys,
        progress_bar=progress_bar, progress_desc=progress_desc,
        leave_progress=leave_progress,
    )


# ── Main experiment ───────────────────────────────────────────────────────────

def experiment(
    groundstate,
    *,
    n,
    h,
    J=1.0,
    k_values,
    chi_values,
    n_trials=60,
    base_seed=1234,
    trial_seed_stride=10000,
    out_dir="data",
):
    """Run the streaming MPS Nyström experiment and return (results, payload).

    Parameters
    ----------
    groundstate : MPS
        DMRG ground state for an n-site periodic TFIM chain.
    n : int
        Number of sites.
    h, J : float
        TFIM transverse field and coupling.
    k_values : sequence of int
        Embedding dimensions to sweep (sorted and de-duplicated internally).
    chi_values : sequence of int
        Bond dimensions of the random MPS sketches.
    n_trials : int
        Number of independent random trials per (chi, k) pair.
    base_seed, trial_seed_stride : int
        RNG seeds: trial i with bond dim chi uses
        ``base_seed + trial_seed_stride * i + 1009 * chi``.
    out_dir : str or Path
        Directory for the JSON checkpoint file (created if absent).

    Returns
    -------
    results : dict[int, dict]
        Keyed by chi (int).  Each value contains relerr_trials,
        spectrum_trials, spectrum_median, etc.
    payload : dict
        Full JSON-serialisable payload (metadata + results).
    """
    import json
    from datetime import datetime
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ell        = n // 2
    k_eval     = np.array(sorted(set(int(k) for k in k_values)), dtype=int)
    spectrum_k = int(k_eval[-1])

    exact_spectrum, exact_entropy = exact_halfchain_spectrum_entropy(groundstate, ell)
    theory           = ptfimCutTheory(n=n, h=h, J=J, ell=ell, numEigs=spectrum_k)
    analytic_entropy = float(theory["entropy"])
    entropy_scale    = max(abs(exact_entropy), np.finfo(np.float64).tiny)

    results = {}
    payload = {
        "metadata": {
            "method":            "mps_nystrom_halfchain_streaming",
            "n":                 int(n),
            "J":                 float(J),
            "h":                 float(h),
            "ell":               int(ell),
            "n_trials":          int(n_trials),
            "chi_values":        [int(x) for x in chi_values],
            "k_values":          k_eval.tolist(),
            "spectrum_k":        spectrum_k,
            "exact_entropy":     float(exact_entropy),
            "analytic_entropy":  float(analytic_entropy),
            "entropy_scale":     float(entropy_scale),
            "base_seed":         int(base_seed),
            "trial_seed_stride": int(trial_seed_stride),
            "exact_spectrum":    exact_spectrum.tolist(),
        },
        "results": {},
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath   = out_dir / f"mps_nystrom_streaming_results_{timestamp}.json"

    for chi in chi_values:
        chi = int(chi)
        relerr_trials   = np.empty((n_trials, len(k_eval)), dtype=np.float64)
        entropy_trials  = np.empty((n_trials, len(k_eval)), dtype=np.float64)
        trace_trials    = np.empty((n_trials, len(k_eval)), dtype=np.float64)
        spectrum_trials = np.empty((n_trials, spectrum_k),  dtype=np.float64)

        for trial_idx in range(n_trials):
            seed_trial = base_seed + trial_seed_stride * trial_idx + 1009 * chi

            lam_by_k = mps_nystrom_halfchain_streaming(
                groundstate, ell, k_values=k_eval, chi=chi,
                seed=seed_trial, tol=None, just_eigs=True,
                progress_desc=f"stream chi={chi} trial={trial_idx+1}/{n_trials}",
                leave_progress=False,
            )

            for k_idx, k in enumerate(k_eval):
                d = np.asarray(lam_by_k[int(k)], dtype=np.float64)
                entropy_hat = vn_entropy_from_eigs(d)
                entropy_trials[trial_idx, k_idx]  = entropy_hat
                trace_trials[trial_idx, k_idx]    = float(np.sum(d))
                relerr_trials[trial_idx, k_idx]   = (
                    abs(entropy_hat - exact_entropy) / entropy_scale
                )

            d_spec = np.asarray(lam_by_k[spectrum_k], dtype=np.float64)
            d_spec = (d_spec[:spectrum_k] if d_spec.size >= spectrum_k
                      else np.pad(d_spec, (0, spectrum_k - d_spec.size)))
            spectrum_trials[trial_idx, :] = d_spec

        entry = {
            "relerr_trials":   relerr_trials.tolist(),
            "entropy_trials":  entropy_trials.tolist(),
            "trace_trials":    trace_trials.tolist(),
            "spectrum_trials": spectrum_trials.tolist(),
            "relerr_median":   np.median(relerr_trials, axis=0).tolist(),
            "relerr_lo":       np.quantile(relerr_trials, 0.25, axis=0).tolist(),
            "relerr_hi":       np.quantile(relerr_trials, 0.75, axis=0).tolist(),
            "spectrum_median": np.median(spectrum_trials, axis=0).tolist(),
        }
        results[chi] = entry
        payload["results"][str(chi)] = entry
        with outpath.open("w") as f:
            json.dump(payload, f, indent=2)

    print(f"Saved to {outpath.resolve()}")
    return results, payload
