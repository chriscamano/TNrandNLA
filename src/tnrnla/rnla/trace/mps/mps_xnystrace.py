import numpy as np
import scipy.linalg as sla

from tnrnla.linalg.utils import sym
from tnrnla.tn.trp import TRP, MPS
from ._helpers import apply_oracle_to_trp

def mps_xnystrace_eigh(
    oracle,
    *,
    k,
    chi=None,
    n=None,
    Om=None,
    rmps_d=2,
    seed=None,
    base_seed=0,
    seed_offset=0,
    dtype=np.float64,
    compress_cutoff=1e-14,
    resphere=True,
    denom_eps=1e-300,
):
    rng = np.random.default_rng(seed if seed is not None else base_seed + seed_offset)

    if Om is None:
        if n is None or chi is None:
            raise ValueError("Need n and chi when Om is not provided.")
        Om = TRP.gaussian(
            n_sites=n,
            k=k,
            chi=chi,
            d=rmps_d,
            dtype=dtype,
            rng=rng,
        )
    elif hasattr(Om, "k") and Om.k != k:
        raise ValueError(f"Expected Om.k = {k}, got {Om.k}.")

    n = Om.n
    N = float(2.0 ** n)
    out_dtype = np.result_type(dtype, np.float64)
    I = np.eye(k, dtype=out_dtype)

    Y = oracle @ Om
    G0 = sym(Om.gram(Om, hermitian=True))
    S0 = sym(Om.gram(Y, hermitian=True))

    real_dtype = np.real(np.asarray(G0)).dtype
    finfo = np.finfo(real_dtype)

    scale = max(
        float(np.max(np.abs(G0))) if G0.size else 0.0,
        float(np.max(np.abs(S0))) if S0.size else 0.0,
        float(finfo.tiny),
    )
    inv_sqrt_scale = 1.0 / np.sqrt(scale)

    G = sym(G0 / scale)
    S = sym(S0 / scale)

    T_scaled = np.linalg.cholesky(G).conj().T
    Tinv_scaled = np.linalg.solve(T_scaled, I)

    B = sym(Tinv_scaled.conj().T @ S @ Tinv_scaled)
    lam = np.linalg.eigvalsh(B)
    mu = max(0.0, -float(lam[0])) + k * finfo.eps * float(np.max(np.abs(lam)))

    Ysh = Y + mu * Om

    R_scaled = np.linalg.cholesky(sym(S + mu * G)).conj().T
    invR_scaled = np.linalg.solve(R_scaled, I)

    Tinv = inv_sqrt_scale * Tinv_scaled
    invR = inv_sqrt_scale * invR_scaled
    M = sym(invR @ invR.conj().T)

    row_scale_sq = 1.0 / np.maximum(
        np.real(np.sum(np.abs(invR) ** 2, axis=1)),
        denom_eps,
    )

    if resphere:
        alpha = (N - k + 1.0) * np.real(np.sum(np.abs(Tinv.conj().T) ** 2, axis=0))
    else:
        alpha = np.ones(k, dtype=np.float64)

    coeffs = np.hstack((invR, M))
    norms_sq = np.maximum(
        np.real(
            _lincomb_srcqb_sqnorms(
                Ysh.cols,
                coeffs,
                cutoff=compress_cutoff,
                sketchdim=2 * max(A.max_bond() for A in Ysh.cols),
                rng=rng,
            )
        ),
        0.0,
    )

    tr_vec = (
        float(np.sum(norms_sq[:k]))
        + row_scale_sq * (alpha - norms_sq[k:])
        - mu * N
    )

    tr = float(np.mean(tr_vec))
    err = float(np.std(tr_vec, ddof=1) / np.sqrt(k)) if k > 1 else 0.0
    return tr, err



def mps_xnystrace_chol_gram(
    oracle,
    *,
    n=None,
    k=None,
    chi=None,
    seed=None,
    dtype=np.float64,
    trp_d=2,
    trp_orth=False,
    trp_normalize=False,
    Omega=None,
    Y=None,
    denom_eps=1e-300,
    resphere=False,
    trp_scaled=True,
):
    """
    Gram-based MPS XNysTrace for PSD operators.

    Assumes TRP.gaussian(..., k=k) uses column scaling 1/sqrt(k) when
    trp_scaled=True. In that case the non-resphered correction is k / d_i.

    If Y is provided (a TRP of pre-computed oracle outputs), the oracle
    is not called and oracle may be None. This enables incremental sketching
    where oracle calls are accumulated across checkpoints without redundancy.
    """
    if Omega is None:
        if n is None or k is None:
            raise ValueError("n and k are required when Omega is not provided")
        seed = 0 if seed is None else seed
        Omega = TRP.gaussian(
            n_sites=n,
            k=k,
            chi=chi,
            d=trp_d,
            seed=seed + 1,
            dtype=dtype,
            orth=trp_orth,
            normalize=trp_normalize,
        )
    elif not hasattr(Omega, "k"):
        raise ValueError("Omega must have attribute `k`")

    k = int(Omega.k)

    if n is None:
        if not hasattr(Omega, "n_sites"):
            raise ValueError("n is required when it cannot be inferred from Omega")
        n = int(Omega.n_sites)

    Y = apply_oracle_to_trp(oracle, Omega) if Y is None else Y

    G = sym(Omega.gram())
    C = sym(Omega.gram(rhs=Y))
    T = sym(Y.gram())

    eps = np.finfo(np.asarray(C).real.dtype).eps

    R = sla.cholesky(G, lower=False, check_finite=False)
    Rinv = sla.solve_triangular(
        R,
        np.eye(k, dtype=R.dtype),
        lower=False,
        check_finite=False,
    )

    # B = sym(Rinv.conj().T @ C @ Rinv)
    # lam = np.linalg.eigvalsh(B)
    # mu = max(0.0, -float(lam[0])) + np.sqrt(eps) * abs(float(lam[-1]))

    B = sym(Rinv.conj().T @ C @ Rinv)
    lam = np.linalg.eigvalsh(B)
    lam_min = float(lam[0])
    lam_max = float(lam[-1])
    tau = np.sqrt(eps) * abs(lam_max)
    mu = max(0.0, tau - lam_min)

    H = sym(C + mu * G)
    S = sym(T + mu * (2.0 * C + mu * G))

    L = sla.cholesky(H, lower=True, check_finite=False)
    M = sym(
        sla.cho_solve(
            (L, True),
            np.eye(k, dtype=H.dtype),
            check_finite=False,
        )
    )

    d = np.maximum(np.real(np.diag(M)), denom_eps)
    u = np.real(np.diag(M @ S @ M))
    t = float(np.real(np.trace(S @ M)))
    N = float(trp_d) ** n

    if resphere:
        w = np.maximum(np.real(np.sum(np.abs(Rinv) ** 2, axis=1)), denom_eps)
        corr = (N - k + 1.0) * w / d
    else:
        corr = ((float(k) if trp_scaled else 1.0) / d)

    samples = np.real_if_close(t - u / d + corr - mu * N)
    trace_est = float(np.mean(samples))
    stderr = 0.0 if k == 1 else float(np.std(samples, ddof=1) / np.sqrt(k))

    return trace_est, stderr

def mps_xnystrace_eigh_gram(
    oracle,
    *,
    n=None,
    k=None,
    chi=None,
    seed=None,
    dtype=np.float64,
    trp_d=2,
    trp_orth=True,
    trp_normalize=False,
    Omega=None,
    denom_eps=1e-300,
    resphere=False,
    eig_rel_tau=5.0,
):
    if Omega is None:
        if n is None or k is None:
            raise ValueError("n and k are required when Omega is not provided")

        base_seed = 0 if seed is None else seed
        Omega = TRP.gaussian(
            n_sites=n,
            k=k,
            chi=chi,
            d=trp_d,
            seed=base_seed + 1,
            dtype=dtype,
            orth=trp_orth,
            normalize=trp_normalize,
        )
    else:
        if not hasattr(Omega, "k"):
            raise ValueError("Omega must be a TRP-like object with attribute `k`")
        k = int(Omega.k)
        if n is None:
            if hasattr(Omega, "n_sites"):
                n = int(Omega.n_sites)
            else:
                raise ValueError("n is required when it cannot be inferred from Omega")

    # Apply the oracle to the sketch.
    Y = apply_oracle_to_trp(oracle, Omega)

    # Raw Gram matrices
    #   G = Omega^* Omega
    #   K = Omega^* Y
    #   Q = Y^* Y
    G = sym(Omega.gram())
    K = sym(Omega.gram(rhs=Y))
    Q = sym(Y.gram())

    # Cholesky factor of G, with G = T^* T since lower=False.
    T = sla.cholesky(G, lower=False, check_finite=False)
    eye_k = np.eye(k, dtype=T.dtype)
    Tinv = sla.solve_triangular(T, eye_k, lower=False, check_finite=False)

    # Correct whitened matrix
    #   B = T^{- *} K T^{-1}
    B = sym(Tinv.T.conj() @ K @ Tinv)

    eigvals_B = np.linalg.eigvalsh(B)
    eps = np.finfo(np.real(np.asarray(G)).dtype).eps

    # Spectral shift
    #   mu = max(0, -lambda_min(B)) + sqrt(eps) * lambda_max(B)
    lam_min_B = float(np.real(eigvals_B[0]))
    lam_max_B = float(np.real(eigvals_B[-1]))
    mu = max(0.0, -lam_min_B) + np.sqrt(eps) * lam_max_B

    # Shifted Gram quantities
    #   H = K + mu G
    #   S = Q + 2 mu K + mu^2 G
    H = sym(K + mu * G)
    S = sym(Q + 2.0 * mu * K + (mu * mu) * G)

    # Whitened shifted core
    #   Ccore = T^{- *} H T^{-1} = B + mu I
    Ccore = sym(Tinv.T.conj() @ H @ Tinv)

    evals_C, U = np.linalg.eigh(Ccore)
    lam_max_C = float(np.real(evals_C[-1]))
    eig_cutoff = float(eig_rel_tau) * eps * lam_max_C
    keep = evals_C > eig_cutoff
    eig_rank = int(np.count_nonzero(keep))

    if eig_rank == 0:
        return float("nan"), float("nan")

    U_r = U[:, keep]
    lam_r = np.real(evals_C[keep])

    # Build Z so that
    #   H^{-1} = Z Z^*
    # with
    #   Z = T^{-1} U_r diag(lam_r^{-1/2})
    R = U_r * (1.0 / np.sqrt(lam_r))[None, :]
    Z = Tinv @ R

    # t1 = tr(S H^{-1}) = tr(Z^* S Z)
    SZ = S @ Z
    Csmall = Z.T.conj() @ SZ
    Csmall = sym(Csmall)
    t1 = float(np.real(np.trace(Csmall)))

    # d_i = (H^{-1})_{ii} = ||row_i(Z)||^2
    d = np.sum(np.abs(Z) ** 2, axis=1)
    d_floor = max(
        float(denom_eps),
        float(np.sqrt(np.finfo(d.dtype).tiny)),
        float(np.finfo(d.dtype).eps * max(np.max(d), 1.0)),
    )
    d = np.maximum(d, d_floor)

    # diag(M) with M = H^{-1} S H^{-1} = Z Csmall Z^*
    ZC = Z @ Csmall
    diagM = np.sum(np.conj(Z) * ZC, axis=1)
    diagM = np.real_if_close(diagM)

    N = trp_d ** n

    if resphere:
        # diag(G^{-1}) = diag(T^{-1} T^{- *})
        ginv_diag = np.sum(np.abs(Tinv) ** 2, axis=1)
        ginv_diag = np.maximum(np.real_if_close(ginv_diag), d_floor)

        trace_samples = t1 - np.real(diagM) / d
        trace_samples += (N - k + 1.0) * ginv_diag / d
        trace_samples -= mu * N
    else:
        trace_samples = t1 - np.real(diagM) / d
        trace_samples += 1.0 / d
        trace_samples -= mu * N

    trace_samples = np.real_if_close(trace_samples)

    trace_est = float(np.mean(trace_samples))
    stderr = float(np.std(trace_samples, ddof=1) / np.sqrt(k)) if k > 1 else 0.0

    return trace_est, stderr