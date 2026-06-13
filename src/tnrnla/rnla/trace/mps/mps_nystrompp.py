
import numpy as np
import scipy.linalg as sla

from tnrnla.tn.trp import TRP
from tnrnla.tn.mpo import MPO
from tnrnla.tn.stopping import Cutoff
from ._helpers import apply_oracle_to_trp, sample_rmps_cols, trp_row_inner_fast
from tnrnla.linalg.utils import sym, safe_cholesky_psd
from tnrnla.tn.contraction.src_lincomb import trp_add_SRC_pairwise


def mps_npp_gram(
    oracle,
    num_queries,
    *,
    n=None,
    probe_chi=None,
    seed=None,
    dtype=np.float64,
    stop_cutoff=1e-14,  # kept for API compatibility
    trp_d=2,
    trp_orth=False,
    trp_normalize=False,
    Omega=None,
    Psi=None,
    resphere=False,
):
    """
    Gram-based MPS Nyström++ trace estimator with optional resphering.

    This version uses the stable shifted Nyström approximation directly,
        Ahat = Y_nu (Omega^* Y_nu)^{-1} Y_nu^* - nu I,
    where
        Y_nu = (A + nu I) Omega = Y + nu Omega.

    With resphering enabled, the residual contribution is averaged after
    weighting. That is the correct scaling when the TRP probes already carry
    the built-in 1/sqrt(m) normalization.
    """
    if (Omega is None) != (Psi is None):
        raise ValueError("Omega and Psi must be provided together")

    # ========= probe generation =========
    if Omega is None:
        if n is None:
            raise ValueError("n is required when Omega and Psi are not provided")
        if probe_chi is None:
            raise ValueError(
                "probe_chi is required when Omega and Psi are not provided"
            )

        N = trp_d ** n
        q = int(num_queries)
        if q < 2:
            raise ValueError("num_queries must be at least 2")

        q = min(q, 2 * N)
        q -= (q & 1)

        k = q // 2
        m = q - k
        base = 0 if seed is None else seed

        Omega = TRP.gaussian(
            n_sites=n,
            k=k,
            chi=probe_chi,
            d=trp_d,
            seed=base + 1,
            dtype=dtype,
            orth=trp_orth,
            normalize=trp_normalize,
        )
        Psi = TRP.gaussian(
            n_sites=n,
            k=m,
            chi=probe_chi,
            d=trp_d,
            seed=base + 2,
            dtype=dtype,
            orth=trp_orth,
            normalize=trp_normalize,
        )
    else:
        k = int(Omega.k)
        m = int(Psi.k)

        if n is None:
            if hasattr(Omega, "n_sites"):
                n = int(Omega.n_sites)
            else:
                raise ValueError(
                    "n is required when Omega and Psi do not expose n_sites"
                )

        N = trp_d ** n

    if k <= 0 or m <= 0:
        raise ValueError("Both Omega and Psi must have at least one column")

    # ========= oracle application =========
    Y = apply_oracle_to_trp(oracle, Omega)

    # ========= Gram data =========
    G = sym(Omega.gram())
    K = sym(Omega.gram(rhs=Y))
    YY = sym(Y.gram())

    OPsi = Omega.gram(rhs=Psi)
    YPsi = Y.gram(rhs=Psi)
    PsiPsi = sym(Psi.gram())

    work_dtype = np.result_type(dtype, np.float64)
    eps = np.finfo(work_dtype).eps
    tiny = np.finfo(work_dtype).tiny

    # ========= stable shifted Nyström =========
    # G = Omega^* Omega
    # K = Omega^* A Omega
    # Y_nu = (A + nu I) Omega
    # M = Omega^* Y_nu = K + nu G
    I = np.eye(k, dtype=work_dtype)

    Rg = sla.cholesky(G, lower=False, check_finite=False)
    Rg_inv = sla.solve_triangular(
        Rg,
        I,
        lower=False,
        check_finite=False,
    )

    B = sym(Rg_inv.conj().T @ K @ Rg_inv)
    lam = np.linalg.eigvalsh(B)

    nu = max(0.0, -float(np.real(lam[0]))) + np.sqrt(eps) * float(
        max(np.real(lam[-1]), 0.0)
    )

    M = sym(K + nu * G)
    C = sla.cholesky(M, lower=False, check_finite=False)

    # S = Y_nu^* Y_nu
    S = sym(YY + 2.0 * nu * K + (nu**2) * G)

    # tr(Ahat) = tr(M^{-1} S) - nu * N
    MinvS = sla.cho_solve((C, False), S, check_finite=False)
    t1 = float(
        np.longdouble(np.real(np.trace(MinvS))) - np.longdouble(nu) * np.longdouble(N)
    )

    # ========= quadratic forms of the low-rank piece =========
    # For each psi_j,
    # psi_j^* Ahat psi_j = || C^{- *} Y_nu^* psi_j ||^2 - nu ||psi_j||^2
    psi_norm2 = np.real(np.diag(PsiPsi)).astype(np.float64, copy=False)

    YnuPsi = YPsi + nu * OPsi
    FstarPsi = sla.solve_triangular(
        C.conj().T,
        YnuPsi,
        lower=True,
        check_finite=False,
    )

    lowrank_samples = np.sum(
        np.abs(FstarPsi) ** 2,
        axis=0,
        dtype=np.longdouble,
    ) - np.longdouble(nu) * psi_norm2.astype(np.longdouble, copy=False)

    # ========= Hutch residual samples =========
    if isinstance(oracle, MPO):
        quad_samples_raw = Psi.quadform(oracle, dtype=dtype)
    else:
        quad_samples_raw = np.fromiter(
            (float(np.real_if_close(oracle.quadform(psi))) for psi in Psi.cols),
            dtype=np.float64,
            count=m,
        )

    residual_samples = (
        np.asarray(quad_samples_raw, dtype=np.longdouble) - lowrank_samples
    )

    # ========= resphering =========
    if resphere:
        codim = N - k

        if codim <= 0:
            correction = 0.0
        else:
            # denom_j = ||(I - P_Omega) psi_j||^2
            #         = ||psi_j||^2 - (Omega^* psi_j)^* G^{-1} (Omega^* psi_j)
            Ginv_OPsi = sla.cho_solve((Rg, False), OPsi, check_finite=False)
            proj_norm2 = np.real(
                np.sum(np.conj(OPsi) * Ginv_OPsi, axis=0)
            ).astype(np.float64, copy=False)

            denom = np.maximum(psi_norm2 - proj_norm2, tiny)
            alpha = np.longdouble(codim) / denom.astype(np.longdouble, copy=False)

            # Important
            # The TRP probes already include the 1/sqrt(m) scaling.
            # After resphering, alpha removes that cancellation, so we must
            # average the weighted residual samples rather than summing them.
            residual_samples = alpha * residual_samples
            correction = float(np.mean(residual_samples, dtype=np.longdouble))
    else:
        # Without resphering, the built-in 1/sqrt(m) probe scaling means the sum
        # already equals the Monte Carlo average in the unscaled convention.
        correction = float(np.sum(residual_samples, dtype=np.longdouble))

    # ========= final accumulation =========
    tr_hat = float(np.longdouble(t1) + np.longdouble(correction))
    return float(np.real(tr_hat)), 0.0

def mps_npp_chol(
    oracle,
    num_queries,
    *,
    n=None,
    probe_chi=None,
    seed=None,
    dtype=np.float64,
    trp_d=2,
    trp_orth=False,
    trp_normalize=False,
    Omega=None,
    Psi=None,
    compress_cutoff=1e-14,
):
    rng = np.random.default_rng(seed)

    if Omega is None:
        q = int(num_queries)
        N_int = trp_d ** n
        q = min(q, 2 * N_int)
        q -= (q & 1)

        k = q // 2
        m = q - k

        base = 0 if seed is None else seed

        Omega = TRP.gaussian(
            n_sites=n,
            k=k,
            chi=probe_chi,
            d=trp_d,
            seed=base + 1,
            dtype=dtype,
            normalize=trp_normalize,
        )
        Psi = TRP.gaussian(
            n_sites=n,
            k=m,
            chi=probe_chi,
            d=trp_d,
            seed=base + 2,
            dtype=dtype,
            normalize=trp_normalize,
        )
    else:
        n = Omega[0].n
        k = int(Omega.k)
        m = int(Psi.k)

    N = float(trp_d ** n)
    out_dtype = np.result_type(dtype, np.float64)
    I = np.eye(k, dtype=out_dtype)

    Y = apply_oracle_to_trp(oracle, Omega)

    G0 = sym(Omega.gram())
    K0 = sym(Omega.gram(rhs=Y))

    real_dtype = np.real(np.asarray(G0)).dtype
    finfo = np.finfo(real_dtype)

    scale = max(
        float(np.max(np.abs(G0))) if G0.size else 0.0,
        float(np.max(np.abs(K0))) if K0.size else 0.0,
        float(finfo.tiny),
    )
    inv_sqrt_scale = 1.0 / np.sqrt(scale)

    G = sym(G0 / scale)
    K = sym(K0 / scale)

    T_scaled = np.linalg.cholesky(G).conj().T
    Tinv_scaled = sla.solve_triangular(
        T_scaled,
        I,
        lower=False,
        check_finite=False,
    )

    B = sym(Tinv_scaled.conj().T @ K @ Tinv_scaled)
    lam = np.linalg.eigvalsh(B)
    mu = max(0.0, -float(lam[0])) + finfo.eps * float(np.max(np.abs(lam)))

    if mu == 0.0:
        Ymu = Y
    else:
        Ymu = Y+mu*Omega

    H_scaled = sym(K + mu * G)
    R_scaled = np.linalg.cholesky(H_scaled).conj().T
    invR_scaled = sla.solve_triangular(
        R_scaled,
        I,
        lower=False,
        check_finite=False,
    )
    invR = inv_sqrt_scale * invR_scaled

    F_col_norms_sq = np.maximum(
        np.real(
            _lincomb_srcqb_sqnorms(
                Ymu.cols,
                invR,
                cutoff=compress_cutoff,
                sketchdim=16,
                sketchincrement=32,
                rng=rng,
            )
        ),
        0.0,
    )
    t1 = float(np.sum(F_col_norms_sq))

    if isinstance(oracle, MPO):
        tZ_raw = Psi.quadform(oracle, dtype=dtype)
    else:
        tZ_raw = np.fromiter(
            (float(np.real_if_close(oracle.quadform(psi))) for psi in Psi.cols),
            dtype=np.float64,
            count=m,
        )
    psi_norm_sq_raw = np.fromiter(
        (float(np.real_if_close(psi.inner_product(psi))) for psi in Psi.cols),
        dtype=np.float64,
        count=m,
    )

    amu_scaled = np.maximum((tZ_raw + mu * psi_norm_sq_raw) / scale, 0.0)
    Bpsi_scaled = Ymu.gram(rhs=Psi) / scale

    U = sla.solve_triangular(
        R_scaled.conj().T,
        Bpsi_scaled,
        lower=True,
        check_finite=False,
    )
    corr_scaled = np.real(np.sum(np.abs(U) ** 2, axis=0))
    resid_samples = scale * np.maximum(amu_scaled - corr_scaled, 0.0)

    tr = float(t1 - mu * N + np.mean(resid_samples))
    err = float(np.std(resid_samples, ddof=1) / np.sqrt(m)) if m > 1 else 0.0
    return tr, err