import numpy as np
from scipy.linalg import eigh_tridiagonal, blas


def _canonical_krylov_dtype(dtype):
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.complexfloating):
        return np.dtype(np.complex128)
    return np.dtype(np.float64)


def _ensure_lanczos_work(work, n, k, vec_dtype=np.float64):
    n = int(n)
    k = int(k)
    vec_dtype = _canonical_krylov_dtype(vec_dtype)

    if work is None:
        return {
            "Q": np.empty((n, k), dtype=vec_dtype, order="F"),
            "alpha": np.empty((k,), dtype=np.float64),
            "beta": np.empty((max(k - 1, 0),), dtype=np.float64),
            "vbuf": np.empty((n,), dtype=vec_dtype),
        }

    Q = work.get("Q", None)
    alpha = work.get("alpha", None)
    beta = work.get("beta", None)
    vbuf = work.get("vbuf", None)

    good = (
        isinstance(Q, np.ndarray)
        and isinstance(alpha, np.ndarray)
        and isinstance(beta, np.ndarray)
        and isinstance(vbuf, np.ndarray)
        and Q.shape == (n, k)
        and alpha.shape == (k,)
        and beta.shape == (max(k - 1, 0),)
        and vbuf.shape == (n,)
        and Q.dtype == vec_dtype
        and alpha.dtype == np.float64
        and beta.dtype == np.float64
        and vbuf.dtype == vec_dtype
        and Q.flags["F_CONTIGUOUS"]
    )

    if good:
        return work

    new_work = {
        "Q": np.empty((n, k), dtype=vec_dtype, order="F"),
        "alpha": np.empty((k,), dtype=np.float64),
        "beta": np.empty((max(k - 1, 0),), dtype=np.float64),
        "vbuf": np.empty((n,), dtype=vec_dtype),
    }

    if isinstance(work, dict):
        work.clear()
        work.update(new_work)
        return work

    return new_work


def _apply_fun_to_tridiag(alpha, beta, *, fun=np.exp, t=1.0, lapack_driver="stev"):
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    evals, evecs = eigh_tridiagonal(
        alpha,
        beta,
        eigvals_only=False,
        lapack_driver=lapack_driver,
        check_finite=False,
    )

    fvals = fun(t * evals)
    u = evecs @ (fvals * evecs[0, :])
    return evals, u


def _exp_amplification_bound(evals, t):
    evals = np.asarray(evals, dtype=np.float64)
    t_real = float(np.real(t))
    return float(np.exp(np.max(t_real * evals)))


def lanczos_fun(
    v,
    matvec,
    *,
    fun=np.exp,
    k: int = 16,
    t=1.0,
    breakdown_tol: float = 1e-12,
    tol: float = 1e-10,
    rel_tol: float = 1e-8,
    change_tol: float = 1e-8,
    min_m: int = 12,
    check_every: int = 6,
    reorth: str = "none",
    auto_eta: float = 0.5,
    force_full_depth: bool = False,
    return_info: bool = False,
    work: dict | None = None,
    lapack_driver: str = "stev",
):
    """
    Lanczos action of f(tA) on v for real or complex Hermitian inputs.

    Assumptions
    - v may be float64 or complex128
    - matvec(v) returns matching dtype and represents a Hermitian operator
    - reorth is disabled
    - this is intended for the fast TDVP/GSE hot path
    """
    del auto_eta

    if reorth != "none":
        raise ValueError("This fast lanczos_fun only supports reorth='none'")

    v0 = np.asarray(v)
    vec_dtype = _canonical_krylov_dtype(v0.dtype)

    if isinstance(matvec, np.ndarray):
        A = np.asarray(matvec, dtype=vec_dtype)
        matvec = A.dot

    v = np.asarray(v0, dtype=vec_dtype)

    k = int(k)
    n = int(v.size)
    min_m = int(min_m)
    check_every = int(check_every)
    tiny = np.finfo(np.float64).tiny

    work = _ensure_lanczos_work(work, n, k, vec_dtype=vec_dtype)
    Q = work["Q"]
    alpha = work["alpha"]
    beta = work["beta"]
    vbuf = work["vbuf"]

    nrm2 = blas.get_blas_funcs("nrm2", arrays=(v,))
    dot = np.vdot

    gamma = float(nrm2(v))
    if gamma == 0.0:
        out = np.zeros_like(v)
        if return_info:
            return out, {
                "m_stop": 0,
                "stop_reason": "zero_input",
                "err_abs": 0.0,
                "err_rel": 0.0,
                "change_rel": 0.0,
                "amp_bound": 1.0,
                "stopped_early": True,
            }
        return out

    Q[:, 0] = v
    Q[:, 0] *= (1.0 / gamma)

    beta_prev = 0.0
    prev_u = None
    prev_m = None
    final_u = None
    final_evals = None
    final_err_abs = None
    final_err_rel = None
    final_amp_bound = None

    if return_info:
        last_info = {
            "m_stop": k,
            "stop_reason": "max_depth",
            "err_abs": np.nan,
            "err_rel": np.nan,
            "change_rel": np.nan,
            "amp_bound": np.nan,
            "stopped_early": False,
        }

    for j in range(k):
        qj = Q[:, j]
        np.copyto(vbuf, qj)

        z = matvec(vbuf)

        if not (isinstance(z, np.ndarray) and z.dtype == vec_dtype and z.ndim == 1):
            z = np.asarray(z, dtype=vec_dtype)

        aj = float(np.real(dot(qj, z)))
        alpha[j] = aj

        z -= aj * qj
        if j > 0:
            z -= beta_prev * Q[:, j - 1]

        bj = float(nrm2(z))
        m = j + 1
        broke = bj < float(breakdown_tol) * gamma

        if (j < k - 1) and (not broke):
            beta[j] = bj
            Q[:, j + 1] = z
            Q[:, j + 1] *= (1.0 / bj)
            beta_prev = bj

        do_check = (m >= min_m) and ((m - min_m) % check_every == 0)
        need_projected_solve = broke or (m == k) or (do_check and (not force_full_depth))

        if not need_projected_solve:
            continue

        evals_m, u_m = _apply_fun_to_tridiag(
            alpha[:m],
            beta[: m - 1],
            fun=fun,
            t=t,
            lapack_driver=lapack_driver,
        )

        y_norm = gamma * float(np.linalg.norm(u_m))
        amp_bound = float(_exp_amplification_bound(evals_m, t)) if fun is np.exp else 1.0

        if m == 1:
            err_abs = np.nan
            err_rel = np.nan
            change_rel = np.inf
        else:
            if j < k - 1:
                residual_coeff = bj * float(abs(u_m[-1]))
            else:
                residual_coeff = float(beta[m - 2]) * float(abs(u_m[-1]))

            err_abs = gamma * residual_coeff
            if fun is np.exp:
                err_rel = amp_bound * err_abs / max(y_norm, tiny)
            else:
                err_rel = err_abs / max(y_norm, tiny)

            if prev_u is None:
                change_rel = np.inf
            else:
                du = np.empty((m,), dtype=u_m.dtype)
                du[:prev_m] = u_m[:prev_m] - prev_u
                du[prev_m:m] = u_m[prev_m:m]
                change_rel = gamma * float(np.linalg.norm(du)) / max(y_norm, tiny)

        # If we already solved the projected problem at full depth, cache it and
        # reuse after the loop to avoid a duplicate tridiagonal eigensolve.
        if m == k:
            final_u = u_m
            final_evals = evals_m
            final_err_abs = err_abs
            final_err_rel = err_rel
            final_amp_bound = amp_bound

        if broke:
            out = gamma * (Q[:, :m] @ u_m)
            if return_info:
                last_info = {
                    "m_stop": m,
                    "stop_reason": "breakdown",
                    "err_abs": 0.0 if np.isnan(err_abs) else float(err_abs),
                    "err_rel": 0.0 if np.isnan(err_rel) else float(err_rel),
                    "change_rel": 0.0 if np.isinf(change_rel) else float(change_rel),
                    "amp_bound": float(amp_bound),
                    "stopped_early": True,
                }
                return out, last_info
            return out

        if do_check and (not force_full_depth):
            rel_ok = (not np.isnan(err_rel)) and (err_rel <= float(rel_tol))
            abs_ok = (not np.isnan(err_abs)) and (err_abs <= float(tol))
            chg_ok = np.isfinite(change_rel) and (change_rel <= float(change_tol))

            if rel_ok and chg_ok:
                out = gamma * (Q[:, :m] @ u_m)
                if return_info:
                    last_info = {
                        "m_stop": m,
                        "stop_reason": "adaptive",
                        "err_abs": float(err_abs),
                        "err_rel": float(err_rel),
                        "change_rel": float(change_rel),
                        "amp_bound": float(amp_bound),
                        "stopped_early": True,
                    }
                    return out, last_info
                return out

            if abs_ok and chg_ok:
                out = gamma * (Q[:, :m] @ u_m)
                if return_info:
                    last_info = {
                        "m_stop": m,
                        "stop_reason": "adaptive_abs_change",
                        "err_abs": float(err_abs),
                        "err_rel": float(err_rel),
                        "change_rel": float(change_rel),
                        "amp_bound": float(amp_bound),
                        "stopped_early": True,
                    }
                    return out, last_info
                return out

            prev_u = np.array(u_m, copy=True)
            prev_m = m

    if final_u is not None:
        u_k = final_u
        evals_k = final_evals
    else:
        evals_k, u_k = _apply_fun_to_tridiag(
            alpha[:k],
            beta[: k - 1],
            fun=fun,
            t=t,
            lapack_driver=lapack_driver,
        )
    out = gamma * (Q[:, :k] @ u_k)

    if return_info:
        if final_u is not None:
            err_abs = final_err_abs
            err_rel = final_err_rel
            amp_bound = final_amp_bound
        else:
            y_norm = gamma * float(np.linalg.norm(u_k))
            if k > 1:
                residual_coeff = float(beta[k - 2]) * float(abs(u_k[-1]))
                err_abs = gamma * residual_coeff
            else:
                err_abs = 0.0

            if fun is np.exp:
                amp_bound = float(_exp_amplification_bound(evals_k, t))
                err_rel = amp_bound * err_abs / max(y_norm, tiny)
            else:
                amp_bound = 1.0
                err_rel = err_abs / max(y_norm, tiny)

        last_info = {
            "m_stop": k,
            "stop_reason": "max_depth",
            "err_abs": float(err_abs),
            "err_rel": float(err_rel),
            "change_rel": np.nan,
            "amp_bound": float(amp_bound),
            "stopped_early": False,
        }
        return out, last_info

    return out


def lan_exp(
    v,
    A,
    t=1.0,
    k: int = 16,
    *,
    work: dict | None = None,
    tol: float = 1e-10,
    rel_tol: float = 1e-8,
    change_tol: float = 1e-8,
    min_m: int = 12,
    check_every: int = 6,
    reorth: str = "none",
    auto_eta: float = 0.5,
    force_full_depth: bool = False,
    return_info: bool = False,
    lapack_driver: str = "stev",
):
    return lanczos_fun(
        v,
        A,
        fun=np.exp,
        k=int(k),
        t=t,
        work=work,
        tol=tol,
        rel_tol=rel_tol,
        change_tol=change_tol,
        min_m=int(min_m),
        check_every=int(check_every),
        reorth=reorth,
        auto_eta=auto_eta,
        force_full_depth=force_full_depth,
        return_info=return_info,
        lapack_driver=lapack_driver,
    )
