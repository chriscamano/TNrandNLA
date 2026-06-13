from __future__ import annotations

# import numpy as np

# from tnrnla.tn.mpo import MPO
# from tnrnla.tn.stopping import Cutoff
# from tnrnla.quantum.basis_enrichment import expand
# from tnrnla import mpo_mps
# from tnrnla.tn.contraction.src import SRC
# from tnrnla.quantum.tdvp1.tdvp import tdvp_implicit


# def tdvp_gse(
#     mps,
#     H_mpo,
#     *,
#     tau,
#     time_type="imag",
#     sweeps=1,
#     tdvp_krylov_depth=8,
#     use_lanczos=True,
#     measure=False,
#     k=5,
#     epsK=1e-4,
#     epsM=1e-4,
#     epsSRC=1e-4,
#     contraction_type="random",
#     sketchincrement=16,
#     sketchdim=16,
#     tau_krylov=None,
#     lambda_est=None,
#     tdvp_module=None,
#     normalize_enriched=False,
#     renormalize_tdvp=False,
#     normalize_tdvp_working_state=True,
#     track_tdvp=False,
#     rng=None,
#     rng_seed=None,
#     return_log_scale=False,
#     track_bonds=False,
#     return_bond_history=False,
#     return_max_bond=False,
#     max_enriched_bond=None,
#     early_stopping=False,
#     stop_tol=1e-8,
#     stop_patience=2,
#     stop_min_sweeps=2,
#     stop_mode="tangent",
#     return_info=False,
#     tdvp_cutoff=None,
# ):
#     """
#     Build a short Krylov-style enrichment space, expand the input MPS into that
#     space, then run TDVP on the enriched state.

#     Notes
#     -----
#     `use_lanczos` and `measure` are kept only for API compatibility.

#     `max_enriched_bond`, if given, caps the enriched bond dimension per site
#     to that value (keeping everything above 1e-14 when fewer directions exist).

#     If `return_log_scale=True`, any normalization applied to the enriched state
#     and any TDVP log-scale output are accumulated and returned.
#     """
#     if time_type not in ("real", "imag"):
#         raise ValueError("time_type must be 'real' or 'imag'")

#     tau = float(tau)
#     if tau <= 0.0:
#         raise ValueError("tau must be positive")

#     if tau_krylov is None:
#         if time_type == "real":
#             tau_krylov = tau
#         else:
#             if lambda_est is None:
#                 raise ValueError(
#                     "lambda_est is required for imaginary time when tau_krylov is None"
#                 )
#             lambda_est = float(lambda_est)
#             if lambda_est <= 0.0:
#                 raise ValueError("lambda_est must be positive")
#             tau_krylov = 1.0 / lambda_est
#     else:
#         tau_krylov = float(tau_krylov)

#     if rng is None:
#         rng = (
#             np.random.default_rng(int(rng_seed))
#             if rng_seed is not None
#             else np.random.default_rng()
#         )

#     bond_history = [] if track_bonds else None
#     if track_bonds:
#         bond_history.append(
#             {
#                 "stage": "input",
#                 "max_bond": int(mps.max_bond()),
#             }
#         )

#     alpha = -1j * tau_krylov if time_type == "real" else -1.0 * tau_krylov
#     A_mpo = MPO.eye(int(mps.N), d=2) + (alpha * H_mpo)
#    # A_mpo.round()

#     basis = [mps]
#     k_vec = mps

#     if contraction_type == "naive":
#         stop = Cutoff(float(epsK))
#         for j in range(int(k) - 1):
#             k_vec = mpo_mps(A_mpo, k_vec, stop=stop, round_type="daas_blas")
#             basis.append(k_vec)
#             if track_bonds:
#                 bond_history.append(
#                     {
#                         "stage": f"krylov_{j + 1}",
#                         "max_bond": int(k_vec.max_bond()),
#                     }
#                 )

#     elif contraction_type == "random":
#         stop = Cutoff(float(epsSRC))
#         for j in range(int(k) - 1):
#             src_seed = int(rng.integers(0, np.iinfo(np.int64).max))
#             k_vec = SRC(
#                 A_mpo,
#                 k_vec,
#                 stop=stop,
#                 finalround=False,
#                 sketchincrement=int(sketchincrement),
#                 sketchdim=int(sketchdim),
#                 rng=rng,
#                 seed=src_seed,
#             )
#             basis.append(k_vec)
#             if track_bonds:
#                 bond_history.append(
#                     {
#                         "stage": f"krylov_{j + 1}",
#                         "max_bond": int(k_vec.max_bond()),
#                     }
#                 )
#     else:
#         raise ValueError("contraction_type must be 'naive' or 'random'")

#     mps_enriched = expand(
#         basis,
#         cutoff=float(epsM),
#         max_enriched_bond=max_enriched_bond,
#     )
#     #print(mps_enriched.max_bond())
#     total_log_scale = 0.0
#     if track_bonds:
#         bond_history.append(
#             {
#                 "stage": "enriched",
#                 "max_bond": int(mps_enriched.max_bond()),
#             }
#         )

#     if bool(normalize_enriched):
#         n2 = mps_enriched.inner_product(mps_enriched)
#         n2r = float(np.real(n2))
#         if n2r <= 0.0:
#             raise ValueError(f"Nonpositive norm encountered in tdvp_gse: {n2}")
#         total_log_scale += 0.5 * np.log(n2r)
#         mps_enriched.normalize()

#         if track_bonds:
#             bond_history.append(
#                 {
#                     "stage": "enriched_normalized",
#                     "max_bond": int(mps_enriched.max_bond()),
#                 }
#             )

#     tdvp_kernel = tdvp_implicit if tdvp_module is None else tdvp_module

#     tdvp_kernel_kwargs = dict(
#         dt=float(tau),
#         sweeps=int(sweeps),
#         krylov_depth=int(tdvp_krylov_depth),
#         time_type=str(time_type),
#         renormalize=bool(renormalize_tdvp),
#         normalize_working_state=bool(normalize_tdvp_working_state),
#         track=bool(track_tdvp),
#         return_log_scale=bool(return_log_scale),
#         early_stopping=bool(early_stopping),
#         stop_tol=float(stop_tol),
#         stop_patience=int(stop_patience),
#         stop_min_sweeps=int(stop_min_sweeps),
#         stop_mode=str(stop_mode),
#         return_info=bool(return_info),
#     )
#     if tdvp_cutoff is not None:
#         tdvp_kernel_kwargs["cutoff"] = Cutoff(float(tdvp_cutoff))
#     tdvp_out = tdvp_kernel(mps_enriched, H_mpo, **tdvp_kernel_kwargs)

#     tdvp_info = None
#     if return_log_scale and return_info:
#         mps_out, dlog_tdvp, tdvp_info = tdvp_out
#         total_log_scale += float(dlog_tdvp)
#     elif return_log_scale:
#         mps_out, dlog_tdvp = tdvp_out
#         total_log_scale += float(dlog_tdvp)
#     elif return_info:
#         mps_out, tdvp_info = tdvp_out
#     else:
#         mps_out = tdvp_out

#     if track_bonds:
#         bond_history.append(
#             {
#                 "stage": "tdvp_out",
#                 "max_bond": int(mps_out.max_bond()),
#             }
#         )

#     out = [mps_out]

#     if return_log_scale:
#         out.append(float(total_log_scale))

#     if return_info:
#         out.append(tdvp_info)

#     if return_bond_history:
#         out.append(bond_history)

#     if return_max_bond:
#         out.append(int(mps_out.max_bond()))

#     if len(out) == 1:
#         return out[0]
#     return tuple(out)
from __future__ import annotations

import numpy as np

from scipy.special import iv

from tnrnla import mpo_mps
from tnrnla.linalg.lra import Cutoff, truncated_svd
from tnrnla.quantum.basis_enrichment import expand
from tnrnla.quantum.tdvp1.tdvp import *
from tnrnla.tn.contraction.srcR import SRC_R
from tnrnla.tn.mpo import MPO
from tnrnla.tn.other.tensor_ops import (
    flatten_left,
    flatten_right,
    _flatten_last2,
    _ascontig,
    qr_right,
    lq_left,
)
from tnrnla.tn.stopping import Cutoff as EnrichCutoff


def _diag_times_rows(s, Vt):
    return s[:, None] * Vt


def _cols_times_diag(U, s):
    return U * s[None, :]


def split_left_tensor_qr_or_svd(
    A,
    *,
    split_mode="qr",
    cutoff=None,
    abstol=0.0,
    check_finite=True,
):
    """
    Split A as A ≈ Q @ C with Q left-isometric.

    In SVD mode, compute a thin QR first,
        M = Q0 R,  R = U S Vh,
    then store Q0 U and return S Vh.
    """
    A = _ascontig(A)

    if split_mode == "qr":
        Q_tensor, C = qr_right(A)
        return _ascontig(Q_tensor), _ascontig(C)

    shape = A.shape

    if A.ndim == 2:
        m, n = shape
        M = A
        left_shape = (m,)
    elif A.ndim == 3:
        ldim, pdim, rdim = shape
        M = A.reshape(ldim * pdim, rdim)
        left_shape = (ldim, pdim)
    else:
        raise ValueError(f"Unsupported tensor rank {A.ndim} in split_left_tensor_qr_or_svd")

    Q0, R = np.linalg.qr(M, mode="reduced")
    U, s, Vt = truncated_svd(
        R,
        stop=cutoff,
        abstol=abstol,
        check_finite=check_finite,
    )

    k = len(s)
    Qmat = Q0 @ U
    C = _diag_times_rows(s, Vt)

    if A.ndim == 2:
        Q_tensor = Qmat.reshape(left_shape[0], k)
    else:
        Q_tensor = Qmat.reshape(left_shape[0], left_shape[1], k)

    return _ascontig(Q_tensor), _ascontig(C)


def split_right_tensor_qr_or_svd(
    A,
    *,
    split_mode="qr",
    cutoff=None,
    abstol=0.0,
    check_finite=True,
):
    """
    Split A as A ≈ C @ Q with Q right-isometric.

    In SVD mode, compute an LQ through QR of M*, 
        M = L Q0,  L = U S Vh,
    then return U S and store Vh Q0.
    """
    A = _ascontig(A)

    if split_mode == "qr":
        C, Q_tensor = lq_left(A)
        return _ascontig(C), _ascontig(Q_tensor)

    shape = A.shape

    if A.ndim == 2:
        ldim, pdim = shape
        M = A
        right_shape = (pdim,)
    elif A.ndim == 3:
        ldim, pdim, rdim = shape
        M = A.reshape(ldim, pdim * rdim)
        right_shape = (pdim, rdim)
    else:
        raise ValueError(f"Unsupported tensor rank {A.ndim} in split_right_tensor_qr_or_svd")

    Qh, Rh = np.linalg.qr(M.conj().T, mode="reduced")
    L = Rh.conj().T
    Q0 = Qh.conj().T

    U, s, Vt = truncated_svd(
        L,
        stop=cutoff,
        abstol=abstol,
        check_finite=check_finite,
    )

    k = len(s)
    C = _cols_times_diag(U, s)
    Qmat = Vt @ Q0

    if A.ndim == 2:
        Q_tensor = Qmat.reshape(k, right_shape[0])
    else:
        Q_tensor = Qmat.reshape(k, right_shape[0], right_shape[1])

    return _ascontig(C), _ascontig(Q_tensor)


def update_left_site(
    state,
    site,
    mpo_views,
    LR,
    *,
    split_mode="qr",
    cutoff=None,
    svd_abstol=0.0,
    svd_check_finite=True,
):
    """
    Left-to-right orthogonalization step.

    In SVD mode, uses QR-SVD:
        M = Q0 R, R = U S Vh,
        state[site] <- Q0 U,
        return C = S Vh.
    """
    if site == 0:
        Q, C = split_left_tensor_qr_or_svd(
            state[0],
            split_mode=split_mode,
            cutoff=cutoff,
            abstol=svd_abstol,
            check_finite=svd_check_finite,
        )
        state[0] = Q
    else:
        Q, C = split_left_tensor_qr_or_svd(
            state[site],
            split_mode=split_mode,
            cutoff=cutoff,
            abstol=svd_abstol,
            check_finite=svd_check_finite,
        )
        state[site] = Q

    if site == 0:
        mpo0 = mpo_views.site[0]
        prod = np.conj(state[0]).T @ mpo0["mat_A_0"]
        LR[0] = prod.reshape(
            prod.shape[0],
            mpo0["A"].shape[1],
            mpo0["A"].shape[2],
        )

        prod2 = flatten_left(LR[0]) @ state[0]
        LR[0] = prod2.reshape(
            LR[0].shape[0],
            LR[0].shape[1],
            prod2.shape[-1],
        )
    else:
        prod = (
            flatten_left(LR[site - 1].transpose(1, 2, 0))
            @ np.conj(state[site]).reshape(state[site].shape[0], -1)
        )
        prod = prod.reshape(
            LR[site - 1].shape[-2],
            LR[site - 1].shape[-1],
            state[site].shape[1],
            state[site].shape[2],
        )
        LR[site] = prod.transpose(3, 2, 0, 1)

        mpo_i = mpo_views.site[site]

        LRt2 = LR[site].transpose(0, 3, 2, 1)
        prod2 = LRt2.reshape(-1, LRt2.shape[-2] * LRt2.shape[-1]) @ mpo_i["mat_A_01"]
        prod2 = prod2.reshape(
            LR[site].shape[0],
            LR[site].shape[-1],
            mpo_i["A"].shape[2],
            mpo_i["A"].shape[3],
        )
        LR[site] = prod2.transpose(0, 2, 3, 1)

        LRt3 = LR[site].transpose(0, 1, 3, 2)
        prod3 = (
            LRt3.reshape(-1, LRt3.shape[-2] * LRt3.shape[-1])
            @ state[site].reshape(-1, state[site].shape[-1])
        )
        LR[site] = prod3.reshape(
            LR[site].shape[0],
            LR[site].shape[1],
            state[site].shape[-1],
        )

    return C


def update_right_site(
    state,
    site,
    mpo_views,
    LR,
    *,
    split_mode="qr",
    cutoff=None,
    svd_abstol=0.0,
    svd_check_finite=True,
):
    """
    Right-to-left orthogonalization step.

    In SVD mode, uses QR-SVD through LQ:
        M = L Q0, L = U S Vh,
        return C = U S,
        state[site] <- Vh Q0.
    """
    if site == get_N(state) - 1:
        C, Q = split_right_tensor_qr_or_svd(
            state[-1],
            split_mode=split_mode,
            cutoff=cutoff,
            abstol=svd_abstol,
            check_finite=svd_check_finite,
        )
        state[-1] = Q
    else:
        C, Q = split_right_tensor_qr_or_svd(
            state[site],
            split_mode=split_mode,
            cutoff=cutoff,
            abstol=svd_abstol,
            check_finite=svd_check_finite,
        )
        state[site] = Q

    if site == mpo_views.N - 1:
        mpo_last = mpo_views.site[-1]
        prod = np.conj(state[-1]) @ mpo_last["mat_T102_0"]
        LR[-1] = prod.reshape(
            prod.shape[0],
            mpo_last["A"].shape[0],
            mpo_last["A"].shape[2],
        )

        prod2 = flatten_left(LR[-1]) @ state[-1].T
        LR[-1] = prod2.reshape(
            LR[-1].shape[0],
            LR[-1].shape[1],
            state[-1].shape[0],
        )
    else:
        prod = (
            LR[site + 1].T.reshape(-1, LR[site + 1].shape[0])
            @ np.conj(state[site]).T.reshape(state[site].shape[-1], -1)
        )
        prod = prod.reshape(
            LR[site + 1].shape[-1],
            LR[site + 1].shape[-2],
            state[site].shape[1],
            state[site].shape[0],
        )
        LR[site] = prod.transpose(3, 2, 1, 0)

        mpo_i = mpo_views.site[site]

        prod2 = (
            _flatten_last2(LR[site].transpose(0, 3, 1, 2))
            @ mpo_i["mat_T1203_last2"]
        )
        prod2 = prod2.reshape(
            LR[site].shape[0],
            LR[site].shape[3],
            mpo_i["A"].shape[0],
            mpo_i["A"].shape[3],
        )
        LR[site] = prod2.transpose(0, 2, 3, 1)

        approx_T = state[site].transpose(1, 2, 0)
        prod3 = (
            LR[site].reshape(-1, LR[site].shape[-2] * LR[site].shape[-1])
            @ approx_T.reshape(-1, approx_T.shape[-1])
        )
        LR[site] = prod3.reshape(
            LR[site].shape[0],
            LR[site].shape[1],
            prod3.shape[1],
        )

    return C


def tdvp_gse(
    mps,
    H_mpo,
    *,
    tau,
    time_type="imag",
    sweeps=1,
    tdvp_krylov_depth=8,
    use_lanczos=True,
    measure=False,
    k=5,
    epsK=1e-4,
    epsM=1e-4,
    epsSRC=1e-4,
    contraction_type="random",
    sketchincrement=16,
    sketchdim=16,
    tau_krylov=None,
    lambda_est=None,
    lambda_min_est=0.0,
    enrichment_mode="power_basis",
    cheb_pad=1.05,
    cheb_include_input=True,
    normalize_enriched=False,
    renormalize_tdvp=False,
    normalize_tdvp_working_state=True,
    track_tdvp=False,
    rng=None,
    rng_seed=None,
    return_log_scale=False,
    track_bonds=False,
    return_bond_history=False,
    return_max_bond=False,
    return_info=False,
    tdvp_cutoff=None,
    split_mode="enriched",
    svd_abstol=0.0,
    svd_check_finite=True,
    enrich_once=True,
    rebuild_envs_each_sweep=False,
):
    del use_lanczos
    del measure
    del cheb_pad
    del cheb_include_input
    del lambda_min_est

    if get_N(mps) < 2:
        raise ValueError("This implementation requires an MPS with N >= 2")

    if time_type not in ("real", "imag"):
        raise ValueError("time_type must be 'real' or 'imag'")

    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be positive")

    sweeps = int(sweeps)
    if sweeps < 1:
        raise ValueError("sweeps must be >= 1")

    _enrich_every_step = not bool(enrich_once)

    _use_enriched_split = str(split_mode).lower() == "enriched"
    if _use_enriched_split:
        split_mode = "svd"

    split_mode = canonical_split_mode(split_mode)
    _fwd_split_mode = [split_mode]
    _bwd_split_mode = canonical_split_mode("qr") if _use_enriched_split else split_mode

    if tau_krylov is None:
        if time_type == "real":
            tau_krylov = tau
        else:
            if lambda_est is None:
                raise ValueError(
                    "lambda_est is required for imaginary time when tau_krylov is None"
                )
            lambda_est = float(lambda_est)
            if lambda_est <= 0.0:
                raise ValueError("lambda_est must be positive")
            tau_krylov = 1.0 / lambda_est
    else:
        tau_krylov = float(tau_krylov)

    if rng is None:
        rng = (
            np.random.default_rng(int(rng_seed))
            if rng_seed is not None
            else np.random.default_rng()
        )

    mpo_views = MPOViews(H_mpo)

    if time_type == "real":
        forward_phase = -1j
        center_phase = +1j
    else:
        forward_phase = -1.0
        center_phase = +1.0

    tdvp_cutoff_obj = Cutoff(1e-14) if tdvp_cutoff is None else Cutoff(float(tdvp_cutoff))

    ctx = TDVPContext(
        mpo_views=mpo_views,
        forward_phase=forward_phase,
        center_phase=center_phase,
        krylov_depth=int(tdvp_krylov_depth),
        track=bool(track_tdvp),
        LR=None,
        left_sweep=True,
        oracle_cache={},
        lanczos_work_cache={},
        split_mode=split_mode,
        cutoff=tdvp_cutoff_obj,
        svd_abstol=float(svd_abstol),
        svd_check_finite=bool(svd_check_finite),
    )

    state = mps.copy()
    if state.orthform != "Left":
        state.orthL()

    total_log_scale = 0.0
    bond_history = [] if track_bonds else None
    step_infos = []
    post_gse_max_bond = None
    post_first_sweep_max_bond = None

    if track_bonds:
        bond_history.append(
            {
                "stage": "input",
                "step": -1,
                "max_bond": int(state.max_bond()),
            }
        )

    if bool(normalize_tdvp_working_state):
        _, dlog0 = rescale_to_unit_norm(
            state,
            track=track_tdvp,
            tag="initial",
            return_log_scale=True,
            center_index=0,
        )
        total_log_scale += dlog0

    def _record_bond(stage, step_idx, vec):
        if track_bonds:
            bond_history.append(
                {
                    "stage": str(stage),
                    "step": int(step_idx),
                    "max_bond": int(vec.max_bond()),
                }
            )

    def _apply_enrichment_mpo(op_mpo, vec):
        if contraction_type == "naive":
            stop_k = EnrichCutoff(float(epsK))
            out = mpo_mps(op_mpo, vec, stop=stop_k, round_type="daas_blas")
        elif contraction_type == "random":
            stop_src = EnrichCutoff(float(epsSRC))
            src_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            out = SRC_R(
                op_mpo,
                vec,
                stop=stop_src,
                finalround=False,
                sketchincrement=int(sketchincrement),
                sketchdim=int(sketchdim),
                rng=rng,
                seed=src_seed,
            )
        else:
            raise ValueError("contraction_type must be 'naive' or 'random'")

        if out.orthform != "Right":
            out.orthR()
        return out

    def _mps_axpby(a, x, b, y):
        try:
            out = (a * x) + (b * y)
        except TypeError:
            try:
                out = (x * a) + (y * b)
            except TypeError:
                out = x.copy()
                out *= a
                tmp = y.copy()
                tmp *= b
                out += tmp

        if out.orthform != "Right":
            out.orthR()
        return out

    def _build_enriched_state(psi, step_idx):
        k_eff = int(k)
        if k_eff < 1:
            raise ValueError("k must be >= 1")

        psi0 = psi.copy()
        if psi0.orthform != "Right":
            psi0.orthR()

        if enrichment_mode == "power_basis":
            alpha = -1j * tau_krylov if time_type == "real" else -1.0 * tau_krylov
            A_mpo = MPO.eye(int(psi.N), d=2) + (alpha * H_mpo)

            k_vec = psi0
            basis = [k_vec]
            _record_bond("power_0", step_idx, k_vec)

            for j in range(k_eff - 1):
                k_vec = _apply_enrichment_mpo(A_mpo, k_vec)
                basis.append(k_vec)
                _record_bond(f"power_{j + 1}", step_idx, k_vec)

        else:
            raise ValueError("This tdvpgse2 variant currently supports enrichment_mode='power_basis'")

        psi_enriched = expand(
            basis,
            cutoff=float(epsM),
        )

        dlog_enriched = 0.0

        if track_bonds:
            bond_history.append(
                {
                    "stage": "enriched",
                    "step": int(step_idx),
                    "max_bond": int(psi_enriched.max_bond()),
                }
            )

        if bool(normalize_enriched):
            n2 = psi_enriched.inner_product(psi_enriched)
            n2r = float(np.real(n2))
            if n2r <= 0.0:
                raise ValueError(
                    f"Nonpositive norm encountered in tdvp_gse enrichment, got {n2}"
                )

            dlog_enriched = 0.5 * np.log(n2r)
            psi_enriched.normalize()

            if track_bonds:
                bond_history.append(
                    {
                        "stage": "enriched_normalized",
                        "step": int(step_idx),
                        "max_bond": int(psi_enriched.max_bond()),
                    }
                )

        return psi_enriched, float(dlog_enriched)

    def _run_tdvp_sweeps(psi, n_sweeps=None):
        nonlocal total_log_scale
        nonlocal post_first_sweep_max_bond

        _sweeps = sweeps if n_sweeps is None else int(n_sweeps)

        if psi.orthform != "Left":
            psi.orthL()

        ctx.LR = compute_right_envs(psi, mpo_views)
        ctx.left_sweep = True
        ctx.oracle_cache = {}

        sweep_idx = 0

        if track_bonds:
            bond_history.append(
                {
                    "stage": "pre_tdvp_step",
                    "step": int(sweep_idx),
                    "max_bond": int(psi.max_bond()),
                }
            )

        for site, edge_flag in sweep_scheduler(psi.N, _sweeps):
            step_idx = min(int(sweep_idx), int(_sweeps - 1))

            if track_tdvp:
                track_norm(f"step {step_idx} before evolve site {site}", psi, True)

            if site == 0:
                step = tau / 2 if edge_flag else tau
                evolve_site(ctx, psi, site, step)

                psi[site], dlog = maybe_normalize_local_tensor(
                    psi[site],
                    enabled=bool(normalize_tdvp_working_state),
                    tag=f"step {step_idx} forward site {site}",
                    track=track_tdvp,
                )
                total_log_scale += dlog

                if edge_flag != -1:
                    C = update_left_site(
                        psi,
                        site,
                        mpo_views,
                        ctx.LR,
                        split_mode=_fwd_split_mode[0],
                        cutoff=ctx.cutoff,
                        svd_abstol=ctx.svd_abstol,
                        svd_check_finite=ctx.svd_check_finite,
                    )

                    C = evolve_center(ctx, C, site, tau)

                    C, dlog = maybe_normalize_local_tensor(
                        C,
                        enabled=bool(normalize_tdvp_working_state),
                        tag=f"step {step_idx} center bond {site}",
                        track=track_tdvp,
                    )
                    total_log_scale += dlog

                    apply_center_left(C, psi, site, mpo_views)

                ctx.left_sweep = True

                if edge_flag == -1:
                    if sweep_idx == 0:
                        post_first_sweep_max_bond = int(psi.max_bond())

                    if track_bonds:
                        bond_history.append(
                            {
                                "stage": "post_tdvp_step",
                                "step": int(sweep_idx),
                                "max_bond": int(psi.max_bond()),
                            }
                        )

                    if return_info:
                        step_infos.append(
                            {
                                "step_index": int(sweep_idx),
                                "used_gse": bool(sweep_idx == 0),
                                "max_bond": int(psi.max_bond()),
                                "forward_split_mode": (
                                    "qr_svd"
                                    if sweep_idx == 0 and _use_enriched_split
                                    else "qr"
                                ),
                                "backward_split_mode": (
                                    "qr" if _use_enriched_split else str(_bwd_split_mode)
                                ),
                            }
                        )

                    sweep_idx += 1

                    if sweep_idx < _sweeps and track_bonds:
                        bond_history.append(
                            {
                                "stage": "pre_tdvp_step",
                                "step": int(sweep_idx),
                                "max_bond": int(psi.max_bond()),
                            }
                        )

            elif 0 < site < psi.N - 1:
                evolve_site(ctx, psi, site, tau / 2)

                psi[site], dlog = maybe_normalize_local_tensor(
                    psi[site],
                    enabled=bool(normalize_tdvp_working_state),
                    tag=f"step {step_idx} forward site {site}",
                    track=track_tdvp,
                )
                total_log_scale += dlog

                if ctx.left_sweep:
                    C = update_left_site(
                        psi,
                        site,
                        mpo_views,
                        ctx.LR,
                        split_mode=_fwd_split_mode[0],
                        cutoff=ctx.cutoff,
                        svd_abstol=ctx.svd_abstol,
                        svd_check_finite=ctx.svd_check_finite,
                    )

                    C = evolve_center(ctx, C, site, tau)

                    C, dlog = maybe_normalize_local_tensor(
                        C,
                        enabled=bool(normalize_tdvp_working_state),
                        tag=f"step {step_idx} center bond {site}",
                        track=track_tdvp,
                    )
                    total_log_scale += dlog

                    apply_center_left(C, psi, site, mpo_views)

                else:
                    C = update_right_site(
                        psi,
                        site,
                        mpo_views,
                        ctx.LR,
                        split_mode=_bwd_split_mode,
                        cutoff=ctx.cutoff,
                        svd_abstol=ctx.svd_abstol,
                        svd_check_finite=ctx.svd_check_finite,
                    )

                    C = evolve_center(ctx, C, site - 1, tau)

                    C, dlog = maybe_normalize_local_tensor(
                        C,
                        enabled=bool(normalize_tdvp_working_state),
                        tag=f"step {step_idx} center bond {site - 1}",
                        track=track_tdvp,
                    )
                    total_log_scale += dlog

                    apply_center_right(C, psi, site)

            else:
                evolve_site(ctx, psi, site, tau)

                psi[site], dlog = maybe_normalize_local_tensor(
                    psi[site],
                    enabled=bool(normalize_tdvp_working_state),
                    tag=f"step {step_idx} forward last site",
                    track=track_tdvp,
                )
                total_log_scale += dlog

                C = update_right_site(
                    psi,
                    site,
                    mpo_views,
                    ctx.LR,
                    split_mode=_bwd_split_mode,
                    cutoff=ctx.cutoff,
                    svd_abstol=ctx.svd_abstol,
                    svd_check_finite=ctx.svd_check_finite,
                )

                C = evolve_center(ctx, C, site - 1, tau)

                C, dlog = maybe_normalize_local_tensor(
                    C,
                    enabled=bool(normalize_tdvp_working_state),
                    tag=f"step {step_idx} center bond {site - 1}",
                    track=track_tdvp,
                )
                total_log_scale += dlog

                apply_center_right(C, psi, site)
                ctx.left_sweep = False

                if _use_enriched_split:
                    _fwd_split_mode[0] = canonical_split_mode("qr")

        return psi, int(sweep_idx)

    if not _enrich_every_step:
        state, dlog_enrich = _build_enriched_state(state, 0)
        total_log_scale += dlog_enrich
        post_gse_max_bond = int(state.max_bond())
        state, sweeps_completed = _run_tdvp_sweeps(state)
    else:
        sweeps_completed = 0
        for step_idx in range(sweeps):
            if _use_enriched_split:
                _fwd_split_mode[0] = split_mode
            state, dlog_enrich = _build_enriched_state(state, step_idx)
            total_log_scale += dlog_enrich
            if step_idx == 0:
                post_gse_max_bond = int(state.max_bond())
            state, n = _run_tdvp_sweeps(state, n_sweeps=1)
            sweeps_completed += n

    state.orthform = "Left"

    if bool(normalize_tdvp_working_state):
        _, dlogf = rescale_to_unit_norm(
            state,
            track=track_tdvp,
            tag="final",
            return_log_scale=True,
            center_index=0,
        )
        total_log_scale += dlogf

    if track_tdvp:
        track_norm("final canonical", state, True)

    info = None
    if return_info:
        info = {
            "requested_sweeps": int(sweeps),
            "sweeps_completed": int(sweeps_completed),
            "total_log_scale": float(total_log_scale),
            "returned_canonical_state": bool(
                bool(normalize_tdvp_working_state)
                and (renormalize_tdvp or return_log_scale)
            ),
            "normalize_working_state": bool(normalize_tdvp_working_state),
            "split_mode": "enriched" if _use_enriched_split else str(ctx.split_mode),
            "protocol": (
                "enrich_every_step_forward_qr_svd"
                if _enrich_every_step
                else "enrich_once_then_first_forward_qr_svd_then_qr"
            ),
            "enrich_once": not _enrich_every_step,
            "rebuild_envs_each_sweep": _enrich_every_step,
            "single_working_state": True,
            "post_gse_max_bond": int(post_gse_max_bond),
            "post_first_sweep_max_bond": (
                int(post_first_sweep_max_bond)
                if post_first_sweep_max_bond is not None
                else int(state.max_bond())
            ),
            "final_max_bond": int(state.max_bond()),
            "cutoff": ctx.cutoff,
            "svd_abstol": float(ctx.svd_abstol),
            "svd_check_finite": bool(ctx.svd_check_finite),
            "gse_k": int(k),
            "epsK": float(epsK),
            "epsM": float(epsM),
            "epsSRC": float(epsSRC),
            "tau": float(tau),
            "tau_krylov": float(tau_krylov),
            "step_info": step_infos,
        }

    if return_log_scale and renormalize_tdvp:
        out_state = state
    elif return_log_scale:
        out_state = state
    elif renormalize_tdvp:
        out_state = state
    else:
        out_state = _apply_log_scale_to_state(state, total_log_scale)

    out = [out_state]

    if return_log_scale:
        out.append(float(total_log_scale))

    if return_info:
        out.append(info)

    if return_bond_history:
        out.append(bond_history)

    if return_max_bond:
        out.append(int(state.max_bond()))

    if len(out) == 1:
        return out[0]

    return tuple(out)