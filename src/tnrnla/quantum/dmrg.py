from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from tnrnla.linalg.lra import truncated_svd
from tnrnla.quantum.groundstate import GroundState
from tnrnla.tn.stopping import Cutoff

from tnrnla.quantum.dmrg_util import contract_blas, make_op_dense, make_op_sparse


def _energy_scalar(e) -> float:
    return float(np.real(np.asarray(e).reshape(-1)[0]))


def _local_residual(op_mv, e: float, psi_vec: np.ndarray) -> float:
    v = np.asarray(psi_vec).reshape(-1)
    hv = op_mv(v) if callable(op_mv) else op_mv @ v
    return float(np.linalg.norm(hv - e * v) / max(np.linalg.norm(v), 1e-30))


def _discarded_weight(mat: np.ndarray, kept_svals: np.ndarray) -> float:
    tot = float(np.linalg.norm(mat, "fro") ** 2)
    if tot <= 0.0:
        return 0.0
    return float(max(0.0, 1.0 - np.sum(np.abs(kept_svals) ** 2) / tot))


def _snapshot_bonds(state) -> List[int]:
    return [int(np.asarray(state[i]).shape[-1]) for i in range(len(state) - 1)]


def _sweep_path(n: int, sweeps: int) -> np.ndarray:
    seq = np.concatenate([np.arange(n - 1), np.arange(n - 3, 0, -1)])
    return np.append(np.tile(seq, sweeps), 0)


def _right_envs(mps, mpo) -> list:
    N = len(mps)
    R = [None] * N
    tmp = contract_blas(mpo[-1], 1, np.conj(mps[-1]), 1)
    R[-1] = contract_blas(tmp.transpose(2, 0, 1), 2, mps[-1], 1)
    for i in range(N - 2, 1, -1):
        Ri = contract_blas(np.conj(mps[i]), 2, R[i + 1], 0)
        tmp = contract_blas(mpo[i], (1, 2), Ri, (1, 2))
        Ri = tmp.transpose(2, 0, 1, 3)
        tmp = contract_blas(mps[i], (1, 2), Ri, (2, 3))
        R[i] = tmp.transpose(1, 2, 0)
    return R


def _update_left_env(LR, k, mpo_k, site_tensor):
    tmp = contract_blas(LR[k - 1], 0, np.conj(site_tensor), 0)
    LR[k] = tmp.transpose(3, 2, 0, 1)
    tmp = contract_blas(LR[k], (2, 1), mpo_k, (0, 1))
    LR[k] = tmp.transpose(0, 2, 3, 1)
    LR[k] = contract_blas(LR[k], (3, 2), site_tensor, (0, 1))


def _update_right_env(LR, k, mpo_k1, site_tensor):
    tmp = contract_blas(LR[k + 2], 0, np.conj(site_tensor), 2)
    LR[k + 1] = tmp.transpose(2, 3, 0, 1)
    tmp = contract_blas(LR[k + 1], (1, 2), mpo_k1, (1, 2))
    LR[k + 1] = tmp.transpose(0, 2, 3, 1)
    LR[k + 1] = contract_blas(LR[k + 1], (2, 3), site_tensor, (1, 2))


def _eigsh(
    op,
    psi: np.ndarray,
    *,
    eig_tol: float,
    maxit: int,
    ncv: int,
    warmstart: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    v0 = np.asarray(psi).reshape(-1).copy()
    if warmstart is not None:
        w = np.asarray(warmstart).reshape(-1)
        if w.size == v0.size:
            wn = np.linalg.norm(w)
            vn = np.linalg.norm(v0)
            if wn > 0.0 and vn > 0.0:
                v0 = 0.8 * (w / wn) + 0.2 * (v0 / vn)

    kw = dict(k=1, which="SA", maxiter=int(maxit), v0=v0, tol=float(eig_tol))
    if ncv is not None:
        kw["ncv"] = int(ncv)
    return spla.eigsh(op, **kw)


def _apply_mpo(mpo, psi):
    if hasattr(mpo, "apply"):
        return mpo.apply(psi)
    if hasattr(mpo, "__matmul__"):
        return mpo @ psi
    raise TypeError("Need mpo.apply(psi) or mpo @ psi to compute variance")


def _state_inner(x, y) -> complex:
    if hasattr(x, "inner_product"):
        return x.inner_product(y)
    if hasattr(x, "dot"):
        return x.dot(y)
    if hasattr(np, "vdot"):
        return np.vdot(x, y)
    raise TypeError("Need a state inner product method to compute variance")


def _state_energy_variance(mpo, psi) -> Tuple[float, float]:
    hpsi = _apply_mpo(mpo, psi)
    e = float(np.real(_state_inner(psi, hpsi)))
    h2 = float(np.real(_state_inner(hpsi, hpsi)))
    var = max(h2 - e * e, 0.0)
    return e, var


def dmrg2(
    mpo,
    mps,
    *,
    sweeps: int,
    stop: Cutoff = Cutoff(1e-14),
    maxit: int = 10,
    eig_tol: float = 1e-8,
    eig_ncv: int = 8,
    sparse_mpo_cores=None,
    return_update_times: bool = False,
    collect_metrics: bool = False,
    metric_energy_ref: Optional[float] = None,
    metric_eval_every: int = 1,
    collect_bond_history: bool = False,
    precontract: bool = True,
    lanczos_warmstart: bool = True,
    variance_tol: Optional[float] = None,
    variance_check_every_sweeps: int = 1,
    variance_per_site: bool = True,
    variance_state: Optional[object] = None,
):
    metric_eval_every = max(1, int(metric_eval_every))
    variance_check_every_sweeps = max(1, int(variance_check_every_sweeps))
    sweeps_eff = int(sweeps)

    if sweeps_eff < 1:
        raise ValueError("sweeps must be >= 1")

    if sparse_mpo_cores is None:
        sparse_mpo_cores = getattr(mpo, "sparse_cores", None)
    use_sparse = sparse_mpo_cores is not None
    if use_sparse and len(sparse_mpo_cores) != len(mpo):
        raise ValueError("sparse_mpo_cores length mismatch")

    timing = dict(
        eigsh_s=0.0,
        svd_s=0.0,
        env_update_s=0.0,
        metrics_s=0.0,
        matvec_s=0.0,
        matvec_calls=0,
        num_updates=0,
        warmstart_attempts=0,
        warmstart_used=0,
        total_core_s=0.0,
        variance_stop_triggered=False,
        variance_stop_update=None,
        variance_stop_reason=None,
        variance_checks=0,
        variance=None,
        variance_per_site=None,
        variance_energy=None,
        variance_completed_sweeps=0,
        bond_dims_history=None,
    )

    def record_mv(dt, *, sparse=False):
        timing["matvec_s"] += float(dt)
        timing["matvec_calls"] += 1

    def _op(boundary, *args):
        builder = make_op_sparse if use_sparse else make_op_dense
        kwargs = dict(
            record_matvec=record_mv,
            contract_fn=contract_blas,
            timer=time.perf_counter,
        )
        if not use_sparse:
            kwargs["precontract"] = bool(precontract)
            kwargs["sweeping_right"] = bool(left_sweep)
        return builder(boundary, *args, **kwargs)

    t0 = time.perf_counter()
    mps.orthR()
    LR = _right_envs(mps, mpo)
    timing["env_init_s"] = time.perf_counter() - t0

    N = int(mps.N)
    cores = sparse_mpo_cores if use_sparse else mpo
    approx = mps.copy()
    Es = []
    update_times = []
    metrics = []
    bond_history = []
    warmstart_cache = {}

    if collect_bond_history:
        timing["bond_dims_initial"] = _snapshot_bonds(approx)

    left_sweep = True
    core_elapsed = 0.0
    completed_sweeps = 0
    path = _sweep_path(N, sweeps_eff)

    for k in path:
        k = int(k)
        t_core = time.perf_counter()
        residual_op = None

        if k == 0:
            psi = contract_blas(approx[0], 1, approx[1], 0)
            op, residual_op = _op("first", cores[0], cores[1], LR[2], psi.shape)
        elif k < N - 2:
            psi = contract_blas(approx[k], 2, approx[k + 1], 0)
            op, residual_op = _op("mid", LR[k - 1], cores[k], cores[k + 1], LR[k + 2], psi.shape)
        else:
            psi = contract_blas(approx[-2], 2, approx[-1], 0)
            op, residual_op = _op("last", LR[-3], cores[-2], cores[-1], psi.shape)

        n_full = int(np.prod(psi.shape))
        ws = warmstart_cache.get((k, n_full)) if bool(lanczos_warmstart) else None
        if ws is not None:
            timing["warmstart_attempts"] += 1

        t_eig = time.perf_counter()
        energy, psivec = _eigsh(
            op,
            psi,
            eig_tol=eig_tol,
            maxit=maxit,
            ncv=eig_ncv,
            warmstart=ws,
        )
        timing["eigsh_s"] += time.perf_counter() - t_eig

        if ws is not None and psivec is not None:
            timing["warmstart_used"] += 1
        if bool(lanczos_warmstart):
            warmstart_cache[(k, n_full)] = np.asarray(psivec).reshape(psi.shape).copy()

        if k == 0:
            mat = np.asarray(psivec).reshape(psi.shape[0], psi.shape[1] * psi.shape[2])
        elif k < N - 2:
            mat = np.asarray(psivec).reshape(
                psi.shape[0] * psi.shape[1],
                psi.shape[2] * psi.shape[3],
            )
        else:
            mat = np.asarray(psivec).reshape(psi.shape[0] * psi.shape[1], psi.shape[2])

        t_svd = time.perf_counter()
        U, svals, Vt = truncated_svd(mat, stop=stop)
        timing["svd_s"] += time.perf_counter() - t_svd

        svals = np.asarray(svals)
        svals_n = svals / max(float(np.linalg.norm(svals)), 1e-30)

        t_env = time.perf_counter()
        if k == 0:
            approx[0] = U
            approx[1] = (svals_n[:, None] * Vt).reshape(Vt.shape[0], psi.shape[1], psi.shape[2])

            LR[0] = contract_blas(np.conj(U), 0, mpo[0], 0)
            LR[0] = contract_blas(LR[0], 2, U, 0)
            left_sweep = True

        elif k < N - 2:
            if left_sweep:
                approx[k] = U.reshape(psi.shape[0], mpo[k].shape[1], U.shape[1])
                approx[k + 1] = (svals_n[:, None] * Vt).reshape(
                    Vt.shape[0], psi.shape[2], psi.shape[3]
                )
                _update_left_env(LR, k, mpo[k], approx[k])
            else:
                approx[k] = (U * svals_n[None, :]).reshape(
                    psi.shape[0], mpo[k].shape[1], U.shape[1]
                )
                approx[k + 1] = Vt.reshape(Vt.shape[0], psi.shape[2], psi.shape[3])
                _update_right_env(LR, k, mpo[k + 1], approx[k + 1])

        else:
            approx[-2] = (U * svals_n[None, :]).reshape(psi.shape[0], approx[-2].shape[1], -1)
            approx[-1] = Vt.reshape(-1, approx[-1].shape[1])

            tmp = contract_blas(mpo[-1], 1, np.conj(approx[-1]), 1)
            LR[-1] = contract_blas(tmp.transpose(2, 0, 1), 2, approx[-1], 1)
            left_sweep = False

        timing["env_update_s"] += time.perf_counter() - t_env

        core_elapsed += time.perf_counter() - t_core
        e = _energy_scalar(energy)
        psi_vec = np.asarray(psivec).reshape(-1)

        Es.append(e)
        if return_update_times:
            update_times.append(core_elapsed)
        if collect_bond_history:
            bond_history.append(_snapshot_bonds(approx))

        update_idx = len(Es) - 1

        if collect_metrics and update_idx % metric_eval_every == 0:
            t_m = time.perf_counter()
            resid = _local_residual(residual_op, e, psi_vec)
            dw = _discarded_weight(mat, svals)
            rel = None
            if metric_energy_ref is not None:
                rel = abs(e - metric_energy_ref) / max(abs(metric_energy_ref), 1e-30)
            metrics.append(
                dict(
                    update=update_idx,
                    site=k,
                    energy=e,
                    local_residual=resid,
                    discarded_weight=dw,
                    rel_energy=rel,
                )
            )
            timing["metrics_s"] += time.perf_counter() - t_m

        if k == 0 and update_idx > 0:
            completed_sweeps += 1
            timing["variance_completed_sweeps"] = completed_sweeps

            if variance_tol is not None and completed_sweeps % variance_check_every_sweeps == 0:
                v_state = approx if variance_state is None else variance_state(approx)
                e_var, var = _state_energy_variance(mpo, v_state)
                scale = float(N) if variance_per_site else 1.0
                eps_var = float(np.sqrt(var) / max(scale, 1.0))

                timing["variance_checks"] += 1
                timing["variance_energy"] = e_var
                timing["variance"] = var
                timing["variance_per_site"] = eps_var if variance_per_site else np.sqrt(var)

                if eps_var <= float(variance_tol):
                    timing["variance_stop_triggered"] = True
                    timing["variance_stop_update"] = update_idx + 1
                    timing["variance_stop_reason"] = "variance"
                    break

    timing["num_updates"] = len(Es)
    timing["total_core_s"] = core_elapsed
    if collect_bond_history:
        timing["bond_dims_history"] = bond_history

    out = GroundState(approx)
    out.canform = getattr(approx, "canform", "None")
    dmrg2.last_timing = timing

    result = [out, Es]
    if return_update_times:
        result.append(update_times)
    if collect_metrics:
        result.append(metrics)
    return tuple(result) if len(result) > 2 else (out, Es)
