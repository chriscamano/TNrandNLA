"""oracle.py — TDVP-GSE oracle for exp(-beta H).

Applies A = exp(-beta (H + b_shift * I)) to MPS probe vectors.
GSE enrichment is performed once on the first sweep; remaining sweeps use
plain TDVP. The spectral shift is restored via the accumulated log scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tnrnla.tn.trp import TRP
from tnrnla.quantum.tdvpgse import tdvp_gse

_FINALS_DIR = Path(__file__).resolve().parents[1]
if str(_FINALS_DIR) not in sys.path:
    sys.path.insert(0, str(_FINALS_DIR))

from tfim import (
    periodic_tfim_zz_x,
    tfim_periodic_log_trace_exp_analytic,
    tfim_periodic_tau_krylov,
    tfim_periodic_trace_exp_analytic,
)

_COLUMN_SEED_STRIDE  = 1_000_003
_DEFAULT_MAX_CHUNK_LOG = 300.0


def reference_trace(n, h, J, beta):
    """Unshifted tr[exp(-beta H)] for periodic TFIM."""
    return tfim_periodic_trace_exp_analytic(int(n), float(h), float(J), float(beta))


def default_shift_b(n, J, h):
    """Spectral shift b ≈ ||H|| so that exp(-beta(H+bI)) stays numerically stable."""
    return float(n) * (abs(float(J)) + abs(float(h)))


def shifted_reference_trace(n, h, J, beta, *, b_shift=None):
    """tr[exp(-beta(H+bI))] = exp(-beta*b) * tr[exp(-beta H)]."""
    b = default_shift_b(n, J, h) if b_shift is None else float(b_shift)
    logz = tfim_periodic_log_trace_exp_analytic(int(n), float(h), float(J), float(beta))
    log_shifted = float(logz - float(beta) * b)
    log_max  = float(np.log(np.finfo(np.float64).max))
    log_tiny = float(np.log(np.finfo(np.float64).tiny))
    if log_shifted >= log_max:
        return float(np.finfo(np.float64).max)
    if log_shifted <= log_tiny:
        return 0.0
    return float(np.exp(log_shifted))


def _scale_state(psi, alpha):
    if alpha == 1.0:
        return psi
    try:
        return alpha * psi
    except Exception:
        pass
    if hasattr(psi, "copy"):
        out = psi.copy()
        try:
            out *= alpha
            return out
        except Exception:
            pass
    if hasattr(psi, "cores"):
        out = psi.copy()
        out.cores[0] = alpha * out.cores[0]
        return out
    raise TypeError("Could not scale MPS state by scalar")


def _apply_log_scale(psi, log_scale, *, max_chunk_log=_DEFAULT_MAX_CHUNK_LOG):
    log_scale = float(log_scale)
    if log_scale == 0.0:
        return psi
    if not np.isfinite(log_scale):
        raise ValueError("log_scale must be finite")
    out, rem = psi, log_scale
    while abs(rem) > max_chunk_log:
        step = float(np.sign(rem)) * max_chunk_log
        out  = _scale_state(out, float(np.exp(step)))
        rem -= step
    return _scale_state(out, float(np.exp(rem)))


def _resolve_eps_enrich(eps_enrich):
    # Legacy int convention maps to tight tolerances.
    if isinstance(eps_enrich, int):
        return 1e-14, 1e-14
    eps = float(eps_enrich)
    return eps, eps


class TDVPGSEExpOracle:
    """Applies A = exp(-beta (H + b_shift * I)) to MPS probe vectors."""

    def __init__(
        self,
        *,
        H_mpo,
        beta,
        b_shift=0.0,
        nsteps=1,
        sweeps=12,
        tdvp_krylov_depth=32,
        gse_k=3,
        tau_krylov,
        epsEnrich=1e-2,
        epsSRC=1e-8,
        gse_max_enriched_bond=None,
        rng_seed=0,
        max_chunk_log=_DEFAULT_MAX_CHUNK_LOG,
        restore_log_scale=True,
        normalize_tdvp_working_state=True,
        track_max_bond=True,
        tdvp_cutoff=1e-14,
        enrich_every_step=False,
    ):
        self.H_mpo      = H_mpo
        self.beta       = float(beta)
        self.b_shift    = float(b_shift)
        self.nsteps     = int(nsteps)
        self.sweeps     = int(sweeps)
        self.tdvp_krylov_depth = int(tdvp_krylov_depth)
        self.gse_k      = int(gse_k)
        self.tau_krylov = float(tau_krylov)
        self.epsK, self.epsM = _resolve_eps_enrich(epsEnrich)
        self.epsSRC     = float(epsSRC)
        self.gse_max_enriched_bond = (
            None if gse_max_enriched_bond is None else int(gse_max_enriched_bond)
        )
        self.rng_seed   = int(rng_seed)
        self.max_chunk_log = float(max_chunk_log)
        self.restore_log_scale = bool(restore_log_scale)
        self.normalize_tdvp_working_state = bool(normalize_tdvp_working_state)
        self.track_max_bond  = bool(track_max_bond)
        self.tdvp_cutoff     = float(tdvp_cutoff)
        self.enrich_every_step = bool(enrich_every_step)

        if self.nsteps < 1:
            raise ValueError("nsteps must be >= 1")
        if self.sweeps < 1:
            raise ValueError("sweeps must be >= 1")
        if self.gse_k < 1:
            raise ValueError("gse_k must be >= 1")

        self.total_steps = int(self.nsteps * self.sweeps)
        self.tau = float(self.beta) / float(self.total_steps)

        self.last_max_bond               = None
        self.last_post_gse_bond          = None
        self.last_post_first_sweep_bond  = None
        self.last_output_memory_gb       = None
        self.last_tdvp_info              = None

    def _tdvp_kwargs(self):
        kwargs = dict(
            k                 = int(self.gse_k),
            contraction_type  = "random",
            sketchincrement   = 100,
            sketchdim         = 100,
            tau_krylov        = float(self.tau_krylov),
            normalize_enriched= False,
            renormalize_tdvp  = True,
            epsK              = float(self.epsK),
            epsM              = float(self.epsM),
            epsSRC            = float(self.epsSRC),
        )
        if self.gse_max_enriched_bond is not None:
            kwargs["max_enriched_bond"] = int(self.gse_max_enriched_bond)
        return kwargs

    def _record_output_memory(self, psi):
        self.last_output_memory_gb = (
            float(psi.memory_gb()) if hasattr(psi, "memory_gb") else None
        )

    def apply_with_log_scale(self, psi0, col_seed=None):
        """Return (psi_out_normalized, log_scale) for A|psi0>."""
        seed = int(self.rng_seed if col_seed is None else col_seed)

        out = tdvp_gse(
            psi0,
            self.H_mpo,
            tau                          = float(self.tau),
            time_type                    = "imag",
            sweeps                       = int(self.total_steps),
            tdvp_krylov_depth            = int(self.tdvp_krylov_depth),
            rng_seed                     = seed,
            return_log_scale             = True,
            return_info                  = True,
            return_max_bond              = bool(self.track_max_bond),
            normalize_tdvp_working_state = bool(self.normalize_tdvp_working_state),
            tdvp_cutoff                  = float(self.tdvp_cutoff),
            split_mode                   = "enriched",
            enrich_once                  = not self.enrich_every_step,
            rebuild_envs_each_sweep      = self.enrich_every_step,
            **self._tdvp_kwargs(),
        )

        psi       = out[0]
        log_scale = float(out[1]) - float(self.beta) * float(self.b_shift)
        info      = out[2] if len(out) >= 3 else None
        max_bond  = out[3] if self.track_max_bond and len(out) >= 4 else None

        self.last_tdvp_info = info
        self.last_max_bond  = int(max_bond) if max_bond is not None else None

        if isinstance(info, dict):
            post_gse = info.get("post_gse_max_bond")
            self.last_post_gse_bond = int(post_gse) if post_gse is not None else None
            post_first = info.get("post_first_sweep_max_bond")
            self.last_post_first_sweep_bond = int(post_first) if post_first is not None else None
        else:
            self.last_post_gse_bond         = None
            self.last_post_first_sweep_bond = None

        return psi, float(log_scale)

    def apply(self, psi0, col_seed=None):
        psi, log_scale = self.apply_with_log_scale(psi0, col_seed=col_seed)
        if self.restore_log_scale:
            psi = _apply_log_scale(psi, log_scale, max_chunk_log=self.max_chunk_log)
        self._record_output_memory(psi)
        return psi

    def __matmul__(self, rhs):
        if not hasattr(rhs, "cols"):
            return self.apply(rhs)
        out_cols, max_bonds = [], []
        for j, col in enumerate(rhs.cols):
            seed = int(self.rng_seed + _COLUMN_SEED_STRIDE * j)
            out_cols.append(self.apply(col, col_seed=seed))
            if self.track_max_bond and self.last_max_bond is not None:
                max_bonds.append(int(self.last_max_bond))
        self.last_max_bond = int(max(max_bonds)) if max_bonds else None
        return TRP._from_parent(rhs, out_cols)


def build_oracle(
    n, J, h, beta,
    *,
    sweeps,
    gse_k=3,
    seed=0,
    tdvp_krylov_depth=32,
    epsEnrich=1e-2,
    epsSRC=1e-8,
    tdvp_cutoff=1e-10,
    nsteps=1,
    b_shift=None,
    tau_krylov=None,
    gse_max_enriched_bond=None,
    enrich_every_step=False,
):
    """Build a TDVPGSEExpOracle for the periodic TFIM."""
    n  = int(n)
    b  = default_shift_b(n, J, h) if b_shift is None else float(b_shift)
    tk = tfim_periodic_tau_krylov(n, J=J, h=h) if tau_krylov is None else float(tau_krylov)
    H  = periodic_tfim_zz_x(n, J=float(J), h=float(h))

    return TDVPGSEExpOracle(
        H_mpo                    = H,
        beta                     = float(beta),
        b_shift                  = float(b),
        nsteps                   = int(nsteps),
        sweeps                   = int(sweeps),
        tdvp_krylov_depth        = int(tdvp_krylov_depth),
        gse_k                    = int(gse_k),
        tau_krylov               = float(tk),
        epsEnrich                = epsEnrich,
        epsSRC                   = float(epsSRC),
        gse_max_enriched_bond    = gse_max_enriched_bond,
        rng_seed                 = int(seed),
        restore_log_scale        = True,
        track_max_bond           = True,
        tdvp_cutoff              = float(tdvp_cutoff),
        enrich_every_step        = bool(enrich_every_step),
    )
