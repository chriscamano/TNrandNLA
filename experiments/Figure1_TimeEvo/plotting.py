"""plotting.py — PostBerkeley experiment plotting.

Main entry point::

    from plotting import plot_racedriver_memory_time
    fig, axes = plot_racedriver_memory_time(out)   # out = run_experiment(...)

Produces a 3-panel figure (memory, time, matvecs vs n).
"""
from pathlib import Path
import sys
import importlib.util

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit as _scipy_curve_fit

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Register Scientific Colour Maps ──────────────────────────────────────────
_CMAPS_DIR = _ROOT / "ScientificColourMaps8"
for _name in ["devon", "oslo", "lipari", "davos", "lajolla", "acton",
              "batlow", "glasgow", "bamako", "bilbao", "navia", "imola"]:
    _cmap_file = _CMAPS_DIR / _name / f"{_name}.py"
    if not _cmap_file.exists():
        continue
    _spec = importlib.util.spec_from_file_location(f"{_name}_module", _cmap_file)
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _cmap_obj = getattr(_mod, f"{_name}_map", None)
    if _cmap_obj is None:
        continue
    for _reg_name, _reg_cmap in [(_name, _cmap_obj), (f"{_name}_r", _cmap_obj.reversed())]:
        try:
            mpl.colormaps.unregister(_reg_name)
        except Exception:
            pass
        mpl.colormaps.register(_reg_cmap, name=_reg_name)


def _fit_expscale(x, y):
    """Fit y = exp(a + b*x) via Levenberg-Marquardt in original y-space.

    Falls back to polyfit-on-log if curve_fit fails.
    Returns a callable predict(xq) -> array.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p_log = np.polyfit(x, np.log(y), 1)          # [b, a] log-space seed

    def _model(xx, a, b):
        return np.exp(a + b * xx)

    try:
        popt, _ = _scipy_curve_fit(
            _model, x, y,
            p0=[float(p_log[1]), float(p_log[0])],
            maxfev=20_000,
        )
        return lambda xq: _model(np.asarray(xq, dtype=float), *popt)
    except Exception:
        return lambda xq: np.exp(np.polyval(p_log, np.asarray(xq, dtype=float)))


from plot_util import (  # noqa: E402
    _darken,
    _dense_memory_gb,
    _csr_memory_gb,
    _extract_records,
    _finite_float_or_none,
    _finite_row,
    _load_dense_sparse_json,
    _marker_mask,
    _plot_curve_with_cutoff,
    _plot_markers_with_cutoff,
    _record_matvecs_to_target,
    _record_sweeps,
    _resolve_cmap,
    _set_pub_rc,
    _summarize,
    plot_relerr_focus_averaged,
)


# ── Style controls: edit these in one place ──────────────────────────────────
AXIS_LABEL_FS = 16
Y_AXIS_LABEL_FS = 16
TITLE_FS = 16
TITLE_Y = 1.03
LEGEND_FS = 20
LEGEND_TOP_FS = 16
TICK_FS = 14
MEM_LABEL_FS = 11
CBAR_LABEL_FS = 22
CBAR_TICK_FS = 18
EPS_TEXT_FS = 19
PROJECTED_DASH = (0, (2.0, 1.6))  # finer projected-line dashes
PROJECTED_LW = 2.0

LINE_MS = 4.5
MEM_MARKER_MS = 6.0          # match default marker size used in plots 2/3
DATA_MARKER_MS = 6.0
LINE_LW_LEFT = 2.0
LINE_LW_RIGHT = 1.8
ANALYTIC_MS = 9.0
ANALYTIC_MEW = 1.4

FIGSIZE = (16, 5)
X_PAD = 1.5
X_TICK_STEP = 10

CMAP_NAME = "devon_r"
TRUNC_LO = 0.40
TRUNC_HI = 0.99

# ─────────────────────────────────────────────────────────────────────────────
# Failure detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kmax_from_race_out(race_out, kmax):
    """Auto-detect kmax from race_out base_cfg if not explicitly provided."""
    if kmax is not None:
        return int(kmax)
    if isinstance(race_out, dict):
        bc = race_out.get("base_cfg", {})
        for key in ("kmax", "qmax"):
            v = bc.get(key)
            if v is not None:
                return int(v)
    return None


_HLINE_COLOR = "gray"
_HLINE_LW    = 1.0
_HLINE_ALPHA = 0.45
_HLINE_TEXT_ALPHA = 1.0


def _draw_budget_lines(ax_time, ax_matvec, *, budget_time_s, kmax):
    budget_z = 0

    if budget_time_s is not None:
        ax_time.axhline(
            float(budget_time_s),
            color=_HLINE_COLOR, linestyle="-", linewidth=_HLINE_LW, alpha=_HLINE_ALPHA, zorder=budget_z,
        )
        ax_time.text(
            0.01, float(budget_time_s), "1 minute",
            transform=ax_time.get_yaxis_transform(),
            color=_HLINE_COLOR, alpha=_HLINE_TEXT_ALPHA, va="bottom", ha="left", fontsize=11, zorder=budget_z,
        )
    ax_time.axhline(
        3600.0,
        color=_HLINE_COLOR, linestyle="-", linewidth=_HLINE_LW, alpha=_HLINE_ALPHA, zorder=budget_z,
    )
    ax_time.text(
        0.01, 3600.0, "1 hour",
        transform=ax_time.get_yaxis_transform(),
        color=_HLINE_COLOR, alpha=_HLINE_TEXT_ALPHA, va="bottom", ha="left", fontsize=11, zorder=budget_z,
    )

    if kmax is not None and ax_matvec is not None:
        ax_matvec.axhline(
            float(kmax),
            color="red", linestyle="-", linewidth=_HLINE_LW, alpha=_HLINE_ALPHA, zorder=budget_z,
        )
        ax_matvec.text(
            0.01, float(kmax), f"Max budget (k={kmax})",
            transform=ax_matvec.get_yaxis_transform(),
            color="red", alpha=_HLINE_TEXT_ALPHA, va="bottom", ha="left", fontsize=11, zorder=budget_z,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main plot: memory / time / matvec vs n
# ─────────────────────────────────────────────────────────────────────────────

def plot_racedriver_memory_time(
    race_out,
    *,
    cmap=CMAP_NAME,
    cmap_lo=0.0,
    cmap_hi=1.0,
    mode="mean_std",
    q_low=0.1,
    q_high=0.9,
    figsize=(16, 4),
    show=True,
    include=("dense", "sparse", "rMPS", "KR"),
    plot_memory_curve=True,
    all_memory_on_earth_gb=1.7e14,
    save_svg_path=None,
    dense_upperbound_n=None,
    sparse_upperbound_n=None,
    extrap_alpha=0.70,
    marker_every=2,
    dense_sparse_npz=None,
    dense_sparse_json=None,
    kmax=None,
    budget_time_s=60.0,
    terminal_marker="o",
    terminal_markersize=DATA_MARKER_MS,
    chi1_extrap_n=70,
    extrapolate=True,
):
    _set_pub_rc()
    records = _extract_records(race_out)
    if chi1_extrap_n is not None:
        records = [
            r for r in records
            if not (int(r.get("chi", -1)) == 1 and int(r.get("n", -1)) == int(chi1_extrap_n))
        ]
    chi_cmap = _resolve_cmap(cmap, cmap_lo=cmap_lo, cmap_hi=cmap_hi)
    # Dense/sparse reference data should come from DENSE_SPARSE3-style JSON.
    # Keep dense_sparse_npz in signature for backward compatibility, but ignore it.
    _ = dense_sparse_npz
    _ds_json = (
        None if dense_sparse_json is None else _load_dense_sparse_json(dense_sparse_json)
    )
    target_accuracy = _finite_float_or_none(records[0].get("target_accuracy", np.nan))

    ns   = np.array(sorted({int(r["n"])   for r in records}), dtype=int)
    chis = np.array(sorted({int(r["chi"]) for r in records}), dtype=int)

    _kmax = _kmax_from_race_out(race_out, kmax)

    _KR_COLOR = "#d08080"

    if len(chis) == 1:
        chi_norm = mpl.colors.Normalize(
            vmin=float(chis[0]) - 0.5, vmax=float(chis[0]) + 0.5)
    else:
        chi_norm = mpl.colors.Normalize(
            vmin=float(np.min(chis)), vmax=float(np.max(chis)))

    chi_colors = {
        int(chi): (_KR_COLOR if int(chi) == 1 else chi_cmap(chi_norm(float(chi))))
        for chi in chis
    }
    chi_rank = {int(chi): i for i, chi in enumerate(chis)}

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax_mem, ax_time, ax_matvec = axes
    _proj_ymax = {}  # ax -> max projected y used to expand y-limits when needed
    _time_sparse_last_y = None
    _time_sparse_n10_y = None
    _time_sparse_n10_lo = None

    def _terminal(ax, x, y, color, *, zorder=None):
        kwargs = {}
        if zorder is not None:
            kwargs["zorder"] = zorder
        ax.plot(
            [x], [y],
            marker=terminal_marker,
            linestyle="None",
            color=color,
            markersize=terminal_markersize,
            markerfacecolor=color,
            markeredgecolor=_darken(color),
            markeredgewidth=0.9,
            **kwargs,
        )

    def _plot_key_curve_markers(ax, x, y, color, *, upperbound_n=None):
        x = np.asarray(x, dtype=int)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(y)
        if upperbound_n is not None:
            valid &= (x <= int(upperbound_n))
        if not np.any(valid):
            return
        xv = x[valid]
        yv = y[valid]
        idx = []
        for n_key in (10, 20):
            hits = np.where(xv == n_key)[0]
            if hits.size:
                idx.append(int(hits[0]))
        idx.append(len(xv) - 1)
        idx = np.array(sorted(set(idx)), dtype=int)
        ax.plot(
            xv[idx], yv[idx],
            marker="o", linestyle="None", color=color,
            markersize=DATA_MARKER_MS,
            markerfacecolor=color,
            markeredgecolor=_darken(color),
            markeredgewidth=0.9,
        )

    def _plot_sparse_three_point_line(ax, x, y, color, *, upperbound_n=None):
        """Plot sparse memory as a 3-point linear interpolation: n=10, n=20, and last."""
        x = np.asarray(x, dtype=int)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(y)
        if upperbound_n is not None:
            valid &= (x <= int(upperbound_n))
        if not np.any(valid):
            return False
        xv = x[valid]
        yv = y[valid]
        idx = []
        for n_key in (10, 20):
            hits = np.where(xv == n_key)[0]
            if hits.size:
                idx.append(int(hits[0]))
        idx.append(len(xv) - 1)
        idx = np.array(sorted(set(idx)), dtype=int)
        if idx.size == 0:
            return False
        ax.plot(xv[idx], yv[idx], lw=2.0, color=color)
        ax.plot(
            xv[idx], yv[idx],
            marker="o", linestyle="None", color=color,
            markersize=DATA_MARKER_MS,
            markerfacecolor=color,
            markeredgecolor=_darken(color),
            markeredgewidth=0.9,
        )
        return True

    n_curve = np.arange(max(int(np.min(ns)), 10), int(np.max(ns)) + 1, dtype=int)

    # Memory curves: use JSON total memory where available, fall back to formula.
    _json_dense  = (_ds_json or {}).get('dense',  {})
    _json_sparse = (_ds_json or {}).get('sparse', {})
    dense_curve  = np.array([
        _json_dense.get(int(n),  _dense_memory_gb(n)) for n in n_curve
    ], dtype=float)
    sparse_curve = np.array([
        _json_sparse.get(int(n), _csr_memory_gb(n))   for n in n_curve
    ], dtype=float)

    # Scatter points per n: JSON total memory takes priority over record estimates.
    dense_pts  = np.full(ns.size, np.nan, dtype=float)
    sparse_pts = np.full(ns.size, np.nan, dtype=float)
    for i, n in enumerate(ns):
        if int(n) in _json_dense:
            dense_pts[i] = _json_dense[int(n)]
        else:
            subset_n = [r for r in records if int(r["n"]) == int(n)]
            dvals = [x for r in subset_n
                     for x in [_finite_float_or_none(r.get("dense_memory_gb_est", np.nan))]
                     if x is not None]
            if dvals:
                dense_pts[i] = float(np.mean(dvals))

        if int(n) in _json_sparse:
            sparse_pts[i] = _json_sparse[int(n)]
        else:
            subset_n = [r for r in records if int(r["n"]) == int(n)]
            svals = [x for r in subset_n
                     for x in [_finite_float_or_none(r.get("sparse_memory_gb_est", np.nan))]
                     if x is not None]
            if svals:
                sparse_pts[i] = float(np.mean(svals))

    legend_handles = []

    if plot_memory_curve and "dense" in include:
        h = _plot_curve_with_cutoff(ax_mem, n_curve, dense_curve, color="black",
                                    label="Dense", upperbound_n=dense_upperbound_n,
                                    extrap_alpha=extrap_alpha)
        if h is not None:
            legend_handles.append(Line2D(
                [0], [0], color="black", lw=2.0, marker="o", markersize=DATA_MARKER_MS,
                markerfacecolor="black", markeredgecolor=_darken("black", factor=0.25),
                markeredgewidth=0.8, label="Dense",
            ))
        _plot_markers_with_cutoff(ax_mem, ns, dense_pts, color="black",
                                  upperbound_n=dense_upperbound_n, marker_every=marker_every,
                                  skip_last=True, markersize=DATA_MARKER_MS)
        _plot_key_curve_markers(ax_mem, n_curve, dense_curve, "black",
                                upperbound_n=dense_upperbound_n)
        _ub = int(dense_upperbound_n) if dense_upperbound_n is not None else int(n_curve[-1])
        _m_sol = n_curve <= _ub
        if np.any(_m_sol):
            ax_mem.plot(n_curve[_m_sol][-1], dense_curve[_m_sol][-1],
                        marker="o", linestyle="None", color="black",
                        markerfacecolor="black", markeredgecolor=_darken("black", factor=0.25),
                        markersize=DATA_MARKER_MS, markeredgewidth=0.8, zorder=5)
        _m_dash = n_curve > _ub
        # No extrapolated white-dot markers on plot 1.

    if plot_memory_curve and "sparse" in include:
        h = _plot_sparse_three_point_line(
            ax_mem, n_curve, sparse_curve, "0.45", upperbound_n=sparse_upperbound_n
        )
        if h:
            legend_handles.append(Line2D(
                [0], [0], color="0.45", lw=2.0, marker="o", markersize=DATA_MARKER_MS,
                markerfacecolor="0.45", markeredgecolor=_darken("0.45", factor=0.25),
                markeredgewidth=0.8, label="Sparse",
            ))
        # Restore sparse extrapolated dashed segment beyond the solid-data cutoff.
        if extrapolate and sparse_upperbound_n is not None:
            _ub = int(sparse_upperbound_n)
            _m_dash = n_curve > _ub
            if np.any(_m_dash):
                _first_dash = np.where(_m_dash)[0][0]
                _start_idx = max(_first_dash - 1, 0)
                ax_mem.plot(
                    n_curve[_start_idx:],
                    sparse_curve[_start_idx:],
                    color="0.45",
                    lw=PROJECTED_LW,
                    ls=PROJECTED_DASH,
                    alpha=extrap_alpha,
                )

    # ── dense/sparse runtime curves on ax_time from DENSE_SPARSE3 JSON ───────
    if _ds_json is not None:
        for _mode, _color, _ub in (
            ('dense', 'black', dense_upperbound_n),
            ('sparse', '0.45', sparse_upperbound_n),
        ):
            _t_map = _ds_json.get(f'{_mode}_t', {})
            if not _t_map:
                continue
            _ns_t = np.array(sorted(_t_map), dtype=int)
            _t_med = np.array([_t_map[n][0] for n in _ns_t], dtype=float)
            _t_lo = np.array([_t_map[n][1] for n in _ns_t], dtype=float)
            _t_hi = np.array([_t_map[n][2] for n in _ns_t], dtype=float)
            _valid = np.isfinite(_t_med) & (_ns_t >= 10)
            if _ub is not None:
                _valid &= (_ns_t <= int(_ub))
            if not np.any(_valid):
                continue
            _xt, _yt = _ns_t[_valid], _t_med[_valid]
            _yt_lo, _yt_hi = _t_lo[_valid], _t_hi[_valid]
            ax_time.plot(_xt, _yt, lw=2.0, color=_color)
            _mk = _marker_mask(_xt, marker_every=marker_every, skip_last=True)
            ax_time.plot(_xt[_mk], _yt[_mk], marker="o", linestyle="None", color=_color,
                         markeredgecolor=_darken(_color), markeredgewidth=0.6,
                         markersize=DATA_MARKER_MS)
            ax_time.fill_between(_xt, _yt_lo, _yt_hi, alpha=0.12, color=_color)
            _terminal(ax_time, _xt[-1], _yt[-1], _color)
            if _mode == 'sparse':
                _time_sparse_last_y = float(_yt[-1])
                _hits_n10 = np.where(_xt == 10)[0]
                if _hits_n10.size:
                    _i10 = int(_hits_n10[0])
                    _time_sparse_n10_y = float(_yt[_i10])
                    _time_sparse_n10_lo = float(_yt_lo[_i10])
            # Exponential-fit projection for dense/sparse runtime beyond the cutoff.
            if extrapolate and _ub is not None and chi1_extrap_n is not None and _xt.size >= 2:
                _n_ext = float(chi1_extrap_n)
                _n_last = float(_xt[-1])
                if _n_ext > _n_last:
                    _fit_x = _xt.astype(float)
                    _fit_y = _yt.astype(float)
                    _fit_mask = np.isfinite(_fit_y) & (_fit_y > 0)
                    _fit_x = _fit_x[_fit_mask]
                    _fit_y = _fit_y[_fit_mask]
                    if _fit_x.size >= 2 and np.all(_fit_y > 0):
                        _pred = _fit_expscale(_fit_x, _fit_y)
                        _y_ext = float(_pred(_n_ext))
                        if np.isfinite(_y_ext) and _y_ext > 0:
                            ax_time.plot(
                                [_n_last, _n_ext], [float(_fit_y[-1]), _y_ext],
                                color=_color, lw=PROJECTED_LW, ls=PROJECTED_DASH, alpha=0.90,
                            )
                            _proj_ymax[ax_time] = max(_proj_ymax.get(ax_time, 0.0), _y_ext)
            if not plot_memory_curve:
                legend_handles.append(
                    Line2D([0], [0], color=_color, lw=2.0, label="Dense" if _mode == 'dense' else "Sparse")
                )

    # ── dense/sparse k (matvecs-to-target) on ax_matvec from JSON ────────────
    if _ds_json is not None:
        for _mode, _color, _ub in (
            ('dense',  'black', dense_upperbound_n),
            ('sparse', '0.45', sparse_upperbound_n),
        ):
            _k_map = _ds_json.get(f'{_mode}_k', {})
            if not _k_map:
                continue
            _ns_k  = np.array(sorted(_k_map), dtype=int)
            _k_med = np.array([_k_map[n][0] for n in _ns_k], dtype=float)
            _k_lo  = np.array([_k_map[n][1] for n in _ns_k], dtype=float)
            _k_hi  = np.array([_k_map[n][2] for n in _ns_k], dtype=float)
            _valid = np.isfinite(_k_med)
            if _ub is not None:
                _valid &= (_ns_k <= int(_ub))
            if not np.any(_valid):
                continue
            _xk, _yk = _ns_k[_valid], _k_med[_valid]
            _yk_lo, _yk_hi = _k_lo[_valid], _k_hi[_valid]
            ax_matvec.plot(_xk, _yk, lw=2.0, color=_color)
            _plot_key_curve_markers(ax_matvec, _xk, _yk, _color)
            ax_matvec.fill_between(_xk, _yk_lo, _yk_hi, alpha=0.12, color=_color)
            _terminal(ax_matvec, _xk[-1], _yk[-1], _color)

    chi_legend_handles = []
    any_failed = False
    _kr_extrap_ymax = _proj_ymax   # keep existing name for downstream y-limit code

    for chi in chis:
        mem_c  = np.full(ns.size, np.nan, dtype=float)
        mem_lo = np.full(ns.size, np.nan, dtype=float)
        mem_hi = np.full(ns.size, np.nan, dtype=float)
        t_c    = np.full(ns.size, np.nan, dtype=float)
        t_lo   = np.full(ns.size, np.nan, dtype=float)
        t_hi   = np.full(ns.size, np.nan, dtype=float)
        mv_c   = np.full(ns.size, np.nan, dtype=float)
        mv_lo  = np.full(ns.size, np.nan, dtype=float)
        mv_hi  = np.full(ns.size, np.nan, dtype=float)
        fk_c   = np.full(ns.size, np.nan, dtype=float)

        for i, n in enumerate(ns):
            subset = [r for r in records
                      if int(r["n"]) == int(n) and int(r["chi"]) == int(chi)]
            if not subset:
                continue
            # Total working-set memory = sketch (Omega+Y) + oracle output MPS.
            # Prefer hit-point values; fall back to final-k values.
            def _total_mem(r):
                s = _finite_float_or_none(r.get("time_to_target_sketch_memory_gb"))
                if s is None:
                    s = _finite_float_or_none(r.get("final_sketch_memory_gb"))
                o = _finite_float_or_none(r.get("time_to_target_oracle_memory_gb"))
                if o is None:
                    o = _finite_float_or_none(r.get("final_oracle_memory_gb"))
                if s is None:
                    return None
                return s + (o if o is not None else 0.0)
            mem_vals = [x for r in subset for x in [_total_mem(r)] if x is not None]
            mem_c[i], mem_lo[i], mem_hi[i] = _summarize(
                mem_vals, mode=mode, q_low=q_low, q_high=q_high)
            t_vals = [x for v in (r.get("time_to_target_sec", None) for r in subset)
                      for x in [_finite_float_or_none(v)] if x is not None]
            t_c[i], t_lo[i], t_hi[i] = _summarize(
                t_vals, mode=mode, q_low=q_low, q_high=q_high)
            mv_vals = [float(v) for r in subset
                       for v in [_record_matvecs_to_target(r)] if v is not None]
            mv_c[i], mv_lo[i], mv_hi[i] = _summarize(
                mv_vals, mode=mode, q_low=q_low, q_high=q_high)
            fk_vals = [x for v in (r.get("final_k", np.nan) for r in subset)
                       for x in [_finite_float_or_none(v)] if x is not None]
            fk_c[i], _, _ = _summarize(
                fk_vals, mode=mode, q_low=q_low, q_high=q_high)

        color     = chi_colors[int(chi)]
        z_chi_terminal = 1.8 + 0.02 * chi_rank[int(chi)]
        z_chi_curve = 2.0 + 0.05 * chi_rank[int(chi)]
        is_kr     = int(chi) == 1
        linestyle = "-"
        marker    = "o"
        chi_label = "Gaussian--Kronecker" if is_kr else f"$\\mathsf{{rMPS}}$"

        plotted   = False

        mask_m = np.isfinite(mem_c)
        if np.any(mask_m):
            xm, mm = ns[mask_m], mem_c[mask_m]
            _tm = (xm < int(chi1_extrap_n)) if (is_kr and chi1_extrap_n is not None) else np.ones(len(xm), dtype=bool)
            ax_mem.plot(xm[_tm], mm[_tm], lw=2.0, color=color, ls=linestyle, zorder=z_chi_curve)
            mk = _marker_mask(xm[_tm], marker_every=marker_every, skip_last=False)
            ax_mem.plot(xm[_tm][mk], mm[_tm][mk], marker=marker, linestyle="None", color=color,
                        markeredgecolor=_darken(color), markeredgewidth=0.6,
                        markersize=DATA_MARKER_MS, zorder=z_chi_curve)
            ax_mem.fill_between(xm[_tm], mem_lo[mask_m][_tm], mem_hi[mask_m][_tm], alpha=0.20, color=color, zorder=z_chi_curve - 0.2)
            plotted = True

        if _kmax is not None:
            exhausted_mask = np.isfinite(fk_c) & (fk_c >= float(_kmax))
        else:
            exhausted_mask = np.zeros(ns.size, dtype=bool)

        mask_t = np.isfinite(t_c)
        if np.any(mask_t):
            xt, tt = ns[mask_t], t_c[mask_t]
            _tt = (xt < int(chi1_extrap_n)) if (is_kr and chi1_extrap_n is not None) else np.ones(len(xt), dtype=bool)
            ax_time.plot(xt[_tt], tt[_tt], lw=2.0, color=color, ls=linestyle)
            _plot_mask_t = mask_t.copy()
            if is_kr and chi1_extrap_n is not None:
                _plot_mask_t &= (ns < int(chi1_extrap_n))
            mk_t_full = np.zeros(ns.size, dtype=bool)
            mk_t_full[np.where(_plot_mask_t)[0]] = _marker_mask(
                ns[_plot_mask_t], marker_every=marker_every, skip_last=False
            )
            hit_t = mk_t_full & ~exhausted_mask
            if np.any(hit_t):
                ax_time.plot(ns[hit_t], t_c[hit_t],
                             marker=marker, linestyle="None", color=color,
                             markeredgecolor=_darken(color), markeredgewidth=0.6,
                             markersize=DATA_MARKER_MS)
            fail_t = mk_t_full & exhausted_mask
            if np.any(fail_t):
                any_failed = True
            for n_f, t_f in zip(ns[fail_t], t_c[fail_t]):
                ax_time.plot(n_f, t_f, marker="x", linestyle="None",
                             color=color, markeredgecolor=_darken(color, factor=0.25),
                             markersize=9, markeredgewidth=2.0, zorder=5)
            ax_time.fill_between(xt[_tt], t_lo[mask_t][_tt], t_hi[mask_t][_tt], alpha=0.20, color=color)
            plotted = True

        mask_mv = np.isfinite(mv_c)
        if np.any(mask_mv):
            xv, mv = ns[mask_mv], mv_c[mask_mv]
            _tv = (xv < int(chi1_extrap_n)) if (is_kr and chi1_extrap_n is not None) else np.ones(len(xv), dtype=bool)
            ax_matvec.plot(xv[_tv], mv[_tv], lw=2.0, color=color, ls=linestyle)
            _plot_mask_mv = mask_mv.copy()
            if is_kr and chi1_extrap_n is not None:
                _plot_mask_mv &= (ns < int(chi1_extrap_n))
            mk_mv_full = np.zeros(ns.size, dtype=bool)
            mk_mv_full[np.where(_plot_mask_mv)[0]] = _marker_mask(
                ns[_plot_mask_mv], marker_every=marker_every, skip_last=False
            )
            hit_mv = mk_mv_full & ~exhausted_mask
            if np.any(hit_mv):
                ax_matvec.plot(ns[hit_mv], mv_c[hit_mv],
                               marker=marker, linestyle="None", color=color,
                               markeredgecolor=_darken(color), markeredgewidth=0.6,
                               markersize=DATA_MARKER_MS)
            fail_mv = mk_mv_full & exhausted_mask
            if np.any(fail_mv):
                any_failed = True
            for n_f, mv_f in zip(ns[fail_mv], mv_c[fail_mv]):
                ax_matvec.plot(n_f, mv_f, marker="x", linestyle="None",
                               color=color, markeredgecolor=_darken(color, factor=0.25),
                               markersize=9, markeredgewidth=2.0, zorder=5)
            ax_matvec.fill_between(xv[_tv], mv_lo[mask_mv][_tv], mv_hi[mask_mv][_tv], alpha=0.20, color=color)
            plotted = True

        if plotted and (not is_kr or "KR" in include):
            chi_legend_handles.append(Line2D(
                [0], [0], color=color, lw=2.0, ls=linestyle,
                marker=marker, markersize=DATA_MARKER_MS, label=chi_label,
                markerfacecolor=color,
                markeredgecolor=_darken(color, factor=0.25),
                markeredgewidth=0.8,
            ))

        # Add back exponential-fit dashed lines for KR (chi=1) on plots 2 and 3 only.
        if extrapolate and is_kr and chi1_extrap_n is not None:
            for _ax, _yc, _mask in ((ax_time, t_c, mask_t), (ax_matvec, mv_c, mask_mv)):
                if not np.any(_mask):
                    continue
                _xf = ns[_mask].astype(float)
                _yf = _yc[_mask]
                _fit_mask = np.isfinite(_yf) & (_yf > 0)
                _xf = _xf[_fit_mask]
                _yf = _yf[_fit_mask]
                if len(_xf) < 2:
                    continue
                _n_ext = float(chi1_extrap_n)
                _pred_mask = _xf != _n_ext
                _xfit = _xf[_pred_mask]
                _yfit = _yf[_pred_mask]
                if len(_xfit) < 2 or not np.all(_yfit > 0):
                    continue
                _pred = _fit_expscale(_xfit, _yfit)
                _y_ext = float(_pred(_n_ext))
                if not np.isfinite(_y_ext) or _y_ext <= 0:
                    continue
                _ax.plot([float(_xfit[-1]), _n_ext], [float(_yfit[-1]), _y_ext],
                         color=color, lw=PROJECTED_LW, ls=PROJECTED_DASH, alpha=0.90, zorder=z_chi_curve - 0.05)
                _kr_extrap_ymax[_ax] = max(_kr_extrap_ymax.get(_ax, 0.0), _y_ext)

    # ── budget lines ──────────────────────────────────────────────────────────
    _draw_budget_lines(ax_time, ax_matvec, budget_time_s=budget_time_s, kmax=_kmax)

    ax_mem.set_xlabel("Tensor order n", fontsize=AXIS_LABEL_FS)
    ax_mem.set_ylabel("Memory (gigabytes)", fontsize=Y_AXIS_LABEL_FS)
    ax_mem.set_title("Memory", fontsize=TITLE_FS, y=TITLE_Y)
    ax_mem.set_yscale("log")
    ax_mem.set_ylim(top=1e16)

    for y_gb, label in (
        (1.0 / 1024.0, "1 MB"),
        (1.0, "1 GB"), (1024.0, "1 TB"),
        (1024.0**2, "1 PB"), (float(all_memory_on_earth_gb), "All memory on earth"),
    ):
        ax_mem.axhline(y_gb, linestyle="-", linewidth=1.0, color="gray", alpha=0.45, zorder=0)
        ax_mem.text(0.01, y_gb, label, transform=ax_mem.get_yaxis_transform(),
                    color="gray", alpha=1.0, va="bottom", ha="left", fontsize=MEM_LABEL_FS, zorder=0)

    ax_time.set_xlabel("Tensor order n", fontsize=AXIS_LABEL_FS)
    ax_time.set_ylabel("Time (s) to reach\n1\\% relative trace error", fontsize=Y_AXIS_LABEL_FS)
    ax_time.set_yscale("log")
    _time_floor_candidates = []
    if _time_sparse_n10_y is not None and np.isfinite(_time_sparse_n10_y) and _time_sparse_n10_y > 0:
        _time_floor_candidates.append(_time_sparse_n10_y)
    if _time_sparse_n10_lo is not None and np.isfinite(_time_sparse_n10_lo) and _time_sparse_n10_lo > 0:
        _time_floor_candidates.append(_time_sparse_n10_lo)
    if _time_floor_candidates:
        ax_time.set_ylim(bottom=min(_time_floor_candidates) * 0.80)
    _time_top = 5e4
    ax_time.set_ylim(top=_time_top)
    ax_time.set_xlim(left=9)
    ax_time.set_title("Time", fontsize=TITLE_FS, y=TITLE_Y)

    ax_matvec.set_xlabel("Tensor order n", fontsize=AXIS_LABEL_FS)
    ax_matvec.set_ylabel(
        "Evaluations to reach\n1\\% relative trace error",
        fontsize=Y_AXIS_LABEL_FS,
    )
    ax_matvec.set_title(
        r"Evaluations of $\boldsymbol{x} \mapsto e^{-\beta \mathbf{H}}\boldsymbol{x}$",
        fontsize=TITLE_FS,
        y=TITLE_Y,
    )
    _mv_base = float(_kmax) + 60 if _kmax is not None else 0.0
    if _mv_base > 0:
        ax_matvec.set_ylim(top=_mv_base * 1.35)

    x_lo = int(np.floor(np.min(ns)))
    x_hi = int(np.ceil(np.max(ns)))
    x_ticks = np.arange(x_lo, x_hi + 1, X_TICK_STEP, dtype=int)
    for ax in (ax_mem, ax_time, ax_matvec):
        ax.set_xticks(x_ticks)

    for ax in (ax_mem, ax_time, ax_matvec):
        ax.tick_params(axis="both", which="both", labelsize=TICK_FS)

    legend_handles.extend(chi_legend_handles)
    if any_failed:
        legend_handles.append(Line2D(
            [0], [0], color="red", lw=0, marker="x",
            markersize=9, markeredgewidth=2.2, label="Budget exhausted",
        ))

    if legend_handles:
        legend = fig.legend(
            handles=legend_handles, loc="upper center", ncol=len(legend_handles),
            frameon=True, fancybox=True, bbox_to_anchor=(0.5, 1.07), fontsize=LEGEND_TOP_FS,
        )
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(1.5)

    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.16, top=0.82, wspace=0.32)

    if save_svg_path is not None:
        fig.savefig(str(save_svg_path), format="svg", bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


# ── hit-rate table helper ─────────────────────────────────────────────────────

def _group_records(records):
    """Return dict  (n, chi) -> list[record]."""
    groups = {}
    for rec in records:
        key = (int(rec["n"]), int(rec["chi"]))
        groups.setdefault(key, []).append(rec)
    return groups


def print_hit_rate_table(out):
    """Print a text table of hit rates by (n, chi)."""
    records    = out.get("records", [])
    groups     = _group_records(records)
    n_values   = sorted({r["n"]   for r in records})
    chi_values = sorted({r["chi"] for r in records})

    # header
    hdr = "  n  |" + "".join(f"  χ={c:2d}" for c in chi_values)
    print(hdr)
    print("-" * len(hdr))

    for n in n_values:
        row = f" {n:3d} |"
        for chi in chi_values:
            recs = groups.get((n, chi), [])
            if not recs:
                row += "    - "
                continue
            hits = sum(1 for r in recs if r.get("hit", False))
            row += f" {hits}/{len(recs)} "
        print(row)
