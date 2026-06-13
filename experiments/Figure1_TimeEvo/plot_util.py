import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def _darken(color, factor=0.25):
    """Return a color darkened by factor (0=unchanged, 1=black)."""
    r, g, b = mpl.colors.to_rgb(color)
    f = 1.0 - factor
    return (r * f, g * f, b * f)


def _set_pub_rc():
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"""
\usepackage{newtxtext}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
""",
            "font.family": "serif",
            "axes.labelsize": 11,
            "legend.fontsize": 20,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "path.simplify": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plasma_cmap():
    return plt.cm.plasma


def _resolve_cmap(cmap, *, cmap_lo=0.0, cmap_hi=1.0):
    cmap_lo = float(cmap_lo)
    cmap_hi = float(cmap_hi)
    if not (0.0 <= cmap_lo <= 1.0 and 0.0 <= cmap_hi <= 1.0 and cmap_lo < cmap_hi):
        raise ValueError("cmap_lo and cmap_hi must satisfy 0 <= cmap_lo < cmap_hi <= 1")

    if cmap is None:
        base = _plasma_cmap()
    elif isinstance(cmap, mpl.colors.Colormap):
        base = cmap
    elif isinstance(cmap, str):
        name = str(cmap).strip()
        if not name:
            base = _plasma_cmap()
        else:
            try:
                import plotly.express as px

                base = None
                for group_name in ("sequential", "diverging", "cyclical"):
                    group = getattr(px.colors, group_name, None)
                    if group is None:
                        continue
                    for attr in dir(group):
                        if attr.startswith("_"):
                            continue
                        if attr.lower() != name.lower():
                            continue
                        scale = getattr(group, attr)
                        if isinstance(scale, (list, tuple)) and len(scale) > 0:
                            base = LinearSegmentedColormap.from_list(
                                f"plotly_{attr.lower()}",
                                list(scale),
                            )
                            break
                    if base is not None:
                        break
            except Exception:
                base = None

            if base is None:
                try:
                    base = mpl.colormaps[name]
                except Exception:
                    raise ValueError(f"Unknown colormap {cmap!r}")
    else:
        raise TypeError("cmap must be None, a colormap name, or a matplotlib colormap")

    if cmap_lo == 0.0 and cmap_hi == 1.0:
        return base

    return LinearSegmentedColormap.from_list(
        f"{getattr(base, 'name', 'cmap')}_trunc_{cmap_lo:.3f}_{cmap_hi:.3f}",
        base(np.linspace(cmap_lo, cmap_hi, 256)),
    )


def _summarize(vals, *, mode="mean_std", q_low=0.25, q_high=0.75):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan

    if mode == "mean_std":
        c = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
        return c, c - s, c + s

    if mode == "median_quantile":
        c = float(np.median(arr))
        lo = float(np.quantile(arr, q_low))
        hi = float(np.quantile(arr, q_high))
        return c, lo, hi

    raise ValueError("mode must be 'mean_std' or 'median_quantile'")


def _finite_float_or_none(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _record_sweeps(record):
    for key in ("tdvp_sweeps", "sweeps"):
        if key in record and record[key] is not None:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                return None
    return None


def _record_matvecs_to_target(record):
    for key in ("matvecs_to_target", "hit_k"):
        if key in record and record[key] is not None:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                return None
    return None


def _marker_mask(x, marker_every=2, skip_last=False):
    x = np.asarray(x)
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    keep = np.zeros(x.size, dtype=bool)
    keep[:: max(1, int(marker_every))] = True
    keep[-1] = not skip_last   # always explicitly set the last entry
    return keep


def _plot_curve_with_cutoff(
    ax,
    x,
    y,
    *,
    color,
    label=None,
    upperbound_n=None,
    linestyle_main="-",
    extrap_alpha=0.70,
    extrap_linestyle=(0, (2.0, 1.6)),
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return None

    plot_kwargs = {
        "lw": 2.0,
        "color": color,
        "linestyle": linestyle_main,
    }
    if label is not None:
        plot_kwargs["label"] = label

    if upperbound_n is None:
        (h,) = ax.plot(x, y, **plot_kwargs)
        return h

    ub = int(upperbound_n)
    solid_mask = x <= ub
    extrap_mask = x > ub

    if np.any(solid_mask):
        (h,) = ax.plot(x[solid_mask], y[solid_mask], **plot_kwargs)
    else:
        (h,) = ax.plot([], [], **plot_kwargs)

    if np.any(extrap_mask):
        first_extrap_idx = int(np.argmax(extrap_mask))
        start_idx = max(first_extrap_idx - 1, 0)
        ax.plot(
            x[start_idx:],
            y[start_idx:],
            lw=2.0,
            color=color,
            linestyle=extrap_linestyle,
            alpha=float(extrap_alpha),
        )

    return h


def _plot_markers_with_cutoff(ax, x, y, *, color, upperbound_n=None, marker_every=2,
                              skip_last=False, markersize=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return

    ec = _darken(color)
    if upperbound_n is None:
        m = _marker_mask(x, marker_every=marker_every, skip_last=skip_last)
        kwargs = {}
        if markersize is not None:
            kwargs["markersize"] = float(markersize)
        ax.plot(x[m], y[m], marker="o", linestyle="None", color=color,
                markeredgecolor=ec, markeredgewidth=0.6, **kwargs)
        return

    ub = int(upperbound_n)
    solid_mask = x <= ub
    extrap_mask = x > ub

    xs, ys = x[solid_mask], y[solid_mask]
    if xs.size > 0:
        ms = _marker_mask(xs, marker_every=marker_every, skip_last=skip_last)
        if np.any(ms):
            kwargs = {}
            if markersize is not None:
                kwargs["markersize"] = float(markersize)
            ax.plot(xs[ms], ys[ms], marker="o", linestyle="None", color=color,
                    markeredgecolor=ec, markeredgewidth=0.6, **kwargs)

    # no markers on the extrapolated portion — dashed line only


def _dense_memory_gb(n, dtype=np.float64):
    dim = 1 << int(n)
    return float(dim * dim * np.dtype(dtype).itemsize) / (1024.0**3)


def _csr_memory_gb(n, dtype=np.float64, index_dtype=np.int32, indptr_dtype=np.int64):
    n = int(n)
    dim = 1 << n
    nnz = dim * (n + 1)
    b_data = nnz * np.dtype(dtype).itemsize
    b_idx = nnz * np.dtype(index_dtype).itemsize
    b_ptr = (dim + 1) * np.dtype(indptr_dtype).itemsize
    return float(b_data + b_idx + b_ptr) / (1024.0**3)


def _first_hit_row(rows, record):
    hit_k = record.get("hit_k", None)
    if hit_k is not None:
        for row in rows:
            if int(row.get("k", -1)) == int(hit_k):
                return row

    target = _finite_float_or_none(record.get("target_accuracy", None))
    if target is not None:
        for row in rows:
            rel = _finite_float_or_none(row.get("relerr", None))
            if rel is not None and rel <= float(target):
                return row

    return None


def _enrich_record_from_case_rows(record, case, rows):
    out = dict(record)

    n = out.get("n", case.get("n", None))
    if n is not None:
        out.setdefault("dense_memory_gb_est", float(_dense_memory_gb(n)))
        out.setdefault("sparse_memory_gb_est", float(_csr_memory_gb(n)))

    if "tdvp_sweeps" in case and case.get("tdvp_sweeps", None) is not None:
        out.setdefault("tdvp_sweeps", int(case["tdvp_sweeps"]))
    if "kmax" in case and case.get("kmax", None) is not None:
        out.setdefault("kmax", int(case["kmax"]))

    if not rows:
        return out

    final_row = rows[-1]
    hit_row = _first_hit_row(rows, out)

    if hit_row is not None:
        if "sketch_memory_gb" in hit_row:
            out.setdefault("time_to_target_sketch_memory_gb", float(hit_row["sketch_memory_gb"]))
        if "omega_memory_gb" in hit_row:
            out.setdefault("time_to_target_omega_memory_gb", float(hit_row["omega_memory_gb"]))
        if "y_memory_gb" in hit_row:
            out.setdefault("time_to_target_y_memory_gb", float(hit_row["y_memory_gb"]))

    if "sketch_memory_gb" in final_row:
        out.setdefault("final_sketch_memory_gb", float(final_row["sketch_memory_gb"]))
    if "omega_memory_gb" in final_row:
        out.setdefault("final_omega_memory_gb", float(final_row["omega_memory_gb"]))
    if "y_memory_gb" in final_row:
        out.setdefault("final_y_memory_gb", float(final_row["y_memory_gb"]))

    return out


def _enrich_records_from_nested_results(race_out, records):
    lookup = {}

    if isinstance(race_out, dict) and "runs" in race_out:
        for run in race_out.get("runs", []):
            for trial_idx, trial in enumerate(run.get("trials", [])):
                for chi, result in trial.get("results", {}).items():
                    case = dict(result.get("case", {}))
                    rows = list(result.get("rows", []))
                    key = (
                        int(case.get("n", run.get("n"))),
                        int(chi),
                        int(trial_idx),
                    )
                    lookup[key] = (case, rows)
    elif isinstance(race_out, dict) and "trials" in race_out:
        for trial_idx, trial in enumerate(race_out.get("trials", [])):
            cfg = trial.get("cfg", {})
            n_val = int(cfg.get("n"))
            for chi, result in trial.get("results", {}).items():
                case = dict(result.get("case", {}))
                rows = list(result.get("rows", []))
                key = (
                    int(case.get("n", n_val)),
                    int(chi),
                    int(trial_idx),
                )
                lookup[key] = (case, rows)

    if not lookup:
        return records

    enriched = []
    for record in records:
        key = (
            int(record["n"]),
            int(record["chi"]),
            int(record.get("trial_index", max(int(record.get("trial", 1)) - 1, 0))),
        )
        case_rows = lookup.get(key, None)
        if case_rows is None:
            enriched.append(dict(record))
            continue
        case, rows = case_rows
        enriched.append(_enrich_record_from_case_rows(record, case, rows))

    return enriched


def _extract_records(race_out):
    if isinstance(race_out, dict) and "records" in race_out:
        records = race_out["records"]
    else:
        records = race_out
    records = list(records)
    if len(records) == 0:
        raise ValueError("No records to plot")
    return _enrich_records_from_nested_results(race_out, records)


def _load_dense_sparse_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)

    ns = np.asarray(z["ns"], dtype=int)
    out = {
        "ns": ns,
        "status_dense": np.asarray(z["status_dense"], dtype=object),
        "status_sparse": np.asarray(z["status_sparse"], dtype=object),
        "dense_times_s": np.asarray(z["dense_times_s"], dtype=float),
        "sparse_times_s": np.asarray(z["sparse_times_s"], dtype=float),
    }

    if "mem_dense_gb_est" in z:
        out["mem_dense_gb_est"] = np.asarray(z["mem_dense_gb_est"], dtype=float)
    if "mem_sparse_gb_est" in z:
        out["mem_sparse_gb_est"] = np.asarray(z["mem_sparse_gb_est"], dtype=float)

    return out


def _load_dense_sparse_json(json_path):
    """Load a DENSE_SPARSE3-style JSON summary.

    Returns a dict with:
      'dense'   / 'sparse'   : {n: total_memory_gb_median}         (memory curves)
      'dense_k' / 'sparse_k' : {n: (k_median, k_q10, k_q90)}       (matvec plot)
      'dense_t' / 'sparse_t' : {n: (t_median, t_q10, t_q90)}       (time plot)
    """
    import json as _json
    with open(json_path) as f:
        data = _json.load(f)
    out = {}
    for mode in ('dense', 'sparse'):
        if mode not in data:
            continue
        m = data[mode]
        out[mode] = {
            int(n): float(total)
            for n, total in zip(m['n'], m['total_memory_gb_median'])
            if total is not None
        }
        out[f'{mode}_k'] = {
            int(n): (float(km), float(klo), float(khi))
            for n, km, klo, khi in zip(
                m['n'], m['k_median'], m['k_q10'], m['k_q90']
            )
            if km is not None
        }
        out[f'{mode}_t'] = {
            int(n): (float(tm), float(tlo), float(thi))
            for n, tm, tlo, thi in zip(
                m['n'], m['runtime_s_median'], m['runtime_s_q10'], m['runtime_s_q90']
            )
            if tm is not None
        }
    return out


def _finite_row(arr2d, j):
    row = np.asarray(arr2d[j], dtype=float)
    return row[np.isfinite(row)]

def _stack_relerr_curves(trials, chi_values):
    chi_values = [int(chi) for chi in chi_values]
    if len(trials) == 0:
        raise ValueError("trials is empty")
    if len(chi_values) == 0:
        raise ValueError("chi_values is empty")

    first_chi = chi_values[0]
    first_rows = trials[0]["results"][first_chi]
    if len(first_rows) == 0:
        raise ValueError("No rows found for first chi in first trial")

    x_key = "k" if "k" in first_rows[0] else "q"
    x = np.array([r[x_key] for r in first_rows], dtype=int)

    mps_stack = {}
    for chi in chi_values:
        mps_stack[int(chi)] = np.vstack(
            [
                np.array(
                    [r["relerr"] for r in trial["results"][int(chi)]],
                    dtype=float,
                )
                for trial in trials
            ]
        )

    return x, x_key, mps_stack


def _mean_min_max(arr2d):
    return (
        np.mean(arr2d, axis=0),
        np.min(arr2d, axis=0),
        np.max(arr2d, axis=0),
    )


def plot_relerr_focus_averaged(trials, chi_values, n):
    _ = int(n)
    _set_pub_rc()

    x, x_key, mps_stack = _stack_relerr_curves(trials, chi_values)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    chis = np.array(sorted(int(c) for c in chi_values), dtype=int)

    cmap_base = plt.cm.Blues
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc_plasma",
        cmap_base(np.linspace(0.15, 0.95, 256)),
    )
    norm = mpl.colors.Normalize(vmin=chis.min(), vmax=chis.max())

    def frac(chi):
        if chis.max() == chis.min():
            return 1.0
        return (chi - chis.min()) / (chis.max() - chis.min())

    for y in [1e-2, 1e-4, 1e-6]:
        ax.axhline(y, color="0.82", linewidth=0.9, linestyle=":", zorder=0)

    for chi in chis:
        arr = mps_stack[int(chi)]
        mean, lo, hi = _mean_min_max(arr)

        t = frac(chi)
        color = cmap(norm(chi))
        lw = 1.0 + 1.2 * (t**1.15)

        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0.0, zorder=1)
        ax.plot(
            x,
            mean,
            marker="o",
            markersize=3.5,
            linewidth=lw,
            color=color,
            zorder=2,
        )

        ax.text(
            x[-1] + 0.22,
            mean[-1],
            rf"$\chi={chi}$",
            color=color,
            fontsize=9,
            va="center",
            ha="left",
        )

    ax.set_yscale("log")
    ax.set_xlabel("k" if x_key == "k" else "q")
    ax.set_ylabel(r"$\mathrm{relerr}(\hat z,z)=\frac{|\hat z-z|}{|z|}$")
    ax.set_xlim(x.min() - 0.2, x.max() + 1.8)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, location="right")
    cbar.set_label(r"$\chi$")

    return fig, ax


__all__ = [
    "plot_racedriver_memory_time",
    "plot_relerr_focus_averaged",
]
