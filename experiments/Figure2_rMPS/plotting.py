import importlib.util
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ── Register Scientific Colour Maps ──────────────────────────────────────────
_CMAPS_DIR = Path(__file__).parent.parent / "ScientificColourMaps8"
for _name in ["oslo", "devon", "lipari", "davos", "lajolla", "acton",
              "batlow", "glasgow", "bamako", "bilbao", "navia", "imola"]:
    _cmap_file = _CMAPS_DIR / _name / f"{_name}.py"
    if not _cmap_file.exists():
        continue
    _spec = importlib.util.spec_from_file_location(f"{_name}_module", _cmap_file)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _cmap = getattr(_mod, f"{_name}_map")
    for _reg_name, _reg_cmap in [(_name, _cmap), (f"{_name}_r", _cmap.reversed())]:
        try:
            mpl.colormaps.unregister(_reg_name)
        except Exception:
            pass
        mpl.colormaps.register(_reg_cmap, name=_reg_name)

# ── Font sizes ────────────────────────────────────────────────────────────────
AXIS_LABEL_FS = 22
LEGEND_FS     = 20
TICK_FS       = 19
CBAR_LABEL_FS = 22
CBAR_TICK_FS  = 18
EPS_TEXT_FS   = 19

# ── Line geometry ─────────────────────────────────────────────────────────────
LINE_LW_LEFT  = 2.0
LINE_LW_RIGHT = 1.8
ANALYTIC_MS   = 9.0
ANALYTIC_MEW  = 1.4

# ── Figure geometry ───────────────────────────────────────────────────────────
FIGSIZE = (16, 5)

# ── Colourmaps ────────────────────────────────────────────────────────────────
MPS_CMAP_NAME      = "devon_r"
MPS_TRUNC_LO       = 0.40
MPS_TRUNC_HI       = 0.99
RMPS_CMAP_NAME     = "coolwarm_r"
RMPS_TRUNC_LO      = 0.0
RMPS_TRUNC_HI      = 1.0
BASELINE_CMAP_NAME = "coolwarm_r"
BASELINE_TRUNC_LO  = 0.0
BASELINE_TRUNC_HI  = 1.0

mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"""
\usepackage{newtxtext}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
""",
    "font.family":        "serif",
    "axes.labelsize":     AXIS_LABEL_FS,
    "legend.fontsize":    LEGEND_FS,
    "xtick.labelsize":    TICK_FS,
    "ytick.labelsize":    TICK_FS,
    "figure.dpi":         600,
    "savefig.dpi":        600,
    "path.simplify":      False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# ── Helpers ───────────────────────────────────────────────────────────────────
class HandlerGradient(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        colors = orig_handle.get_colors()
        lw = orig_handle.get_linewidths()[0]
        x = np.linspace(0, width, len(colors) + 1)
        y = np.full_like(x, height / 2)
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        return [LineCollection(segs, colors=colors, linewidth=lw, transform=trans)]


def _gradient_line_handle(cmap, u0=0.15, u1=0.90, n=96, lw=3.0):
    x = np.linspace(0.0, 1.0, n)
    u = np.linspace(u0, u1, n)
    segs = np.stack([
        np.column_stack([x[:-1], np.ones(n - 1)]),
        np.column_stack([x[1:],  np.ones(n - 1)]),
    ], axis=1)
    return LineCollection(segs, colors=[cmap(float(t)) for t in u[:-1]], linewidths=lw)


def _jitter_duplicates(x, jitter_frac=0.012):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    step = jitter_frac * max(float(np.max(x) - np.min(x)), 1.0)
    x_plot = x.copy()
    for val, count in zip(*np.unique(x, return_counts=True)):
        if count > 1:
            idx = np.where(x == val)[0]
            x_plot[idx] += (np.arange(count) - 0.5 * (count - 1)) * step
    return x_plot


def _positive_log_limits(*arrays, pad=1.8, eps=1e-16):
    vals = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
    vals = [a[np.isfinite(a) & (a > eps)] for a in vals if a.size]
    if not vals:
        return eps, 1.0
    flat = np.concatenate(vals)
    ymin = float(np.min(flat)) / pad
    ymax = float(np.max(flat)) * pad
    return max(ymin, eps), max(ymax, 10.0 * max(ymin, eps))


# ── Main plot function ────────────────────────────────────────────────────────
def plot_nystrom(
    res,
    eps=1e-16,
    figsize=FIGSIZE,
    wspace=0.35,
    right_pad=0.08,
    xpad_frac=0.04,
    jitter_frac=0.012,
    max_xticks=12,
    xtick_rotation=0,
    show_colorbars=True,
    cbar_label=r"Bond dimension $\chi$",
    plot_nystrom=True,
):
    k_raw  = np.asarray(res["k_list"], dtype=float)
    k_plot = _jitter_duplicates(k_raw, jitter_frac=jitter_frac)
    chis        = sorted(int(c) for c in res["chi_list"])
    chi_var_list = sorted(int(c) for c in res["chi_var_list"])
    chi_vals    = np.asarray(chi_var_list, dtype=float)

    # ── Build truncated colourmaps ────────────────────────────────────────────
    blues = mpl.colors.LinearSegmentedColormap.from_list(
        f"{MPS_CMAP_NAME}_trunc",
        mpl.colormaps.get_cmap(MPS_CMAP_NAME)(np.linspace(MPS_TRUNC_LO, MPS_TRUNC_HI, 256)),
    )
    rmps_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{RMPS_CMAP_NAME}_trunc",
        mpl.colormaps.get_cmap(RMPS_CMAP_NAME)(np.linspace(RMPS_TRUNC_LO, RMPS_TRUNC_HI, 256)),
    )
    _baseline = mpl.colormaps.get_cmap(BASELINE_CMAP_NAME)
    kr_color    = _baseline(BASELINE_TRUNC_LO)
    gauss_color = _baseline(BASELINE_TRUNC_HI)

    chi_to_blue = {
        chi: blues(i / max(len(chis) - 1, 1))
        for i, chi in enumerate(chis)
    }

    # ── Layout ────────────────────────────────────────────────────────────────
    if plot_nystrom:
        fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=figsize, constrained_layout=False)
        ax0, ax1 = _ax1, _ax0
        fig.subplots_adjust(left=0.075, right=1.0 - right_pad, bottom=0.16, top=0.80, wspace=wspace + 0.08)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] / 2, figsize[1]), constrained_layout=False)
        ax0 = None
        fig.subplots_adjust(left=0.15, right=1.0 - right_pad, bottom=0.16, top=0.80)

    # ── Left panel: MPS Nyström error ─────────────────────────────────────────
    if plot_nystrom:
        y_gauss  = np.asarray(res["dense_nuc_median"], dtype=float) + eps
        lo_gauss = np.asarray(res["dense_nuc_p10"],    dtype=float)
        hi_gauss = np.asarray(res["dense_nuc_p90"],    dtype=float)

        h_gauss = ax0.plot(
            k_plot, y_gauss,
            color="black", marker="o", markersize=13.0,
            markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.7,
            linestyle="-", linewidth=1.8, zorder=6,
        )[0]
        ax0.fill_between(k_plot, np.maximum(lo_gauss, eps), hi_gauss,
                         color="black", alpha=0.10, linewidth=0.0, zorder=1)

        for chi in chis:
            y  = np.asarray(res["mps_tn_rel_fro_median"][chi], dtype=float) + eps
            lo = np.asarray(res["mps_tn_rel_fro_p10"][chi],    dtype=float)
            hi = np.asarray(res["mps_tn_rel_fro_p90"][chi],    dtype=float)
            color = chi_to_blue[chi]
            dark  = tuple(max(0.0, c * 0.65) if i < 3 else c for i, c in enumerate(color))
            ax0.plot(k_plot, y, color=color, marker="o", markersize=5.0,
                     markeredgecolor=dark, markeredgewidth=0.8,
                     linewidth=1.55, linestyle="-", zorder=4)
            ax0.fill_between(k_plot, np.maximum(lo, eps), hi,
                             color=color, alpha=0.13, linewidth=0.0, zorder=2)

        ax0.set_title(r"\textsc{Nystr\"{o}m} approximation error", fontsize=AXIS_LABEL_FS - 3)
        ax0.set_xlabel(r"Embedding dimension $k$",                  fontsize=AXIS_LABEL_FS)
        ax0.set_ylabel(r"$\|\mathbf{A}-\widehat{\mathbf{A}}\|_*/\|\mathbf{A}\|_*$", fontsize=AXIS_LABEL_FS)
        ax0.set_yscale("log")

        ticks = [10, 30, 50, 70, 90]
        ax0.set_xticks(ticks)
        ax0.set_xticklabels([str(v) for v in ticks], rotation=xtick_rotation)
        xpad = xpad_frac * max(float(np.max(k_plot) - np.min(k_plot)), 1.0)
        ax0.set_xlim(float(np.min(k_plot)) - xpad, float(np.max(k_plot)) + xpad)
        ax0.set_ylim(*_positive_log_limits(
            y_gauss, lo_gauss, hi_gauss,
            *[np.asarray(res["mps_tn_rel_fro_median"][chi], dtype=float) for chi in chis],
            eps=eps,
        ))

    # ── Right panel: rMPS quadratic form variance ──────────────────────────────
    v_rmps = np.asarray(res["quad_median_dense_rmps"], dtype=float) + eps
    t = ((chi_vals - chi_vals.min()) / (chi_vals.max() - chi_vals.min())
         if chi_vals.size > 1 else np.array([0.5]))
    rmps_colors = [rmps_cmap(float(tt)) for tt in t]

    if chi_vals.size >= 2:
        segs = np.stack([
            np.column_stack([chi_vals[:-1], v_rmps[:-1]]),
            np.column_stack([chi_vals[1:],  v_rmps[1:]]),
        ], axis=1)
        ax1.add_collection(LineCollection(segs, colors=rmps_colors[:-1], linewidths=2.8, zorder=6))
        ax1.scatter(chi_vals, v_rmps, c=rmps_colors, s=34, marker="o",
                    edgecolors=[tuple(c * 0.5 if i < 3 else c for i, c in enumerate(col)) for col in rmps_colors],
                    linewidths=0.35, zorder=7)
    else:
        ax1.plot(chi_vals, v_rmps, color=rmps_colors[0], marker="o", markersize=5, linewidth=2.4, zorder=6)

    has_gauss = res.get("gauss_var_median") is not None
    gmedian   = float(res["gauss_var_median"]) + eps if has_gauss else None
    kr_idx    = int(np.argmin(np.abs(chi_vals - 1.0))) if chi_vals.size else 0
    kr_median = float(v_rmps[kr_idx]) if v_rmps.size else gmedian

    if has_gauss:
        ax1.axhline(gmedian,   color=gauss_color, linewidth=2.0, zorder=4)
    if kr_median is not None:
        ax1.axhline(kr_median, color=kr_color,    linewidth=2.0, zorder=4)

    # Theory curve from Theorem 1.2
    matrix_dim    = 2 ** 10
    subspace_dim  = 20
    matrix_eps    = res["eps"]
    tr_A          = subspace_dim * (1.0 + matrix_eps) + (matrix_dim - subspace_dim) * matrix_eps
    frob_sq_A     = subspace_dim * (1.0 + matrix_eps) ** 2 + (matrix_dim - subspace_dim) * matrix_eps ** 2
    tensor_order  = 10

    def _var_bound(chi):
        a = (1.0 + 1.0 / chi) ** (tensor_order - 1)
        b = (1.0 + 2.0 / chi) ** (tensor_order - 1)
        return (2.0 * a * frob_sq_A + (3.0 * b - 2.0 * a - 1.0) * tr_A ** 2) / tr_A ** 2

    v_theory = np.maximum(_var_bound(chi_vals), eps) if chi_vals.size else np.array([])
    h_theory = (
        ax1.plot(chi_vals, v_theory, color="black", linestyle="--",
                 linewidth=1.8, alpha=0.6, zorder=5)[0]
        if v_theory.size
        else Line2D([], [], color="black", linestyle="--", linewidth=1.8, alpha=0.6)
    )

    # Vertical reference lines at multiples of n
    n_sites  = int(res["n"])
    chi_min  = float(np.min(chi_vals))
    chi_max  = float(np.max(chi_vals))
    mult_max = max(int(np.floor(chi_max / max(n_sites, 1))), 1)
    for mult in range(1, mult_max + 1):
        xpos = float(mult * n_sites)
        if chi_min - 0.5 <= xpos <= chi_max + 0.5:
            ax1.axvline(xpos, color="0.75", linewidth=1.0, alpha=0.35, zorder=0)
            label = r"$\chi=n$" if mult == 1 else rf"$\chi={mult}n$"
            _trans = ax1.get_xaxis_transform() + ScaledTranslation(-4.5 / 72, 0, fig.dpi_scale_trans)
            ax1.text(xpos, 1, label, transform=_trans, rotation=90,
                     color="0.45", fontsize=14, ha="right", va="top", clip_on=False)

    ax1.axvline(1.0,  color="0.70", linewidth=1.0, alpha=0.50, zorder=0)
    ax1.axvline(60.0, color="0.55", linewidth=1.2, alpha=0.65, zorder=0)
    _6n_trans = ax1.get_xaxis_transform() + ScaledTranslation(-4.5 / 72, 0, fig.dpi_scale_trans)
    ax1.text(60.0, 1, r"$\chi=6n$", transform=_6n_trans, rotation=90,
             color="0.35", fontsize=14, ha="right", va="top", clip_on=False)

    ax1.set_title("Relative quadratic form variance", fontsize=AXIS_LABEL_FS - 3)
    ax1.set_xlabel(r"Bond dimension $\chi$", fontsize=AXIS_LABEL_FS)
    ax1.set_ylabel(
        r"$\mathrm{Var}\left(\boldsymbol{\omega}^{\top}\mathbf{A}\boldsymbol{\omega}\right)/\operatorname{Tr}\left(\mathbf{A}\right)^2$",
        fontsize=AXIS_LABEL_FS,
    )
    ax1.set_yscale("log")
    ax1.set_xlim(chi_min - 0.5, max(chi_max, 60.0) + 0.5)
    xticks = np.arange(10.0 * np.floor(chi_min / 10.0), 10.0 * np.ceil(chi_max / 10.0) + 1.0, 10.0)
    ax1.set_xticks(xticks[(xticks >= chi_min - 5.0) & (xticks <= chi_max + 5.0)])
    _baseline_vals = [v for v in [gmedian, kr_median] if v is not None]
    ax1.set_ylim(*_positive_log_limits(
        v_rmps,
        v_theory if v_theory.size else None,
        np.array(_baseline_vals) if _baseline_vals else None,
        eps=eps,
    ))

    if has_gauss:
        ax1.text(0.98, gmedian, "Gaussian", transform=ax1.get_yaxis_transform(),
                 color=gauss_color, fontsize=EPS_TEXT_FS, ha="right", va="bottom", clip_on=True)
    if kr_median is not None:
        ax1.text(0.98, kr_median, "Gaussian–Kronecker", transform=ax1.get_yaxis_transform(),
                 color=kr_color, fontsize=EPS_TEXT_FS, ha="right", va="bottom", clip_on=True)

    # ── Legends & colourmaps ──────────────────────────────────────────────────
    legend_kw = dict(
        loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2,
        frameon=True, fancybox=True, framealpha=1.0,
        edgecolor="black", facecolor="white",
        borderpad=0.35, handlelength=2.25, handletextpad=0.55,
        fontsize=LEGEND_FS, handler_map={LineCollection: HandlerGradient()},
    )

    if plot_nystrom:
        ax0.legend(
            handles=[h_gauss, _gradient_line_handle(blues, u0=0.0, u1=1.0, lw=3.0)],
            labels=[r"Gaussian \textsc{Nystr\"{o}m}", r"$\mathsf{MPS}$ \textsc{GramNystr\"{o}m}"],
            **legend_kw,
        )

    ax1.legend(
        handles=[h_theory, _gradient_line_handle(rmps_cmap, u0=0.0, u1=1.0, lw=3.0)],
        labels=["Theorem 1.2", r"$\mathsf{rMPS}$"],
        **{**legend_kw, "ncol": 4},
    )

    if show_colorbars:
        cbar_kw = dict(fraction=0.035, pad=0.02)
        if plot_nystrom and chis:
            sm = mpl.cm.ScalarMappable(mpl.colors.Normalize(float(min(chis)), float(max(chis))), blues)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax0, **cbar_kw)
            cb.set_label(cbar_label, rotation=90, labelpad=8, fontsize=CBAR_LABEL_FS)
            cb.ax.tick_params(labelsize=CBAR_TICK_FS)
        if chi_vals.size:
            sm = mpl.cm.ScalarMappable(mpl.colors.Normalize(chi_vals.min(), chi_vals.max()), rmps_cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax1, **cbar_kw)
            cb.set_label(cbar_label, rotation=90, labelpad=8, fontsize=CBAR_LABEL_FS)
            cb.ax.tick_params(labelsize=CBAR_TICK_FS)

    for ax in filter(None, (ax0, ax1)):
        ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
        ax.grid(False)

    return fig, (ax0, ax1)
