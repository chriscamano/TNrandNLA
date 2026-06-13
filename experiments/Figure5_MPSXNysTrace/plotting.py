"""
plotting.py
-----------
Publication-quality plotting utilities for Figure 5.

All matplotlib/display code lives here; no experiment-running or tnrnla
imports are present.  Load JSON snapshots with `load_experiment_json`, then
call `plot_experiment_multi` / `add_shared_legend` / `finalize_figure_layout`.
"""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D


# ══════════════════════════════════════════════════════════════════════════
# Palette + style constants
# ══════════════════════════════════════════════════════════════════════════

# Colorblind-friendlier sequential palettes.
# Hue identifies the estimator. Lightness identifies chi.
METHOD_PALETTES = {
    "hutch": [
        "#D8F0E4",
        "#8DD3C7",
        "#1B9E77",
        "#005A32",
    ],
    "nystrom++": [
        "#F4D6EA",
        "#D89BC8",
        "#B54C96",
        "#6A1B5D",
    ],
    "xnystrace": [
        "#DCE9F7",
        "#8DBBE8",
        "#377EB8",
        "#084081",
    ],
}

METHOD_BASES = {
    "hutch": "#1B9E77",
    "nystrom++": "#7B4AB5",
    "xnystrace": "#377EB8",
}


def make_palette_cmap(name, colors):
    return LinearSegmentedColormap.from_list(name, colors, N=256)


METHOD_CMAPS = {
    "hutch": make_palette_cmap("hutch_green", METHOD_PALETTES["hutch"]),
    "nystrom++": make_palette_cmap("npp_purple", METHOD_PALETTES["nystrom++"]),
    "xnystrace": make_palette_cmap("xnys_blue", METHOD_PALETTES["xnystrace"]),
}

AXIS_LABEL_FS = 22
TITLE_FS = 22
LEGEND_FS = 20
TICK_FS = 19
YTICK_FS = 19

LINE_MS = 5.0
LINE_LW = 2.0
MARKER_MEW = 1.15
EDGE_DARKEN = 0.68
FILL_ALPHA = 0.2
MARKEVERY = 1

CMAP_LO = 0.3
CMAP_HI = 1.0

X_PAD = 2
FIGSIZE = (7.2, 4.4)

METHOD_STYLE = {
    "hutch": dict(marker="s", ls="-"),
    "nystrom++": dict(marker="D", ls="-"),
    "xnystrace": dict(marker="o", ls="-"),
    "nystrom++_resph": dict(marker="v", ls="--"),
    "xnystrace_resph": dict(marker="^", ls="--"),
}

METHOD_DISPLAY_NAMES = {
    "hutch": r"$\mathsf{MPS}$ Girard–Hutchinson",
    "nystrom++": r"$\mathsf{MPS}$ Nyström++",
    "xnystrace": r"$\mathsf{MPS}$ XNysTrace",
    "nystrom++_resph": r"$\mathsf{MPS}$ Nyström++ (resph.)",
    "xnystrace_resph": r"$\mathsf{MPS}$ XNysTrace (resph.)",
}

ALL_METHODS_WITH_RESPH = [
    "hutch",
    "nystrom++",
    "xnystrace",
    "nystrom++_resph",
    "xnystrace_resph",
]

# Draw order used when filtering/sorting method lists
plot_order = [
    "hutch",
    "nystrom++",
    "xnystrace",
    "nystrom++_resph",
    "xnystrace_resph",
]


# ══════════════════════════════════════════════════════════════════════════
# Colour helpers
# ══════════════════════════════════════════════════════════════════════════

def darken(color, factor=0.75):
    r, g, b = to_rgb(color)
    return (factor * r, factor * g, factor * b)


def setup_plot_style():
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
            "axes.labelsize": AXIS_LABEL_FS,
            "axes.titlesize": TITLE_FS,
            "legend.fontsize": LEGEND_FS,
            "xtick.labelsize": TICK_FS,
            "ytick.labelsize": TICK_FS,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "path.simplify": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ══════════════════════════════════════════════════════════════════════════
# Chi color sampling
# ══════════════════════════════════════════════════════════════════════════

def _chi_colors(chi_list, cmap):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    n = len(chi_list)
    positions = [CMAP_HI] if n == 1 else np.linspace(CMAP_LO, CMAP_HI, n)
    return {chi: cmap(pos) for chi, pos in zip(chi_list, positions)}


# ══════════════════════════════════════════════════════════════════════════
# Legend handler
# ══════════════════════════════════════════════════════════════════════════

class HandlerColormapLine(HandlerBase):
    def __init__(
        self,
        cmap,
        n=64,
        lw=3.0,
        marker="o",
        ms=7.0,
        mew=0.8,
        chi_values=None,
        **kw,
    ):
        super().__init__(**kw)
        self.cmap = cmap
        self.n = n
        self.lw = lw
        self.marker = marker
        self.ms = ms
        self.mew = mew
        self.chi_values = chi_values or [1, 4, 8, 16]

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        from matplotlib.text import Text

        y = ydescent + 0.5 * height
        y -= height * 0.2

        xs = np.linspace(xdescent, xdescent + width, self.n + 1)

        artists = [
            Line2D(
                [xs[i], xs[i + 1]],
                [y, y],
                transform=trans,
                color=self.cmap(i / max(self.n - 1, 1)),
                lw=self.lw,
                solid_capstyle="butt",
            )
            for i in range(self.n)
        ]

        t_chi = Text(
            xdescent + 0.5 * width,
            y + height * 1.5,
            r"$\chi$",
            transform=trans,
            ha="center",
            va="top",
            fontsize=13,
            color="black",
        )
        artists.append(t_chi)

        n_chi = len(self.chi_values)
        cmap_positions = np.linspace(0.0, 1.0, n_chi)
        x_positions = np.linspace(xs[0], xs[-1], n_chi)
        label_y = y - height * 0.55

        for chi_val, x_pos, cmap_pos in zip(
            self.chi_values,
            x_positions,
            cmap_positions,
        ):
            col = self.cmap(cmap_pos)

            artists.append(
                Line2D(
                    [x_pos],
                    [y],
                    transform=trans,
                    marker=self.marker,
                    markersize=self.ms * 0.9,
                    markerfacecolor=col,
                    markeredgecolor=darken(col, EDGE_DARKEN),
                    markeredgewidth=self.mew,
                    linestyle="None",
                )
            )

            artists.append(
                Text(
                    x_pos,
                    label_y,
                    str(chi_val),
                    transform=trans,
                    ha="center",
                    va="top",
                    fontsize=fontsize * 0.55,
                    color="black",
                )
            )

        return artists


# ══════════════════════════════════════════════════════════════════════════
# Y-limit helpers
# ══════════════════════════════════════════════════════════════════════════

def _panel_log_ylim_top_at_one(results, pad_decades=0.06, floor=None):
    tiny = np.finfo(np.float64).tiny
    y_lo = []
    y_hi = []

    if not results:
        return None

    for chi in results:
        for name, data in results[chi].items():
            data = np.asarray(data, dtype=np.float64)

            if data.size == 0:
                continue

            if data.ndim == 1:
                data = data[np.newaxis, :]

            with np.errstate(invalid="ignore"):
                lo = np.nanpercentile(data, 10, axis=0)
                hi = np.nanpercentile(data, 90, axis=0)

            lo = lo[np.isfinite(lo)]
            hi = hi[np.isfinite(hi)]

            if lo.size == 0 or hi.size == 0:
                continue

            lo = np.clip(lo, tiny, None)
            hi = np.clip(hi, tiny, None)

            y_lo.append(np.min(lo))
            y_hi.append(np.max(hi))

    if not y_lo:
        return None

    upper = 5.0
    lower = 10.0 ** (np.log10(min(y_lo)) - pad_decades)

    if floor is not None:
        lower = max(lower, floor)

    if not np.isfinite(lower) or lower <= 0 or lower >= upper:
        lower = upper / 1e6

    return lower, upper


# ══════════════════════════════════════════════════════════════════════════
# Main plotting function
# ══════════════════════════════════════════════════════════════════════════

def plot_experiment_multi(
    k_vals,
    results,
    trace_true,
    kind,
    *,
    ax=None,
    show=True,
    subplot_title=None,
    y_limits=None,
):
    setup_plot_style()

    standalone = ax is None

    if standalone:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    else:
        fig = ax.figure

    k_vals = np.asarray(k_vals).ravel()
    tiny = np.finfo(np.float64).tiny
    title = subplot_title or kind
    no_data = not results or k_vals.size == 0

    if no_data or not sorted(results.keys()):
        ax.set_title(title, pad=10)
        ax.set_xlabel(r"Number of queries $t$")
        ax.set_ylabel("Relative Trace Error")
        ax.grid(False)
        ax.set_axisbelow(True)

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.9)

        ax.text(
            0.5,
            0.5,
            "Data not yet available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=TITLE_FS,
            alpha=0.75,
        )

        if standalone:
            plt.tight_layout()
            if show:
                plt.show()
            return fig, ax

        return ax

    chi_list = sorted(results.keys())
    method_names = list(next(iter(results.values())).keys())

    def _get_cmap(name):
        if name in METHOD_CMAPS:
            return METHOD_CMAPS[name]
        base = name.replace("_resph", "")
        return METHOD_CMAPS.get(base, plt.get_cmap("Greys"))

    chi_colors = {
        name: _chi_colors(chi_list, _get_cmap(name))
        for name in method_names
    }

    plotted_any = False

    for name in method_names:
        style = METHOD_STYLE.get(name, {})
        linestyle = style.get("ls", "-")
        marker = style.get("marker", "o")
        colors = chi_colors[name]

        for chi in chi_list:
            data = np.asarray(results[chi][name], dtype=np.float64)
            color = colors[chi]

            if data.size == 0:
                continue

            if data.ndim == 1:
                data = data[np.newaxis, :]

            with np.errstate(invalid="ignore"):
                median = np.nanmedian(data, axis=0)
                lo = np.nanpercentile(data, 10, axis=0)
                hi = np.nanpercentile(data, 90, axis=0)

            if k_vals.shape[0] != median.shape[0]:
                continue

            mask = np.isfinite(median)

            if not np.any(mask):
                continue

            xs = k_vals[mask]
            ys_med = np.clip(median[mask], tiny, None)
            ys_lo = np.clip(lo[mask], tiny, None)
            ys_hi = np.clip(hi[mask], tiny, None)

            if xs.size == 0:
                continue

            plotted_any = True
            edge_color = darken(color, EDGE_DARKEN)

            ax.semilogy(
                xs,
                ys_med,
                linestyle=linestyle,
                marker=marker,
                ms=LINE_MS,
                lw=LINE_LW,
                color=color,
                markerfacecolor=color,
                markeredgecolor=edge_color,
                markeredgewidth=MARKER_MEW,
                markevery=MARKEVERY,
                alpha=1.0,
                label=f"{name}  χ={chi}",
            )

            ax.fill_between(
                xs,
                ys_lo,
                ys_hi,
                color=color,
                alpha=FILL_ALPHA,
                edgecolor="none",
            )

    ax.set_xlabel(r"Number of queries $t$")
    ax.set_ylabel("Relative Trace Error")
    ax.set_title(title, pad=10)

    if plotted_any and k_vals.size > 0:
        ax.set_xlim(k_vals.min() - X_PAD, k_vals.max() + X_PAD)

        if y_limits is not None:
            ax.set_ylim(*y_limits)

    ax.grid(False)
    ax.set_axisbelow(True)

    ax.tick_params(axis="x", bottom=True, top=False, labelsize=TICK_FS)
    ax.tick_params(axis="y", left=True, right=False, labelsize=YTICK_FS, labelleft=True)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.9)

    if standalone:
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    return ax


def plot_experiment(k_vals, results, trace_true, kind, *, ax=None, show=True):
    wrapped = {
        1: {
            name: v[np.newaxis, :]
            for name, v in results.items()
        }
    }

    return plot_experiment_multi(
        k_vals,
        wrapped,
        trace_true,
        kind,
        ax=ax,
        show=show,
    )


# ══════════════════════════════════════════════════════════════════════════
# Shared legend
# ══════════════════════════════════════════════════════════════════════════

def add_shared_legend(
    fig,
    axes,
    *,
    method_names=None,
    ncol=None,
    y=1.03,
    fontsize=LEGEND_FS,
    handlelength=3.0,
    borderpad=0.55,
    labelspacing=0.35,
    columnspacing=1.8,
    chi_values=None,
):
    if method_names is None:
        method_names = ["hutch", "nystrom++", "xnystrace"]

    if chi_values is None:
        chi_values = [1, 4, 8, 16]

    handles = []
    labels = []
    handler_map = {}

    for name in method_names:
        cmap = METHOD_CMAPS.get(name, plt.get_cmap("Greys"))

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        truncated = LinearSegmentedColormap.from_list(
            f"_trunc_{name}",
            cmap(np.linspace(CMAP_LO, CMAP_HI, 256)),
        )

        marker = METHOD_STYLE.get(name, {}).get("marker", "o")
        proxy = Line2D([], [], lw=LINE_LW)

        handles.append(proxy)
        labels.append(METHOD_DISPLAY_NAMES.get(name, name))

        handler_map[proxy] = HandlerColormapLine(
            truncated,
            n=64,
            lw=LINE_LW + 1.0,
            marker=marker,
            chi_values=chi_values,
        )

    return fig.legend(
        handles,
        labels,
        handler_map=handler_map,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol or len(handles),
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        fontsize=fontsize,
        handlelength=handlelength,
        borderpad=borderpad,
        labelspacing=labelspacing,
        columnspacing=columnspacing,
    )


def finalize_figure_layout(fig, *, top=0.88):
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])


# ══════════════════════════════════════════════════════════════════════════
# Final composite figure
# ══════════════════════════════════════════════════════════════════════════

def final_plot(
    loaded_experiments,
    *,
    chi_subset=None,
    show=True,
):
    """Build the full Figure 5 composite (3 main panels + mini spectra + legend).

    Parameters
    ----------
    loaded_experiments : list of (kind, k_vals, results, tr_true, title)
        One entry per benchmark, in the order they should appear left-to-right.
        Each entry is the tuple returned by ``load_experiment_json`` with a
        display title appended.
    chi_subset : list of int, optional
        Which chi values to include.  Defaults to [1, 4, 8, 16].
    show : bool
        Whether to call ``plt.show()`` before returning.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if chi_subset is None:
        chi_subset = [1, 4, 8, 16]

    setup_plot_style()

    fig = plt.figure(figsize=(20, 4))

    outer = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[3.6, 0.5],
        wspace=0.06,
    )

    gs_left = outer[0].subgridspec(1, 3, wspace=0.26)
    axes = [fig.add_subplot(gs_left[i]) for i in range(3)]

    gs_right = outer[1].subgridspec(3, 1, hspace=0.8)
    axes_right = [fig.add_subplot(gs_right[i]) for i in range(3)]

    # ── Main panels ───────────────────────────────────────────────────────
    for ax, (kind, k_vals, res, tr, title) in zip(axes, loaded_experiments):
        chi_list = sorted(chi for chi in res.keys() if chi in chi_subset)
        available_methods = list(next(iter(res.values())).keys())

        ordered_methods = [m for m in plot_order if m in available_methods]
        ordered_methods += [m for m in available_methods if m not in ordered_methods]

        res_ordered = {
            chi: {m: res[chi][m] for m in ordered_methods}
            for chi in chi_list
        }

        plot_experiment_multi(
            k_vals, res_ordered, tr, kind,
            ax=ax, show=False, subplot_title=title,
        )

        ylim = _panel_log_ylim_top_at_one(res_ordered, pad_decades=0.06)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.margins(x=0.03)
        ax.set_ylabel("Relative Trace Error")
        ax.set_xlabel(r"Number of queries $t$")
        ax.tick_params(axis="y", labelleft=True)

    # ── Mini spectrum panels ──────────────────────────────────────────────
    N_bench = 50
    d_bench = 2
    n_eigs  = 200

    exp_spectrum = 0.7 ** np.arange(n_eigs)
    il_spectrum  = dirichlet_inverse_laplacian_eigenvalues_first(N_bench, n_eigs)

    bp, vals = staircase_auto_spec(N=N_bench, d=d_bench, seed=0)
    staircase_spectrum = np.empty(n_eigs, dtype=np.float64)
    boundaries = list(bp) + [n_eigs]
    prev = 0
    for j, bnd in enumerate(boundaries):
        bnd = min(bnd, n_eigs)
        if prev < bnd:
            staircase_spectrum[prev:bnd] = vals[j]
        prev = bnd
        if prev >= n_eigs:
            break
    if prev < n_eigs:
        staircase_spectrum[prev:] = vals[-1]

    spectra = [
        (exp_spectrum,       "Exponential"),
        (il_spectrum,        "Inverse Laplacian"),
        (staircase_spectrum, "Staircase"),
    ]

    for ax_r, (spec, mini_title) in zip(axes_right, spectra):
        spec_sorted     = np.sort(spec)[::-1]
        spec_normalized = spec_sorted / spec_sorted[0]

        ax_r.semilogy(np.arange(len(spec_normalized)), spec_normalized,
                      color="black", lw=1.0)
        ax_r.set_title(mini_title, fontsize=13, fontstyle="normal",
                       fontweight="normal", pad=4)
        ax_r.set_xticks(np.arange(0, n_eigs + 1, 50))
        ax_r.tick_params(axis="both", labelsize=9)
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(False)
        for spine in ["left", "bottom"]:
            ax_r.spines[spine].set_linewidth(0.6)

    # ── Shared legend ─────────────────────────────────────────────────────
    leg = add_shared_legend(
        fig, axes,
        method_names=["hutch", "nystrom++", "xnystrace"],
        y=1.2, ncol=3, chi_values=chi_subset,
    )

    # ── "Spectra (normalized)" label above the mini-panel column ─────────
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    right_boxes  = [ax.get_position() for ax in axes_right]
    x_center     = 0.5 * (min(bb.x0 for bb in right_boxes)
                          + max(bb.x1 for bb in right_boxes))

    legend_text_boxes = [
        txt.get_window_extent(renderer=renderer).transformed(
            fig.transFigure.inverted()
        )
        for txt in leg.get_texts()
    ]
    legend_text_y = np.mean([0.5 * (bb.y0 + bb.y1) for bb in legend_text_boxes])
    legend_fs     = leg.get_texts()[0].get_fontsize()

    title_y             = legend_text_y - 0.012
    subtitle_y          = legend_text_y - 0.044
    underline_y         = subtitle_y - 0.020
    underline_halfwidth = 0.060

    fig.text(x_center, title_y, "Spectra (normalized)",
             ha="center", va="center", fontsize=legend_fs)

    spectra_underline = Line2D(
        [x_center - underline_halfwidth, x_center + underline_halfwidth],
        [underline_y, underline_y],
        transform=fig.transFigure,
        color="black", lw=0.9, solid_capstyle="butt",
    )
    fig.add_artist(spectra_underline)

    finalize_figure_layout(fig, top=0.88)

    if show:
        plt.show()

    return fig


# ══════════════════════════════════════════════════════════════════════════
# Spectrum helpers (for mini-panel plots)
# ══════════════════════════════════════════════════════════════════════════

def dirichlet_inverse_laplacian_eigenvalues_first(N, m, dtype=np.float64):
    """Return the first `m` eigenvalues of the inverse 1-D Dirichlet Laplacian.

    The discrete grid has ``n_plus_1 = 2**N + 1`` points (including the two
    Dirichlet endpoints), giving ``2**N - 1`` interior nodes.

    Parameters
    ----------
    N : int
        Number of MPS sites; grid size is ``2**N + 1``.
    m : int
        Number of eigenvalues to return (in ascending eigenvalue order of the
        *Laplacian*, i.e. descending order of the *inverse* Laplacian).
    dtype : numpy dtype

    Returns
    -------
    ndarray of shape (m,)
        Eigenvalues ``1 / lambda_k`` for ``k = 1, …, m``.
    """
    n_plus_1 = 2**N + 1
    k = np.arange(1, m + 1, dtype=np.float64)
    half_angle = (np.pi * k) / (2.0 * n_plus_1)
    lam = 4.0 * np.sin(half_angle) ** 2
    return (1.0 / lam).astype(dtype, copy=False)


def staircase_auto_spec(
    N=100,
    d=2,
    seed=0,
    first_block=32,
    num_levels=3,
    block_growth=1.7,
    level_decay=0.15,
    head_val=1.0,
    tail_val=0.0,
):
    """Compute the breakpoints and level values for the staircase spectrum.

    This mirrors the logic inside ``build_mpo`` for the ``"staircase"`` kind
    but does NOT construct the MPO — it only returns the spectrum spec so that
    mini-panel plots can be drawn without the tnrnla dependency.

    Returns
    -------
    breakpoints : tuple of int
        Cumulative sizes of the constant-value blocks (exclusive upper bounds).
    values : tuple of float
        One value per block plus the final tail value.
    """
    n = d ** N
    lengths = []
    running = 0

    for j in range(num_levels):
        Lj = max(int(round(first_block * (block_growth ** j))), 1)
        max_allowed = (n - 1) - running - (num_levels - 1 - j)

        if max_allowed <= 0:
            break

        Lj = min(Lj, max_allowed)
        lengths.append(Lj)
        running += Lj

        if running >= n - 1:
            break

    breakpoints = tuple(int(x) for x in np.cumsum(lengths, dtype=np.int64))

    values = tuple(
        [float(head_val * (level_decay ** j)) for j in range(len(lengths))]
        + [float(tail_val)]
    )

    return breakpoints, values


# ══════════════════════════════════════════════════════════════════════════
# JSON loader
# ══════════════════════════════════════════════════════════════════════════

def load_experiment_json(path):
    """Load a JSON snapshot written by run_experiment_multi.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    k_vals   : ndarray[int]
    results  : dict[int chi, dict[str method, ndarray(n_runs, n_k)]]
    tr_true  : float
    kind     : str
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    k_vals  = np.array(data["config"]["k_vals"], dtype=int)
    tr_true = float(data["trace_true"])
    kind    = data["kind"]

    results = {
        int(chi_str): {
            method: np.array(arr, dtype=np.float64)
            for method, arr in method_dict.items()
        }
        for chi_str, method_dict in data["relative_trace_error"].items()
    }

    return k_vals, results, tr_true, kind
