"""
plot_split.py
-------------
Produce two matched figures from the Figure 5 data:
  Fig A – MPS Nystrom++ only
  Fig B – Nystrom++ + XNysTrace

Y-axis range is determined jointly from both method sets so the scale
does not shift between the two figures.

Run from the Figure5_MPSXNysTrace directory:
    python plot_split.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── make sure plotting.py is importable ──────────────────────────────────────
HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plotting import (
    load_experiment_json,
    plot_experiment_multi,
    add_shared_legend,
    finalize_figure_layout,
    _panel_log_ylim_top_at_one,
    dirichlet_inverse_laplacian_eigenvalues_first,
    staircase_auto_spec,
    plot_order,
    setup_plot_style,
    METHOD_CMAPS,
    METHOD_DISPLAY_NAMES,
    METHOD_STYLE,
    CMAP_LO, CMAP_HI,
    LINE_LW,
    HandlerColormapLine,
)

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR   = HERE / "data"
CHI_SUBSET = [1, 4, 8, 16]
SHOW       = True

EXPERIMENT_DEFS = [
    ("exp",               "Exponential"),
    ("inverse_laplacian", "Inverse Laplacian"),
    ("staircase",         "Staircase"),
]

# ── load data ─────────────────────────────────────────────────────────────────
loaded = {}
for kind, title in EXPERIMENT_DEFS:
    matches = sorted(DATA_DIR.glob(f"{kind}_*.json"))
    if not matches:
        raise FileNotFoundError(f"No {kind}_*.json found in {DATA_DIR}")
    k_vals, results, tr_true, _ = load_experiment_json(matches[-1])
    loaded[kind] = (k_vals, results, tr_true, title)
    print(f"Loaded {matches[-1].name}")


# ── helper: filter results to specific methods ────────────────────────────────
def filter_methods(results, keep):
    out = {}
    for chi, method_dict in results.items():
        filtered = {m: v for m, v in method_dict.items() if m in keep}
        if filtered:
            out[chi] = filtered
    return out


def filter_chi(results, chi_subset):
    return {chi: v for chi, v in results.items() if chi in chi_subset}


# ── compute shared y-limits across both method sets ───────────────────────────
# "Both" means the union of nystrom++ and xnystrace — that determines the range.
def compute_combined_ylim(loaded_exps, methods_a, methods_b, chi_subset):
    """Return (ylo, yhi) covering data from methods_a ∪ methods_b."""
    all_lo, all_hi = [], []

    for kind, title in loaded_exps:
        k_vals, results, tr_true, _ = loaded[kind]
        res_chi = filter_chi(results, chi_subset)
        combined_methods = set(methods_a) | set(methods_b)
        res_filtered = filter_methods(res_chi, combined_methods)
        lim = _panel_log_ylim_top_at_one(res_filtered, pad_decades=0.06)
        if lim is not None:
            all_lo.append(lim[0])
            all_hi.append(lim[1])

    if not all_lo:
        return None

    return min(all_lo), max(all_hi)


shared_ylim = compute_combined_ylim(
    EXPERIMENT_DEFS,
    methods_a=["nystrom++"],
    methods_b=["nystrom++", "xnystrace"],
    chi_subset=CHI_SUBSET,
)
print(f"Shared y-limits: {shared_ylim}")


# ── mini spectrum data (for right-column panels) ──────────────────────────────
def build_spectra():
    N_bench, d_bench, n_eigs = 50, 2, 200

    exp_spec = 0.7 ** np.arange(n_eigs)
    il_spec  = dirichlet_inverse_laplacian_eigenvalues_first(N_bench, n_eigs)

    bp, vals = staircase_auto_spec(N=N_bench, d=d_bench, seed=0)
    sc_spec = np.empty(n_eigs, dtype=np.float64)
    boundaries = list(bp) + [n_eigs]
    prev = 0
    for j, bnd in enumerate(boundaries):
        bnd = min(bnd, n_eigs)
        if prev < bnd:
            sc_spec[prev:bnd] = vals[j]
        prev = bnd
        if prev >= n_eigs:
            break
    if prev < n_eigs:
        sc_spec[prev:] = vals[-1]

    return [
        (exp_spec, "Exponential"),
        (il_spec,  "Inverse Laplacian"),
        (sc_spec,  "Staircase"),
    ]


SPECTRA = build_spectra()


# ── generic figure builder ────────────────────────────────────────────────────
def build_figure(method_names, ylim, legend_y=1.2):
    setup_plot_style()

    fig = plt.figure(figsize=(20, 4))

    outer = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[3.6, 0.5],
        wspace=0.06,
    )
    gs_left  = outer[0].subgridspec(1, 3, wspace=0.26)
    axes     = [fig.add_subplot(gs_left[i]) for i in range(3)]

    gs_right   = outer[1].subgridspec(3, 1, hspace=0.8)
    axes_right = [fig.add_subplot(gs_right[i]) for i in range(3)]

    # ── main panels ───────────────────────────────────────────────────────────
    for ax, (kind, title) in zip(axes, EXPERIMENT_DEFS):
        k_vals, results, tr_true, _ = loaded[kind]

        res_chi = filter_chi(results, CHI_SUBSET)
        res_filt = filter_methods(res_chi, method_names)

        ordered = [m for m in plot_order if m in method_names]
        res_ord = {
            chi: {m: res_filt[chi][m] for m in ordered if m in res_filt[chi]}
            for chi in sorted(res_filt)
        }

        plot_experiment_multi(
            k_vals, res_ord, tr_true, kind,
            ax=ax, show=False, subplot_title=title,
        )

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.margins(x=0.03)
        ax.set_ylabel("Relative Trace Error")
        ax.set_xlabel(r"Number of queries $t$")
        ax.tick_params(axis="y", labelleft=True)

    # ── mini spectrum panels ──────────────────────────────────────────────────
    n_eigs = 200
    for ax_r, (spec, mini_title) in zip(axes_right, SPECTRA):
        spec_norm = np.sort(spec)[::-1]
        spec_norm = spec_norm / spec_norm[0]

        ax_r.semilogy(np.arange(len(spec_norm)), spec_norm, color="black", lw=1.0)
        ax_r.set_title(mini_title, fontsize=13, fontstyle="normal",
                       fontweight="normal", pad=4)
        ax_r.set_xticks(np.arange(0, n_eigs + 1, 50))
        ax_r.tick_params(axis="both", labelsize=9)
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(False)
        for spine in ["left", "bottom"]:
            ax_r.spines[spine].set_linewidth(0.6)

    # ── shared legend ─────────────────────────────────────────────────────────
    leg = add_shared_legend(
        fig, axes,
        method_names=method_names,
        y=legend_y, ncol=len(method_names), chi_values=CHI_SUBSET,
    )

    # ── "Spectra (normalized)" label above the mini-panel column ─────────────
    from matplotlib.lines import Line2D as _L2D
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    right_boxes = [ax.get_position() for ax in axes_right]
    x_center = 0.5 * (min(bb.x0 for bb in right_boxes)
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

    spectra_underline = _L2D(
        [x_center - underline_halfwidth, x_center + underline_halfwidth],
        [underline_y, underline_y],
        transform=fig.transFigure,
        color="black", lw=0.9, solid_capstyle="butt",
    )
    fig.add_artist(spectra_underline)

    finalize_figure_layout(fig, top=0.88)
    return fig


# ── Figure A: MPS Nystrom++ only ──────────────────────────────────────────────
print("\nBuilding Figure A (MPS Nystrom++ only)...")
fig_a = build_figure(method_names=["nystrom++"], ylim=shared_ylim)
fig_a.savefig(
    HERE / "figures" / "FigA_MPSNystromPP.pdf",
    format="pdf", bbox_inches="tight", pad_inches=0.05,
)
print("Saved figures/FigA_MPSNystromPP.pdf")

# ── Figure B: Nystrom++ + XNysTrace ──────────────────────────────────────────
print("Building Figure B (Nystrom++ + XNysTrace)...")
fig_b = build_figure(method_names=["nystrom++", "xnystrace"], ylim=shared_ylim)
fig_b.savefig(
    HERE / "figures" / "FigB_NystromPP_XNysTrace.pdf",
    format="pdf", bbox_inches="tight", pad_inches=0.05,
)
print("Saved figures/FigB_NystromPP_XNysTrace.pdf")

if SHOW:
    plt.show()
