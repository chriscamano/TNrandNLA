import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from pathlib import Path
import importlib.util


# ── Scientific colourmap loader ───────────────────────────────────────────────

def load_scientific_cmaps(names, search_roots=None):
    if search_roots is None:
        search_roots = [
            Path(__file__).resolve().parent.parent / "ScientificColourMaps8",
        ]
    for cmap_name in names:
        cmap_file = None
        for root in search_roots:
            candidate = root / cmap_name / f"{cmap_name}.py"
            if candidate.exists():
                cmap_file = candidate
                break
        if cmap_file is None:
            continue
        spec = importlib.util.spec_from_file_location(f"{cmap_name}_module", cmap_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cmap = getattr(mod, f"{cmap_name}_map")
        cmap_r = cmap.reversed()
        for name in [cmap_name, f"{cmap_name}_r"]:
            try:
                mpl.colormaps.unregister(name)
            except Exception:
                pass
        mpl.colormaps.register(cmap, name=cmap_name)
        mpl.colormaps.register(cmap_r, name=f"{cmap_name}_r")


# ── Helpers ───────────────────────────────────────────────────────────────────

def darken(color, factor=0.8):
    r, g, b, a = color
    return (r * factor, g * factor, b * factor, a)


def resolve_k_axis(result_entry, global_k_values, expected_len):
    if isinstance(result_entry, dict) and "k_values" in result_entry:
        k_local = np.asarray(result_entry["k_values"]).ravel()
        if k_local.size == expected_len:
            return k_local
        raise ValueError(
            f"results entry 'k_values' has length {k_local.size}, "
            f"but relerr data has length {expected_len}."
        )
    k_global = np.asarray(global_k_values).ravel()
    if k_global.size >= expected_len:
        return k_global[:expected_len]
    return np.arange(expected_len)


class HandlerColormapLine(HandlerBase):
    def __init__(self, cmap, n=64, lw=3.0, **kwargs):
        super().__init__(**kwargs)
        self.cmap = cmap
        self.n = n
        self.lw = lw

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        y = ydescent + 0.5 * height
        xs = np.linspace(xdescent, xdescent + width, self.n + 1)
        artists = []
        for i in range(self.n):
            artists.append(Line2D(
                [xs[i], xs[i + 1]], [y, y],
                transform=trans,
                color=self.cmap(i / max(self.n - 1, 1)),
                lw=self.lw,
                solid_capstyle="butt",
            ))
        return artists


# ── Main plot function ────────────────────────────────────────────────────────

def plot_nystrom_results(
    results,
    chi_values,
    k_values,
    exact_spectrum,
    *,
    figsize=(16, 5),
    cmap_name="devon_r",
    trunc_lo=0.40,
    trunc_hi=0.99,
    x_pad=1.5,
    axis_label_fs=22,
    legend_fs=20,
    tick_fs=19,
    cbar_label_fs=22,
    cbar_tick_fs=18,
    eps_text_fs=19,
    line_ms=4.5,
    line_lw_left=2.0,
    line_lw_right=1.8,
    analytic_ms=9.0,
    analytic_mew=1.4,
    savepath=None,
):
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Nimbus Roman"],
            "text.latex.preamble": r"""
\usepackage{newtxtext}
\usepackage{mathtools}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{mathrsfs}
\usepackage{bm}
\usepackage{bbm}
\usepackage{dsfont}
\usepackage{eucal}
\usepackage{xspace}
\usepackage{nicefrac}
\usepackage{relsize}
\usepackage{mleftright}

\newcommand{\matop}[2]{\operatorname{mat}_{#1}\!\left(#2\right)}
\newcommand{\mat}[1]{\operatorname{mat}\!\left(#1\right)}
\newcommand{\vecop}[1]{\operatorname{vec}\!\left(#1\right)}
\newcommand{\tenop}[1]{\operatorname{ten}\!\left(#1\right)}
\newcommand{\bigKron}[1]{\bigotimes_{r=1}^{#1}}

\newcommand{\rmps}{\ensuremath{\mathsf{rMPS}}\xspace}
\newcommand{\mps}{\ensuremath{\mathsf{MPS}}\xspace}
\newcommand{\mpo}{\ensuremath{\mathsf{MPO}}\xspace}
\newcommand{\rmpss}{\ensuremath{\mathsf{rMPSs}}\xspace}
\newcommand{\mpss}{\ensuremath{\mathsf{MPSs}}\xspace}
\newcommand{\mpos}{\ensuremath{\mathsf{MPOs}}\xspace}
""",
            "axes.labelsize": axis_label_fs,
            "legend.fontsize": legend_fs,
            "xtick.labelsize": tick_fs,
            "ytick.labelsize": tick_fs,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "path.simplify": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    tiny = np.finfo(np.float64).tiny
    eps_line = np.sqrt(np.finfo(np.float64).eps)

    exact_eigs = np.asarray(exact_spectrum, dtype=np.float64)
    available_lengths = [
        len(np.asarray(results[chi]["spectrum_median"]))
        for chi in chi_values
        if results[chi].get("spectrum_median") is not None
    ]
    if not available_lengths:
        raise ValueError("No spectrum_median found in results.")

    base_rank = min(len(exact_eigs), min(available_lengths))
    exact_base = exact_eigs[:base_rank]
    above_eps = np.flatnonzero(exact_base >= eps_line)
    if above_eps.size == 0:
        raise ValueError("Exact spectrum is entirely below sqrt(machine epsilon).")
    plot_rank = int(above_eps[-1]) + 2
    exact_visible = exact_base[:plot_rank]

    base_cmap = plt.get_cmap(cmap_name)
    truncated_cmap = LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc",
        base_cmap(np.linspace(trunc_lo, trunc_hi, 256)),
    )
    colors = truncated_cmap(np.linspace(0.0, 1.0, len(chi_values)))
    analytic_color = (0.0, 0.0, 0.0, 0.85)
    n_chi = len(chi_values)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left panel: relative VN entropy error
    for color_idx, chi in enumerate(chi_values):
        color = colors[color_idx]
        relerr = np.asarray(results[chi]["relerr_trials"], dtype=np.float64)
        if relerr.ndim != 2:
            raise ValueError(f"results[{chi!r}]['relerr_trials'] must be 2D.")
        med = np.median(relerr, axis=0)
        q10 = np.quantile(relerr, 0.10, axis=0)
        q90 = np.quantile(relerr, 0.90, axis=0)
        k_plot = resolve_k_axis(results[chi], k_values, med.size)
        axes[0].semilogy(k_plot, np.clip(med, tiny, None),
                         linestyle="-", marker="o", ms=line_ms, lw=line_lw_left,
                         color=color, markeredgecolor=darken(color), markeredgewidth=0.8)
        axes[0].fill_between(k_plot, np.clip(q10, tiny, None), np.clip(q90, tiny, None),
                              color=color, alpha=0.18, linewidth=0.0)

    axes[0].set_xlabel(r"Embedding Dimension $k$", fontsize=axis_label_fs)
    axes[0].set_ylabel("Relative VN Entropy Error", fontsize=axis_label_fs)
    k_arr = np.asarray(k_values).ravel()
    k_ticks = np.arange(int(np.ceil(k_arr[0] / 10) * 10), int(k_arr[-1]) + 1, 10)
    axes[0].set_xticks(k_ticks)
    axes[0].set_xlim(k_arr[0] - x_pad, k_arr[-1] + x_pad)
    axes[0].tick_params(labelsize=tick_fs)
    axes[0].grid(False)

    # Right panel: eigenvalue spectrum
    for color_idx, chi in enumerate(chi_values):
        color = colors[color_idx]
        spec_med = np.asarray(results[chi]["spectrum_median"], dtype=np.float64)
        spec_trials = results[chi].get("spectrum_trials")
        if spec_trials is not None:
            spec_trials = np.asarray(spec_trials, dtype=np.float64)
            sq10 = np.quantile(spec_trials, 0.10, axis=0)
            sq90 = np.quantile(spec_trials, 0.90, axis=0)
        else:
            sq10 = sq90 = spec_med.copy()

        this_rank = min(plot_rank, len(spec_med))
        idx = np.arange(this_rank)
        raw = spec_med[:this_rank]
        y_med = np.maximum(raw, eps_line)
        y_q10 = np.maximum(sq10[:this_rank], eps_line)
        y_q90 = np.maximum(sq90[:this_rank], eps_line)
        at_eps = raw <= eps_line
        above = ~at_eps

        z_main = 10 + (n_chi - 1 - color_idx)
        z_tail = 10 + color_idx

        axes[1].fill_between(idx, y_q10, y_q90, color=color, alpha=0.18,
                              linewidth=0.0, zorder=min(z_main, z_tail) - 0.2)

        if np.any(at_eps):
            tail_start = np.flatnonzero(at_eps)[0]
            if tail_start > 0:
                axes[1].semilogy(idx[:tail_start + 1], y_med[:tail_start + 1],
                                 linestyle="-", marker="None", lw=line_lw_right,
                                 color=color, zorder=z_main)
            axes[1].semilogy(idx[tail_start:], y_med[tail_start:],
                             linestyle="-", marker="None", lw=line_lw_right,
                             color=color, zorder=z_tail)
        else:
            axes[1].semilogy(idx, y_med, linestyle="-", marker="None",
                             lw=line_lw_right, color=color, zorder=z_main)

        if np.any(above):
            axes[1].semilogy(idx[above], y_med[above], linestyle="None", marker="o",
                             ms=line_ms, color=color, markeredgecolor=darken(color),
                             markeredgewidth=0.8, zorder=z_main + 0.05)
        if np.any(at_eps):
            axes[1].semilogy(idx[at_eps], y_med[at_eps], linestyle="None", marker="o",
                             ms=line_ms, color=color, markeredgecolor=darken(color),
                             markeredgewidth=0.8, zorder=z_tail + 0.05)

    axes[1].semilogy(np.arange(plot_rank), np.maximum(exact_visible, eps_line),
                     color=analytic_color, linestyle="None", marker="o",
                     ms=analytic_ms, markerfacecolor="none",
                     markeredgewidth=analytic_mew, markeredgecolor=analytic_color,
                     clip_on=False, zorder=100)

    axes[1].axhline(y=eps_line, color="grey", linestyle="--", lw=1.3, alpha=0.7, zorder=10)
    axes[1].text(-0.85, eps_line * 1.35, r"$\sqrt{\varepsilon_{\mathrm{mach}}}$",
                 fontsize=eps_text_fs, va="bottom", ha="left", color="grey")
    axes[1].set_xlim(-1.25, plot_rank - 0.25)
    axes[1].set_ylim(0.8 * eps_line, 3.0 * np.max(exact_visible))
    axes[1].set_xlabel("Eigenvalue Index", fontsize=axis_label_fs)
    axes[1].set_ylabel(r"Eigenvalue $\lambda_i$", fontsize=axis_label_fs)
    axes[1].tick_params(labelsize=tick_fs)
    axes[1].grid(False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Colorbars
    norm = Normalize(vmin=chi_values[0], vmax=chi_values[-1])
    for ax in axes:
        sm = ScalarMappable(cmap=truncated_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Bond Dimension " + r"$\chi$", fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)
        cbar.outline.set_linewidth(0.6)

    # Legend
    gradient_proxy = Line2D([], [], lw=3.0)
    dmrg_proxy = Line2D([], [], color=analytic_color, linestyle="None", marker="o",
                        markersize=analytic_ms, markerfacecolor="none",
                        markeredgewidth=analytic_mew, markeredgecolor=analytic_color)

    leg = fig.legend(
        handles=[gradient_proxy, dmrg_proxy],
        labels=[r"\mps \textsc{GramNystr\"om}", "True Eigenvalues"],
        handler_map={gradient_proxy: HandlerColormapLine(truncated_cmap, n=64, lw=3.0)},
        loc="upper center", bbox_to_anchor=(0.5, 1.08),
        frameon=True, fancybox=True, shadow=False,
        fontsize=legend_fs, framealpha=1.0, edgecolor="black", facecolor="white",
        ncol=2, columnspacing=1.8, handlelength=3.0,
    )
    leg.get_texts()[1].set_color(analytic_color)

    plt.subplots_adjust(top=0.92, wspace=0.28)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if savepath is not None:
        mpl.rcParams["svg.fonttype"] = "path"
        fig.savefig(savepath, format=Path(savepath).suffix.lstrip("."),
                    facecolor="white", bbox_inches="tight", pad_inches=0.08)

    plt.show()
    return fig, axes
