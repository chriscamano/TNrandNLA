from pathlib import Path
import importlib.util
import json
from math import comb

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
import matplotlib.gridspec as gridspec


SQRT2_COLOR = "#C1121F"
GAUSS_COLOR = "#000000"


AXIS_LABEL_FS = 26
Y_AXIS_LABEL_FS = 26
TITLE_FS = 22
LEGEND_FS = 25
TICK_FS = 18
MEM_LABEL_FS = 11
CBAR_LABEL_FS = 22
CBAR_TICK_FS = 18
EPS_TEXT_FS = 19

LINE_MS = 4.5
LINE_LW = 2.2
MARKER_MEW = 0.8

CMAP_NAME = "Blues"
TRUNC_LO = 0.4
TRUNC_HI = 0.99

REF_LS = (0, (4, 2))
REF_LW = 2.0
PLOT3_YMIN = 4e-2


def _set_pub_rc() -> None:
    mpl.rcParams.update({
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
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
    })


def _get_key(data, *names):
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(f"None of these keys were found: {names}")


def _register_scientific_colormaps(repo_root: Path = None) -> None:
    _cmaps_dir = Path(__file__).resolve().parent.parent / "ScientificColourMaps8"
    for cmap_name in [
        "oslo", "devon", "lipari", "davos", "lajolla", "acton",
        "batlow", "glasgow", "bamako", "bilbao", "lapaz",
    ]:
        cmap_file = _cmaps_dir / cmap_name / f"{cmap_name}.py"
        if not cmap_file.exists():
            continue

        spec = importlib.util.spec_from_file_location(f"{cmap_name}_module", cmap_file)
        cmap_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cmap_module)

        cmap = getattr(cmap_module, f"{cmap_name}_map")
        cmap_r = cmap.reversed()

        for name in [cmap_name, f"{cmap_name}_r"]:
            try:
                mpl.colormaps.unregister(name)
            except Exception:
                pass

        mpl.colormaps.register(cmap, name=cmap_name)
        mpl.colormaps.register(cmap_r, name=f"{cmap_name}_r")


def darken(color, factor=0.75):
    r, g, b = to_rgb(color)
    return (factor * r, factor * g, factor * b)


def auto_log_ylim(*arrays, pad_decades=0.06):
    vals = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1e-16, 1.0

    log_lo = np.log10(vals.min())
    log_hi = np.log10(vals.max())
    return 10.0 ** (log_lo - pad_decades), 10.0 ** (log_hi + pad_decades)


def _ising_open_xx_sector_weights(n, beta):
    m = np.arange(n, dtype=int)
    mult = 2.0 * np.array([comb(n - 1, int(j)) for j in m], dtype=np.float64)
    log_weights = np.log(mult) + beta * ((n - 1) - 2.0 * m)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    return weights, mult


def gaussian_median_relative_error_open_xx(n, beta, k, nsamples=1_000_000, batch=20_000, seed=1234):
    rng = np.random.default_rng(seed)
    weights, mult = _ising_open_xx_sector_weights(n=n, beta=beta)
    dfs = k * mult.astype(np.float64)

    samples = np.empty(nsamples, dtype=np.float64)
    for start in range(0, nsamples, batch):
        stop = min(start + batch, nsamples)
        b = stop - start
        chi2 = rng.chisquare(df=dfs, size=(b, dfs.size))
        samples[start:stop] = np.abs((chi2 / dfs) @ weights - 1.0)

    return np.median(samples)


def gaussian_median_relative_error_curve_open_xx(n, beta_list, k_vals, nsamples=200_000, batch=20_000, base_seed=1234):
    out = np.empty((len(beta_list), len(k_vals)), dtype=np.float64)
    for bi, beta in enumerate(beta_list):
        for ki, k in enumerate(k_vals):
            out[bi, ki] = gaussian_median_relative_error_open_xx(
                n=n,
                beta=beta,
                k=int(k),
                nsamples=nsamples,
                batch=batch,
                seed=base_seed + 1000 * bi + ki,
            )
    return out


class HandlerShortLine(HandlerBase):
    """Draws a centered line at a fraction of the handle width."""

    def __init__(self, color, lw, ls, scale=0.5, **kw):
        super().__init__(**kw)
        self.color, self.lw, self.ls, self.scale = color, lw, ls, scale

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        w = width * self.scale
        cx = xdescent + width * 0.5
        y = ydescent + height * 0.5
        return [Line2D([cx - w * 0.5, cx + w * 0.5], [y, y], transform=trans,
                       color=self.color, lw=self.lw, linestyle=self.ls,
                       dash_capstyle="round")]


class HandlerColormapLine(HandlerBase):
    """Legend handler that draws a gradient bar with chi markers and labels."""

    def __init__(self, cmap, n=64, lw=3.0, marker="o", ms=7.0, mew=0.8,
                 chi_values=None, **kw):
        super().__init__(**kw)
        self.cmap = cmap
        self.n = n
        self.lw = lw
        self.marker = marker
        self.ms = ms
        self.mew = mew
        self.chi_values = chi_values or [1, 2, 4, 8, 16]

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        y = ydescent + 0.5 * height
        y -= height * 0.2
        xs = np.linspace(xdescent, xdescent + width, self.n + 1)

        artists = [
            Line2D([xs[i], xs[i + 1]], [y, y], transform=trans,
                   color=self.cmap(i / max(self.n - 1, 1)),
                   lw=self.lw, solid_capstyle="butt")
            for i in range(self.n)
        ]

        artists.append(Text(
            xdescent + 0.5 * width, y + height * 1.6, r"$\chi$",
            transform=trans, ha="center", va="top",
            fontsize=fontsize * 0.85, color="black",
        ))

        n_chi = len(self.chi_values)
        x_positions = np.linspace(xs[0], xs[-1], n_chi)
        cmap_positions = np.linspace(0.0, 1.0, n_chi)
        label_y = y - height * 0.55

        for chi_val, x_pos, cmap_pos in zip(self.chi_values, x_positions, cmap_positions):
            col = self.cmap(cmap_pos)
            artists.append(Line2D(
                [x_pos], [y], transform=trans,
                marker=self.marker, markersize=self.ms * 0.9,
                markerfacecolor=col, markeredgecolor=darken(col, 0.75),
                markeredgewidth=self.mew, linestyle="None",
            ))
            artists.append(Text(
                x_pos, label_y, str(chi_val),
                transform=trans, ha="center", va="top",
                fontsize=fontsize * 0.68, color="black",
            ))

        return artists


def plot_hutch_median(
    json_path: Path | str = Path(__file__).resolve().parent / "collate.json",
    output_pdf: Path | str | None = Path(__file__).resolve().parent / "trace_error_plot_median.pdf",
    panel_ylims: list[tuple[float | None, float | None] | None] | None = None,
    manual_ylims_by_beta: dict[float, tuple[float | None, float | None]] | None = None,
    show: bool = True,
):
    json_path = Path(json_path)

    _set_pub_rc()
    _register_scientific_colormaps()

    with json_path.open("r") as f:
        data = json.load(f)

    k_vals = np.asarray(_get_key(data, "K_VALS", "k_vals", "k_list", "ell_vals"), dtype=float)
    chi_list = np.asarray(_get_key(data, "CHI_LIST", "chi_list", "chis"), dtype=float)
    beta_list = np.asarray(_get_key(data, "BETA_LIST", "beta_list", "betas"), dtype=float)

    rel_trace_err = np.asarray(
        _get_key(data, "rel_trace_err", "relative_trace_error", "rel_err"),
        dtype=float,
    )

    gauss_rel_median_exact = None
    if "gauss_rel_median_exact" in data:
        gauss_rel_median_exact = np.asarray(data["gauss_rel_median_exact"], dtype=float)

    eps = 1e-300
    median_rte = np.maximum(np.median(rel_trace_err, axis=2), eps)
    q10_rte = np.maximum(np.quantile(rel_trace_err, 0.10, axis=2), eps)
    q25_rte = np.maximum(np.quantile(rel_trace_err, 0.25, axis=2), eps)
    q75_rte = np.maximum(np.quantile(rel_trace_err, 0.75, axis=2), eps)
    q90_rte = np.maximum(np.quantile(rel_trace_err, 0.90, axis=2), eps)

    base_cmap = plt.get_cmap(CMAP_NAME)
    truncated_cmap = LinearSegmentedColormap.from_list(
        f"{CMAP_NAME}_trunc",
        base_cmap(np.linspace(TRUNC_LO, TRUNC_HI, 256)),
    )
    norm = Normalize(vmin=float(np.min(chi_list)), vmax=float(np.max(chi_list)))

    n = 50
    if gauss_rel_median_exact is None:
        gauss_rel_median_exact = gaussian_median_relative_error_curve_open_xx(
            n=n,
            beta_list=beta_list,
            k_vals=k_vals,
            nsamples=200_000,
            batch=20_000,
            base_seed=2026,
        )

    ell_vals = np.asarray(k_vals, dtype=float)
    valid = ell_vals > 0
    red_ref_curve = 1.0 / np.sqrt(ell_vals[valid])
    shared_ymax = 2.0

    fig = plt.figure(figsize=(19.6, 5.9))
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=len(beta_list),
        width_ratios=[1] * len(beta_list),
        wspace=0.24,
    )
    axs = [fig.add_subplot(gs[0, bi]) for bi in range(len(beta_list))]

    for bi, beta in enumerate(beta_list):
        ax = axs[bi]
        plot_fn = ax.loglog
        draw_order = list(range(len(chi_list)))[::-1]

        for ci in draw_order:
            chi = chi_list[ci]
            color = truncated_cmap(norm(chi))

            ax.fill_between(
                k_vals,
                q25_rte[bi, ci, :],
                q75_rte[bi, ci, :],
                color=color,
                alpha=0.25,
                linewidth=0.5,
                edgecolor=(*mpl.colors.to_rgb(color), 0.3),
            )

            plot_fn(
                k_vals,
                median_rte[bi, ci, :],
                linestyle="-",
                marker="o",
                ms=LINE_MS,
                lw=LINE_LW,
                color=color,
                markeredgecolor=darken(color),
                markeredgewidth=MARKER_MEW,
                clip_on=False,
            )

        gauss_rel_median = gauss_rel_median_exact[bi, valid]

        plot_fn(
            ell_vals[valid],
            red_ref_curve,
            color=SQRT2_COLOR,
            lw=REF_LW,
            ls=REF_LS,
            dash_capstyle="round",
            zorder=5,
        )

        plot_fn(
            ell_vals[valid],
            gauss_rel_median,
            color=GAUSS_COLOR,
            lw=REF_LW,
            ls=REF_LS,
            dash_capstyle="round",
            zorder=5,
        )

        panel_ymin, _ = auto_log_ylim(
            q10_rte[bi, :, :],
            q90_rte[bi, :, :],
            median_rte[bi, :, :],
            red_ref_curve,
            gauss_rel_median,
        )

        if np.isclose(beta, 0.1):
            suffix = " (flat spectrum)"
        elif np.isclose(beta, 1.0):
            suffix = " (slow decay)"
        elif np.isclose(beta, 10.0):
            suffix = " (fast decay)"
        else:
            suffix = ""
        ax.set_title(rf"$\beta = {beta:g}$" + suffix)
        ax.set_xlabel(r"Matrix-$\mathsf{MPS}$ products $\ell$")
        ax.set_ylabel("Relative trace error", fontsize=Y_AXIS_LABEL_FS)
        ax.set_xscale("log")
        manual_override = None
        if manual_ylims_by_beta is not None:
            for b_key, lims in manual_ylims_by_beta.items():
                if np.isclose(float(beta), float(b_key)):
                    manual_override = lims
                    break

        if manual_override is not None:
            yb, yt = manual_override
            if yb is None:
                yb = panel_ymin
            if yt is None:
                yt = shared_ymax
            ax.set_ylim(bottom=yb, top=yt)
        elif panel_ylims is not None and bi < len(panel_ylims) and panel_ylims[bi] is not None:
            yb, yt = panel_ylims[bi]
            if yb is None:
                yb = panel_ymin
            if yt is None:
                yt = shared_ymax
            ax.set_ylim(bottom=yb, top=yt)
        else:
            is_beta10_panel = np.isclose(float(beta), 10.0) or (bi == len(beta_list) - 1)
            if is_beta10_panel:
                ax.set_ylim(bottom=PLOT3_YMIN, top=shared_ymax)
            else:
                ax.set_ylim(panel_ymin, shared_ymax)

        y_lo, y_hi = ax.get_ylim()
        y_lo = max(float(y_lo), 1e-300)
        y_hi = max(float(y_hi), y_lo * 10.0)
        ymax_decade = int(np.ceil(np.log10(y_hi)))
        ymin_decade = int(np.floor(np.log10(y_lo)))
        major_ticks = 10.0 ** np.arange(ymin_decade, ymax_decade)
        if major_ticks.size == 0:
            major_ticks = np.array([10.0 ** ymin_decade], dtype=float)
        major_ticks = major_ticks[(major_ticks >= y_lo) & (major_ticks <= y_hi)]
        if major_ticks.size == 0:
            major_ticks = np.array([y_lo], dtype=float)

        ax.set_yticks(major_ticks)
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        ax.grid(False)
        ax.set_axisbelow(True)
        ax.tick_params(
            which="both",
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelleft=True,
            labelsize=TICK_FS,
            direction="out",
            width=0.8,
            length=4,
            pad=5,
        )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.18,
        top=0.84,
        wspace=0.24,
    )

    gradient_proxy = Line2D([], [], lw=LINE_LW, color=truncated_cmap(0.75), marker="o", ms=LINE_MS)
    sqrt2_proxy = Line2D([], [], color=SQRT2_COLOR, lw=REF_LW, ls=REF_LS)
    gauss_proxy = Line2D([], [], color=GAUSS_COLOR, lw=REF_LW, ls=REF_LS)

    chi_handler = HandlerColormapLine(
        cmap=truncated_cmap,
        n=64,
        lw=LINE_LW * 1.2,
        marker="o",
        ms=LINE_MS * 1.5,
        mew=MARKER_MEW,
        chi_values=[int(c) for c in chi_list],
    )

    labels = [
        r"$\mathsf{MPS}$ Girard–Hutchinson",
        r"$1/\sqrt{\ell}$",
        r"Gaussian (theoretical)",
    ]

    legend = fig.legend(
        handles=[gradient_proxy, sqrt2_proxy, gauss_proxy],
        labels=labels,
        handler_map={
            gradient_proxy: chi_handler,
            sqrt2_proxy: HandlerShortLine(SQRT2_COLOR, REF_LW, REF_LS, scale=0.7),
            gauss_proxy: HandlerShortLine(GAUSS_COLOR, REF_LW, REF_LS, scale=0.7),
        },
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        fontsize=LEGEND_FS,
        handlelength=4.0,
        borderpad=0.8,
        labelspacing=0.62,
        columnspacing=1.9,
        handletextpad=0.62,
    )

    mpl.rcParams["svg.fonttype"] = "none"

    if output_pdf is not None:
        fig.savefig(
            str(output_pdf),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            edgecolor="none",
        )

    if show:
        plt.show()

    return fig, axs


if __name__ == "__main__":
    plot_hutch_median()
