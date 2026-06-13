import numpy as np
import scipy.sparse as sps

from tnrnla.tn.mpo import MPO
from tnrnla.quantum.hamiltonians import mpo_from_fsm


# ---------------------------------------------------------------------
# Single-site operators
# ---------------------------------------------------------------------

I2 = np.eye(2, dtype=np.float64)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)


# ---------------------------------------------------------------------
# Bit-order helper
# ---------------------------------------------------------------------

def _bit_position_map(n, ordering):
    """
    Return a function i -> bit position in the packed integer state.

    ordering="msb" means site 0 is the most significant bit.
    ordering="lsb" means site 0 is the least significant bit.
    """
    n = int(n)
    ordering = str(ordering).lower()

    if ordering == "lsb":
        return lambda i: int(i)
    if ordering == "msb":
        return lambda i: int(n - 1 - i)
    raise ValueError("ordering must be 'lsb' or 'msb'")


# ---------------------------------------------------------------------
# Sparse TFIM in the Z Z + X convention
# ---------------------------------------------------------------------

def tfim_open_sparse_zz_x(n, h, J=1.0, *, dtype=np.float64, ordering="msb"):
    r"""
    Open-boundary TFIM in the Z Z + X convention

        H = -J \sum_{i=1}^{n-1} Z_i Z_{i+1} - h \sum_{i=1}^n X_i

    represented in the computational Z basis.
    """
    n = int(n)
    dim = 1 << n
    dt = np.dtype(dtype)
    bitpos = _bit_position_map(n, ordering)

    states = np.arange(dim, dtype=np.uint32)

    bits = np.empty((n, dim), dtype=np.uint32)
    for i in range(n):
        bits[i] = (states >> np.uint32(bitpos(i))) & np.uint32(1)

    spins = 1.0 - 2.0 * bits.astype(np.float64, copy=False)

    zz_sum = np.zeros(dim, dtype=np.float64)
    for i in range(n - 1):
        zz_sum += spins[i] * spins[i + 1]

    diag = (-float(J)) * zz_sum

    flip_masks = np.array([1 << bitpos(i) for i in range(n)], dtype=np.uint32)
    off_cols = (states[:, None] ^ flip_masks[None, :]).reshape(-1).astype(np.int32, copy=False)

    nnz = n + 1
    indptr = np.arange(dim + 1, dtype=np.int64) * nnz

    cols = np.empty(dim * nnz, dtype=np.int32)
    data = np.empty(dim * nnz, dtype=dt)

    cols[0::nnz] = states.astype(np.int32, copy=False)
    data[0::nnz] = diag.astype(dt, copy=False)

    cols.reshape(dim, nnz)[:, 1:] = off_cols.reshape(dim, n)
    data.reshape(dim, nnz)[:, 1:] = dt.type(-float(h))

    return sps.csr_matrix((data, cols, indptr), shape=(dim, dim))


def tfim_periodic_sparse_zz_x(n, h, J=1.0, *, dtype=np.float64, ordering="msb"):
    r"""
    Periodic-boundary TFIM in the Z Z + X convention

        H = -J \sum_{i=1}^{n} Z_i Z_{i+1} - h \sum_{i=1}^{n} X_i,
        with Z_{n+1} \equiv Z_1.

    Equivalently,

        H = -J \sum_{i=1}^{n} Z_i Z_{\bmod(i,n)+1} - h \sum_{i=1}^{n} X_i.

    represented in the computational Z basis.
    """
    n = int(n)
    dim = 1 << n
    dt = np.dtype(dtype)
    bitpos = _bit_position_map(n, ordering)

    states = np.arange(dim, dtype=np.uint32)

    bits = np.empty((n, dim), dtype=np.uint32)
    for i in range(n):
        bits[i] = (states >> np.uint32(bitpos(i))) & np.uint32(1)

    spins = 1.0 - 2.0 * bits.astype(np.float64, copy=False)

    zz_sum = np.zeros(dim, dtype=np.float64)
    for i in range(n):
        zz_sum += spins[i] * spins[(i + 1) % n]

    diag = (-float(J)) * zz_sum

    flip_masks = np.array([1 << bitpos(i) for i in range(n)], dtype=np.uint32)
    off_cols = (states[:, None] ^ flip_masks[None, :]).reshape(-1).astype(np.int32, copy=False)

    nnz = n + 1
    indptr = np.arange(dim + 1, dtype=np.int64) * nnz

    cols = np.empty(dim * nnz, dtype=np.int32)
    data = np.empty(dim * nnz, dtype=dt)

    cols[0::nnz] = states.astype(np.int32, copy=False)
    data[0::nnz] = diag.astype(dt, copy=False)

    cols.reshape(dim, nnz)[:, 1:] = off_cols.reshape(dim, n)
    data.reshape(dim, nnz)[:, 1:] = dt.type(-float(h))

    return sps.csr_matrix((data, cols, indptr), shape=(dim, dim))


# ---------------------------------------------------------------------
# MPO TFIM in the Z Z + X convention
# ---------------------------------------------------------------------

def tfim_zz_x(N, J=1.0, h=1.0, local_ops=False):
    r"""
    Open-boundary MPO for

        H = -J \sum_{i=1}^{N-1} Z_i Z_{i+1} - h \sum_{i=1}^{N} X_i.
    """
    fsm = {
        (0, 0): I2,
        (0, 1): -float(J) * Z,
        (1, 2): Z,
        (0, 2): -float(h) * X,
        (2, 2): I2,
    }
    cores = mpo_from_fsm(fsm, 3, int(N), source=0, target=2)
    if local_ops:
        return cores
    return MPO(cores)


def mpo_wrap_Z1_ZN(N, coeff, *, dtype=np.float64):
    """
    MPO for coeff * Z_1 Z_N in 1-indexed notation,
    i.e. coeff * Z_0 Z_{N-1} in 0-indexed Python notation.
    """
    N = int(N)
    if N < 2:
        raise ValueError("Need N >= 2")

    d = 2
    D = 2
    I = np.asarray(I2, dtype=dtype)
    Zop = np.asarray(Z, dtype=dtype)

    cores = []

    W0 = np.zeros((d, D, d), dtype=dtype)
    W0[:, 0, :] = I
    W0[:, 1, :] = coeff * Zop
    cores.append(W0)

    for _ in range(N - 2):
        W = np.zeros((D, d, D, d), dtype=dtype)
        W[0, :, 0, :] = I
        W[1, :, 1, :] = I
        cores.append(W)

    WN = np.zeros((D, d, d), dtype=dtype)
    WN[1, :, :] = Zop
    cores.append(WN)

    return MPO(cores)


def periodic_tfim_zz_x(N, J=1.0, h=1.0, local_ops=False):
    r"""
    Periodic-boundary MPO for

        H = -J \sum_{i=1}^{N} Z_i Z_{i+1} - h \sum_{i=1}^{N} X_i,
        with Z_{N+1} \equiv Z_1.

    This is implemented as the open-chain MPO plus the wrap term
    -J Z_1 Z_N.
    """
    H_open = tfim_zz_x(int(N), J=float(J), h=float(h), local_ops=False)
    H_wrap = mpo_wrap_Z1_ZN(int(N), coeff=-float(J), dtype=np.float64)

    try:
        H = H_open + H_wrap
    except Exception:
        H = H_open.add(H_wrap, subtract=False)

    if local_ops:
        return H.cores if hasattr(H, "cores") else H
    return H


def periodic_tfim_split(N, J=1.0, h=1.0):
    r"""
    Return the periodic MPO together with its open-chain and wrap pieces.

        H_periodic = H_open + H_wrap

    where

        H_open = -J \sum_{i=1}^{N-1} Z_i Z_{i+1} - h \sum_{i=1}^{N} X_i
        H_wrap = -J Z_1 Z_N.
    """
    H_open = tfim_zz_x(int(N), J=float(J), h=float(h), local_ops=False)
    H_wrap = mpo_wrap_Z1_ZN(int(N), coeff=-float(J), dtype=np.float64)

    try:
        H = H_open + H_wrap
    except Exception:
        H = H_open.add(H_wrap, subtract=False)

    return H, H_open, H_wrap


def _log_parity_sums_from_eps(eps, beta):
    beta = float(beta)
    logZe = 0.0
    logZo = -np.inf
    for e in np.asarray(eps, dtype=np.float64):
        y = -beta * float(e)
        logZe_new = np.logaddexp(logZe, logZo + y)
        logZo_new = np.logaddexp(logZo, logZe + y)
        logZe, logZo = logZe_new, logZo_new
    return float(logZe), float(logZo)


def tfim_periodic_analytic_observables(n, h, J=1.0, beta=0.5):
    """
    Return (E_avg, E_var) for the periodic TFIM analytically.

    Uses the analytically computed partition function and numerical
    differentiation of log Z with respect to beta:
        <E>    = -d/d beta  log Z
        Var(E) =  d^2/d beta^2  log Z
    """
    n = int(n)
    h, J, beta = float(h), float(J), float(beta)
    db = max(abs(beta) * 1e-5, 1e-8)
    logZm = tfim_periodic_log_trace_exp_analytic(n, h, J, beta - db)
    logZ0 = tfim_periodic_log_trace_exp_analytic(n, h, J, beta)
    logZp = tfim_periodic_log_trace_exp_analytic(n, h, J, beta + db)
    E_avg = -(logZp - logZm) / (2 * db)
    E_var = (logZp - 2 * logZ0 + logZm) / db**2
    return float(E_avg), float(E_var)


def tfim_periodic_E_max_analytic(n, h, J=1.0):
    """
    Exact maximum eigenvalue of the periodic TFIM Hamiltonian in O(n) time.

    Uses the quasi-particle dispersion directly — no 2^n enumeration.

    Sector 1 (anti-periodic BC): all eps1_k > 0, so
        E_max1 = -E0_1 = 0.5 * sum(eps1)   [all modes occupied]

    Sector 2 (periodic BC): eps2 can be negative at k=0 and k=π (h>J),
    so maximize by occupying only positive-energy modes:
        E_max2 = E0_2 + sum(max(eps2_k, 0))
    """
    n = int(n)
    if n % 2:
        raise ValueError("analytic periodic TFIM E_max requires n even")
    h, J = float(h), float(J)

    ks1 = np.arange(-n + 1, n, 2, dtype=np.float64) * np.pi / n
    eps1 = 2.0 * np.sqrt(J * J + h * h + 2.0 * J * h * np.cos(ks1))
    E_max1 = 0.5 * float(np.sum(eps1))

    ks2 = np.arange(-n // 2, n // 2, 1, dtype=np.float64) * 2.0 * np.pi / n
    eps2 = 2.0 * np.sqrt(J * J + h * h + 2.0 * J * h * np.cos(ks2))
    eps2[0] = -2.0 * (J + h)
    eps2[n // 2] = 2.0 * (J - h)
    E0_2 = -0.5 * float(np.sum(eps2))
    E_max2 = E0_2 + float(np.sum(np.maximum(eps2, 0.0)))

    return max(E_max1, E_max2)


def tfim_periodic_E_min_analytic(n, h, J=1.0):
    """
    Exact minimum eigenvalue (ground state energy) of the periodic TFIM.

    Sector 1: E_min1 = -0.5 * sum(eps1)  (all modes unoccupied)
    Sector 2: E_min2 = -0.5 * sum(|eps2|)  (occupy only negative-energy modes)
    """
    n = int(n)
    if n % 2:
        raise ValueError("analytic periodic TFIM E_min requires n even")
    h, J = float(h), float(J)

    ks1 = np.arange(-n + 1, n, 2, dtype=np.float64) * np.pi / n
    eps1 = 2.0 * np.sqrt(J * J + h * h + 2.0 * J * h * np.cos(ks1))
    E_min1 = -0.5 * float(np.sum(eps1))

    ks2 = np.arange(-n // 2, n // 2, 1, dtype=np.float64) * 2.0 * np.pi / n
    eps2 = 2.0 * np.sqrt(J * J + h * h + 2.0 * J * h * np.cos(ks2))
    eps2[0] = -2.0 * (J + h)
    eps2[n // 2] = 2.0 * (J - h)
    E_min2 = -0.5 * float(np.sum(np.abs(eps2)))

    return min(E_min1, E_min2)


def tfim_periodic_tau_krylov(n, h, J=1.0):
    """
    Exact Krylov step size for imaginary-time GSE enrichment.

    Uses tau = 1 / (E_max - E_min) so that all eigenvalues of
    A = I - tau*H lie in (0, 1), guaranteeing ||A||_2 < 1 and
    preventing Krylov vector overflow.

    With tau = 1/E_max (old formula), A has eigenvalue
    1 + |E_min|/E_max > 1 at the ground state, causing the Krylov
    vectors to grow by ~||A||^j per step and eventually overflow.
    """
    E_max = tfim_periodic_E_max_analytic(n, h=h, J=J)
    E_min = tfim_periodic_E_min_analytic(n, h=h, J=J)
    return 1.0 / (E_max - E_min)


def tfim_periodic_trace_exp_analytic(n, h, J=1.0, beta=0.5):
    """Return tr[exp(-beta H)] with bounded conversion from log-space."""
    logZ = tfim_periodic_log_trace_exp_analytic(n, h, J=J, beta=beta)
    log_max = float(np.log(np.finfo(np.float64).max))
    log_tiny = float(np.log(np.finfo(np.float64).tiny))
    if logZ >= log_max:
        return float(np.finfo(np.float64).max)
    if logZ <= log_tiny:
        return 0.0
    return float(np.exp(logZ))


def tfim_periodic_log_trace_exp_analytic(n, h, J=1.0, beta=0.5):
    """Return log(tr[exp(-beta H)]) for periodic TFIM (even n)."""
    n = int(n)
    if n % 2:
        raise ValueError("analytic periodic TFIM trace requires n even")

    h = float(h)
    J = float(J)
    beta = float(beta)

    ks1 = (np.arange(-n + 1, n, 2, dtype=np.float64) * np.pi / n)
    eps1 = 2.0 * np.sqrt(J * J + h * h + 2.0 * J * h * np.cos(ks1))
    E0_1 = -0.5 * float(np.sum(eps1))

    ks2 = (np.arange(-n // 2, n // 2, 1, dtype=np.float64) * 2.0 * np.pi / n)
    eps2 = 2.0 * np.sqrt(J * J + h * h + 2.0 * J * h * np.cos(ks2))
    eps2[0] = -2.0 * (J + h)
    eps2[n // 2] = 2.0 * (J - h)
    E0_2 = -0.5 * float(np.sum(eps2))

    logZe1, _ = _log_parity_sums_from_eps(eps1, beta)
    _, logZo2 = _log_parity_sums_from_eps(eps2, beta)

    logZ1 = (-beta * E0_1) + logZe1
    logZ2 = (-beta * E0_2) + logZo2
    return float(np.logaddexp(logZ1, logZ2))


def tfim_periodic_eigs_analytic(n, h, J=1.0):
    n = int(n); h = float(h); J = float(J)
    if n % 2:
        raise ValueError('Not implemented for n odd (matches MATLAB construction).')

    states = np.arange(1 << n, dtype=np.uint64)[:, None]
    bits = ((states >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int8)
    parity = (bits.sum(axis=1) & 1).astype(np.int8)

    ks1 = (np.arange(-n + 1, n, 2, dtype=np.float64) * np.pi / n)
    eps1 = 2.0 * np.sqrt(J * J + h * h + 2.0 * J * h * np.cos(ks1))
    E0_1 = -0.5 * eps1.sum()
    E_even = bits[parity == 0, :] @ eps1.reshape(-1, 1) + E0_1

    ks2 = (np.arange(-n / 2, n / 2, 1, dtype=np.float64) * 2.0 * np.pi / n)
    eps2 = 2.0 * np.sqrt(J * J + h * h + 2.0 * J * h * np.cos(ks2))
    eps2[0] = -2.0 * (J + h)
    eps2[n // 2] = 2.0 * (J - h)
    E0_2 = -0.5 * eps2.sum()
    E_odd = bits[parity == 1, :] @ eps2.reshape(-1, 1) + E0_2

    E = np.concatenate([E_even.ravel(), E_odd.ravel()]).astype(np.float64, copy=False)
    E.sort()
    return E


def plot_periodic_tfim_exp_spectrum_from_cfg(
    cfg,
    *,
    eig_fn=tfim_periodic_eigs_analytic,
    cmap_name="Greens",
    cmap_lo=0.25,
    cmap_hi=0.90,
    n_step=None,
    n_max=None,
    max_exact_n=22,
    max_points_per_curve=20000,
    min_positive=1e-300,
    add_colorbar=False,
    colorbar_tick_step=None,
    colorbar_label="System Size n",
    show_legend=True,
    ax=None,
    show=True,
):
    """
    Plot spectra of exp(-beta H) for periodic TFIM on one log-scale axis.

    Parameters
    ----------
    cfg : dict
        Must contain at least keys: "n_values", "h", "beta", optionally "J".
    eig_fn : callable
        Function returning sorted many-body energies E for a given n, h, J.
    n_step : int or None
        If provided, replace cfg["n_values"] by range(min_n, max_n + 1, n_step).
        Useful to "show n in steps of 2", etc.
    n_max : int or None
        Optional hard cap for n when building stepped values.
    max_exact_n : int
        Hard safety cap for exact many-body enumeration. Values above are skipped.
    max_points_per_curve : int
        If a spectrum is larger, downsample by index for plotting clarity.
    min_positive : float
        Floor before log scaling to avoid zeros from floating-point underflow.
    add_colorbar : bool
        If True, add a colorbar on the right mapping color -> system size n.
    colorbar_tick_step : int or None
        If provided and colorbar is enabled, set colorbar ticks in this step size.
    colorbar_label : str
        Label for the colorbar.
    show_legend : bool
        If True, draw a legend for plotted n curves.
    ax : matplotlib.axes.Axes or None
        Existing axis to draw on; if None, create a new figure+axis.
    show : bool
        If True, call plt.show() when done.

    Returns
    -------
    dict with keys:
        "fig", "ax", "spectra", "skipped"
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    n_values = list(cfg.get("n_values", []))
    if not n_values:
        raise ValueError("cfg must provide non-empty 'n_values'.")
    n_values = [int(n) for n in n_values]

    if n_step is not None:
        n_step = int(n_step)
        if n_step <= 0:
            raise ValueError("n_step must be positive when provided.")
        n_min = int(min(n_values))
        n_hi = int(max(n_values))
        if n_max is not None:
            n_hi = min(n_hi, int(n_max))
        n_values = list(range(n_min, n_hi + 1, n_step))

    h = float(cfg["h"])
    J = float(cfg.get("J", 1.0))
    beta = float(cfg["beta"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    if not (0.0 <= float(cmap_lo) < float(cmap_hi) <= 1.0):
        raise ValueError("Require 0 <= cmap_lo < cmap_hi <= 1.")
    base_cmap = plt.get_cmap(cmap_name)
    truncated_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc_{float(cmap_lo):.2f}_{float(cmap_hi):.2f}",
        base_cmap(np.linspace(float(cmap_lo), float(cmap_hi), 256)),
    )
    color_norm = mpl.colors.Normalize(vmin=float(min(n_values)), vmax=float(max(n_values)))

    spectra = {}
    skipped = []

    for i, n in enumerate(n_values):
        n = int(n)
        if n > int(max_exact_n):
            skipped.append(
                (n, f"skipped: n={n} exceeds max_exact_n={int(max_exact_n)} for exact 2^n enumeration")
            )
            continue
        if n % 2:
            skipped.append((n, "skipped: analytic periodic formula here requires even n"))
            continue

        try:
            E = np.asarray(eig_fn(n=n, h=h, J=J), dtype=np.float64)
        except Exception as exc:
            skipped.append((n, f"skipped: {exc}"))
            continue

        lam = np.exp(-beta * E)
        lam = np.sort(np.maximum(lam, float(min_positive)))[::-1]

        if lam.size > int(max_points_per_curve):
            idx = np.linspace(0, lam.size - 1, int(max_points_per_curve), dtype=np.int64)
            x = idx + 1
            y = lam[idx]
        else:
            x = np.arange(1, lam.size + 1, dtype=np.int64)
            y = lam

        spectra[n] = {"x": x, "lambda": y, "full_size": int(lam.size)}
        ax.plot(x, y, color=truncated_cmap(color_norm(float(n))), lw=1.1, alpha=0.95, label=f"n={n}")
        if y.size >= 1:
            ax.plot(x[-1], y[-1], marker="o", color="red", markersize=4.0, linestyle="None", zorder=5)
        if y.size >= 2:
            ax.plot(x[-2], y[-2], marker="o", color="red", markersize=4.0, linestyle="None", zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel(r"$\lambda_i\left(e^{-\beta H}\right)$")
    ax.set_title(f"Periodic TFIM exp(-beta H) Spectrum (h={h:g}, J={J:g}, beta={beta:g})")
    ax.grid(True, which="both", alpha=0.25)

    if spectra and show_legend:
        ax.legend(frameon=False, ncol=2, fontsize=9)
    if add_colorbar:
        sm = mpl.cm.ScalarMappable(norm=color_norm, cmap=truncated_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(str(colorbar_label))
        if colorbar_tick_step is not None:
            tstep = int(colorbar_tick_step)
            if tstep <= 0:
                raise ValueError("colorbar_tick_step must be positive when provided.")
            ticks = np.arange(int(min(n_values)), int(max(n_values)) + 1, tstep, dtype=int)
            cbar.set_ticks(ticks)
    if show:
        plt.show()

    return {"fig": fig, "ax": ax, "spectra": spectra, "skipped": skipped}


import numpy as np


def ptfim_rdm_analytic_eigs(
    n,
    h,
    J=1.0,
    ell=None,
    start=0,
    tol=1e-14,
    return_nus=False,
    return_entropy=False,
):
    """
    Exact contiguous-block RDM data for the periodic TFIM ground state.

    The spin Hamiltonian is
        H = -J sum_j Z_j Z_{j+1} - h sum_j X_j

    Entanglement is computed through the equivalent rotated free-fermion form
        H' = -J sum_j X_j X_{j+1} - h sum_j Z_j.

    Assumptions
      - n is even
      - the ground state is the finite-size even-parity Gaussian state
      - the block [start, start + ell) does not wrap around the chain

    Returns
      - lam by default
      - (lam, nu) if return_nus=True
      - (lam, S) if return_entropy=True
      - (lam, nu, S) if both flags are True

    where
      - lam are the exact RDM eigenvalues in descending order
      - nu are the single-particle entanglement parameters
      - S is the exact von Neumann entropy
    """
    n = int(n)
    h = float(h)
    J = float(J)

    if n % 2:
        raise ValueError("This implementation assumes n even.")

    if ell is None:
        ell = n // 2
    ell = int(ell)
    start = int(start)

    if not (0 <= start < n):
        raise ValueError("start must satisfy 0 <= start < n")
    if not (1 <= ell <= n):
        raise ValueError("ell must satisfy 1 <= ell <= n")
    if start + ell > n:
        raise ValueError("This helper assumes a contiguous block without wraparound.")

    # Majorana quadratic form in the even-parity sector
    # gamma = (a_1, b_1, ..., a_n, b_n)
    A = np.zeros((2 * n, 2 * n), dtype=np.float64)

    # onsite field term
    for j in range(n):
        a = 2 * j
        b = 2 * j + 1
        A[a, b] = 2.0 * h
        A[b, a] = -2.0 * h

    # nearest-neighbor coupling
    for j in range(n - 1):
        b = 2 * j + 1
        a_next = 2 * (j + 1)
        A[b, a_next] = 2.0 * J
        A[a_next, b] = -2.0 * J

    # periodic boundary in the even-parity sector
    A[2 * n - 1, 0] = -2.0 * J
    A[0, 2 * n - 1] = 2.0 * J

    # Ground-state covariance
    # Gamma_{mn} = (i/2) < [gamma_m, gamma_n] >
    Hm = 1j * A
    evals, U = np.linalg.eigh(Hm)

    scale = max(np.max(np.abs(evals)), 1.0)
    sgn = np.sign(np.real(evals))
    sgn[np.abs(evals) <= tol * scale] = 0.0

    Gamma = 1j * ((U * sgn[None, :]) @ U.conj().T)
    Gamma = np.real_if_close(Gamma, tol=1000)
    Gamma = np.asarray(np.real(Gamma), dtype=np.float64)
    Gamma = 0.5 * (Gamma - Gamma.T)

    # Restrict to the block
    idx = np.arange(2 * start, 2 * (start + ell))
    Gamma_A = Gamma[np.ix_(idx, idx)]

    # i Gamma_A has eigenvalues ±nu_j
    vals = np.linalg.eigvalsh(1j * Gamma_A)
    vals = np.real_if_close(vals)
    vals = np.asarray(np.real(vals), dtype=np.float64)

    nu = np.sort(vals)[ell:]
    nu = np.clip(nu, 0.0, 1.0)

    # Exact RDM spectrum
    lam = np.array([1.0], dtype=np.float64)
    for x in nu:
        p0 = 0.5 * (1.0 + x)
        p1 = 0.5 * (1.0 - x)
        lam = np.kron(lam, np.array([p0, p1], dtype=np.float64))

    lam = np.sort(lam)[::-1]

    if not return_nus and not return_entropy:
        return lam

    out = [lam]

    if return_nus:
        out.append(nu)

    if return_entropy:
        p_plus = 0.5 * (1.0 + nu)
        p_minus = 0.5 * (1.0 - nu)
        p_plus = p_plus[p_plus > tol]
        p_minus = p_minus[p_minus > tol]
        S = -np.sum(p_plus * np.log(p_plus)) - np.sum(p_minus * np.log(p_minus))
        out.append(float(np.real(S)))

    return tuple(out)
