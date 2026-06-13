
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from tnrnla.linalg import lan_exp
from tnrnla.tn.other.tensor_ops import (
    flatten_right,
    flatten_left,
    _flatten_last2,
    _ascontig,
    qr_right,
    lq_left,
)
from tnrnla.linalg.lra import truncated_svd,Cutoff


oracle_module = None
oracle_error = None


def get_N(obj) -> int:
    return int(getattr(obj, "N", len(obj)))
def canonical_split_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in {"qr", "svd"}:
        raise ValueError("split_mode must be either 'qr' or 'svd'")
    return mode
def _diag_times_rows(s, Vt):
    """
    Return diag(s) @ Vt without explicitly forming diag(s).
    """
    return s[:, None] * Vt
def _cols_times_diag(U, s):
    """
    Return U @ diag(s) without explicitly forming diag(s).
    """
    return U * s[None, :]


def split_left_tensor_qr_or_svd(A, *, split_mode="qr", cutoff=None, abstol=0.0, check_finite=True):
    """
    Split A as A ≈ Q @ C with Q left-isometric.

    For rank-2 A shape (d, r), view as matrix (d, r).
    For rank-3 A shape (l, d, r), view as matrix (l*d, r).

    Returns
    -------
    Q_tensor, C
        Q_tensor has same rank as A and right bond dim = kept rank.
        C has shape (kept_rank, old_right_dim).
    """
    A = _ascontig(A)
    shape = A.shape

    if A.ndim == 2:
        m, n = shape
        M = A
        left_shape = (m,)
    elif A.ndim == 3:
        ldim, pdim, rdim = shape
        M = A.reshape(ldim * pdim, rdim)
        left_shape = (ldim, pdim)
        n = rdim
    else:
        raise ValueError(f"Unsupported tensor rank {A.ndim} in split_left_tensor_qr_or_svd")

    if split_mode == "qr":
        Q_tensor, C = qr_right(A)
        return _ascontig(Q_tensor), _ascontig(C)

    U, s, Vt = truncated_svd(M, stop=cutoff, abstol=abstol, check_finite=check_finite)
    k = len(s)

    if A.ndim == 2:
        Q_tensor = U.reshape(left_shape[0], k)
    else:
        Q_tensor = U.reshape(left_shape[0], left_shape[1], k)

    C = _diag_times_rows(s, Vt)
    return _ascontig(Q_tensor), _ascontig(C)


def split_right_tensor_qr_or_svd(A, *, split_mode="qr", cutoff=None, abstol=0.0, check_finite=True):
    """
    Split A as A ≈ C @ Q with Q right-isometric.

    For rank-2 A shape (l, d), view as matrix (l, d).
    For rank-3 A shape (l, d, r), view as matrix (l, d*r).

    Returns
    -------
    C, Q_tensor
        C has shape (old_left_dim, kept_rank).
        Q_tensor has same rank as A and left bond dim = kept rank.
    """
    A = _ascontig(A)
    shape = A.shape

    if A.ndim == 2:
        ldim, pdim = shape
        M = A
        right_shape = (pdim,)
    elif A.ndim == 3:
        ldim, pdim, rdim = shape
        M = A.reshape(ldim, pdim * rdim)
        right_shape = (pdim, rdim)
    else:
        raise ValueError(f"Unsupported tensor rank {A.ndim} in split_right_tensor_qr_or_svd")

    if split_mode == "qr":
        C, Q_tensor = lq_left(A)
        return _ascontig(C), _ascontig(Q_tensor)

    U, s, Vt = truncated_svd(M, stop=cutoff, abstol=abstol, check_finite=check_finite)
    k = len(s)

    C = _cols_times_diag(U, s)

    if A.ndim == 2:
        Q_tensor = Vt.reshape(k, right_shape[0])
    else:
        Q_tensor = Vt.reshape(k, right_shape[0], right_shape[1])

    return _ascontig(C), _ascontig(Q_tensor)

def load_oracle_module():
    global oracle_module, oracle_error

    if oracle_module is not None:
        return oracle_module
    if oracle_error is not None:
        raise oracle_error

    here = None
    spec = globals().get("__spec__", None)
    if spec is not None and getattr(spec, "origin", None):
        here = os.path.dirname(spec.origin)
    if not here:
        here = os.getcwd()

    build_dir = os.path.join(here, "build")
    if os.path.isdir(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)

    try:
        import tdvp_oracle_prefused as pf

        oracle_module = pf
        return pf
    except Exception as exc:
        oracle_module = None
        oracle_error = exc
        raise


class MPOViews:
    def __init__(self, mpo):
        self.N = get_N(mpo)
        self.site = []

        for i in range(self.N):
            A = _ascontig(mpo[i])
            data = {"A": A, "ndim": A.ndim}

            if A.ndim == 3:
                T021 = _ascontig(A.transpose(0, 2, 1))
                T120 = _ascontig(A.transpose(1, 2, 0))
                T102 = _ascontig(A.transpose(1, 0, 2))

                data["T021"] = T021
                data["T120"] = T120
                data["T102"] = T102

                data["mat_A_0"] = flatten_right(A)
                data["mat_T021_last"] = flatten_left(T021)
                data["mat_T120_last"] = flatten_left(T120)
                data["mat_T102_0"] = flatten_right(T102)

            elif A.ndim == 4:
                T0312 = _ascontig(A.transpose(0, 3, 1, 2))
                T1203 = _ascontig(A.transpose(1, 2, 0, 3))

                data["T0312"] = T0312
                data["T1203"] = T1203

                wl, wr = A.shape[0], A.shape[1]
                data["mat_A_0"] = flatten_right(A)
                data["mat_A_01"] = _ascontig(A.reshape(wl * wr, -1))
                data["mat_T0312_last2"] = _flatten_last2(T0312)
                data["mat_T1203_last2"] = _flatten_last2(T1203)

            else:
                raise ValueError(f"Unsupported MPO tensor rank at site {i}")

            self.site.append(data)


@dataclass
class TDVPContext:
    mpo_views: MPOViews
    forward_phase: object
    center_phase: object
    krylov_depth: int
    track: bool
    LR: list | None = None
    left_sweep: bool = True
    oracle_cache: dict | None = None
    lanczos_work_cache: dict | None = None
    split_mode: str = "qr"
    cutoff: object | None = None
    svd_abstol: float = 0.0
    svd_check_finite: bool = True

@dataclass
class StopState:
    enabled: bool = False
    tol: float = 1e-8
    patience: int = 2
    min_sweeps: int = 2
    mode: str = "projective"

    stable_sweeps: int = 0
    sweeps_done: int = 0
    stopped: bool = False
    stop_reason: str | None = None

    prev_state: object | None = None

    current_site_sqsum: float = 0.0
    current_site_count: int = 0
    current_site_max: float = 0.0

    metric_history: list[float] = field(default_factory=list)
    projective_history: list[float] = field(default_factory=list)
    cosine_history: list[float] = field(default_factory=list)
    site_rms_history: list[float] = field(default_factory=list)
    site_max_history: list[float] = field(default_factory=list)


def mps_norm2(state):
    return state.inner_product(state)


def rescale_to_unit_norm(state, track=False, tag="", return_log_scale=False, center_index=0):
    N = get_N(state)
    c = int(center_index)
    if c < 0 or c >= N:
        raise IndexError(f"center_index={c} is out of bounds for an MPS with N={N}")

    n2 = mps_norm2(state)
    n2r = float(np.real(n2))

    if n2r <= 0.0:
        if track:
            print("track norm2 nonpositive", tag, n2)
        if return_log_scale:
            return n2, 0.0
        return n2

    s = np.sqrt(n2r)
    state[c] = state[c] / s
    log_scale = 0.5 * np.log(n2r)

    if return_log_scale:
        return n2, log_scale
    return n2


def track_norm(tag, state, track):
    if not track:
        return
    n2 = mps_norm2(state)
    print("track", tag, "norm2", float(np.real(n2)))


def local_norm2(X):
    arr = np.asarray(X)
    return float(np.real(np.vdot(arr.reshape(-1), arr.reshape(-1))))


def normalize_local_tensor(X, tag="", track=False):
    n2 = local_norm2(X)
    if (not np.isfinite(n2)) or n2 <= 0.0:
        raise ValueError(f"Nonpositive or nonfinite local norm in {tag}, got {n2}")
    s = np.sqrt(n2)
    if track:
        print("track", tag, "local norm2", n2)
    return X / s, float(np.log(s))


def maybe_normalize_local_tensor(X, *, enabled=True, tag="", track=False):
    if not bool(enabled):
        return X, 0.0
    return normalize_local_tensor(X, tag=tag, track=track)


def make_lanczos_work(n: int, k: int, dtype):
    dtype = np.dtype(dtype)
    return {
        "Q": np.empty((int(n), int(k)), dtype=dtype, order="F"),
        "alpha": np.empty((int(k),), dtype=np.float64),
        "beta": np.empty((max(int(k) - 1, 0),), dtype=np.float64),
        "vbuf": np.empty((int(n),), dtype=dtype),
    }


def lanczos_work_key(vec, k: int) -> tuple:
    arr = np.asarray(vec)
    return (int(arr.size), int(k), np.dtype(arr.dtype).str)


def get_lanczos_work(ctx, vec, k: int):
    if ctx.lanczos_work_cache is None:
        ctx.lanczos_work_cache = {}

    arr = np.asarray(vec)
    key = lanczos_work_key(arr, k)

    work = ctx.lanczos_work_cache.get(key, None)
    if work is None:
        work = make_lanczos_work(arr.size, k, arr.dtype)
        ctx.lanczos_work_cache[key] = work

    return work


def canonical_stop_mode(mode: str) -> str:
    mode = str(mode)

    if mode == "center":
        return "projective"

    if mode == "tangent":
        return "hybrid"

    if mode in {"site", "projective", "hybrid"}:
        return mode

    raise ValueError(
        "stop_mode must be one of "
        "'projective', 'hybrid', 'site' "
        "(legacy aliases 'center' -> 'projective', 'tangent' -> 'hybrid')"
    )


def projective_distance(a, b):
    aa = a.inner_product(a)
    bb = b.inner_product(b)

    naa = float(np.real(aa))
    nbb = float(np.real(bb))

    if (not np.isfinite(naa)) or (not np.isfinite(nbb)) or naa <= 0.0 or nbb <= 0.0:
        return np.inf, 0.0

    ov = a.inner_product(b)
    denom = np.sqrt(naa * nbb)
    if (not np.isfinite(denom)) or denom <= 0.0:
        return np.inf, 0.0

    cos = float(np.abs(ov) / denom)
    cos = min(1.0, max(0.0, cos))

    dist = float(np.sqrt(max(0.0, 2.0 - 2.0 * cos)))
    return dist, cos


def observe_site_proxy(stop, value):
    if not stop.enabled:
        return

    stop.current_site_count += 1

    if not np.isfinite(value):
        stop.current_site_sqsum = np.inf
        stop.current_site_max = np.inf
        return

    v = float(value)
    stop.current_site_sqsum += v * v
    stop.current_site_max = max(stop.current_site_max, v)


def current_site_rms(stop):
    if stop.current_site_count <= 0:
        return 0.0

    if not np.isfinite(stop.current_site_sqsum):
        return np.inf

    return float(np.sqrt(stop.current_site_sqsum / float(stop.current_site_count)))


def reset_sweep_observables(stop):
    stop.current_site_sqsum = 0.0
    stop.current_site_count = 0
    stop.current_site_max = 0.0


def finish_sweep(stop, state, track=False):
    if not stop.enabled:
        return False

    stop.sweeps_done += 1

    if stop.prev_state is None:
        proj = np.nan
        cos = np.nan
    else:
        proj, cos = projective_distance(state, stop.prev_state)

    site_rms = current_site_rms(stop)
    site_max = float(stop.current_site_max) if stop.current_site_count > 0 else 0.0

    if stop.mode == "projective":
        metric = proj
    elif stop.mode == "site":
        metric = site_rms
    elif stop.mode == "hybrid":
        if np.isnan(proj):
            metric = np.inf
        else:
            metric = max(float(proj), float(site_rms))
    else:
        raise ValueError(f"Unsupported stop mode {stop.mode!r}")

    stop.projective_history.append(float(proj) if np.isfinite(proj) else float("nan"))
    stop.cosine_history.append(float(cos) if np.isfinite(cos) else float("nan"))
    stop.site_rms_history.append(float(site_rms) if np.isfinite(site_rms) else np.inf)
    stop.site_max_history.append(float(site_max) if np.isfinite(site_max) else np.inf)
    stop.metric_history.append(float(metric) if np.isfinite(metric) else np.inf)

    if track:
        print(
            "track sweep",
            stop.sweeps_done,
            "metric",
            stop.metric_history[-1],
            "projective",
            stop.projective_history[-1],
            "site_rms",
            stop.site_rms_history[-1],
            "site_max",
            stop.site_max_history[-1],
            "cosine",
            stop.cosine_history[-1],
        )

    needed_sweeps = int(stop.min_sweeps)
    if stop.mode in {"projective", "hybrid"}:
        needed_sweeps = max(needed_sweeps, 2)

    metric_finite = np.isfinite(metric)
    if stop.sweeps_done >= needed_sweeps and metric_finite and float(metric) < stop.tol:
        stop.stable_sweeps += 1
    else:
        stop.stable_sweeps = 0

    if stop.stable_sweeps >= stop.patience:
        stop.stopped = True
        stop.stop_reason = f"{stop.mode}_metric_below_tol"

    stop.prev_state = state.copy()
    reset_sweep_observables(stop)

    return stop.stopped


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

    raise TypeError("Could not scale state by a scalar")


def _apply_log_scale_to_state(psi, log_scale, *, max_chunk_log=300.0):
    ls = float(log_scale)
    if ls == 0.0:
        return psi
    if not np.isfinite(ls):
        raise ValueError("log_scale must be finite")

    chunk = abs(float(max_chunk_log))
    if chunk <= 0.0:
        raise ValueError("max_chunk_log must be > 0")

    out = psi
    rem = ls
    while abs(rem) > chunk:
        step = np.sign(rem) * chunk
        out = _scale_state(out, float(np.exp(step)))
        rem -= step

    return _scale_state(out, float(np.exp(rem)))


def compute_right_envs(state, mpo_views):
    N = get_N(state)
    R = [None] * N

    mpo_last = mpo_views.site[-1]
    mpo_last_A = mpo_last["A"]
    mpo_last_T021_mat = mpo_last["mat_T021_last"]

    prod = mpo_last_T021_mat @ np.conj(state[-1]).T
    prod = prod.reshape(
        mpo_last_A.shape[0],
        mpo_last_A.shape[2],
        state[-1].shape[0],
    )
    R[-1] = prod.transpose(2, 0, 1)

    prod = flatten_left(R[-1]) @ state[-1].T
    R[-1] = prod.reshape(
        R[-1].shape[0],
        R[-1].shape[1],
        state[-1].shape[0],
    )

    for i in range(N - 2, 0, -1):
        Aconj = np.conj(state[i])
        prod = flatten_left(Aconj) @ flatten_right(R[i + 1])
        R[i] = prod.reshape(
            Aconj.shape[0],
            Aconj.shape[1],
            R[i + 1].shape[1],
            R[i + 1].shape[2],
        )

        mpo_i = mpo_views.site[i]
        mpo_i_A = mpo_i["A"]
        mpo_i_T0312_mat = mpo_i["mat_T0312_last2"]

        prod = mpo_i_T0312_mat @ _flatten_last2(R[i].transpose(1, 2, 0, 3))
        prod = prod.reshape(
            mpo_i_A.shape[0],
            mpo_i_A.shape[3],
            R[i].shape[0],
            R[i].shape[-1],
        )
        R[i] = prod.transpose(2, 0, 1, 3)

        mps_flat = state[i].reshape(state[i].shape[0], -1)
        B = R[i].reshape(R[i].shape[0] * R[i].shape[1], -1)
        R[i] = (B @ mps_flat.T).reshape(R[i].shape[0], R[i].shape[1], -1)

    return R


def update_left_site(state, site, mpo_views, LR, *, split_mode="qr", cutoff=None, svd_abstol=0.0, svd_check_finite=True):
    """
    Left-to-right orthogonalization step.

    Produces
        state[site] <- left-isometric tensor Q
        returns C
    where
        old_tensor ≈ Q @ C
    and for SVD mode
        C = S Vh.
    """
    if site == 0:
        Q, C = split_left_tensor_qr_or_svd(
            state[0],
            split_mode=split_mode,
            cutoff=cutoff,
            abstol=svd_abstol,
            check_finite=svd_check_finite,
        )
        state[0] = Q
    else:
        Q, C = split_left_tensor_qr_or_svd(
            state[site],
            split_mode=split_mode,
            cutoff=cutoff,
            abstol=svd_abstol,
            check_finite=svd_check_finite,
        )
        state[site] = Q

    if site == 0:
        mpo0 = mpo_views.site[0]
        prod = np.conj(state[0]).T @ mpo0["mat_A_0"]
        LR[0] = prod.reshape(
            prod.shape[0],
            mpo0["A"].shape[1],
            mpo0["A"].shape[2],
        )

        prod2 = flatten_left(LR[0]) @ state[0]
        LR[0] = prod2.reshape(
            LR[0].shape[0],
            LR[0].shape[1],
            prod2.shape[-1],
        )
    else:
        prod = (
            flatten_left(LR[site - 1].transpose(1, 2, 0))
            @ np.conj(state[site]).reshape(state[site].shape[0], -1)
        )
        prod = prod.reshape(
            LR[site - 1].shape[-2],
            LR[site - 1].shape[-1],
            state[site].shape[1],
            state[site].shape[2],
        )
        LR[site] = prod.transpose(3, 2, 0, 1)

        mpo_i = mpo_views.site[site]

        LRt2 = LR[site].transpose(0, 3, 2, 1)
        prod2 = LRt2.reshape(-1, LRt2.shape[-2] * LRt2.shape[-1]) @ mpo_i["mat_A_01"]
        prod2 = prod2.reshape(
            LR[site].shape[0],
            LR[site].shape[-1],
            mpo_i["A"].shape[2],
            mpo_i["A"].shape[3],
        )
        LR[site] = prod2.transpose(0, 2, 3, 1)

        LRt3 = LR[site].transpose(0, 1, 3, 2)
        prod3 = (
            LRt3.reshape(-1, LRt3.shape[-2] * LRt3.shape[-1])
            @ state[site].reshape(-1, state[site].shape[-1])
        )
        LR[site] = prod3.reshape(
            LR[site].shape[0],
            LR[site].shape[1],
            state[site].shape[-1],
        )

    return C


def update_right_site(state, site, mpo_views, LR, *, split_mode="qr", cutoff=None, svd_abstol=0.0, svd_check_finite=True):
    """
    Right-to-left orthogonalization step.

    Produces
        state[site] <- right-isometric tensor Q
        returns C
    where
        old_tensor ≈ C @ Q
    and for SVD mode
        C = U S.
    """
    if site == get_N(state) - 1:
        C, Q = split_right_tensor_qr_or_svd(
            state[-1],
            split_mode=split_mode,
            cutoff=cutoff,
            abstol=svd_abstol,
            check_finite=svd_check_finite,
        )
        state[-1] = Q
    else:
        C, Q = split_right_tensor_qr_or_svd(
            state[site],
            split_mode=split_mode,
            cutoff=cutoff,
            abstol=svd_abstol,
            check_finite=svd_check_finite,
        )
        state[site] = Q

    if site == mpo_views.N - 1:
        mpo_last = mpo_views.site[-1]
        prod = np.conj(state[-1]) @ mpo_last["mat_T102_0"]
        LR[-1] = prod.reshape(
            prod.shape[0],
            mpo_last["A"].shape[0],
            mpo_last["A"].shape[2],
        )

        prod2 = flatten_left(LR[-1]) @ state[-1].T
        LR[-1] = prod2.reshape(
            LR[-1].shape[0],
            LR[-1].shape[1],
            state[-1].shape[0],
        )
    else:
        prod = (
            LR[site + 1].T.reshape(-1, LR[site + 1].shape[0])
            @ np.conj(state[site]).T.reshape(state[site].shape[-1], -1)
        )
        prod = prod.reshape(
            LR[site + 1].shape[-1],
            LR[site + 1].shape[-2],
            state[site].shape[1],
            state[site].shape[0],
        )
        LR[site] = prod.transpose(3, 2, 1, 0)

        mpo_i = mpo_views.site[site]

        prod2 = (
            _flatten_last2(LR[site].transpose(0, 3, 1, 2))
            @ mpo_i["mat_T1203_last2"]
        )
        prod2 = prod2.reshape(
            LR[site].shape[0],
            LR[site].shape[3],
            mpo_i["A"].shape[0],
            mpo_i["A"].shape[3],
        )
        LR[site] = prod2.transpose(0, 2, 3, 1)

        approx_T = state[site].transpose(1, 2, 0)
        prod3 = (
            LR[site].reshape(-1, LR[site].shape[-2] * LR[site].shape[-1])
            @ approx_T.reshape(-1, approx_T.shape[-1])
        )
        LR[site] = prod3.reshape(
            LR[site].shape[0],
            LR[site].shape[1],
            prod3.shape[1],
        )

    return C

def apply_center_left(C, state, site, mpo_views):
    if site == mpo_views.N - 2:
        state[-1] = C @ state[-1]
    else:
        prod = C @ state[site + 1].reshape(state[site + 1].shape[0], -1)
        state[site + 1] = prod.reshape(
            prod.shape[0],
            state[site + 1].shape[1],
            state[site + 1].shape[2],
        )


def apply_center_right(C, state, site):
    if site == 1:
        state[0] = state[0] @ C
    else:
        prod = state[site - 1].reshape(-1, state[site - 1].shape[-1]) @ C
        state[site - 1] = prod.reshape(
            state[site - 1].shape[0],
            state[site - 1].shape[1],
            C.shape[1],
        )


def oracle_key(site: int, dtype: np.dtype, left_sweep: bool) -> tuple:
    if dtype == np.float64:
        dt = "D"
    elif dtype == np.complex128:
        dt = "Z"
    else:
        dt = "U"
    direction = "R" if left_sweep else "L"
    return (site, dt, direction)


def get_oracle(ctx, site: int, dtype: np.dtype):
    if ctx.oracle_cache is None:
        ctx.oracle_cache = {}

    key = oracle_key(site, dtype, ctx.left_sweep)
    hit = ctx.oracle_cache.get(key, None)
    if hit is not None:
        return hit

    pf = load_oracle_module()

    if dtype == np.float64:
        oracle = pf.RightOracleD() if ctx.left_sweep else pf.LeftOracleD()
    elif dtype == np.complex128:
        oracle = pf.RightOracleZ() if ctx.left_sweep else pf.LeftOracleZ()
    else:
        raise TypeError("Oracle path supports float64 and complex128 only")

    ctx.oracle_cache[key] = oracle
    return oracle


def make_site_matvec(ctx, state, site):
    mpo_views = ctx.mpo_views
    LR = ctx.LR

    if site == 0:
        mpo0 = mpo_views.site[0]

        LR1_flatL = flatten_left(LR[1])
        mpo0_T120_mat = mpo0["mat_T120_last"]
        d_phys = mpo0["A"].shape[2]
        r_dim = LR[1].shape[2]
        L0, L1 = LR[1].shape[0], LR[1].shape[1]

        def matvec(psi):
            tmp = psi.reshape(d_phys, r_dim)
            prod = LR1_flatL @ tmp.T
            tmp2 = prod.reshape(L0, L1, d_phys)
            return (tmp2.reshape(L0, -1) @ mpo0_T120_mat).T.reshape(-1)

        return matvec

    if site == mpo_views.N - 1:
        mpo_last = mpo_views.site[-1]

        LRm2_flatL = flatten_left(LR[-2])
        mpo_last_T021_mat = mpo_last["mat_T021_last"]
        left_dim = LR[-2].shape[2]
        d_phys = mpo_last["A"].shape[2]
        L0, L1 = LR[-2].shape[0], LR[-2].shape[1]

        def matvec(psi):
            tmp = psi.reshape(left_dim, d_phys)
            prod = LRm2_flatL @ tmp
            tmp2 = prod.reshape(L0, L1, d_phys)
            return (tmp2.reshape(L0, -1) @ mpo_last_T021_mat).reshape(-1)

        return matvec

    pf = load_oracle_module()

    mpo_i = mpo_views.site[site]
    W = mpo_i["A"]
    if W.ndim != 4:
        raise ValueError("Interior site requires rank 4 MPO tensor")

    A = state[site]
    dtype = A.dtype

    is_real64 = dtype == np.float64
    is_cplx128 = dtype == np.complex128
    if (not is_real64) and (not is_cplx128):
        raise TypeError("Oracle path supports float64 and complex128 only")

    Lenv = _ascontig(LR[site - 1])
    Renv = _ascontig(LR[site + 1])
    Wc = _ascontig(W)

    oracle = get_oracle(ctx, site, dtype)

    if ctx.left_sweep:
        if is_real64:
            WR = pf.precontract_WRD(Wc, Renv)
            oracle.update(Lenv, WR)
        else:
            WR = pf.precontract_WRZ(Wc, Renv)
            oracle.update(Lenv, WR)
    else:
        if is_real64:
            LW = pf.precontract_LWD(Lenv, Wc)
            oracle.update(LW, Renv)
        else:
            LW = pf.precontract_LWZ(Lenv, Wc)
            oracle.update(LW, Renv)

    def matvec(v):
        return oracle(v)

    return matvec


def apply_site_heff(ctx, state, site, tensor=None):
    X = state[site] if tensor is None else tensor
    matvec = make_site_matvec(ctx, state, site)
    return matvec(np.asarray(X).reshape(-1)).reshape(np.asarray(X).shape)


def evolve_site(ctx, state, site, timestep):
    X = state[site]
    matvec = make_site_matvec(ctx, state, site)
    x0 = X.reshape(-1)
    work = get_lanczos_work(ctx, x0, ctx.krylov_depth)

    state[site] = lan_exp(
        x0,
        matvec,
        t=ctx.forward_phase * timestep,
        k=ctx.krylov_depth,
        work=work,
        force_full_depth=False,
    ).reshape(X.shape)


def make_center_matvec(ctx, C, bond):
    LR = ctx.LR

    LRk_flatL = flatten_left(LR[bond])
    LRkp1_t120_mat = flatten_left(LR[bond + 1].transpose(1, 2, 0))

    Cshape = C.shape
    L0, L1 = LR[bond].shape[0], LR[bond].shape[1]
    ncol = Cshape[-1]

    def matvec(Cvec):
        tmp = Cvec.reshape(Cshape)
        prod = LRk_flatL @ tmp
        tmp2 = prod.reshape(L0, L1, ncol)
        return (tmp2.reshape(L0, -1) @ LRkp1_t120_mat).reshape(-1)

    return matvec


def apply_center_heff(ctx, C, bond):
    matvec = make_center_matvec(ctx, C, bond)
    return matvec(np.asarray(C).reshape(-1)).reshape(np.asarray(C).shape)


def evolve_center(ctx, C, bond, timestep):
    matvec = make_center_matvec(ctx, C, bond)
    c0 = C.reshape(-1)
    work = get_lanczos_work(ctx, c0, ctx.krylov_depth)

    Ce = lan_exp(
        c0,
        matvec,
        t=ctx.center_phase * timestep / 2,
        k=ctx.krylov_depth,
        work=work,
        force_full_depth=False,
    )
    return Ce.reshape(C.shape)


def rayleigh_residual(X, HX):
    x = np.asarray(X).reshape(-1)
    hx = np.asarray(HX).reshape(-1)

    denom = np.vdot(x, x)
    denom_real = float(np.real(denom))
    if (not np.isfinite(denom_real)) or denom_real <= 0.0:
        return np.inf

    lam = np.vdot(x, hx) / denom
    norm_x = np.sqrt(denom_real)
    return float(np.linalg.norm(hx - lam * x) / norm_x)


def site_residual(ctx, state, site):
    HX = apply_site_heff(ctx, state, site, state[site])
    return rayleigh_residual(state[site], HX)


def center_residual(ctx, C, bond):
    HC = apply_center_heff(ctx, C, bond)
    return rayleigh_residual(C, HC)


def sweep_scheduler(n, sweeps):
    sequence = np.concatenate([np.arange(n), np.arange(n - 2, 0, -1)])
    repeated = np.tile(sequence, sweeps)
    repeated = np.append(repeated, 0)
    edge_flags = np.zeros(repeated.shape, dtype=int)
    edge_flags[0] = 1
    edge_flags[-1] = -1
    return zip(list(repeated), list(edge_flags))


def tdvp(
    mps,
    mpo,
    dt,
    sweeps,
    krylov_depth=10,
    verbose=False,
    time_type="real",
    renormalize=False,
    track=False,
    return_log_scale=False,
    normalize_working_state=True,
    early_stopping=False,
    stop_tol=1e-8,
    stop_patience=2,
    stop_min_sweeps=2,
    stop_mode="projective",
    return_info=False,
    split_mode="svd",
    cutoff=Cutoff(1e-14),
    svd_abstol=0.0,
    svd_check_finite=True,
):
  
    if get_N(mps) < 2:
        raise ValueError("This implementation requires an MPS with N >= 2")

    # Early stopping is intentionally disabled: always run all requested sweeps.
    del early_stopping
    del stop_tol
    del stop_patience
    del stop_min_sweeps
    del stop_mode
    split_mode = canonical_split_mode(split_mode)

    mpo_views = MPOViews(mpo)

    if time_type == "real":
        forward_phase = -1j
        center_phase = +1j
    elif time_type == "imag":
        forward_phase = -1.0
        center_phase = +1.0
    else:
        raise ValueError("time_type must be 'real' or 'imag'")

    ctx = TDVPContext(
        mpo_views=mpo_views,
        forward_phase=forward_phase,
        center_phase=center_phase,
        krylov_depth=int(krylov_depth),
        track=bool(track),
        LR=None,
        left_sweep=True,
        oracle_cache={},
        lanczos_work_cache={},
        split_mode=split_mode,
        cutoff=cutoff,
        svd_abstol=float(svd_abstol),
        svd_check_finite=bool(svd_check_finite),
    )

    if mps.orthform != "Left":
        mps.orthL()

    state = mps.copy()
    total_log_scale = 0.0

    if bool(normalize_working_state):
        _, dlog0 = rescale_to_unit_norm(
            state,
            track=track,
            tag="initial",
            return_log_scale=True,
            center_index=0,
        )
        total_log_scale += dlog0

    ctx.LR = compute_right_envs(state, mpo_views)
    ctx.left_sweep = True

    for site, edge_flag in sweep_scheduler(mps.N, sweeps):
        if verbose:
            print("-------site", site)
            state.show()

        track_norm(f"before evolve site {site}", state, track)

        if site == 0:
            step = dt / 2 if edge_flag else dt
            evolve_site(ctx, state, site, step)

            state[site], dlog = maybe_normalize_local_tensor(
                state[site],
                enabled=bool(normalize_working_state),
                tag=f"forward site {site}",
                track=track,
            )
            total_log_scale += dlog

            track_norm("after forward site 0", state, track)

            if edge_flag != -1:
                C = update_left_site(
                    state,
                    site,
                    mpo_views,
                    ctx.LR,
                    split_mode=ctx.split_mode,
                    cutoff=ctx.cutoff,
                    svd_abstol=ctx.svd_abstol,
                    svd_check_finite=ctx.svd_check_finite,
                )
                track_norm("after split and env update site 0", state, track)

                C = evolve_center(ctx, C, site, dt)

                C, dlog = maybe_normalize_local_tensor(
                    C,
                    enabled=bool(normalize_working_state),
                    tag=f"center bond {site}",
                    track=track,
                )
                total_log_scale += dlog

                if track:
                    print(
                        "track C norm2 site 0",
                        float(np.real(np.vdot(C.reshape(-1), C.reshape(-1)))),
                    )

                apply_center_left(C, state, site, mpo_views)
                track_norm("after C apply site 0", state, track)

            ctx.left_sweep = True

        elif 0 < site < mps.N - 1:
            evolve_site(ctx, state, site, dt / 2)

            state[site], dlog = maybe_normalize_local_tensor(
                state[site],
                enabled=bool(normalize_working_state),
                tag=f"forward site {site}",
                track=track,
            )
            total_log_scale += dlog

            track_norm(f"after forward site {site}", state, track)

            if ctx.left_sweep:
                C = update_left_site(
                    state,
                    site,
                    mpo_views,
                    ctx.LR,
                    split_mode=ctx.split_mode,
                    cutoff=ctx.cutoff,
                    svd_abstol=ctx.svd_abstol,
                    svd_check_finite=ctx.svd_check_finite,
                )
                track_norm(f"after split and env update site {site}", state, track)

                C = evolve_center(ctx, C, site, dt)

                C, dlog = maybe_normalize_local_tensor(
                    C,
                    enabled=bool(normalize_working_state),
                    tag=f"center bond {site}",
                    track=track,
                )
                total_log_scale += dlog

                if track:
                    print(
                        "track C norm2 site",
                        site,
                        float(np.real(np.vdot(C.reshape(-1), C.reshape(-1)))),
                    )

                apply_center_left(C, state, site, mpo_views)
                track_norm(f"after C apply site {site}", state, track)

            else:
                C = update_right_site(
                    state,
                    site,
                    mpo_views,
                    ctx.LR,
                    split_mode=ctx.split_mode,
                    cutoff=ctx.cutoff,
                    svd_abstol=ctx.svd_abstol,
                    svd_check_finite=ctx.svd_check_finite,
                )
                track_norm(f"after split and env update site {site}", state, track)

                C = evolve_center(ctx, C, site - 1, dt)

                C, dlog = maybe_normalize_local_tensor(
                    C,
                    enabled=bool(normalize_working_state),
                    tag=f"center bond {site - 1}",
                    track=track,
                )
                total_log_scale += dlog

                if track:
                    print(
                        "track C norm2 site",
                        site - 1,
                        float(np.real(np.vdot(C.reshape(-1), C.reshape(-1)))),
                    )

                apply_center_right(C, state, site)
                track_norm(f"after C apply site {site}", state, track)

        else:
            evolve_site(ctx, state, site, dt)

            state[site], dlog = maybe_normalize_local_tensor(
                state[site],
                enabled=bool(normalize_working_state),
                tag="forward last site",
                track=track,
            )
            total_log_scale += dlog

            track_norm("after forward last site", state, track)

            C = update_right_site(
                state,
                site,
                mpo_views,
                ctx.LR,
                split_mode=ctx.split_mode,
                cutoff=ctx.cutoff,
                svd_abstol=ctx.svd_abstol,
                svd_check_finite=ctx.svd_check_finite,
            )
            track_norm("after split and env update last site", state, track)

            C = evolve_center(ctx, C, site - 1, dt)

            C, dlog = maybe_normalize_local_tensor(
                C,
                enabled=bool(normalize_working_state),
                tag=f"center bond {site - 1}",
                track=track,
            )
            total_log_scale += dlog

            if track:
                print(
                    "track C norm2 site",
                    site - 1,
                    float(np.real(np.vdot(C.reshape(-1), C.reshape(-1)))),
                )

            apply_center_right(C, state, site)
            track_norm("after C apply last site", state, track)

            ctx.left_sweep = False

    state.orthform = "Left"

    if bool(normalize_working_state):
        _, dlogf = rescale_to_unit_norm(
            state,
            track=track,
            tag="final",
            return_log_scale=True,
            center_index=0,
        )
        total_log_scale += dlogf

    track_norm("final canonical", state, track)

    if return_info:
        info = {
            "requested_sweeps": int(sweeps),
            "sweeps_completed": int(sweeps),
            "stopped_early": False,
            "stop_reason": None,
            "total_log_scale": float(total_log_scale),
            "returned_canonical_state": bool(
                bool(normalize_working_state) and (renormalize or return_log_scale)
            ),
            "normalize_working_state": bool(normalize_working_state),
            "split_mode": str(ctx.split_mode),
            "cutoff": ctx.cutoff,
            "svd_abstol": float(ctx.svd_abstol),
            "svd_check_finite": bool(ctx.svd_check_finite),
            "residual_history": [],
            "final_metric": None,
            "final_residual": None,
            "final_projective": None,
            "final_site_rms": None,
            "final_site_max": None,
        }

        if return_log_scale:
            return state, float(total_log_scale), info

        if renormalize:
            return state, info

        return _apply_log_scale_to_state(state, total_log_scale), info

    if return_log_scale:
        return state, float(total_log_scale)

    if renormalize:
        return state

    return _apply_log_scale_to_state(state, total_log_scale)

tdvp_implicit = tdvp
tdvp_implicit = tdvp
