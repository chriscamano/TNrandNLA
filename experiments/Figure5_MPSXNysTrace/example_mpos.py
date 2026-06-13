import numpy as np

from tnrnla import Cutoff, MPO


BENCHMARK_KINDS = (
    "exp",
    "step",
    "inverse_laplacian",
    "staircase",
)

BENCHMARK_DISPLAY_NAMES = {
    "exp": r"$\mathtt{exp}$",
    "step": r"$\mathtt{step}$",
    "inverse_laplacian": r"$\mathtt{inverse\_laplacian}$",
    "staircase": r"$\mathtt{staircase}$",
}


# ============================================================
# Diagonal benchmark helpers
# ============================================================


def _digits_base_d(x, N, d):
    out = [0] * N
    for k in range(N - 1, -1, -1):
        out[k] = x % d
        x //= d
    return out


def _automaton_to_diagonal_mpo(local_mats, left_vec, right_vec, dtype=float):
    """
    Convert a finite-state automaton representation of a diagonal operator
    into an MPO.

    local_mats[k][s] is the transition matrix used at site k for local state s.
    """
    N = len(local_mats)
    d = len(local_mats[0])

    if N == 1:
        M0 = local_mats[0]
        val = np.zeros((d, d), dtype=dtype)
        for s in range(d):
            val[s, s] = left_vec @ M0[s] @ right_vec
        return MPO([val], dtype=dtype)

    mpo = []

    M0 = local_mats[0]
    r1 = M0[0].shape[1]
    G0 = np.zeros((d, r1, d), dtype=dtype)
    for s in range(d):
        G0[s, :, s] = left_vec @ M0[s]
    mpo.append(G0)

    for k in range(1, N - 1):
        Mk = local_mats[k]
        rl, rr = Mk[0].shape
        Gk = np.zeros((rl, d, rr, d), dtype=dtype)
        for s in range(d):
            Gk[:, s, :, s] = Mk[s]
        mpo.append(Gk)

    MN = local_mats[-1]
    rl = MN[0].shape[0]
    GN = np.zeros((rl, d, d), dtype=dtype)
    for s in range(d):
        GN[:, s, s] = MN[s] @ right_vec
    mpo.append(GN)

    return MPO(mpo, dtype=dtype)


def _local_exp_automaton(N, d, alpha, dtype=float):
    local_mats = []
    for k in range(N):
        weight = d ** (N - 1 - k)
        Mk = []
        for s in range(d):
            Mk.append(np.array([[alpha ** (s * weight)]], dtype=dtype))
        local_mats.append(Mk)
    left = np.array([1.0], dtype=dtype)
    right = np.array([1.0], dtype=dtype)
    return local_mats, left, right


# ============================================================
# Exact diagonal PSD benchmarks
# ============================================================


def synthetic_exp_mpo(
    N=100,
    d=2,
    dtype=np.float64,
    alpha=0.8,
):
    """
    Exact diagonal exponential spectrum

        lambda_i = alpha^i.
    """
    local_mats, left, right = _local_exp_automaton(
        N=N,
        d=d,
        alpha=alpha,
        dtype=dtype,
    )
    return _automaton_to_diagonal_mpo(local_mats, left, right, dtype=dtype)


def _constant_diagonal_mpo(N, d, value, dtype=float):
    local_mats = []
    for _ in range(N):
        local_mats.append([np.array([[1.0]], dtype=dtype) for _ in range(d)])
    left = np.array([1.0], dtype=dtype)
    right = np.array([float(value)], dtype=dtype)
    return _automaton_to_diagonal_mpo(local_mats, left, right, dtype=dtype)


def synthetic_staircase_mpo(
    N=100,
    d=2,
    dtype=np.float64,
    breakpoints=(64, 128, 256, 512),
    values=(1.0, 0.1, 0.03, 0.01, 0.0),
):
    """
    Exact diagonal PSD staircase spectrum.

    With breakpoints (K1, ..., Km) and values (v0, ..., vm), this builds

        lambda_i = v0   for 0 <= i < K1
                 = v1   for K1 <= i < K2
                 = ...
                 = vm   for Km <= i < d**N

    for i = 0, ..., d**N - 1.

    The values should be nonnegative for PSD. For a nonincreasing staircase,
    also take
        v0 >= v1 >= ... >= vm >= 0.
    """
    n = d ** N

    breakpoints = tuple(int(k) for k in breakpoints)
    values = tuple(float(v) for v in values)

    if len(values) != len(breakpoints) + 1:
        raise ValueError("values must have length len(breakpoints) + 1")

    if any(v < 0 for v in values):
        raise ValueError("all values must be nonnegative for PSD")

    if any(k <= 0 or k >= n for k in breakpoints):
        raise ValueError("breakpoints must lie strictly between 0 and d**N")

    if any(breakpoints[j] >= breakpoints[j + 1] for j in range(len(breakpoints) - 1)):
        raise ValueError("breakpoints must be strictly increasing")

    A = _constant_diagonal_mpo(N=N, d=d, value=values[-1], dtype=dtype)

    for K, dv in zip(breakpoints, np.diff(values)):
        coeff = -float(dv)  # coeff = values[j-1] - values[j]
        if coeff == 0.0:
            continue

        local_mats, left, right = _local_step_automaton_rank2(
            N=N,
            d=d,
            K=K,
            lo=0.0,
            dtype=dtype,
        )
        Step = _automaton_to_diagonal_mpo(local_mats, left, coeff * right, dtype=dtype)
        A = A + Step

    return A


def synthetic_staircase_mpo_auto(
    N=100,
    d=2,
    dtype=np.float64,
    first_block=32,
    num_levels=3,
    block_growth=1.7,
    level_decay=0.15,
    head_val=1.0,
    tail_val=0.0,
    jitter=0.0,
    seed=0,
    return_spec=False,
):
    """
    Exact diagonal PSD staircase spectrum with automatically generated
    breakpoints and values.

    The first finite block has length `first_block`. Subsequent block lengths
    grow approximately geometrically with factor `block_growth`. The block
    values decay geometrically with factor `level_decay`.

    Spectrum shape

        lambda_i = v0 on the first block
                 = v1 on the second block
                 = ...
                 = vm on the remaining tail

    where
        vj = head_val * level_decay**j,   j = 0, ..., num_levels-1
    and the final infinite tail value is `tail_val`.

    If return_spec=True, also return the generated breakpoints and values.
    """
    n = d ** N

    if first_block <= 0:
        raise ValueError("first_block must be positive")
    if num_levels < 1:
        raise ValueError("num_levels must be at least 1")
    if block_growth <= 1.0:
        raise ValueError("block_growth must exceed 1")
    if not (0.0 <= level_decay <= 1.0):
        raise ValueError("level_decay must lie in [0, 1]")
    if head_val < 0 or tail_val < 0:
        raise ValueError("head_val and tail_val must be nonnegative for PSD")
    if jitter < 0.0:
        raise ValueError("jitter must be nonnegative")

    rng = np.random.default_rng(seed)

    lengths = []
    running = 0

    for j in range(num_levels):
        Lj = int(round(first_block * (block_growth ** j)))

        if jitter > 0.0:
            delta = int(round(jitter * Lj))
            if delta > 0:
                Lj += int(rng.integers(-delta, delta + 1))

        Lj = max(Lj, 1)

        # keep cumulative support strictly below n
        remaining_levels = num_levels - 1 - j
        max_allowed = (n - 1) - running - remaining_levels
        if max_allowed <= 0:
            break

        Lj = min(Lj, max_allowed)
        lengths.append(Lj)
        running += Lj

        if running >= n - 1:
            break

    if not lengths:
        raise ValueError("automatic block construction produced no finite blocks")

    breakpoints = tuple(int(x) for x in np.cumsum(lengths, dtype=np.int64))
    values = tuple(
        [float(head_val * (level_decay ** j)) for j in range(len(lengths))]
        + [float(tail_val)]
    )

    mpo = synthetic_staircase_mpo(
        N=N,
        d=d,
        dtype=dtype,
        breakpoints=breakpoints,
        values=values,
    )

    if return_spec:
        return mpo, breakpoints, values

    return mpo


def _local_step_automaton_rank2(
    N,
    d,
    K,
    lo,
    dtype=float,
):
    """
    Exact rank-2 automaton for the diagonal step spectrum

        value(i) = 1.0,  if i < K
                 = lo,   if i >= K

    where i is the lexicographic index in base d.
    """
    n = d ** N
    K = int(min(max(K, 0), n))
    lo = float(lo)

    if lo < 0:
        raise ValueError("lo must be nonnegative for PSD")

    if K == 0:
        local_mats = []
        for _ in range(N):
            local_mats.append([np.array([[1.0]], dtype=dtype) for _ in range(d)])
        left = np.array([1.0], dtype=dtype)
        right = np.array([lo], dtype=dtype)
        return local_mats, left, right

    if K == n:
        local_mats = []
        for _ in range(N):
            local_mats.append([np.array([[1.0]], dtype=dtype) for _ in range(d)])
        left = np.array([1.0], dtype=dtype)
        right = np.array([1.0], dtype=dtype)
        return local_mats, left, right

    thresh = _digits_base_d(K - 1, N, d)
    delta = 1.0 - lo
    local_mats = []

    for k in range(N):
        c = thresh[k]
        Mk = []
        for s in range(d):
            T = np.zeros((2, 2), dtype=dtype)

            # State is a row vector [const_term, eq_flag].
            # Update rule
            #   const' = const + eq_flag * delta * 1{s < c}
            #   eq'    = eq_flag * 1{s = c}
            T[0, 0] = 1.0
            T[1, 0] = delta if s < c else 0.0
            T[1, 1] = 1.0 if s == c else 0.0

            Mk.append(T)
        local_mats.append(Mk)

    left = np.array([lo, 1.0], dtype=dtype)
    right = np.array([1.0, delta], dtype=dtype)
    return local_mats, left, right


def synthetic_step_mpo(
    N=100,
    d=2,
    dtype=np.float64,
    rank=64,
    tail_val=0.0,
):
    """
    Exact diagonal PSD spectrum with

        lambda_i = 1.0,      i < rank
        lambda_i = tail_val, i >= rank

    for i = 0, ..., d**N - 1.

    For 0 < rank < d**N and tail_val != 1, this construction has exact MPO
    bond dimension 2.
    """
    n = d ** N
    rank = int(min(max(rank, 0), n))
    tail_val = float(tail_val)

    if tail_val < 0:
        raise ValueError("tail_val must be nonnegative for PSD")

    local_mats, left, right = _local_step_automaton_rank2(
        N=N,
        d=d,
        K=rank,
        lo=tail_val,
        dtype=dtype,
    )

    return _automaton_to_diagonal_mpo(local_mats, left, right, dtype=dtype)


# ============================================================
# Exact inverse Dirichlet Laplacian MPO
# ============================================================


def dense_dirichlet_laplacian(N, dtype=np.float64):
    n = 2 ** N
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        A[i, i] = 2.0
        if i > 0:
            A[i, i - 1] = -1.0
        if i + 1 < n:
            A[i, i + 1] = -1.0
    return A


def dirichlet_laplacian_eigenvalues(N, dtype=np.float64):
    n = 2 ** N
    k = np.arange(1, n + 1, dtype=np.float64)
    lam = 2.0 - 2.0 * np.cos(np.pi * k / (n + 1))
    return lam.astype(dtype, copy=False)


def dirichlet_inverse_laplacian_eigenvalues(N, dtype=np.float64):
    lam = dirichlet_laplacian_eigenvalues(N, dtype=np.float64)
    return (1.0 / lam).astype(dtype, copy=False)


def _row_blocks_to_first_core(blocks, dtype):
    r = len(blocks)
    d = blocks[0].shape[0]
    G = np.zeros((d, r, d), dtype=dtype)
    for j, B in enumerate(blocks):
        G[:, j, :] = B
    return G


def _block_matrix_to_middle_core(block_mat, dtype):
    rl = len(block_mat)
    rr = len(block_mat[0])
    d = block_mat[0][0].shape[0]
    G = np.zeros((rl, d, rr, d), dtype=dtype)
    for i in range(rl):
        for j in range(rr):
            G[i, :, j, :] = block_mat[i][j]
    return G


def _col_blocks_to_last_core(blocks, dtype):
    rl = len(blocks)
    d = blocks[0].shape[0]
    G = np.zeros((rl, d, d), dtype=dtype)
    for i, B in enumerate(blocks):
        G[i, :, :] = B
    return G


def synthetic_dirichlet_inverse_laplacian_mpo(
    N,
    *,
    d=2,
    dtype=np.float64,
):
    """
    Exact analytic rank-5 QTT/MPO for the inverse 1D Dirichlet Laplacian.
    """
    if d != 2:
        raise ValueError("synthetic_dirichlet_inverse_laplacian_mpo assumes d=2")
    if N < 1:
        raise ValueError("N must be at least 1")

    I = np.array([[1.0, 0.0],
                  [0.0, 1.0]], dtype=dtype)
    P = np.array([[0.0, 1.0],
                  [1.0, 0.0]], dtype=dtype)
    E = np.array([[1.0, 1.0],
                  [1.0, 1.0]], dtype=dtype)
    F = np.array([[1.0, -1.0],
                  [-1.0, 1.0]], dtype=dtype)
    K = np.array([[-1.0, 0.0],
                  [0.0, 1.0]], dtype=dtype)
    L = np.array([[0.0, -1.0],
                  [1.0, 0.0]], dtype=dtype)
    Z = np.zeros((2, 2), dtype=dtype)

    if N == 1:
        return MPO([(1.0 / 3.0) * (I + E)], dtype=dtype)

    def xi(k):
        return (2 ** (k - 1) + 1.0) / (2 ** k + 1.0)

    def eta(k):
        return (2 ** (k - 2)) / (2 ** k + 1.0)

    def zeta(k):
        return ((2 ** (k - 1) + 1.0) / (2 ** (k - 1))) * xi(k)

    def Gamma(k):
        xk = xi(k)
        zk = zeta(k)
        return [
            0.25 * xk * I + 0.25 * zk * P,
            xk * I - zk * P,
            -0.5 * xk * K,
            0.5 * zk * L,
        ]

    def V(k):
        xk = xi(k)
        ek = eta(k)
        return [
            [2.0 * E,                 Z,                 Z,                 Z],
            [2.0 * (ek ** 2) * F,     2.0 * (xk ** 2) * E,  2.0 * xk * ek * K,  2.0 * xk * ek * L],
            [4.0 * ek * K,            Z,                 2.0 * xk * E,      Z],
            [-4.0 * ek * L,           Z,                 Z,                 2.0 * xk * E],
        ]

    cores = []

    first_blocks = [I] + Gamma(N)
    cores.append(_row_blocks_to_first_core(first_blocks, dtype=dtype))

    for k in range(N - 1, 1, -1):
        Gk = [[None for _ in range(5)] for _ in range(5)]

        Gk[0][0] = I
        gk = Gamma(k)
        for j in range(4):
            Gk[0][j + 1] = gk[j]

        for i in range(1, 5):
            Gk[i][0] = Z

        Vk = V(k)
        for i in range(4):
            for j in range(4):
                Gk[i + 1][j + 1] = Vk[i][j]

        cores.append(_block_matrix_to_middle_core(Gk, dtype=dtype))

    W1_blocks = [
        (1.0 / 3.0) * (I + E),
        2.0 * E,
        (1.0 / 18.0) * F,
        (2.0 / 3.0) * K,
        (-2.0 / 3.0) * L,
    ]
    cores.append(_col_blocks_to_last_core(W1_blocks, dtype=dtype))

    return MPO(cores, dtype=dtype)


# ============================================================
# Public builders
# ============================================================


def synthetic_benchmark_mpo(
    kind="exp",
    N=100,
    d=2,
    seed=0,
    dtype=np.float64,
):
    if kind == "exp":
        return synthetic_exp_mpo(
            N=N,
            dtype=dtype,
            alpha=0.7,
        )

    if kind == "step":
        return synthetic_step_mpo(
            N=N,
            dtype=np.float64,
            rank=20,
            tail_val=1e-3,
        )

    if kind == "inverse_laplacian":
        return synthetic_dirichlet_inverse_laplacian_mpo(
            N=N,
            dtype=dtype,
        )

    if kind == "staircase":
        return synthetic_staircase_mpo_auto(
            N=N,
            d=d,
            dtype=dtype,
            seed=seed,
        )

    raise ValueError(f"unknown benchmark kind {kind!r}")


def build_mpo(kind, N, d, seed=0, dtype=np.float64):
    if kind not in BENCHMARK_KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {BENCHMARK_KINDS}")

    mpo = synthetic_benchmark_mpo(
        kind=kind,
        N=N,
        d=d,
        seed=seed,
        dtype=dtype,
    )
    return mpo, float(np.real(mpo.trace()))