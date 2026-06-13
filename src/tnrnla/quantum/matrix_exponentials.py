import numpy as np
import itertools
import math

from tnrnla.tn.mpo import MPO



def classical_ising_exp(n, beta, pauli="Z"):
    if n < 2:
        raise ValueError("n must be at least 2")
    P = Z if pauli == "Z" else X
    ch = np.cosh(beta)
    sh = np.sinh(beta)
    a = np.sqrt(ch)
    b = np.sqrt(sh)
    c = np.sqrt(ch * sh)
    d = 2
    D = 2
    tensors = []
    L = np.zeros((d, D, d), dtype=float)
    L[:, 0, :] = a *  np.eye(2)
    L[:, 1, :] = b * P
    tensors.append(L)
    for _ in range(1, n - 1):
        M = np.zeros((D, d, D, d), dtype=float)
        M[0, :, 0, :] = ch *  np.eye(2)
        M[0, :, 1, :] = c * P
        M[1, :, 0, :] = c * P
        M[1, :, 1, :] = sh * np.eye(2)
        tensors.append(M)
    R = np.zeros((D, d, d), dtype=float)
    R[0, :, :] = a * I
    R[1, :, :] = b * P
    tensors.append(R)

    return MPO(tensors)


def ptfim_zz_x_matrix_exponential_mpo(
    n_sites,
    dt,
    J=1.0,
    h=1.0,
    order=5,
    split_order=2,
    dtype=np.float64,
):
    """
    Build an MPO product approximation to exp(-dt H) for the periodic TFIM

        H = J * sum_i Z_i Z_{i+1} - h * sum_i X_i

    with periodic boundary conditions, using
    - an FSM/Taylor bulk construction for the open-chain part
    - an exact MPO for the periodic wrap term J Z_1 Z_N
    - Suzuki recursive composition of Strang splitting

    Returns
    -------
    tuple[MPO, ...]
        Ordered MPO factors whose product approximates exp(-dt H).
    """
    if n_sites < 2:
        raise ValueError("n_sites must be >= 2")
    if split_order % 2 != 0 or split_order < 2:
        raise ValueError("split_order must be an even integer >= 2")
    if order < 1:
        raise ValueError("order must be >= 1")

    I2 = np.eye(2, dtype=dtype)
    X = np.array([[0.0, 1.0],
                  [1.0, 0.0]], dtype=dtype)
    Z = np.array([[1.0,  0.0],
                  [0.0, -1.0]], dtype=dtype)

    def identity_mpo():
        return MPO.eye(n_sites, d=2, dtype=dtype)

    def tfim_open_graph():
        # Encodes bulk operator
        #
        #   H_bulk = J * sum_{i=1}^{N-1} Z_i Z_{i+1} - h * sum_{i=1}^N X_i
        #
        return {
            (0, 0): I2,
            (0, 1): float(J) * Z,
            (1, 2): Z,
            (0, 2): -float(h) * X,
            (2, 2): I2,
        }

    def mpo_from_fsm(graph, k, start_level):
        d = next(iter(graph.values())).shape[0]

        A = np.zeros((k, d, k, d), dtype=dtype)
        for (i, j), op in graph.items():
            A[i, :, j, :] = np.asarray(op, dtype=dtype)

        W0 = np.zeros((d, k, d), dtype=dtype)
        for beta in range(k):
            W0[:, beta, :] = A[start_level, :, beta, :]

        Wmid = A.copy()

        WN = np.zeros((k, d, d), dtype=dtype)
        for alpha in range(k):
            WN[alpha, :, :] = A[alpha, :, start_level, :]

        if n_sites == 1:
            return MPO([W0], dtype=dtype)
        if n_sites == 2:
            return MPO([W0, WN], dtype=dtype)
        return MPO([W0] + [Wmid.copy() for _ in range(n_sites - 2)] + [WN], dtype=dtype)

    def wrap_zz_mpo(coeff=1.0):
        D = 2
        tensors = []

        W0 = np.zeros((2, D, 2), dtype=dtype)
        W0[:, 0, :] = I2
        W0[:, 1, :] = coeff * Z
        tensors.append(W0)

        for _ in range(n_sites - 2):
            W = np.zeros((D, 2, D, 2), dtype=dtype)
            W[0, :, 0, :] = I2
            W[1, :, 1, :] = I2
            tensors.append(W)

        WN = np.zeros((D, 2, 2), dtype=dtype)
        WN[1, :, :] = Z
        tensors.append(WN)

        return MPO(tensors, dtype=dtype)

    def exp_wrap(t):
        alpha = float(t) * float(J)
        return np.cosh(alpha) * identity_mpo() - np.sinh(alpha) * wrap_zz_mpo(1.0)

    def fsm_power_graph(base_graph, k, power):
        edges_out = {i: [] for i in range(k)}
        for (i, j), op in base_graph.items():
            edges_out[i].append((j, op))

        states = list(np.ndindex(*([k] * power)))
        idx = {s: n for n, s in enumerate(states)}
        graph_p = {}

        for s in states:
            out_lists = [edges_out[s[m]] for m in range(power)]
            for choices in itertools.product(*out_lists):
                t = tuple(ch[0] for ch in choices)
                op = choices[0][1]
                for m in range(1, power):
                    op = op @ choices[m][1]
                key = (idx[s], idx[t])
                graph_p[key] = graph_p.get(key, 0.0) + op

        return graph_p, idx

    def foldback_graph(graph_p, idx, tau, start_state=0, done_state=2):
        start_tuple = tuple([start_state] * order)
        start_idx = idx[start_tuple]
        alive = set(idx.values())
        tau = float(tau)

        for a in range(1, order + 1):
            w = (tau ** a) * math.factorial(order - a) / math.factorial(order)

            for done_pos in itertools.combinations(range(order), a):
                b = [start_state] * order
                for p in done_pos:
                    b[p] = done_state
                b = tuple(b)
                b_idx = idx.get(b, None)

                if b_idx is None or b_idx not in alive:
                    continue

                incoming = [(u, op) for (u, v), op in graph_p.items() if v == b_idx and u in alive]
                for u, op in incoming:
                    key = (u, start_idx)
                    graph_p[key] = graph_p.get(key, 0.0) + w * op

                to_delete = [(u, v) for (u, v) in list(graph_p.keys()) if u == b_idx or v == b_idx]
                for key in to_delete:
                    graph_p.pop(key, None)

                alive.remove(b_idx)

        return graph_p, alive, start_idx

    def exp_bulk(t):
        base_graph = tfim_open_graph()
        graph_p, idx = fsm_power_graph(base_graph, k=3, power=order)
        graph_p, alive, start_idx = foldback_graph(graph_p, idx, tau=t)

        remap = {old: new for new, old in enumerate(sorted(alive))}
        graph_c = {}
        for (u, v), op in graph_p.items():
            if u in alive and v in alive:
                uu = remap[u]
                vv = remap[v]
                graph_c[(uu, vv)] = graph_c.get((uu, vv), 0.0) + op

        return mpo_from_fsm(
            graph=graph_c,
            k=len(alive),
            start_level=remap[start_idx],
        )

    def suzuki_comp_coeff(p):
        x = 1.0 / (2.0 - 2.0 ** (1.0 / (2.0 * p + 1.0)))
        y = -(2.0 ** (1.0 / (2.0 * p + 1.0))) * x
        return x, y

    def build_step_sizes(p, t):
        if p == 1:
            return [t]
        x, y = suzuki_comp_coeff(p - 1)
        sx = build_step_sizes(p - 1, x * t)
        sy = build_step_sizes(p - 1, y * t)
        return sx + sy + sx

    def strang_triplet(t):
        return (0.5 * t, t, 0.5 * t)

    p = split_order // 2
    step_sizes = build_step_sizes(p, float(dt))
    schedule = [strang_triplet(t) for t in step_sizes]

    parts = []
    for t_wrap_l, t_bulk, t_wrap_r in schedule:
        parts.append(exp_wrap(t_wrap_l))
        parts.append(exp_bulk(t_bulk))
        parts.append(exp_wrap(t_wrap_r))

    return tuple(parts)