import numpy as np

from tnrnla.linalg.utils import supfind
from .randomvector import rvec


def _normalize(v, eps=1e-30):
    nrm = float(np.linalg.norm(v))
    if nrm <= float(eps):
        return None
    return v / nrm


def _orth_cols(X):
    Q, _ = np.linalg.qr(X, mode="reduced")
    return Q


def _sample_matrix(n, k, *, vec_type, rng):
    X = rvec(int(n), mode=vec_type, seed=rng).sample(int(k))
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    return X


def adap_hpp(
    matrix_or_size,
    matvec_oracle=None,
    epsilon=None,
    delta=None,
    *,
    vec_type="gaussian",
    seed=None,
):
    """Adaptive Hutch++ variant matching the provided MATLAB logic."""
    # Old style: adap_hpp(n, Afun, epsilon, delta)
    # Matrix style: adap_hpp(A, epsilon, delta)
    if callable(matvec_oracle):
        n = int(matrix_or_size)
        Afun = matvec_oracle
        eps = float(epsilon)
        delta = float(delta)
    else:
        A = np.asarray(matrix_or_size)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("matrix input must be square")
        n = int(A.shape[0])
        Afun = lambda X: A @ X
        if delta is None:
            if matvec_oracle is None or epsilon is None:
                raise ValueError("for matrix input use adap_hpp(A, epsilon, delta)")
            eps = float(matvec_oracle)
            delta = float(epsilon)
        else:
            eps = float(epsilon)
            delta = float(delta)

    if n < 1:
        raise ValueError("matrix_size must be >= 1")
    if eps <= 0.0:
        raise ValueError("epsilon must be > 0")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")

    rng = np.random.default_rng(seed)
    C = 4.0 * np.log(2.0 / delta) / (eps * eps)
    fnc = []

    y = Afun(_sample_matrix(n, 1, vec_type=vec_type, rng=rng))
    q = _normalize(np.asarray(y).reshape(-1))
    if q is None:
        return 0.0, 0, 0, 0
    Q = q[:, None]

    x = np.asarray(Afun(Q)).reshape(-1)
    t = np.vdot(q, x)
    c = t * t
    trest1 = t
    iteration = 1
    b = np.vdot(x, x)
    fnc.append(2.0 * iteration + C * (c - 2.0 * b))

    while True:
        y = Afun(_sample_matrix(n, 1, vec_type=vec_type, rng=rng))
        y = np.asarray(y).reshape(-1)
        qt = y - Q @ (Q.T.conj() @ y)

        if float(np.linalg.norm(qt)) < 1e-10:
            lowrank_matvecs = int(2 * iteration)
            trest_matvecs = 0
            total_matvecs = int(lowrank_matvecs + trest_matvecs)
            return float(np.real(trest1)), total_matvecs, lowrank_matvecs, trest_matvecs

        qt = _normalize(qt)
        qt = qt - Q @ (Q.T.conj() @ qt)
        q = _normalize(qt)
        if q is None:
            lowrank_matvecs = int(2 * iteration)
            trest_matvecs = 0
            total_matvecs = int(lowrank_matvecs + trest_matvecs)
            return float(np.real(trest1)), total_matvecs, lowrank_matvecs, trest_matvecs

        Q_old = Q
        Q = np.column_stack((Q, q))
        x = np.asarray(Afun(q[:, None])).reshape(-1)

        b = b + np.vdot(x, x)
        t = np.vdot(q, x)
        trest1 = trest1 + t
        c = c + 2.0 * np.linalg.norm(Q_old.T.conj() @ x) ** 2 + t * t

        iteration += 1
        fnc.append(2.0 * iteration + C * (c - 2.0 * b))

        if iteration > 2 and fnc[-2] < fnc[-1] and fnc[-3] < fnc[-2]:
            break

    lowrank_matvecs = int(2 * iteration)

    it2 = 0
    t_acc = 0.0
    trest2_vals = []

    while True:
        psi = _sample_matrix(n, 1, vec_type=vec_type, rng=rng).reshape(-1)
        y = psi - Q @ (Q.T.conj() @ psi)
        y = np.asarray(Afun(y[:, None])).reshape(-1)
        y = y - Q @ (Q.T.conj() @ y)

        trest2_vals.append(np.vdot(psi, y))
        t_acc = t_acc + np.vdot(y, y)
        est_frob = t_acc / float(it2 + 1)
        alpha = max(supfind(it2 + 1, delta), 1e-12)
        M = int(np.ceil(C * np.real(est_frob) / alpha))

        if (it2 + 1) > M:
            break
        it2 += 1

    trest2 = np.mean(trest2_vals) if trest2_vals else 0.0
    trest_matvecs = int(it2)
    trest = trest1 + trest2
    total_matvecs = int(lowrank_matvecs + trest_matvecs)

    return float(np.real(trest)), total_matvecs, lowrank_matvecs, trest_matvecs


def block_adap_hpp(
    matrix_or_size,
    matvec_oracle=None,
    epsilon=None,
    delta=None,
    block_size=None,
    *,
    vec_type="gaussian",
    seed=None,
):
    """Block adaptive Hutch++ variant matching the provided MATLAB logic."""
    # Old style: block_adap_hpp(n, Afun, epsilon, delta, block_size)
    # Matrix style: block_adap_hpp(A, epsilon, delta, block_size)
    if callable(matvec_oracle):
        n = int(matrix_or_size)
        Afun = matvec_oracle
        eps = float(epsilon)
        delta_prob = float(delta)
        bsz = int(block_size)
    else:
        A = np.asarray(matrix_or_size)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("matrix input must be square")
        n = int(A.shape[0])
        Afun = lambda X: A @ X
        if block_size is None:
            # Matrix positional style maps as (A, epsilon, delta, block_size)
            if matvec_oracle is None or epsilon is None or delta is None:
                raise ValueError("for matrix input use block_adap_hpp(A, epsilon, delta, block_size)")
            eps = float(matvec_oracle)
            delta_prob = float(epsilon)
            bsz = int(delta)
        else:
            eps = float(epsilon)
            delta_prob = float(delta)
            bsz = int(block_size)

    if n < 1:
        raise ValueError("matrix_size must be >= 1")
    if eps <= 0.0:
        raise ValueError("epsilon must be > 0")
    if not (0.0 < delta_prob < 1.0):
        raise ValueError("delta must be in (0,1)")
    if bsz < 1:
        raise ValueError("block_size must be >= 1")

    rng = np.random.default_rng(seed)
    C = 4.0 * np.log(2.0 / delta_prob) / (eps * eps)
    fnc = []

    AOmega = np.asarray(Afun(_sample_matrix(n, bsz, vec_type=vec_type, rng=rng)))
    Q = _orth_cols(AOmega)

    AQ = np.asarray(Afun(Q))
    QtAQ = Q.T.conj() @ AQ
    trest1 = np.trace(QtAQ)
    c = np.linalg.norm(QtAQ, ord="fro") ** 2
    iteration = int(Q.shape[1])
    b = np.linalg.norm(AQ, ord="fro") ** 2
    fnc.append(2.0 * iteration + C * (c - 2.0 * b))

    while True:
        AOmega = np.asarray(Afun(_sample_matrix(n, bsz, vec_type=vec_type, rng=rng)))
        qt = AOmega - Q @ (Q.T.conj() @ AOmega)

        if float(np.linalg.norm(qt, ord="fro") / np.sqrt(float(bsz))) < 1e-10:
            lowrank_matvecs = int(2 * iteration)
            trest_matvecs = 0
            total_matvecs = int(lowrank_matvecs + trest_matvecs)
            return float(np.real(trest1)), total_matvecs, lowrank_matvecs, trest_matvecs

        qt = _orth_cols(qt)
        qt = qt - Q @ (Q.T.conj() @ qt)
        q = _orth_cols(qt)

        Q_old = Q
        Q = np.column_stack((Q, q))
        Aq = np.asarray(Afun(q))

        b = b + np.linalg.norm(Aq, ord="fro") ** 2
        qtAq = q.T.conj() @ Aq
        trest1 = trest1 + np.trace(qtAq)
        c = c + 2.0 * np.linalg.norm(Q_old.T.conj() @ Aq, ord="fro") ** 2 + np.linalg.norm(qtAq, ord="fro") ** 2

        prev = fnc[-1]
        iteration += int(q.shape[1])
        cur = 2.0 * iteration + C * (c - 2.0 * b)
        fnc.append(cur)

        if iteration > bsz and prev < cur:
            break

    lowrank_matvecs = int(2 * iteration)

    it2 = 0
    t_acc = 0.0
    trest2_vals = []

    while True:
        psi = _sample_matrix(n, bsz, vec_type=vec_type, rng=rng)
        y = psi - Q @ (Q.T.conj() @ psi)
        y = np.asarray(Afun(y))
        y = y - Q @ (Q.T.conj() @ y)

        trest2_vals.append(np.trace(psi.T.conj() @ y) / float(bsz))
        t_acc = t_acc + np.trace(y.T.conj() @ y)
        est_frob = t_acc / float(it2 + bsz)
        alpha = max(supfind(it2 + bsz, delta_prob), 1e-12)
        M = int(np.ceil(C * np.real(est_frob) / alpha))

        if (it2 + bsz) > M:
            break
        it2 += bsz

    trest2 = np.mean(trest2_vals) if trest2_vals else 0.0
    trest_matvecs = int(it2)
    trest = trest1 + trest2
    total_matvecs = int(lowrank_matvecs + trest_matvecs)

    return float(np.real(trest)), total_matvecs, lowrank_matvecs, trest_matvecs
