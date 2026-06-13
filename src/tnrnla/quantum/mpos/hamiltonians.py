import numpy as np
import numpy.linalg as la
import itertools

from tnrnla.tn.mpo import MPO
from tnrnla.tn.stopping import Cutoff
from tnrnla.quantum.mpos.util import (
    I2, X, Y, Z,
    sz, sp, sm,
    Sz, Sp, Sm, I3,
    mpo_from_fsm, mpo_from_fsm2,
)


def toric_code(N):
    pass


def ALKT(N, local_ops=False):
    Hl = np.zeros((14, 3, 14, 3))
    Hl[0, :, 0, :] = I3
    Hl[1, :, 0, :] = Sz
    Hl[2, :, 0, :] = Sm
    Hl[3, :, 0, :] = Sp
    Hl[4, :, 0, :] = np.matmul(Sz, Sz)
    Hl[5, :, 0, :] = np.matmul(Sz, Sm)
    Hl[6, :, 0, :] = np.matmul(Sz, Sp)
    Hl[7, :, 0, :] = np.matmul(Sm, Sz)
    Hl[8, :, 0, :] = np.matmul(Sm, Sm)
    Hl[9, :, 0, :] = np.matmul(Sm, Sp)
    Hl[10, :, 0, :] = np.matmul(Sp, Sz)
    Hl[11, :, 0, :] = np.matmul(Sp, Sm)
    Hl[12, :, 0, :] = np.matmul(Sp, Sp)
    Hl[13, :, 1, :] = Sz
    Hl[13, :, 2, :] = 0.5 * Sp
    Hl[13, :, 3, :] = 0.5 * Sm
    Hl[13, :, 4, :] = 1 / 3 * np.matmul(Sz, Sz)
    Hl[13, :, 5, :] = 1 / 6 * np.matmul(Sz, Sp)
    Hl[13, :, 6, :] = 1 / 6 * np.matmul(Sz, Sm)
    Hl[13, :, 7, :] = 1 / 6 * np.matmul(Sp, Sz)
    Hl[13, :, 8, :] = 1 / 12 * np.matmul(Sp, Sp)
    Hl[13, :, 9, :] = 1 / 12 * np.matmul(Sp, Sm)
    Hl[13, :, 10, :] = 1 / 6 * np.matmul(Sm, Sz)
    Hl[13, :, 11, :] = 1 / 12 * np.matmul(Sm, Sp)
    Hl[13, :, 12, :] = 1 / 12 * np.matmul(Sm, Sm)
    Hl[13, :, 13, :] = I3
    H = [Hl for l in range(N)]
    H[0] = Hl[-1:np.shape(Hl)[0], :, :, :]
    H[0] = H[0].reshape(H[0].shape[1], H[0].shape[2], H[0].shape[3])
    H[N - 1] = Hl[:, :, 0:1, :]
    H[N - 1] = H[N - 1].reshape(H[N - 1].shape[0], H[N - 1].shape[1], H[N - 1].shape[3])
    return MPO(H)


def Madjumdar_Gosh(N, local_ops=False):
    Hl = np.zeros((8, 2, 8, 2))
    Hl[0, :, 0, :] = I2
    Hl[1, :, 0, :] = sz
    Hl[2, :, 0, :] = sm
    Hl[3, :, 0, :] = sp
    Hl[4, :, 1, :] = I2
    Hl[5, :, 2, :] = I2
    Hl[6, :, 3, :] = I2
    Hl[7, :, 1, :] = sz
    Hl[7, :, 2, :] = 0.5 * sp
    Hl[7, :, 3, :] = 0.5 * sm
    Hl[7, :, 4, :] = 0.5 * sz
    Hl[7, :, 5, :] = 1 / 4 * sp
    Hl[7, :, 6, :] = 1 / 4 * sm
    Hl[7, :, 7, :] = I2
    H = [Hl for l in range(N)]
    H[0] = Hl[-1:np.shape(Hl)[0], :, :, :]
    H[0] = H[0].reshape(H[0].shape[1], H[0].shape[2], H[0].shape[3])
    H[N - 1] = Hl[:, :, 0:1, :]
    H[N - 1] = H[N - 1].reshape(H[N - 1].shape[0], H[N - 1].shape[1], H[N - 1].shape[3])
    if local_ops:
        return H
    return MPO(H)


def Heis_zeeman(N, J=1, h=1, local_ops=False):
    Hl = np.zeros((5, 3, 5, 3))
    Hl[0, :, 0, :] = I3
    Hl[1, :, 0, :] = Sz
    Hl[2, :, 0, :] = Sm
    Hl[3, :, 0, :] = Sp
    Hl[4, :, 0, :] = -h * Sz
    Hl[4, :, 1, :] = J * Sz
    Hl[4, :, 2, :] = J / 2 * Sp
    Hl[4, :, 3, :] = J / 2 * Sm
    Hl[4, :, 4, :] = I3
    H = [Hl for l in range(N)]
    H[0] = Hl[-1:np.shape(Hl)[0], :, :, :]
    H[0] = H[0].reshape(H[0].shape[1], H[0].shape[2], H[0].shape[3])
    H[N - 1] = Hl[:, :, 0:1, :]
    H[N - 1] = H[N - 1].reshape(H[N - 1].shape[0], H[N - 1].shape[1], H[N - 1].shape[3])
    if local_ops:
        return H
    return MPO(H)


def isotropic_heisenberg(N, Jx=1.0, Jy=1.0, Jz=1.0, local_ops=False):
    d = 2
    chi = 5

    W = np.zeros((chi, d, chi, d), dtype=complex)
    W[0, :, 0, :] = I2
    W[4, :, 4, :] = I2
    W[1, :, 0, :] = X
    W[2, :, 0, :] = Y
    W[3, :, 0, :] = Z
    W[4, :, 1, :] = 0.25 * Jx * X
    W[4, :, 2, :] = 0.25 * Jy * Y
    W[4, :, 3, :] = 0.25 * Jz * Z

    H = [W[4, :, :, :].copy()]
    for _ in range(N - 2):
        H.append(W.copy())
    H.append(W[:, :, 0, :].copy())

    if local_ops:
        return H
    return MPO(H)


def tfim_xx_z(N, J=1, h=1, local_ops=False):
    fsm = {
        (0, 0): I2,
        (0, 1): -J * X,
        (1, 2): X,
        (0, 2): -h * Z,
        (2, 2): I2,
    }
    cores = mpo_from_fsm(fsm, 3, N, source=0, target=2)
    if local_ops:
        return cores
    return MPO(cores)


def tfim_zz_x(N, J=1.0, h=1.0, local_ops=False):
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


def mpo_wrap_X1_XN(N: int, coeff: float, *, dtype=np.float64):
    N = int(N)
    if N < 2:
        raise ValueError("Need N >= 2 for a wrap term")
    d, D = 2, 2
    I = np.asarray(I2, dtype=dtype)
    Xop = np.asarray(X, dtype=dtype)

    W0 = np.zeros((d, D, d), dtype=dtype)
    W0[:, 0, :] = I
    W0[:, 1, :] = coeff * Xop
    cores = [W0]

    for _ in range(N - 2):
        W = np.zeros((D, d, D, d), dtype=dtype)
        W[0, :, 0, :] = I
        W[1, :, 1, :] = I
        cores.append(W)

    WN = np.zeros((D, d, d), dtype=dtype)
    WN[1, :, :] = Xop
    cores.append(WN)

    return MPO(cores)


def periodic_tfim(N, J=1.0, h=1.0, local_ops=False):
    H_open = tfim_zz_x(int(N), J=float(J), h=float(h), local_ops=False)
    H_wrap = mpo_wrap_X1_XN(int(N), coeff=-float(J), dtype=np.float64)
    try:
        H = H_open + H_wrap
    except Exception:
        H = H_open.add(H_wrap, subtract=False)
    if local_ops:
        return H.cores if hasattr(H, "cores") else H
    return H


def heisenberg(N, variant="heis", Jx=1, Jy=1, Jz=1, alpha=1, beta=1.1, gamma=1.2, local_ops=False):
    X_ = np.array([[0, 1], [1, 0]])
    Y_ = np.array([[0, -1j], [1j, 0]])
    Z_ = np.array([[1, 0], [0, -1]])

    if variant == "heis":
        graph = {
            (0, 0): np.eye(2),
            (0, 1): Jx * X_, (0, 2): Jy * Y_, (0, 3): Jz * Z_,
            (1, 4): Jx * X_, (2, 4): Jy * Y_, (3, 4): Jz * Z_,
            (4, 4): np.eye(2),
        }
        H = mpo_from_fsm(graph, 5, N, source=0, target=4)
        return H if local_ops else MPO(H)

    elif variant == "heis_next":
        graph = {
            (0, 0): np.eye(2),
            (0, 1): X_, (0, 2): Y_, (0, 3): Z_,
            (1, 7): alpha * X_, (2, 7): alpha * Y_, (3, 7): alpha * Z_,
            (1, 4): beta * np.eye(2), (2, 5): beta * np.eye(2), (3, 6): beta * np.eye(2),
            (4, 7): X_, (5, 7): Y_, (6, 7): Z_,
            (7, 7): np.eye(2),
        }
        H = mpo_from_fsm(graph, 8, N, source=0, target=7)
        return H if local_ops else MPO(H)

    elif variant == "heis_next_next":
        graph = {
            (0, 0): np.eye(2),
            (0, 1): X_, (0, 2): Y_, (0, 3): Z_,
            (1, 10): alpha * X_, (2, 10): alpha * Y_, (3, 10): alpha * Z_,
            (1, 4): np.eye(2), (2, 5): np.eye(2), (3, 6): np.eye(2),
            (4, 7): gamma * np.eye(2), (5, 8): gamma * np.eye(2), (6, 9): gamma * np.eye(2),
            (7, 10): X_, (8, 10): Y_, (9, 10): Z_,
            (4, 10): beta * X_, (5, 10): beta * Y_, (6, 10): beta * Z_,
            (10, 10): np.eye(2),
        }
        H = mpo_from_fsm(graph, 11, N, source=0, target=10)
        return H if local_ops else MPO(H)


def esprit(N, interaction, stop=Cutoff(1e-8), small_eig_cutoff=1e-10):
    r = stop.outputdim if (stop.outputdim is not None) else stop.mindim
    r = min(r, (N - 1) // 2)
    while True:
        F = np.zeros((N - r, r))
        for i, j in itertools.product(range(N - r), range(r)):
            F[i, j] = interaction(i + j + 1)
        W, _, _, _ = la.lstsq(F[:-1, :], F[1:, :])
        bases, _ = la.eig(W)
        bases = np.real(bases)
        bases = bases[bases > small_eig_cutoff * la.norm(bases, np.inf)]

        F = np.zeros((N - 1, len(bases)))
        b = np.zeros(N - 1)
        for i in range(N - 1):
            for j in range(len(bases)):
                F[i, j] = np.power(bases[j], i)
            b[i] = interaction(i + 1)
        coeffs, _, _, _ = la.lstsq(F, b)

        if (stop.outputdim is not None and r >= stop.outputdim) or r >= stop.maxdim or r == (N - 1) // 2:
            return bases, coeffs

        error = la.norm(b - F @ coeffs, np.inf) / la.norm(b, np.inf)
        if error <= stop.cutoff:
            return bases, coeffs

        r += 1


def long_range_tfim(N, h=1.0, interaction=lambda dist: dist ** -2, stop=Cutoff(1e-8)):
    bases, coeffs = esprit(N, interaction, stop=stop)
    fsm = {
        (0, 0): I2,
        **{(0, i + 1): coeffs[i] * X for i in range(len(bases))},
        **{(i + 1, i + 1): bases[i] * I2 for i in range(len(bases))},
        **{(i + 1, len(bases) + 1): X for i in range(len(bases))},
        (0, len(bases) + 1): h * Z,
        (len(bases) + 1, len(bases) + 1): I2,
    }
    return MPO(mpo_from_fsm(fsm, len(bases) + 2, N, source=0, target=len(bases) + 1))


def cluster(N, K=1, h=1, local_ops=False):
    fsm = {
        (0, 0): I2,
        (0, 1): K * X,
        (1, 2): X,
        (2, 3): X,
        (0, 3): h * Z,
        (3, 3): I2,
    }
    H = mpo_from_fsm(fsm, 4, N, source=0, target=3)
    if local_ops:
        return H
    return MPO(H)


def long_range_XY_model(N, J=1, alpha=1, local_ops=False, stop=Cutoff(1e-10)):
    bases, coeffs = esprit(N, lambda dist: dist ** -alpha, stop=stop)
    r = len(bases)
    start = 2 * r
    stp = 2 * r + 1
    fsm = {
        (start, start): I2,
        **{(start, i): J / 2 * coeffs[i] * X for i in range(r)},
        **{(i, i): bases[i] * I2 for i in range(r)},
        **{(i, stp): X for i in range(r)},
        **{(start, i + r): J / 2 * coeffs[i] * Y for i in range(r)},
        **{(i + r, i + r): bases[i] * I2 for i in range(r)},
        **{(i + r, stp): Y for i in range(r)},
        (stp, stp): I2,
    }
    H = mpo_from_fsm(fsm, 2 * r + 2, N, source=start, target=stp)
    if local_ops:
        return H
    return MPO(H)
