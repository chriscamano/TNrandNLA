import numpy as np

# Pauli matrices
I2 = np.eye(2)
X = np.array([[0, 1.], [1, 0]])
Y = np.array([[0, -1.j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# spin-1/2 operators
sz = np.zeros((2, 2))
sz[0, 0] = 1 / 2
sz[1, 1] = -1 / 2
sp = np.zeros((2, 2))
sp[0, 1] = 1
sm = np.zeros((2, 2))
sm[1, 0] = 1

# spin-1 operators
Sz = np.zeros((3, 3))
Sz[0, 0] = 1
Sz[2, 2] = -1
Sp = np.zeros((3, 3))
Sp[0, 1] = np.sqrt(2)
Sp[1, 2] = np.sqrt(2)
Sm = np.zeros((3, 3))
Sm[1, 0] = np.sqrt(2)
Sm[2, 1] = np.sqrt(2)
I3 = np.eye(3)


def mpo_from_fsm(graph, k, n, source=-1, target=0):
    assert len(graph) > 0
    d = graph[list(graph.keys())[0]].shape[0]

    A = np.zeros((k, d, k, d), dtype=float)
    for j, i in graph.keys():
        A[i, :, j, :] = graph[(j, i)]

    return [A[target, :, :, :]] + (n - 2) * [np.copy(A)] + [A[:, :, source, :]]


def mpo_from_fsm2(graph, k, n, source=-1, target=0):
    assert len(graph) > 0
    d = next(iter(graph.values())).shape[0]
    A = np.zeros((k, d, k, d), dtype=complex)
    for (i, j), op in graph.items():
        A[i, :, j, :] = op
    if source < 0:
        source += k
    W0 = np.zeros((d, k, d), dtype=complex)
    for alpha in range(k):
        W0[:, alpha, :] = A[source, :, alpha, :]
    Wmid = np.zeros((k, d, k, d), dtype=complex)
    for i in range(k):
        for j in range(k):
            Wmid[i, :, j, :] = A[i, :, j, :]
    Wlast = np.zeros((k, d, d), dtype=complex)
    for i in range(k):
        Wlast[i, :, :] = A[i, :, target, :]
    if n == 1:
        return [W0]
    if n == 2:
        return [W0, Wlast]
    return [W0] + [Wmid.copy() for _ in range(n - 2)] + [Wlast]
