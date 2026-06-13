
import numpy as np
from tnrnla.linalg.lra import truncated_svd
from tnrnla.linalg.orth import lq
from tnrnla.tn.stopping import Cutoff
from tnrnla.tn.mps import MPS


def _compress_residual_block(L, Qt, chi0, cutoff, max_total_bond=None):
    """
    Keep the first-state subspace exactly, then compress only the residual block
    L[chi0:, chi0:] using your truncated_svd approach.
    """
    dt = np.result_type(np.asarray(L).dtype, np.asarray(Qt).dtype)
    L = np.asarray(L, dtype=dt)
    Qt = np.asarray(Qt, dtype=dt)

    if chi0 < 0 or chi0 > L.shape[0]:
        raise ValueError(f"Invalid chi0={chi0} for L with shape {L.shape}")

    keep = L[:, :chi0]
    qkeep = Qt[:chi0, :]

    tail_rows = L.shape[0] - chi0
    tail_cols = L.shape[1] - chi0

    if tail_rows <= 0 or tail_cols <= 0:
        return keep, qkeep

    L11 = L[chi0:, chi0:]

    # Same absolute truncation logic as before, with an early skip
    # when the residual is already negligible.
    normL = np.linalg.norm(L)
    if normL == 0:
        return keep, qkeep

    abstol = cutoff * normL
    if cutoff > 0:
        tail_norm = np.linalg.norm(L11)
        if tail_norm <= abstol:
            return keep, qkeep

    stop_rule = Cutoff(cutoff)
    if max_total_bond is not None:
        max_total_bond = int(max_total_bond)
        max_extra = max_total_bond - int(chi0)
        if max_extra <= 0:
            return keep, qkeep
        stop_rule = Cutoff(cutoff, maxdim=max_extra)

    U, S, Vt = truncated_svd(
        L11,
        stop=stop_rule,
        abstol=abstol,
    )

    U = np.asarray(U, dtype=dt)
    S = np.asarray(S, dtype=dt)
    Vt = np.asarray(Vt, dtype=dt)

    r = U.shape[1]
    if r == 0:
        return keep, qkeep

    B = np.empty((L.shape[0], chi0 + r), dtype=dt)
    B[:, :chi0] = keep
    B[:chi0, chi0:] = 0
    B[chi0:, chi0:] = U * S

    T = np.empty((chi0 + r, Qt.shape[1]), dtype=dt)
    T[:chi0, :] = qkeep
    T[chi0:, :] = Vt @ Qt[chi0:, :]

    return B, T


def expand(states, cutoff=1e-14, max_enriched_bond=None):
    if not states:
        raise ValueError("expand requires at least one state")

    N = states[0].N
    if any(st.N != N for st in states):
        raise ValueError("All states must have the same number of sites")
    for st in states:
        if st.orthform != "Right":
            #print(st.orthform)
            st.orthR()

    # Trivial case
    if len(states) == 1:
        return MPS([np.array(states[0][i], copy=True) for i in range(N)])

    # Precompute site metadata once
    site_meta = {}
    for i in range(1, N - 1):
        d = states[0][i].shape[1]

        right_slices = []
        left_slices = []

        r0 = 0
        l0 = 0
        for st in states:
            A = np.asarray(st[i])
            if A.ndim != 3:
                raise ValueError(f"Interior site {i} must be rank-3, got shape {A.shape}")
            if A.shape[1] != d:
                raise ValueError(
                    f"Physical dimension mismatch at site {i}. "
                    f"Expected {d}, got {A.shape[1]}"
                )

            chiL, _, chiR = A.shape
            right_slices.append(slice(r0, r0 + chiR))
            left_slices.append(slice(l0, l0 + chiL))
            r0 += chiR
            l0 += chiL

        site_meta[i] = {
            "d": d,
            "right_slices": right_slices,
            "left_slices": left_slices,
            "total_right": r0,
            "total_left": l0,
        }

    mps_out = [None] * N

    # Last site
    last_dtype = np.result_type(*[np.asarray(st[-1]).dtype for st in states])
    boundary = np.vstack([np.asarray(st[-1], dtype=last_dtype) for st in states])

    L, Qt = lq(boundary)
    chi0 = states[0][-1].shape[0]
    B, Tlast = _compress_residual_block(
        L, Qt, chi0, cutoff, max_total_bond=max_enriched_bond
    )
    mps_out[-1] = np.asarray(Tlast, dtype=np.result_type(np.asarray(Tlast).dtype, last_dtype))

    # Sweep right to left through interior sites
    for i in range(N - 2, 0, -1):
        meta = site_meta[i]
        d = meta["d"]
        right_slices = meta["right_slices"]
        left_slices = meta["left_slices"]

        if meta["total_right"] != B.shape[0]:
            raise ValueError(
                f"Bond mismatch at site {i}. "
                f"Expected stacked right dimension {meta['total_right']}, got {B.shape[0]}"
            )

        dt = np.result_type(B.dtype, *[np.asarray(st[i]).dtype for st in states])
        if B.dtype != dt:
            B = np.asarray(B, dtype=dt)

        rank = B.shape[1]

        # Directly build the flattened stacked tensor for LQ.
        # This avoids
        #   Bnew = ...
        #   blocks = ...
        #   concatenate(blocks, axis=0)
        Bflat = np.empty((meta["total_left"], d * rank), dtype=dt)

        for st, rs, ls in zip(states, right_slices, left_slices):
            A = np.asarray(st[i], dtype=dt)
            chiL, _, chiR = A.shape

            prod = A.reshape(chiL * d, chiR) @ B[rs, :]
            Bflat[ls, :] = prod.reshape(chiL, d * rank)

        L, Qt = lq(Bflat)
        chi0 = states[0][i].shape[0]
        B, T = _compress_residual_block(
            L, Qt, chi0, cutoff, max_total_bond=max_enriched_bond
        )
        mps_out[i] = T.reshape(T.shape[0], d, -1)

    # First site
    A0 = np.asarray(states[0][0], dtype=np.result_type(np.asarray(states[0][0]).dtype, B.dtype))
    if A0.shape[1] > B.shape[0]:
        raise ValueError(
            f"First-site contraction mismatch. "
            f"A0.shape={A0.shape}, B.shape={B.shape}"
        )

    mps_out[0] = A0 @ B[:A0.shape[1], :]

    return MPS(mps_out)
