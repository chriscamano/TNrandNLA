"""Matvec/operator builders shared by experimental DMRG2 variants."""

import time

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from tnrnla.tn.other.tensor_ops import flatten_left, flatten_right


def contract_blas(A, axesA, B, axesB):
    axesA = (axesA,) if isinstance(axesA, int) else tuple(axesA)
    axesB = (axesB,) if isinstance(axesB, int) else tuple(axesB)

    freeA = tuple(i for i in range(A.ndim) if i not in axesA)
    freeB = tuple(i for i in range(B.ndim) if i not in axesB)

    A_t = A.transpose(freeA + axesA)
    B_t = B.transpose(axesB + freeB)

    kA = int(np.prod([A.shape[i] for i in axesA], dtype=np.int64)) if axesA else 1
    kB = int(np.prod([B.shape[i] for i in axesB], dtype=np.int64)) if axesB else 1
    if kA != kB:
        raise ValueError(f"Contracted dimensions mismatch: {kA} vs {kB}")
    k = kA

    C2 = (
        np.ascontiguousarray(A_t).reshape(-1, k)
        @ np.ascontiguousarray(B_t).reshape(k, -1)
    )

    out_shape = tuple(A.shape[i] for i in freeA) + tuple(B.shape[i] for i in freeB)
    return C2.reshape(out_shape)


def _timer_now(timer):
    return time.perf_counter() if timer is None else timer()


def _record(record_matvec, dt, sparse):
    if record_matvec is not None:
        record_matvec(float(dt), sparse=bool(sparse))


def _contract_last_axis(a, b):
    """Contract last axis of `a` with last axis of `b` via one GEMM.

    a: (..., n), b: (..., n) -> out: a.shape[:-1] + b.shape[:-1]
    """
    n = int(a.shape[-1])
    if int(b.shape[-1]) != n:
        raise ValueError(f"Last-axis mismatch: {a.shape[-1]} vs {b.shape[-1]}")
    a2 = np.ascontiguousarray(a).reshape(-1, n)
    b2 = np.ascontiguousarray(b).reshape(-1, n)
    out2 = a2 @ b2.T
    return out2.reshape(a.shape[:-1] + b.shape[:-1])


def _prepare_last_axis_rhs(b):
    """Prepare RHS once for repeated _contract_last_axis calls with fixed `b`."""
    b = np.asarray(b)
    n = int(b.shape[-1])
    b2 = np.ascontiguousarray(b).reshape(-1, n)
    return np.ascontiguousarray(b2.T), tuple(int(s) for s in b.shape[:-1])


def _contract_last_axis_prepared(a, rhs_t, rhs_prefix):
    n = int(a.shape[-1])
    if int(rhs_t.shape[0]) != n:
        raise ValueError(f"Last-axis mismatch: {a.shape[-1]} vs {rhs_t.shape[0]}")
    a2 = np.ascontiguousarray(a).reshape(-1, n)
    out2 = a2 @ rhs_t
    return out2.reshape(a.shape[:-1] + tuple(rhs_prefix))


def _record_detail(record_matvec_detail, dt, *, sparse, boundary, phase):
    if record_matvec_detail is not None:
        record_matvec_detail(
            float(dt),
            sparse=bool(sparse),
            boundary=str(boundary),
            phase=str(phase),
        )


def _record_setup(record_op_setup, *, boundary, **meta):
    if record_op_setup is not None:
        payload = dict(meta)
        payload["boundary"] = str(boundary)
        record_op_setup(**payload)


def _accumulate_dtype(obj, dtypes):
    if isinstance(obj, dict):
        core = obj.get("core", None)
        if core is not None:
            if sps.issparse(core):
                dtypes.append(core.dtype)
            else:
                dtypes.append(np.asarray(core).dtype)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _accumulate_dtype(item, dtypes)
        return
    arr = np.asarray(obj) if hasattr(obj, "shape") or hasattr(obj, "dtype") else None
    if arr is not None:
        dtypes.append(arr.dtype)


def _infer_operator_dtype(*args):
    dtypes = [np.float64]
    for a in args:
        _accumulate_dtype(a, dtypes)
    return np.result_type(*dtypes)


def _build_env_symmetry_blocks(L_shape, qbond, qW):
    """Build Z2 block indices for env matmul rows/cols.

    L_shape is expected to be (X, D, X). Flattened env matmul uses:
      Lmat shape (X, X*D), with col index c = b*D + w.
    """
    if qbond is None or qW is None:
        return None
    X0, D, X1 = (int(s) for s in L_shape)
    qb = np.asarray(qbond, dtype=np.int8).reshape(-1)
    qw = np.asarray(qW, dtype=np.int8).reshape(-1)
    if qb.size != X0 or qb.size != X1 or qw.size != D:
        return None

    # Flattened (b, w) index mapping for env input rows.
    b_idx = np.repeat(np.arange(X1, dtype=np.int64), D)
    w_idx = np.tile(np.arange(D, dtype=np.int64), X1)
    q_in = np.bitwise_xor(qb[b_idx], qw[w_idx])

    out0 = np.flatnonzero(qb == 0)
    out1 = np.flatnonzero(qb == 1)
    in0 = np.flatnonzero(q_in == 0)
    in1 = np.flatnonzero(q_in == 1)
    return ((out0, in0), (out1, in1))


def _phys_charge_vector(n, provided=None):
    if provided is None:
        return (np.arange(int(n), dtype=np.int8) & 1)
    q = np.asarray(provided, dtype=np.int8).reshape(-1)
    if q.size < int(n):
        raise ValueError(f"phys charge vector too short: got {q.size}, need >= {int(n)}")
    return q[: int(n)]


def _build_core_symmetry_blocks(row_qch, n_row_phys, col_qch, n_col_phys, *, qphys_row=None, qphys_col=None):
    qrow = np.asarray(row_qch, dtype=np.int8).reshape(-1)
    qcol = np.asarray(col_qch, dtype=np.int8).reshape(-1)
    qr_phys = _phys_charge_vector(n_row_phys, qphys_row)
    qc_phys = _phys_charge_vector(n_col_phys, qphys_col)
    if qrow.size == 0 or qcol.size == 0:
        return None

    row_charge = np.bitwise_xor(np.repeat(qrow, int(n_row_phys)), np.tile(qr_phys, qrow.size))
    col_charge = np.bitwise_xor(np.repeat(qcol, int(n_col_phys)), np.tile(qc_phys, qcol.size))
    r0 = np.flatnonzero(row_charge == 0)
    c0 = np.flatnonzero(col_charge == 0)
    r1 = np.flatnonzero(row_charge == 1)
    c1 = np.flatnonzero(col_charge == 1)
    return ((r0, c0), (r1, c1))


def _sym_block_mm(W, B, blocks, *, dtype_eff):
    if blocks is None:
        return np.asarray(W @ B)
    out = np.zeros((int(W.shape[0]), int(B.shape[1])), dtype=dtype_eff)
    for r_idx, c_idx in blocks:
        if r_idx.size == 0 or c_idx.size == 0:
            continue
        rhs = B[c_idx, :]
        if sps.issparse(W):
            out[r_idx, :] = np.asarray(W[r_idx][:, c_idx] @ rhs)
        else:
            out[r_idx, :] = np.asarray(np.asarray(W)[np.ix_(r_idx, c_idx)] @ rhs)
    return out


def _build_block_ops(W, blocks):
    if blocks is None:
        return None
    out = []
    is_sparse = sps.issparse(W)
    for r_idx, c_idx in blocks:
        if r_idx.size == 0 or c_idx.size == 0:
            out.append(None)
            continue
        if is_sparse:
            out.append(W[r_idx][:, c_idx].tocsr())
        else:
            out.append(np.asarray(W)[np.ix_(r_idx, c_idx)])
    return out


def _block_partition_labels(n, blocks):
    labels = np.full(int(n), -1, dtype=np.int8)
    for bi, (r_idx, _) in enumerate(blocks):
        if r_idx.size:
            labels[r_idx] = np.int8(bi)
    return labels


def _is_block_compatible(W, blocks, *, tol=1e-13):
    if blocks is None:
        return False
    nrow, ncol = int(W.shape[0]), int(W.shape[1])
    row_labels = _block_partition_labels(nrow, blocks)
    col_blocks = tuple((np.asarray(c_idx, dtype=np.int64) for _, c_idx in blocks))
    col_labels = np.full(int(ncol), -1, dtype=np.int8)
    for bi, c_idx in enumerate(col_blocks):
        if c_idx.size:
            col_labels[c_idx] = np.int8(bi)

    if sps.issparse(W):
        coo = W.tocoo(copy=False)
        if coo.nnz == 0:
            return True
        row_b = row_labels[coo.row]
        col_b = col_labels[coo.col]
        vals = np.abs(np.asarray(coo.data))
        total = float(np.sum(vals))
        off = float(np.sum(vals[row_b != col_b]))
        return off <= float(tol) * max(1.0, total)

    A = np.asarray(W)
    total = float(np.sum(np.abs(A)))
    if total == 0.0:
        return True
    mask = np.zeros_like(A, dtype=bool)
    for r_idx, c_idx in blocks:
        if r_idx.size and c_idx.size:
            mask[np.ix_(r_idx, c_idx)] = True
    off = float(np.sum(np.abs(A[~mask])))
    return off <= float(tol) * max(1.0, total)


def _build_validated_block_ops(W, blocks, *, tol=1e-13):
    if blocks is None:
        return None
    if not _is_block_compatible(W, blocks, tol=tol):
        return None
    return _build_block_ops(W, blocks)


def _sym_block_mm_precomputed(B, blocks, block_ops, *, nrow, dtype_eff, out=None):
    if blocks is None or block_ops is None:
        raise ValueError("blocks and block_ops must be provided for precomputed block mm")
    if out is None:
        out = np.zeros((int(nrow), int(B.shape[1])), dtype=dtype_eff)
    else:
        out.fill(0.0)
    for bi, (r_idx, c_idx) in enumerate(blocks):
        op = block_ops[bi]
        if op is None or r_idx.size == 0 or c_idx.size == 0:
            continue
        out[r_idx, :] = np.asarray(op @ B[c_idx, :])
    return out


def _make_op_dense_precontract_first(W0, W1, R, psi_shape, *, dtype_eff, record_matvec=None, timer=None):
    t1 = contract_blas(W1, 2, R, 1)
    RWW = contract_blas(W0, 1, t1, 0)
    RWW_t = np.ascontiguousarray(RWW.transpose(1, 3, 5, 0, 2, 4))
    K = int(np.prod(RWW_t.shape[:3], dtype=np.int64))
    N = int(np.prod(RWW_t.shape[3:], dtype=np.int64))
    RWW_2d = RWW_t.reshape(K, N)
    n = int(np.prod(psi_shape, dtype=np.int64))

    def _mv(v):
        t0 = _timer_now(timer)
        out = np.asarray(v, dtype=dtype_eff).reshape(1, K) @ RWW_2d
        dt = _timer_now(timer) - t0
        _record(record_matvec, dt, sparse=False)
        return out.reshape(-1)

    return spla.LinearOperator((n, n), matvec=_mv, dtype=dtype_eff), _mv


def _make_op_dense_precontract_last(L, Wk, Wk1, psi_shape, *, dtype_eff, record_matvec=None, timer=None):
    t1 = contract_blas(L, 1, Wk, 0)
    LWW = contract_blas(t1, 3, Wk1, 0)
    LWW_t = np.ascontiguousarray(LWW.transpose(1, 3, 5, 0, 2, 4))
    K = int(np.prod(LWW_t.shape[:3], dtype=np.int64))
    N = int(np.prod(LWW_t.shape[3:], dtype=np.int64))
    LWW_2d = LWW_t.reshape(K, N)
    n = int(np.prod(psi_shape, dtype=np.int64))

    def _mv(v):
        t0 = _timer_now(timer)
        out = np.asarray(v, dtype=dtype_eff).reshape(1, K) @ LWW_2d
        dt = _timer_now(timer) - t0
        _record(record_matvec, dt, sparse=False)
        return out.reshape(-1)

    return spla.LinearOperator((n, n), matvec=_mv, dtype=dtype_eff), _mv


def _make_op_dense_precontract_mid_lr(Wk, Wk1, L, R, psi_shape, *, dtype_eff, record_matvec=None, timer=None):
    t1 = contract_blas(L, 1, Wk, 0)
    LWW = contract_blas(t1, 3, Wk1, 0)

    Xo, Xi, db, dk, kb, F, jk = LWW.shape
    LWW_t = np.ascontiguousarray(LWW.transpose(1, 3, 6, 0, 2, 4, 5))
    K = int(Xi * dk * jk)
    N_lww = int(Xo * db * kb * F)
    LWW_T = np.ascontiguousarray(LWW_t.reshape(K, N_lww).T)

    y = int(R.shape[0])
    chi_R = int(R.shape[2])
    R_2d_T = np.ascontiguousarray(R.reshape(y, -1).T)

    n = int(np.prod(psi_shape, dtype=np.int64))
    out_mid = int(Xo * db * kb)

    def _mv(v):
        t0 = _timer_now(timer)
        p = np.asarray(v, dtype=dtype_eff).reshape(-1, chi_R)
        t1_local = LWW_T @ p
        t1_local = t1_local.reshape(out_mid, F * chi_R)
        out = t1_local @ R_2d_T
        dt = _timer_now(timer) - t0
        _record(record_matvec, dt, sparse=False)
        return out.reshape(-1)

    return spla.LinearOperator((n, n), matvec=_mv, dtype=dtype_eff), _mv


def _make_op_dense_precontract_mid_rl(Wk, Wk1, L, R, psi_shape, *, dtype_eff, record_matvec=None, timer=None):
    t1 = contract_blas(Wk1, 2, R, 1)
    RWW = contract_blas(Wk, 2, t1, 0)

    D, db, dk, kb, jk, y, chi = RWW.shape
    RWW_t = np.ascontiguousarray(RWW.transpose(2, 4, 6, 0, 1, 3, 5))
    K = int(dk * jk * chi)
    N_rww = int(D * db * kb * y)
    RWW_2d = RWW_t.reshape(K, N_rww)

    Xo = int(L.shape[0])
    chi_L = int(psi_shape[0])
    L_mat_T = np.ascontiguousarray(L.transpose(2, 1, 0).reshape(-1, Xo).T)

    n = int(np.prod(psi_shape, dtype=np.int64))

    def _mv(v):
        t0 = _timer_now(timer)
        p = np.asarray(v, dtype=dtype_eff).reshape(chi_L, K)
        t1_local = p @ RWW_2d
        t1_local = t1_local.reshape(chi_L * D, db * kb * y)
        out = L_mat_T @ t1_local
        dt = _timer_now(timer) - t0
        _record(record_matvec, dt, sparse=False)
        return out.reshape(-1)

    return spla.LinearOperator((n, n), matvec=_mv, dtype=dtype_eff), _mv


def make_op_dense(
    boundary,
    *op_args,
    record_matvec=None,
    record_matvec_detail=None,
    contract_fn=contract_blas,
    timer=None,
    precontract=False,
    sweeping_right=True,
):
    """Build dense local effective operator with boundary switch.

    boundary:
      - "first": (mpo0, mpo1, R2, psi_shape)
      - "mid": (Lkm1, mpok, mpok1, Rkp2, psi_shape)
      - "last": (LNm3, mpoNm2, mpoNm1, psi_shape)
    """
    dtype_eff = _infer_operator_dtype(*op_args)

    if precontract:
        if boundary == "first":
            mpo0, mpo1, R2, psi_shape = op_args
            return _make_op_dense_precontract_first(
                mpo0,
                mpo1,
                R2,
                psi_shape,
                dtype_eff=dtype_eff,
                record_matvec=record_matvec,
                timer=timer,
            )
        if boundary == "mid":
            Lkm1, mpok, mpok1, Rkp2, psi_shape = op_args
            if sweeping_right:
                return _make_op_dense_precontract_mid_lr(
                    mpok,
                    mpok1,
                    Lkm1,
                    Rkp2,
                    psi_shape,
                    dtype_eff=dtype_eff,
                    record_matvec=record_matvec,
                    timer=timer,
                )
            return _make_op_dense_precontract_mid_rl(
                mpok,
                mpok1,
                Lkm1,
                Rkp2,
                psi_shape,
                dtype_eff=dtype_eff,
                record_matvec=record_matvec,
                timer=timer,
            )
        if boundary == "last":
            LNm3, mpoNm2, mpoNm1, psi_shape = op_args
            return _make_op_dense_precontract_last(
                LNm3,
                mpoNm2,
                mpoNm1,
                psi_shape,
                dtype_eff=dtype_eff,
                record_matvec=record_matvec,
                timer=timer,
            )
        raise ValueError(f"Unknown boundary mode for dense op: {boundary}")

    if boundary == "first":
        mpo0, mpo1, R2, psi_shape = op_args
        n = int(np.prod(psi_shape, dtype=np.int64))

        def _mv(v):
            t_mv0 = _timer_now(timer)
            p = np.asarray(v).reshape(psi_shape)
            t0 = _timer_now(timer)
            # Specialized hot contraction: (l,k,j) x (y,E,j) over j -> (l,k,y,E)
            t1 = _contract_last_axis(p, R2)
            t1_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t1_dt, sparse=False, boundary=boundary, phase="contract_1")
            t0 = _timer_now(timer)
            t2 = contract_fn(t1, (1, 3), mpo1, (3, 2))
            t2_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t2_dt, sparse=False, boundary=boundary, phase="contract_2")
            t0 = _timer_now(timer)
            t3 = contract_fn(t2, (0, 2), mpo0, (2, 1))
            t3_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t3_dt, sparse=False, boundary=boundary, phase="contract_3")
            t0 = _timer_now(timer)
            out = t3.transpose(2, 1, 0)
            out_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, out_dt, sparse=False, boundary=boundary, phase="output")
            total_dt = _timer_now(timer) - t_mv0
            _record(record_matvec, total_dt, sparse=False)
            _record_detail(record_matvec_detail, total_dt, sparse=False, boundary=boundary, phase="total")
            return np.ascontiguousarray(out).reshape(-1)

    elif boundary == "mid":
        Lkm1, mpok, mpok1, Rkp2, psi_shape = op_args
        n = int(np.prod(psi_shape, dtype=np.int64))

        def _mv(v):
            t_mv0 = _timer_now(timer)
            p = np.asarray(v).reshape(psi_shape)
            t0 = _timer_now(timer)
            # Specialized hot contraction: (X,d,k,j) x (y,F,j) over j -> (X,d,k,y,F)
            t1 = _contract_last_axis(p, Rkp2)
            t1_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t1_dt, sparse=False, boundary=boundary, phase="contract_1")
            t0 = _timer_now(timer)
            t2 = contract_fn(t1, (2, 4), mpok1, (3, 2))
            t2_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t2_dt, sparse=False, boundary=boundary, phase="contract_2")
            t0 = _timer_now(timer)
            t3 = contract_fn(t2, (1, 3), mpok, (3, 2))
            t3_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t3_dt, sparse=False, boundary=boundary, phase="contract_3")
            t0 = _timer_now(timer)
            t4 = contract_fn(t3, (0, 3), Lkm1, (2, 1))
            t4_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t4_dt, sparse=False, boundary=boundary, phase="contract_4")
            t0 = _timer_now(timer)
            out = t4.transpose(3, 2, 1, 0)
            out_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, out_dt, sparse=False, boundary=boundary, phase="output")
            total_dt = _timer_now(timer) - t_mv0
            _record(record_matvec, total_dt, sparse=False)
            _record_detail(record_matvec_detail, total_dt, sparse=False, boundary=boundary, phase="total")
            return np.ascontiguousarray(out).reshape(-1)

    elif boundary == "last":
        LNm3, mpoNm2, mpoNm1, psi_shape = op_args
        n = int(np.prod(psi_shape, dtype=np.int64))

        def _mv(v):
            t_mv0 = _timer_now(timer)
            p = np.asarray(v).reshape(psi_shape)
            t0 = _timer_now(timer)
            t1 = contract_fn(p, 2, mpoNm1, 2)
            t1_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t1_dt, sparse=False, boundary=boundary, phase="contract_1")
            t0 = _timer_now(timer)
            t2 = contract_fn(t1, (1, 2), mpoNm2, (3, 2))
            t2_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t2_dt, sparse=False, boundary=boundary, phase="contract_2")
            t0 = _timer_now(timer)
            t3 = contract_fn(t2, (0, 2), LNm3, (2, 1))
            t3_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t3_dt, sparse=False, boundary=boundary, phase="contract_3")
            t0 = _timer_now(timer)
            out = t3.transpose(2, 1, 0)
            out_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, out_dt, sparse=False, boundary=boundary, phase="output")
            total_dt = _timer_now(timer) - t_mv0
            _record(record_matvec, total_dt, sparse=False)
            _record_detail(record_matvec_detail, total_dt, sparse=False, boundary=boundary, phase="total")
            return np.ascontiguousarray(out).reshape(-1)

    else:
        raise ValueError(f"Unknown boundary mode for dense op: {boundary}")

    op = spla.LinearOperator((n, n), matvec=_mv, dtype=dtype_eff)
    return op, _mv


def make_op_sparse(
    boundary,
    *op_args,
    record_matvec=None,
    record_matvec_detail=None,
    record_op_setup=None,
    contract_fn=contract_blas,
    timer=None,
    env_sparse_threshold=0.35,
    env_qbond=None,
    env_qW=None,
    use_z2_block_oracle=False,
    qW_left=None,
    qW_mid=None,
    qW_right=None,
    qphys=None,
):
    """Build sparse-core local effective operator with boundary switch.

    boundary:
      - "first": (core0, core1, R2, psi_shape)
      - "mid": (Lkm1, corek, corek1, Rkp2, psi_shape)
      - "last": (LNm3, coreNm2, coreNm1, psi_shape)
    """
    dtype_eff = _infer_operator_dtype(*op_args)

    if boundary == "first":
        core0, core1, R2, psi_shape = op_args
        n = int(np.prod(psi_shape, dtype=np.int64))
        W0 = core0["core"]
        W1 = core1["core"]
        if sps.issparse(W0) and not sps.isspmatrix_csr(W0):
            W0 = W0.tocsr(copy=False)
        if sps.issparse(W1) and not sps.isspmatrix_csr(W1):
            W1 = W1.tocsr(copy=False)
        d = int(core0["shape"][0])
        D = int(core0["shape"][1])
        l = int(psi_shape[0])
        y = int(R2.shape[0])
        E = int(core1["shape"][2])
        j = int(core1["shape"][3])
        k = int(core1["shape"][1])
        R2_rhs_T, R2_prefix = _prepare_last_axis_rhs(R2)
        b1_buf = np.empty((E * j, l * y), dtype=dtype_eff)
        b2_buf = np.empty((D * l, y * k), dtype=dtype_eff)
        out_ring = [np.empty(psi_shape, dtype=dtype_eff) for _ in range(2)]
        out_idx = 0
        _record_setup(
            record_op_setup,
            boundary=boundary,
            n=int(n),
            mpo_left_sparse=bool(sps.issparse(W0)),
            mpo_right_sparse=bool(sps.issparse(W1)),
        )

        def _mv(v):
            nonlocal out_idx
            t_mv0 = _timer_now(timer)
            p = np.asarray(v).reshape(psi_shape)
            t0 = _timer_now(timer)
            # Specialized hot contraction: (l,k,j) x (y,E,j) over j -> (l,k,y,E)
            t1 = _contract_last_axis_prepared(p, R2_rhs_T, R2_prefix)
            t1_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t1_dt, sparse=True, boundary=boundary, phase="contract_1")

            t0 = _timer_now(timer)
            np.copyto(b1_buf, t1.transpose(3, 1, 0, 2).reshape(E * j, l * y))
            u1 = np.asarray(W1 @ b1_buf)
            t2 = u1.reshape(D, k, l, y).transpose(2, 3, 0, 1)
            sp1_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, sp1_dt, sparse=True, boundary=boundary, phase="sparse_mm_1")

            t0 = _timer_now(timer)
            np.copyto(b2_buf, t2.transpose(2, 0, 1, 3).reshape(D * l, y * k))
            u2 = np.asarray(W0 @ b2_buf)
            out = out_ring[out_idx]
            out_idx = (out_idx + 1) % len(out_ring)
            np.copyto(out, u2.reshape(d, y, k).transpose(0, 2, 1))
            sp2_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, sp2_dt, sparse=True, boundary=boundary, phase="sparse_mm_2")
            total_dt = _timer_now(timer) - t_mv0
            _record(record_matvec, total_dt, sparse=True)
            _record_detail(record_matvec_detail, total_dt, sparse=True, boundary=boundary, phase="total")
            return out.reshape(-1)

    elif boundary == "mid":
        Lkm1, corek, corek1, Rkp2, psi_shape = op_args
        n = int(np.prod(psi_shape, dtype=np.int64))
        Wk = corek["core"]
        Wk1 = corek1["core"]
        if sps.issparse(Wk) and not sps.isspmatrix_csr(Wk):
            Wk = Wk.tocsr(copy=False)
        if sps.issparse(Wk1) and not sps.isspmatrix_csr(Wk1):
            Wk1 = Wk1.tocsr(copy=False)
        D = int(corek["shape"][0])
        d = int(corek["shape"][1])
        E = int(corek["shape"][2])
        l = int(corek["shape"][3])
        k = int(corek1["shape"][1])
        F = int(corek1["shape"][2])
        j = int(corek1["shape"][3])
        X = int(psi_shape[0])
        y = int(Rkp2.shape[0])
        Rkp2_rhs_T, Rkp2_prefix = _prepare_last_axis_rhs(Rkp2)

        Lmat = np.ascontiguousarray(Lkm1.transpose(0, 2, 1).reshape(Lkm1.shape[0], -1))
        Lmat_nnz = int(np.count_nonzero(Lmat))
        Lmat_size = int(Lmat.size)
        Lmat_density = float(Lmat_nnz / max(1, Lmat_size))
        Lmat_sparse = None
        if Lmat.size:
            if Lmat_density <= float(env_sparse_threshold):
                Lmat_sparse = sps.csr_matrix(Lmat)
        env_sym_blocks = _build_env_symmetry_blocks(Lkm1.shape, env_qbond, env_qW)
        Wk1_blocks = None
        Wk_blocks = None
        if bool(use_z2_block_oracle) and (qW_left is not None) and (qW_mid is not None) and (qW_right is not None):
            Wk1_blocks = _build_core_symmetry_blocks(
                qW_mid, k, qW_right, j, qphys_row=qphys, qphys_col=qphys
            )
            Wk_blocks = _build_core_symmetry_blocks(
                qW_left, d, qW_mid, l, qphys_row=qphys, qphys_col=qphys
            )
        Wk1_block_ops = _build_validated_block_ops(Wk1, Wk1_blocks, tol=1e-13)
        Wk_block_ops = _build_validated_block_ops(Wk, Wk_blocks, tol=1e-13)
        Lmat_blocks_dense = None
        Lmat_blocks_sparse = None
        if env_sym_blocks is not None:
            Lmat_blocks_dense = []
            Lmat_blocks_sparse = [] if Lmat_sparse is not None else None
            for out_idx, in_idx in env_sym_blocks:
                Lmat_blocks_dense.append(Lmat[np.ix_(out_idx, in_idx)])
                if Lmat_blocks_sparse is not None:
                    Lmat_blocks_sparse.append(Lmat_sparse[out_idx][:, in_idx].tocsr())
        _record_setup(
            record_op_setup,
            boundary=boundary,
            n=int(n),
            mpo_left_sparse=bool(sps.issparse(Wk)),
            mpo_right_sparse=bool(sps.issparse(Wk1)),
            env_storage="dense",
            env_nnz=int(Lmat_nnz),
            env_size=int(Lmat_size),
            env_density=float(Lmat_density),
            env_use_sparse=bool(Lmat_sparse is not None),
            env_sparse_threshold=float(env_sparse_threshold),
            env_use_symmetry=bool(env_sym_blocks is not None),
            mpo_use_symmetry=bool((Wk1_block_ops is not None) and (Wk_block_ops is not None)),
        )
        b1_buf = np.empty((F * j, X * l * y), dtype=dtype_eff)
        b2_buf = np.empty((E * l, X * y * k), dtype=dtype_eff)
        b3_buf = np.empty((X * D, y * k * d), dtype=dtype_eff)
        out_ring = [np.empty(psi_shape, dtype=dtype_eff) for _ in range(2)]
        out_idx = 0
        u1_buf = np.empty((int(Wk1.shape[0]), X * l * y), dtype=dtype_eff) if Wk1_block_ops is not None else None
        u2_buf = np.empty((int(Wk.shape[0]), X * y * k), dtype=dtype_eff) if Wk_block_ops is not None else None
        env_buf = np.empty((int(Lkm1.shape[0]), y * k * d), dtype=dtype_eff) if env_sym_blocks is not None else None

        def _mv(v):
            nonlocal out_idx
            t_mv0 = _timer_now(timer)
            p = np.asarray(v).reshape(psi_shape)
            t0 = _timer_now(timer)
            # Specialized hot contraction: (X,d,k,j) x (y,F,j) over j -> (X,d,k,y,F)
            t1 = _contract_last_axis_prepared(p, Rkp2_rhs_T, Rkp2_prefix)
            t1_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, t1_dt, sparse=True, boundary=boundary, phase="contract_1")

            t0 = _timer_now(timer)
            np.copyto(b1_buf, t1.transpose(4, 2, 0, 1, 3).reshape(F * j, X * l * y))
            if Wk1_blocks is not None and Wk1_block_ops is not None:
                u1 = _sym_block_mm_precomputed(
                    b1_buf, Wk1_blocks, Wk1_block_ops, nrow=Wk1.shape[0], dtype_eff=dtype_eff, out=u1_buf
                )
                sp1_phase = "sparse_mm_1_z2"
            else:
                u1 = np.asarray(Wk1 @ b1_buf)
                sp1_phase = "sparse_mm_1"
            t2 = u1.reshape(E, k, X, l, y).transpose(2, 3, 4, 0, 1)
            sp1_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, sp1_dt, sparse=True, boundary=boundary, phase=sp1_phase)

            t0 = _timer_now(timer)
            np.copyto(b2_buf, t2.transpose(3, 1, 0, 2, 4).reshape(E * l, X * y * k))
            if Wk_blocks is not None and Wk_block_ops is not None:
                u2 = _sym_block_mm_precomputed(
                    b2_buf, Wk_blocks, Wk_block_ops, nrow=Wk.shape[0], dtype_eff=dtype_eff, out=u2_buf
                )
                sp2_phase = "sparse_mm_2_z2"
            else:
                u2 = np.asarray(Wk @ b2_buf)
                sp2_phase = "sparse_mm_2"
            t3 = u2.reshape(D, d, X, y, k).transpose(2, 3, 4, 0, 1)
            sp2_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, sp2_dt, sparse=True, boundary=boundary, phase=sp2_phase)

            t0 = _timer_now(timer)
            np.copyto(b3_buf, t3.transpose(0, 3, 1, 2, 4).reshape(X * D, y * k * d))
            if env_sym_blocks is not None and Lmat_blocks_dense is not None:
                u3 = env_buf
                u3.fill(0.0)
                for bi, (env_out_idx, in_idx) in enumerate(env_sym_blocks):
                    if env_out_idx.size == 0 or in_idx.size == 0:
                        continue
                    rhs = b3_buf[in_idx, :]
                    if Lmat_blocks_sparse is not None:
                        u3[env_out_idx, :] = np.asarray(Lmat_blocks_sparse[bi] @ rhs)
                    else:
                        u3[env_out_idx, :] = np.asarray(Lmat_blocks_dense[bi] @ rhs)
                env_phase = "env_mm_z2"
            else:
                if Lmat_sparse is None:
                    u3 = np.asarray(Lmat @ b3_buf)
                    env_phase = "env_mm_dense"
                else:
                    u3 = np.asarray(Lmat_sparse @ b3_buf)
                    env_phase = "env_mm_sparse"
            out = out_ring[out_idx]
            out_idx = (out_idx + 1) % len(out_ring)
            np.copyto(out, u3.reshape(Lkm1.shape[0], y, k, d).transpose(0, 3, 2, 1))
            dmm_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, dmm_dt, sparse=True, boundary=boundary, phase=env_phase)
            total_dt = _timer_now(timer) - t_mv0
            _record(record_matvec, total_dt, sparse=True)
            _record_detail(record_matvec_detail, total_dt, sparse=True, boundary=boundary, phase="total")
            return out.reshape(-1)

    elif boundary == "last":
        LNm3, coreNm2, coreNm1, psi_shape = op_args
        n = int(np.prod(psi_shape, dtype=np.int64))
        WNm2 = coreNm2["core"]
        WNm1 = coreNm1["core"]
        if sps.issparse(WNm2) and not sps.isspmatrix_csr(WNm2):
            WNm2 = WNm2.tocsr(copy=False)
        if sps.issparse(WNm1) and not sps.isspmatrix_csr(WNm1):
            WNm1 = WNm1.tocsr(copy=False)
        D = int(coreNm2["shape"][0])
        d = int(coreNm2["shape"][1])
        E = int(coreNm2["shape"][2])
        l = int(psi_shape[1])
        k = int(coreNm1["shape"][1])
        j = int(coreNm1["shape"][2])
        b1_buf = np.empty((j, psi_shape[0] * psi_shape[1]), dtype=dtype_eff)
        b2_buf = np.empty((E * psi_shape[1], psi_shape[0] * k), dtype=dtype_eff)
        b3_buf = np.empty((psi_shape[0] * D, k * d), dtype=dtype_eff)

        Lmat = np.ascontiguousarray(LNm3.transpose(0, 2, 1).reshape(LNm3.shape[0], -1))
        Lmat_nnz = int(np.count_nonzero(Lmat))
        Lmat_size = int(Lmat.size)
        Lmat_density = float(Lmat_nnz / max(1, Lmat_size))
        Lmat_sparse = None
        if Lmat.size:
            if Lmat_density <= float(env_sparse_threshold):
                Lmat_sparse = sps.csr_matrix(Lmat)
        env_sym_blocks = _build_env_symmetry_blocks(LNm3.shape, env_qbond, env_qW)
        WNm1_blocks = None
        WNm2_blocks = None
        if bool(use_z2_block_oracle) and (qW_left is not None) and (qW_mid is not None):
            # WNm1: rows=(E,k), cols=(j)
            WNm1_blocks = _build_core_symmetry_blocks(
                qW_mid, k, np.asarray([0], dtype=np.int8), j, qphys_row=qphys, qphys_col=qphys
            )
            # WNm2: rows=(D,d), cols=(E,l)
            WNm2_blocks = _build_core_symmetry_blocks(
                qW_left, d, qW_mid, l, qphys_row=qphys, qphys_col=qphys
            )
        WNm1_block_ops = _build_validated_block_ops(WNm1, WNm1_blocks, tol=1e-13)
        WNm2_block_ops = _build_validated_block_ops(WNm2, WNm2_blocks, tol=1e-13)
        Lmat_blocks_dense = None
        Lmat_blocks_sparse = None
        if env_sym_blocks is not None:
            Lmat_blocks_dense = []
            Lmat_blocks_sparse = [] if Lmat_sparse is not None else None
            for out_idx, in_idx in env_sym_blocks:
                Lmat_blocks_dense.append(Lmat[np.ix_(out_idx, in_idx)])
                if Lmat_blocks_sparse is not None:
                    Lmat_blocks_sparse.append(Lmat_sparse[out_idx][:, in_idx].tocsr())
        _record_setup(
            record_op_setup,
            boundary=boundary,
            n=int(n),
            mpo_left_sparse=bool(sps.issparse(WNm2)),
            mpo_right_sparse=bool(sps.issparse(WNm1)),
            env_storage="dense",
            env_nnz=int(Lmat_nnz),
            env_size=int(Lmat_size),
            env_density=float(Lmat_density),
            env_use_sparse=bool(Lmat_sparse is not None),
            env_sparse_threshold=float(env_sparse_threshold),
            env_use_symmetry=bool(env_sym_blocks is not None),
            mpo_use_symmetry=bool((WNm1_block_ops is not None) and (WNm2_block_ops is not None)),
        )
        out_ring = [np.empty(psi_shape, dtype=dtype_eff) for _ in range(2)]
        out_idx = 0
        u1_buf = np.empty((int(WNm1.shape[0]), psi_shape[0] * psi_shape[1]), dtype=dtype_eff) if WNm1_block_ops is not None else None
        u2_buf = np.empty((int(WNm2.shape[0]), psi_shape[0] * k), dtype=dtype_eff) if WNm2_block_ops is not None else None
        env_buf = np.empty((int(LNm3.shape[0]), k * d), dtype=dtype_eff) if env_sym_blocks is not None else None

        def _mv(v):
            nonlocal out_idx
            t_mv0 = _timer_now(timer)
            p = np.asarray(v).reshape(psi_shape)

            t0 = _timer_now(timer)
            np.copyto(b1_buf, p.transpose(2, 0, 1).reshape(j, psi_shape[0] * psi_shape[1]))
            if WNm1_blocks is not None and WNm1_block_ops is not None:
                u1 = _sym_block_mm_precomputed(
                    b1_buf, WNm1_blocks, WNm1_block_ops, nrow=WNm1.shape[0], dtype_eff=dtype_eff, out=u1_buf
                )
                sp1_phase = "sparse_mm_1_z2"
            else:
                u1 = np.asarray(WNm1 @ b1_buf)
                sp1_phase = "sparse_mm_1"
            t1 = u1.reshape(E, k, psi_shape[0], psi_shape[1]).transpose(2, 3, 0, 1)
            sp1_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, sp1_dt, sparse=True, boundary=boundary, phase=sp1_phase)

            t0 = _timer_now(timer)
            np.copyto(b2_buf, t1.transpose(2, 1, 0, 3).reshape(E * psi_shape[1], psi_shape[0] * k))
            if WNm2_blocks is not None and WNm2_block_ops is not None:
                u2 = _sym_block_mm_precomputed(
                    b2_buf, WNm2_blocks, WNm2_block_ops, nrow=WNm2.shape[0], dtype_eff=dtype_eff, out=u2_buf
                )
                sp2_phase = "sparse_mm_2_z2"
            else:
                u2 = np.asarray(WNm2 @ b2_buf)
                sp2_phase = "sparse_mm_2"
            t2 = u2.reshape(D, d, psi_shape[0], k).transpose(2, 3, 0, 1)
            sp2_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, sp2_dt, sparse=True, boundary=boundary, phase=sp2_phase)

            t0 = _timer_now(timer)
            np.copyto(b3_buf, t2.transpose(0, 2, 1, 3).reshape(psi_shape[0] * D, k * d))
            if env_sym_blocks is not None and Lmat_blocks_dense is not None:
                u3 = env_buf
                u3.fill(0.0)
                for bi, (env_out_idx, in_idx) in enumerate(env_sym_blocks):
                    if env_out_idx.size == 0 or in_idx.size == 0:
                        continue
                    rhs = b3_buf[in_idx, :]
                    if Lmat_blocks_sparse is not None:
                        u3[env_out_idx, :] = np.asarray(Lmat_blocks_sparse[bi] @ rhs)
                    else:
                        u3[env_out_idx, :] = np.asarray(Lmat_blocks_dense[bi] @ rhs)
                env_phase = "env_mm_z2"
            else:
                if Lmat_sparse is None:
                    u3 = np.asarray(Lmat @ b3_buf)
                    env_phase = "env_mm_dense"
                else:
                    u3 = np.asarray(Lmat_sparse @ b3_buf)
                    env_phase = "env_mm_sparse"
            out = out_ring[out_idx]
            out_idx = (out_idx + 1) % len(out_ring)
            np.copyto(out, u3.reshape(LNm3.shape[0], k, d).transpose(0, 2, 1))
            dmm_dt = _timer_now(timer) - t0
            _record_detail(record_matvec_detail, dmm_dt, sparse=True, boundary=boundary, phase=env_phase)
            total_dt = _timer_now(timer) - t_mv0
            _record(record_matvec, total_dt, sparse=True)
            _record_detail(record_matvec_detail, total_dt, sparse=True, boundary=boundary, phase="total")
            return out.reshape(-1)

    else:
        raise ValueError(f"Unknown boundary mode for sparse op: {boundary}")

    op = spla.LinearOperator((n, n), matvec=_mv, dtype=dtype_eff)
    return op, _mv
