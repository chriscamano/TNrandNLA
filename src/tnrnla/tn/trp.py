
# TRP.py
import numpy as np

from .mps import MPS
from .stopping import Cutoff
from .contraction.src_cacheview import SRCViewCache
from .contraction.src import SRC

from .compression.round_src import src_rounding
from .other.TRP_helper import (
    hermitize_inplace,
    lincomb_exact_block,
    SRC_with_tv,
)

class TRP:
    __array_priority__ = 10000

    def __init__(self, mps_list, dtype=None, orthform=None):
        if not isinstance(mps_list, (list, tuple)):
            raise ValueError("TRP expects a list or tuple of MPS objects.")
        if len(mps_list) == 0:
            raise ValueError("TRP requires at least one MPS column.")
        if not all(hasattr(m, "tensors") for m in mps_list):
            raise TypeError("All entries must be MPS instances.")

        n0 = len(mps_list[0])
        A0 = np.asarray(mps_list[0][0])
        d0 = int(A0.shape[0])

        for m in mps_list:
            if len(m) != n0:
                raise ValueError("All MPS entries must have the same number of sites.")
            if int(np.asarray(m[0]).shape[0]) != d0:
                raise ValueError("All MPS entries must have the same physical dimension.")

        self.cols = list(mps_list)
        self._n = int(n0)
        self._d = int(d0)
        self._k = int(len(self.cols))

        if orthform is not None:
            orthform = str(orthform).lower()
            if orthform not in {"up", "down"}:
                raise ValueError("orthform must be 'up', 'down', or None.")
        self.orthform = orthform

        if dtype is None:
            dtypes = []
            for m in self.cols:
                md = getattr(m, "dtype", None)
                if md is None:
                    md = np.asarray(m[0]).dtype
                dtypes.append(np.dtype(md))
            self.dtype = np.result_type(*dtypes)
        else:
            self.dtype = np.dtype(dtype)

        self._mpo_batch_size = 16
        self._invalidate_gram_cache()

    @classmethod
    def _from_parent(cls, parent, cols):
        obj = object.__new__(cls)
        obj.cols = list(cols)
        obj._n = parent._n
        obj._d = parent._d
        obj._k = len(obj.cols)
        obj.orthform = parent.orthform
        obj.dtype = parent.dtype
        obj._mpo_batch_size = getattr(parent, "_mpo_batch_size", 16)
        obj._invalidate_gram_cache()
        return obj

    @property
    def d(self):
        return self._d

    @property
    def n(self):
        return self._n

    @property
    def k(self):
        return self._k

    @property
    def shape(self):
        return (self.d ** self.n, self.k)

    @property
    def T(self):
        return self.dagger()

    def __len__(self):
        return self.k

    def __repr__(self):
        return (
            f"TRP(shape={(self.d ** self.n, self.k)}, "
            f"d={self.d}, n={self.n}, dtype={self.dtype}, orthform={self.orthform})"
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        out = kwargs.get("out", None)
        if out is not None:
            return NotImplemented

        if ufunc is np.multiply:
            a, b = inputs
            if a is self and np.isscalar(b):
                return self * b
            if b is self and np.isscalar(a):
                return self * a
            return NotImplemented

        if ufunc is np.divide:
            a, b = inputs
            if a is self and np.isscalar(b):
                return self / b
            return NotImplemented

        if ufunc is np.add:
            a, b = inputs
            if a is self and hasattr(b, "cols"):
                return self + b
            if b is self and hasattr(a, "cols"):
                return a + self
            return NotImplemented

        if ufunc is np.subtract:
            a, b = inputs
            if a is self and hasattr(b, "cols"):
                return self - b
            if b is self and hasattr(a, "cols"):
                return a - self
            return NotImplemented

        if ufunc is np.negative:
            (x,) = inputs
            if x is self:
                return -self
            return NotImplemented

        return NotImplemented

    def set_mpo_batch_size(self, batch_size):
        self._mpo_batch_size = int(batch_size)
        return self

    def _invalidate_gram_cache(self):
        self._gram_cache_valid = False

        self._gram_cache_left0 = None
        self._gram_cache_left0_conjT = None

        self._gram_cache_mid_mat = None
        self._gram_cache_mid_conj_mat = None
        self._gram_cache_mid_bc = None

        self._gram_cache_right_last = None
        self._gram_cache_right_conj = None
        return self

    def invalidate_ip_cache(self):
        return self._invalidate_gram_cache()

    def invalidate_rhs_cache(self):
        return self._invalidate_gram_cache()

    @staticmethod
    def _prepare_col_gram_data(m):
        N = int(m.N)

        a0 = np.asarray(m[0])
        left0 = a0
        left0_conjT = np.conjugate(a0).T

        mid_mat = []
        mid_conj_mat = []
        mid_bc = []

        for i in range(1, N - 1):
            ai = np.asarray(m[i])
            mid_mat.append(ai.reshape(ai.shape[0], -1))
            mid_conj_mat.append(np.conjugate(ai).reshape(-1, ai.shape[-1]))
            mid_bc.append((ai.shape[1], ai.shape[2]))

        al = np.asarray(m[-1])
        right_last = al
        right_conj = np.ravel(np.conjugate(al))

        return left0, left0_conjT, mid_mat, mid_conj_mat, mid_bc, right_last, right_conj

    def _build_gram_cache(self):
        left0 = []
        left0_conjT = []

        mid_mat = []
        mid_conj_mat = []
        mid_bc = []

        right_last = []
        right_conj = []

        for m in self.cols:
            (
                a0,
                a0_conjT,
                mats,
                mats_conj,
                bcs,
                al,
                al_conj,
            ) = self._prepare_col_gram_data(m)

            left0.append(a0)
            left0_conjT.append(a0_conjT)

            mid_mat.append(mats)
            mid_conj_mat.append(mats_conj)
            mid_bc.append(bcs)

            right_last.append(al)
            right_conj.append(al_conj)

        self._gram_cache_left0 = left0
        self._gram_cache_left0_conjT = left0_conjT

        self._gram_cache_mid_mat = mid_mat
        self._gram_cache_mid_conj_mat = mid_conj_mat
        self._gram_cache_mid_bc = mid_bc

        self._gram_cache_right_last = right_last
        self._gram_cache_right_conj = right_conj

        self._gram_cache_valid = True
        return self

    def _get_gram_cache(self):
        if not self._gram_cache_valid:
            self._build_gram_cache()
        return self

    @staticmethod
    def _inner_product_prepared_pair(
        *,
        self_left0_conjT,
        self_mid_conj_mat,
        self_right_conj,
        rhs_left0,
        rhs_mid_mat,
        rhs_mid_bc,
        rhs_right_last,
    ):
        site = self_left0_conjT @ rhs_left0

        for t in range(len(self_mid_conj_mat)):
            other_mat = rhs_mid_mat[t]
            b, c = rhs_mid_bc[t]

            temp = site @ other_mat
            temp = temp.reshape(site.shape[0], b, c)

            temp_transposed = temp.transpose(2, 0, 1)
            temp_reshaped = temp_transposed.reshape(temp_transposed.shape[0], -1)

            site = (temp_reshaped @ self_mid_conj_mat[t]).T

        site = site @ rhs_right_last
        return np.dot(site.ravel(), self_right_conj)

    @staticmethod
    def _mps_inner_product_direct(lhs, rhs):
        site = np.asarray(lhs[0]).conj().T @ np.asarray(rhs[0])

        for i in range(1, int(lhs.N) - 1):
            rhs_i = np.asarray(rhs[i])
            rhs_mat = rhs_i.reshape(rhs_i.shape[0], -1)

            temp = site @ rhs_mat
            temp = temp.reshape(site.shape[0], rhs_i.shape[1], rhs_i.shape[2])

            temp_transposed = temp.transpose(2, 0, 1)
            temp_reshaped = temp_transposed.reshape(temp_transposed.shape[0], -1)

            lhs_i = np.asarray(lhs[i])
            lhs_mat = lhs_i.conj().reshape(-1, lhs_i.shape[-1])

            site = (temp_reshaped @ lhs_mat).T

        site = site @ np.asarray(rhs[-1])
        return np.dot(site.ravel(), np.asarray(lhs[-1]).conj().ravel())


    @staticmethod
    def _inner_products_prepared_block(
        *,
        self_left0_conjT,
        self_mid_conj_mat,
        self_right_conj,
        rhs_left0_block,
        rhs_mid_mat_block,
        rhs_mid_bc,
        rhs_right_last_block,
    ):
        site = np.matmul(self_left0_conjT, rhs_left0_block)

        for t, lhs_mat in enumerate(self_mid_conj_mat):
            b, c = rhs_mid_bc[t]

            temp = np.matmul(site, rhs_mid_mat_block[t])
            temp = temp.reshape(site.shape[0], site.shape[1], b, c)
            temp = temp.transpose(0, 3, 1, 2).reshape(site.shape[0], c, -1)

            site = np.matmul(temp, lhs_mat).transpose(0, 2, 1)

        site = np.matmul(site, rhs_right_last_block)
        return site.reshape(site.shape[0], -1) @ self_right_conj
    def gram(
        self,
        rhs=None,
        *,
        hermitian=False,
        out_dtype=None,
        use_cache=True,
        assume_hermitian=False,
    ):
        if rhs is None:
            rhs = self
            hermitian = True

        if self.n != rhs.n or self.d != rhs.d:
            raise ValueError("TRP dimension mismatch.")

        k1 = int(self.k)
        k2 = int(rhs.k)

        if (hermitian or assume_hermitian) and k1 != k2:
            raise ValueError("Hermitian Gram requires matching column counts.")

        self_gram = rhs is self
        do_triangular = bool(hermitian) or bool(assume_hermitian)

        if use_cache:
            lhs_obj = self._get_gram_cache()
            rhs_obj = lhs_obj if self_gram else rhs._get_gram_cache()

            lhs_left0_conjT = lhs_obj._gram_cache_left0_conjT
            lhs_mid_conj_mat = lhs_obj._gram_cache_mid_conj_mat
            lhs_right_conj = lhs_obj._gram_cache_right_conj

            rhs_left0 = rhs_obj._gram_cache_left0
            rhs_mid_mat = rhs_obj._gram_cache_mid_mat
            rhs_mid_bc = rhs_obj._gram_cache_mid_bc
            rhs_right_last = rhs_obj._gram_cache_right_last
        else:
            lhs_left0_conjT = []
            lhs_mid_conj_mat = []
            lhs_right_conj = []

            for m in self.cols:
                _, a0_conjT, _, mats_conj, _, _, al_conj = self._prepare_col_gram_data(m)
                lhs_left0_conjT.append(a0_conjT)
                lhs_mid_conj_mat.append(mats_conj)
                lhs_right_conj.append(al_conj)

            rhs_left0 = []
            rhs_mid_mat = []
            rhs_mid_bc = []
            rhs_right_last = []

            for m in rhs.cols:
                a0, _, mats, _, bcs, al, _ = self._prepare_col_gram_data(m)
                rhs_left0.append(a0)
                rhs_mid_mat.append(mats)
                rhs_mid_bc.append(bcs)
                rhs_right_last.append(al)

        if out_dtype is None:
            base = np.result_type(self.dtype, rhs.dtype)
            if np.issubdtype(base, np.complexfloating):
                out_dtype = np.complex128
            else:
                v0 = self._mps_inner_product_direct(self.cols[0], rhs.cols[0])
                out_dtype = np.complex128 if np.iscomplexobj(v0) else np.float64
        else:
            out_dtype = np.dtype(out_dtype)

        mat = np.empty((k1, k2), dtype=out_dtype)
        inner_block = self._inner_products_prepared_block

        groups = {}
        for j, m in enumerate(rhs.cols):
            sig = tuple(np.asarray(m[t]).shape for t in range(self.n))
            groups.setdefault(sig, []).append(j)

        packed_groups = []
        for idxs in groups.values():
            idxs = np.asarray(idxs, dtype=np.int64)

            left0_block = np.stack([rhs_left0[j] for j in idxs], axis=0)
            mid_block = [
                np.stack([rhs_mid_mat[j][t] for j in idxs], axis=0)
                for t in range(max(self.n - 2, 0))
            ]
            right_last_block = np.stack([rhs_right_last[j] for j in idxs], axis=0)

            packed_groups.append(
                (idxs, left0_block, mid_block, rhs_mid_bc[idxs[0]], right_last_block)
            )

        if do_triangular:
            if self_gram:
                for i in range(k1):
                    mat[i, i] = self._mps_inner_product_direct(self.cols[i], self.cols[i])

                for i in range(k1):
                    for idxs, left0_block, mid_block, mid_bc_block, right_last_block in packed_groups:
                        mask = idxs > i
                        if not np.any(mask):
                            continue

                        vals = inner_block(
                            self_left0_conjT=lhs_left0_conjT[i],
                            self_mid_conj_mat=lhs_mid_conj_mat[i],
                            self_right_conj=lhs_right_conj[i],
                            rhs_left0_block=left0_block[mask],
                            rhs_mid_mat_block=[x[mask] for x in mid_block],
                            rhs_mid_bc=mid_bc_block,
                            rhs_right_last_block=right_last_block[mask],
                        )

                        js = idxs[mask]
                        mat[i, js] = vals
                        mat[js, i] = np.conjugate(vals)

                return mat

            for i in range(k1):
                for idxs, left0_block, mid_block, mid_bc_block, right_last_block in packed_groups:
                    mask = idxs >= i
                    if not np.any(mask):
                        continue

                    vals = inner_block(
                        self_left0_conjT=lhs_left0_conjT[i],
                        self_mid_conj_mat=lhs_mid_conj_mat[i],
                        self_right_conj=lhs_right_conj[i],
                        rhs_left0_block=left0_block[mask],
                        rhs_mid_mat_block=[x[mask] for x in mid_block],
                        rhs_mid_bc=mid_bc_block,
                        rhs_right_last_block=right_last_block[mask],
                    )

                    js = idxs[mask]
                    mat[i, js] = vals
                    mat[js, i] = np.conjugate(vals)

            return mat

        for i in range(k1):
            for idxs, left0_block, mid_block, mid_bc_block, right_last_block in packed_groups:
                vals = inner_block(
                    self_left0_conjT=lhs_left0_conjT[i],
                    self_mid_conj_mat=lhs_mid_conj_mat[i],
                    self_right_conj=lhs_right_conj[i],
                    rhs_left0_block=left0_block,
                    rhs_mid_mat_block=mid_block,
                    rhs_mid_bc=mid_bc_block,
                    rhs_right_last_block=right_last_block,
                )
                mat[i, idxs] = vals

        return mat
    



    def __matmul__(self, rhs):
        if isinstance(rhs, (list, tuple, np.ndarray)):
            coeffs = np.asarray(rhs)

            if coeffs.ndim == 2:
                out_dtype = np.result_type(self.dtype, coeffs.dtype)
                C = coeffs.astype(out_dtype, copy=False)

                if C.shape[0] != self.k:
                    raise ValueError(f"Expected coeff matrix with shape ({self.k}, r), got {C.shape}")

                r = int(C.shape[1])
                out_cols = [lincomb_exact_block(self.cols, C[:, j]) for j in range(r)]
                out = TRP._from_parent(self, out_cols)
                out.dtype = out_dtype
                return out

            coeffs = coeffs.ravel()
            out_dtype = np.result_type(self.dtype, coeffs.dtype)
            coeffs = coeffs.astype(out_dtype, copy=False)

            if coeffs.ndim != 1 or coeffs.shape[0] != self.k:
                raise ValueError(f"Expected 1D coeff vector of length {self.k}, got {coeffs.shape}")

            mps_out = lincomb_exact_block(self.cols, coeffs)
            mps_out.round()
            return mps_out

        if isinstance(rhs, MPS):
            rhs_dtype = getattr(rhs, "dtype", None)
            if rhs_dtype is None:
                rhs_dtype = np.asarray(rhs[0]).dtype

            out_dtype = np.result_type(self.dtype, np.dtype(rhs_dtype))
            v = np.empty((self.k,), dtype=out_dtype)
            for i, m in enumerate(self.cols):
                v[i] = self._mps_inner_product_direct(m, rhs)
            return v

        if hasattr(rhs, "tensors"):
            return self.apply_mpo_src(rhs)

        if hasattr(rhs, "cols"):
            if rhs is self:
                return self.gram(None, hermitian=True)
            return self.gram(rhs, hermitian=False)

        raise TypeError(f"Unsupported RHS type for TRP @  {type(rhs)}")

    def apply_mpo_src(
        self,
        mpo,
        *,
        stop=None,
        sketchdim=None,
        sketchincrement=None,
        finalround=False,
        accuracychecks=False,
        dtype=np.float64,
        batch_cols=16,
        bucket=True,
        seed=0,
    ):
        if sketchdim is None:
            sketchdim = min(128, mpo.max_bond() * self[0].max_bond())
        if sketchincrement is None:
            sketchincrement = sketchdim
        if not hasattr(mpo, "tensors"):
            raise TypeError("apply_mpo_src expects an MPO like object with .tensors")
        if int(len(mpo.tensors)) != self.n:
            raise ValueError("MPO and TRP site count mismatch")

        if stop is None:
            stop = Cutoff(1e-14)

        batch_cols = int(batch_cols)
        if batch_cols <= 0:
            raise ValueError("batch_cols must be positive")

        if bucket:
            groups = {}
            for j, m in enumerate(self.cols):
                sig = tuple(np.asarray(m[i]).shape for i in range(self.n))
                groups.setdefault(sig, []).append(j)
        else:
            groups = {None: list(range(self.k))}

        out_cols = [None] * self.k

        def _make_tv(rep_mps):
            tv = SRCViewCache.term_views(mpo, rep_mps, assume_identity=False)

            if tv.H0_flat is not None and not tv.H0_flat.flags["C_CONTIGUOUS"]:
                tv.H0_flat = np.ascontiguousarray(tv.H0_flat)

            if tv.H_env_T_erl is not None:
                for kk in range(len(tv.H_env_T_erl)):
                    a = tv.H_env_T_erl[kk]
                    if not a.flags["C_CONTIGUOUS"]:
                        tv.H_env_T_erl[kk] = np.ascontiguousarray(a)

            if tv.H_sk is not None:
                for kk in range(len(tv.H_sk)):
                    a = tv.H_sk[kk]
                    if not a.flags["C_CONTIGUOUS"]:
                        tv.H_sk[kk] = np.ascontiguousarray(a)

            if tv.H_cap is not None:
                for kk in range(1, self.n - 1):
                    a = tv.H_cap[kk]
                    if a is not None and not a.flags["C_CONTIGUOUS"]:
                        tv.H_cap[kk] = np.ascontiguousarray(a)

            return tv

        def _src_with_tv_batch(tv, idx_chunk):
            cols_chunk = [self.cols[j] for j in idx_chunk]

            results = []
            for col in cols_chunk:
                results.append(
                    SRC_with_tv(
                        tv,
                        mpo,
                        col,
                        stop=stop,
                        sketchdim=int(sketchdim),
                        sketchincrement=int(sketchincrement),
                        finalround=bool(finalround),
                        accuracychecks=bool(accuracychecks),
                        dtype=dtype,
                    )
                )
            return results

        for _, idxs in groups.items():
            rep = self.cols[idxs[0]]
            tv = _make_tv(rep)

            for p in range(0, len(idxs), batch_cols):
                idx_chunk = idxs[p : p + batch_cols]
                res_chunk = _src_with_tv_batch(tv, idx_chunk)
                for j, r in zip(idx_chunk, res_chunk):
                    out_cols[j] = r

        out = TRP._from_parent(self, out_cols)
        out.invalidate_ip_cache()
        out.invalidate_rhs_cache()
        return out

    def src_round(self, stop=None, **kwargs):
        """
        Columnwise SRC compression by applying the identity MPO to each column.

        This mirrors the standalone src_rounding(mps, stop, ...) pattern but runs it
        on every MPS in self.cols and returns self (in place), similar to TRP.round
        and TRP.src_round.
        """
        if stop is None:
            stop_obj = Cutoff(1e-10)
        else:
            if hasattr(stop, "cutoff") or hasattr(stop, "maxdim") or hasattr(stop, "outputdim"):
                stop_obj = stop
            else:
                stop_obj = Cutoff(float(stop))

        finalround = kwargs.pop("finalround", None)
        sketchdim = int(kwargs.pop("sketchdim", 16))
        sketchincrement = int(kwargs.pop("sketchincrement", 16))

        if self.k == 0:
            return self

        from .mpo import MPO
        d = int(np.asarray(self.cols[0][0]).shape[0])
        mpo = MPO.eye(self.n, d=d)

        self.cols = [
            SRC(
                mpo,
                col,
                stop=stop_obj,
                finalround=finalround,
                sketchdim=sketchdim,
                sketchincrement=sketchincrement,
                **kwargs,
            )
            for col in self.cols
        ]

        self.orthform = None
        self.invalidate_ip_cache()
        self.invalidate_rhs_cache()
        return self

    # ======================================================================
    # Batched quadratic forms against an MPO
    # ======================================================================

    @staticmethod
    def _cache_mpo_for_quadform(mpo, dtype):
        """Pre-compute and store every MPO tensor reshape needed for quadform.

        This is called once per (MPO object, dtype) pair.  All k probe
        evaluations then reuse the cached matrices, avoiding redundant
        np.asarray / transpose / reshape work inside the hot loop.
        """
        tensors = mpo.tensors
        N = len(tensors)

        # site 0 — shape (d_bra, Dw_R, d_ket)
        W0 = np.asarray(tensors[0], dtype=dtype, order="C")
        # transpose(2,1,0): (d_ket, Dw_R, d_bra)  →  reshape: (d_ket, Dw_R*d_bra)
        W0_mat = np.ascontiguousarray(
            W0.transpose(2, 1, 0).reshape(W0.shape[2], -1)
        )

        # bulk sites 1..N-2 — shape (Dw_L, d_bra, Dw_R, d_ket)
        Wi_mats   = []
        Wi_shapes = []
        for i in range(1, N - 1):
            Wi = np.asarray(tensors[i], dtype=dtype, order="C")
            # transpose(0,2,1,3): (Dw_L, Dw_R, d_bra, d_ket)
            # reshape: (Dw_L*Dw_R*d_bra, d_ket)
            Wi_mats.append(np.ascontiguousarray(
                Wi.transpose(0, 2, 1, 3).reshape(
                    Wi.shape[0] * Wi.shape[2] * Wi.shape[1], Wi.shape[3]
                )
            ))
            Wi_shapes.append(Wi.shape)

        # last site — shape (Dw_N, d_bra, d_ket)
        WN = np.asarray(tensors[-1], dtype=dtype, order="C")
        # transpose(2,0,1): (d_ket, Dw_N, d_bra)  →  reshape: (d_ket, Dw_N*d_bra)
        WN_mat = np.ascontiguousarray(
            WN.transpose(2, 0, 1).reshape(WN.shape[2], -1)
        )

        return {
            "N":        N,
            "dtype":    dtype,
            "W0_mat":   W0_mat,
            "W0_shape": W0.shape,
            "Wi_mats":  Wi_mats,
            "Wi_shapes":Wi_shapes,
            "WN_mat":   WN_mat,
            "WN_shape": WN.shape,
        }

    # def quadform(self, mpo, dtype=None):
    #     """Compute <col_j | mpo | col_j> for every probe column j.

    #     MPO tensor reshapes are cached by ``(id(mpo), dtype)`` so that
    #     repeated calls with the same MPO — as in ``mps_hutch`` — pay the
    #     setup cost only once.

    #     Parameters
    #     ----------
    #     mpo : MPO
    #     dtype : numpy dtype, optional
    #         Working precision.  Defaults to
    #         ``np.result_type(self.dtype, mpo.dtype)``.

    #     Returns
    #     -------
    #     out : ndarray, shape (k,), dtype float64
    #         ``out[j] = Re(<col_j | mpo | col_j>)``
    #     """
    #     from .mpo import MPO as _MPO_cls
    #     if not isinstance(mpo, _MPO_cls):
    #         raise TypeError(
    #             "TRP.quadform only supports MPO operators; got "
    #             + type(mpo).__name__
    #         )

    #     if dtype is None:
    #         dtype = np.result_type(self.dtype, mpo.dtype)
    #     dtype = np.dtype(dtype)

    #     # rebuild cache only when MPO object or dtype changes
    #     cache_key = (id(mpo), dtype)
    #     if getattr(self, "_qf_mpo_cache_key", None) != cache_key:
    #         self._qf_mpo_cache     = self._cache_mpo_for_quadform(mpo, dtype)
    #         self._qf_mpo_cache_key = cache_key

    #     cache     = self._qf_mpo_cache
    #     N         = cache["N"]
    #     W0_mat    = cache["W0_mat"]
    #     W0_shape  = cache["W0_shape"]
    #     Wi_mats   = cache["Wi_mats"]
    #     Wi_shapes = cache["Wi_shapes"]
    #     WN_mat    = cache["WN_mat"]
    #     WN_shape  = cache["WN_shape"]

    #     do_conj = np.issubdtype(dtype, np.complexfloating)
    #     out = np.empty(self._k, dtype=np.float64)

    #     for j, v in enumerate(self.cols):
    #         # ── left boundary (site 0) ────────────────────────────────────
    #         K0 = np.asarray(v.tensors[0], dtype=dtype, order="C")   # (d, Dk0)
    #         B0 = np.conjugate(K0) if do_conj else K0                # (d, Db0)

    #         # (Dk0, d) @ (d, Dw_R*d) → (Dk0, Dw_R, d) → transpose → (d, Dw_R, Dk0)
    #         T0 = (K0.T @ W0_mat).reshape(
    #             K0.shape[1], W0_shape[1], W0_shape[0]
    #         ).transpose(2, 1, 0)

    #         # (Db0, d) @ (d, Dw_R*Dk0) → (Db0, Dw_R, Dk0)
    #         E = (B0.T @ T0.reshape(W0_shape[0], -1)).reshape(
    #             B0.shape[1], W0_shape[1], K0.shape[1]
    #         )

    #         # ── bulk sites ────────────────────────────────────────────────
    #         for site_idx, (Wi_mat, Wi_shape) in enumerate(zip(Wi_mats, Wi_shapes)):
    #             Ki = np.asarray(
    #                 v.tensors[site_idx + 1], dtype=dtype, order="C"
    #             )                                                    # (DkL, d, DkR)
    #             Bi = np.conjugate(Ki) if do_conj else Ki

    #             DwL, _, DwR, _ = Wi_shape
    #             DkL, _d, DkR   = Ki.shape

    #             # (d, DkL*DkR)
    #             Ki_mat = Ki.transpose(1, 0, 2).reshape(_d, DkL * DkR)
    #             # (DwL*DwR*d, DkL*DkR)
    #             T_raw  = Wi_mat @ Ki_mat
    #             # → (DwL, DkL, DwR, d, DkR)
    #             T = T_raw.reshape(DwL, DwR, Wi_shape[1], DkL, DkR).transpose(
    #                 0, 3, 1, 2, 4
    #             )

    #             DbL = E.shape[0]
    #             DbR = Bi.shape[2]

    #             # (DwL*DkL, DbL)
    #             E_mat  = E.transpose(1, 2, 0).reshape(DwL * DkL, DbL)
    #             # (DbL, d*DbR)
    #             Bi_mat = Bi.reshape(DbL, Wi_shape[1] * DbR)
    #             # (DwL, DkL, d, DbR)
    #             U = (E_mat @ Bi_mat).reshape(DwL, DkL, Wi_shape[1], DbR)

    #             # (DbR, DwL*DkL*d)
    #             U_mat = U.transpose(3, 0, 1, 2).reshape(
    #                 DbR, DwL * DkL * Wi_shape[1]
    #             )
    #             # (DwL*DkL*d, DwR*DkR)
    #             T_mat = T.transpose(0, 1, 3, 2, 4).reshape(
    #                 DwL * DkL * Wi_shape[1], DwR * DkR
    #             )
    #             # (DbR, DwR, DkR)
    #             E = (U_mat @ T_mat).reshape(DbR, DwR, DkR)

    #         # ── right boundary (last site) ────────────────────────────────
    #         KN = np.asarray(v.tensors[-1], dtype=dtype, order="C")  # (DkN, d)
    #         BN = np.conjugate(KN) if do_conj else KN

    #         DwN = WN_shape[0]
    #         DkN = KN.shape[0]

    #         # (DkN, d) @ (d, DwN*d_bra) → (DkN, DwN, d_bra) → (DwN, DkN, d_bra)
    #         Tn = (KN @ WN_mat).reshape(DkN, DwN, WN_shape[1]).transpose(1, 0, 2)

    #         # (DbN, DwN*DkN) @ (DwN*DkN, d_bra) → (DbN, d_bra)
    #         V = E.reshape(BN.shape[0], DwN * DkN) @ Tn.reshape(
    #             DwN * DkN, WN_shape[1]
    #         )

    #         out[j] = float(np.real_if_close((V * BN).sum()))

    #     return out
    def quadform(self, mpo, dtype=None):
        """Compute <col_j | mpo | col_j> for every probe column j.

        MPO tensor reshapes are cached by ``(id(mpo), dtype)`` so that
        repeated calls with the same MPO, as in ``mps_hutch``, pay the
        setup cost only once.

        Parameters
        ----------
        mpo : MPO
        dtype : numpy dtype, optional
            Working precision. Defaults to
            ``np.result_type(self.dtype, mpo.dtype)``.

        Returns
        -------
        out : ndarray, shape (k,), dtype float64
            ``out[j] = Re(<col_j | mpo | col_j>)``
        """
        from .mpo import MPO as _MPO_cls
        if not isinstance(mpo, _MPO_cls):
            raise TypeError(
                "TRP.quadform only supports MPO operators; got "
                + type(mpo).__name__
            )

        if dtype is None:
            dtype = np.result_type(self.dtype, mpo.dtype)
        dtype = np.dtype(dtype)

        # rebuild cache only when MPO object or dtype changes
        cache_key = (id(mpo), dtype)
        if getattr(self, "_qf_mpo_cache_key", None) != cache_key:
            self._qf_mpo_cache = self._cache_mpo_for_quadform(mpo, dtype)
            self._qf_mpo_cache_key = cache_key

        cache = self._qf_mpo_cache
        N = cache["N"]
        W0_mat = cache["W0_mat"]
        W0_shape = cache["W0_shape"]
        Wi_mats = cache["Wi_mats"]
        Wi_shapes = cache["Wi_shapes"]
        WN_mat = cache["WN_mat"]
        WN_shape = cache["WN_shape"]

        do_conj = np.issubdtype(dtype, np.complexfloating)
        out = np.empty(self._k, dtype=np.float64)

        for j, v in enumerate(self.cols):
            # left boundary
            K0 = np.asarray(v.tensors[0], dtype=dtype, order="C")                                   # (d,Dk0)
            B0 = np.conjugate(K0) if do_conj else K0                                                # (d,Dk0)

            T0 = (K0.T @ W0_mat).reshape(K0.shape[1], W0_shape[1], W0_shape[0]).transpose(2, 1, 0) # (Dk0,d)@(d,DwR*d)->(Dk0,DwR*d)->(Dk0,DwR,d)->(d,DwR,Dk0)

            E = (B0.T @ T0.reshape(W0_shape[0], -1)).reshape(B0.shape[1], W0_shape[1], K0.shape[1]) # (Dk0,d)@(d,DwR*Dk0)->(Dk0,DwR*Dk0)->(Dk0,DwR,Dk0)

            # bulk sites
            for site_idx, (Wi_mat, Wi_shape) in enumerate(zip(Wi_mats, Wi_shapes)):
                Ki = np.asarray(v.tensors[site_idx + 1], dtype=dtype, order="C")                   # (DkL,d,DkR)
                Bi = np.conjugate(Ki) if do_conj else Ki                                            # (DkL,d,DkR)

                DwL, _, DwR, _ = Wi_shape
                DkL, _d, DkR = Ki.shape

                Ki_mat = Ki.transpose(1, 0, 2).reshape(_d, DkL * DkR)                               # (DkL,d,DkR)->(d,DkL,DkR)->(d,DkL*DkR)
                T_raw = Wi_mat @ Ki_mat                                                             # (DwL*DwR*d,d)@(d,DkL*DkR)->(DwL*DwR*d,DkL*DkR)
                T = T_raw.reshape(DwL, DwR, Wi_shape[1], DkL, DkR).transpose(0, 3, 1, 2, 4)        # (DwL*DwR*d,DkL*DkR)->(DwL,DwR,d,DkL,DkR)->(DwL,DkL,DwR,d,DkR)

                DbL = E.shape[0]
                DbR = Bi.shape[2]

                E_mat = E.transpose(1, 2, 0).reshape(DwL * DkL, DbL)                                # (DbL,DwL,DkL)->(DwL,DkL,DbL)->(DwL*DkL,DbL)
                Bi_mat = Bi.reshape(DbL, Wi_shape[1] * DbR)                                         # (DbL,d,DbR)->(DbL,d*DbR)
                U = (E_mat @ Bi_mat).reshape(DwL, DkL, Wi_shape[1], DbR)                            # (DwL*DkL,DbL)@(DbL,d*DbR)->(DwL*DkL,d*DbR)->(DwL,DkL,d,DbR)

                U_mat = U.transpose(3, 0, 1, 2).reshape(DbR, DwL * DkL * Wi_shape[1])               # (DwL,DkL,d,DbR)->(DbR,DwL,DkL,d)->(DbR,DwL*DkL*d)
                T_mat = T.transpose(0, 1, 3, 2, 4).reshape(DwL * DkL * Wi_shape[1], DwR * DkR)     # (DwL,DkL,DwR,d,DkR)->(DwL,DkL,d,DwR,DkR)->(DwL*DkL*d,DwR*DkR)
                E = (U_mat @ T_mat).reshape(DbR, DwR, DkR)                                           # (DbR,DwL*DkL*d)@(DwL*DkL*d,DwR*DkR)->(DbR,DwR*DkR)->(DbR,DwR,DkR)

            # right boundary
            KN = np.asarray(v.tensors[-1], dtype=dtype, order="C")                                  # (DkN,d)
            BN = np.conjugate(KN) if do_conj else KN                                                # (DkN,d)

            DwN = WN_shape[0]
            DkN = KN.shape[0]

            Tn = (KN @ WN_mat).reshape(DkN, DwN, WN_shape[1]).transpose(1, 0, 2)                    # (DkN,d)@(d,DwN*d)->(DkN,DwN*d)->(DkN,DwN,d)->(DwN,DkN,d)

            V = E.reshape(BN.shape[0], DwN * DkN) @ Tn.reshape(DwN * DkN, WN_shape[1])              # (DbR,DwN,DkN)->(DbR,DwN*DkN), (DwN,DkN,d)->(DwN*DkN,d), (DbR,DwN*DkN)@(DwN*DkN,d)->(DbR,d)

            out[j] = float(np.real_if_close((V * BN).sum()))                                         # (DbR,d)*(DkN,d)->scalar

        return out

    def __add__(self, other):
        if not hasattr(other, "cols"):
            return NotImplemented
        if self.n != other.n or self.d != other.d or self.k != other.k:
            raise ValueError("TRP dimension mismatch in addition.")
        cols = [a + b for a, b in zip(self.cols, other.cols)]
        out = TRP._from_parent(self, cols)
        out.invalidate_ip_cache()
        out.invalidate_rhs_cache()
        return out

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        if not hasattr(other, "cols"):
            return NotImplemented
        if self.n != other.n or self.d != other.d or self.k != other.k:
            raise ValueError("TRP dimension mismatch in subtraction.")
        cols = [a - b for a, b in zip(self.cols, other.cols)]
        out = TRP._from_parent(self, cols)
        out.invalidate_ip_cache()
        out.invalidate_rhs_cache()
        return out

    def __neg__(self):
        return (-1) * self

    def copy(self):
        return TRP._from_parent(self, self.cols.copy())

    def deepcopy(self):
        cloned = []
        for m in self.cols:
            new_tensors = [np.array(t, copy=True) for t in m.tensors]
            cloned.append(
                type(m)(
                    new_tensors,
                    orthform=getattr(m, "orthform", None),
                    rounded=getattr(m, "rounded", False),
                    dtype=getattr(m, "dtype", None),
                )
            )
        out = TRP._from_parent(self, cloned)
        out.invalidate_ip_cache()
        out.invalidate_rhs_cache()
        return out

    def to_dense(self):
        cols = []
        for j in range(self.k):
            v = np.asarray(self.cols[j].to_dense()).reshape(-1)
            cols.append(v)
        return np.column_stack(cols)

    def norm(self, ord="fro"):
        if ord != "fro":
            raise ValueError("TRP.norm only supports ord='fro'.")
        s2 = 0.0
        for j in range(self.k):
            nj = float(self.cols[j].norm())
            s2 += nj * nj
        return float(np.sqrt(max(s2, 0.0)))

    def round(self, stop=Cutoff(1e-14), **kwargs):
        for m in self.cols:
            m.round(stop=stop, **kwargs)
        self.orthform = None
        self.invalidate_ip_cache()
        self.invalidate_rhs_cache()
        return self

    def __getitem__(self, j):
        if isinstance(j, slice):
            return TRP._from_parent(self, self.cols[j])
        if not isinstance(j, int):
            raise TypeError("Index must be int or slice.")
        if not (0 <= j < self.k):
            raise IndexError("column index out of range.")
        return self.cols[j]

    def __setitem__(self, j, value):
        if not (0 <= j < self.k):
            raise IndexError("column index out of range.")
        if not hasattr(value, "tensors"):
            raise TypeError("Assigned value must be an MPS.")
        if len(value) != self.n:
            raise ValueError("Assigned MPS must have the same number of sites.")
        if int(np.asarray(value[0]).shape[0]) != self.d:
            raise ValueError("Assigned MPS must have the same physical dimension.")
        self.cols[j] = value
        self.invalidate_ip_cache()
        self.invalidate_rhs_cache()

    def astype(self, dtype):
        dtype = np.dtype(dtype)
        casted = []
        for m in self.cols:
            new_tensors = [np.asarray(t, dtype=dtype) for t in m.tensors]
            casted.append(
                type(m)(
                    new_tensors,
                    orthform=getattr(m, "orthform", None),
                    rounded=getattr(m, "rounded", False),
                    dtype=dtype,
                )
            )
        new = TRP._from_parent(self, casted)
        new.dtype = dtype
        new.invalidate_ip_cache()
        new.invalidate_rhs_cache()
        return new

    def dagger(self):
        if not all(hasattr(m, "dagger") for m in self.cols):
            raise AttributeError("All MPS columns must implement dagger().")
        dag_cols = [m.dagger() for m in self.cols]
        out = TRP._from_parent(self, dag_cols)
        out.invalidate_ip_cache()
        out.invalidate_rhs_cache()
        return out

    def orthRL(self):
        for j, m in enumerate(self.cols):
            m.orthL()
        self.orthform = "down"
        self.invalidate_ip_cache()
        self.invalidate_rhs_cache()
        return self

    def orthLR(self):
        for j, m in enumerate(self.cols):
            m.orthR()
        self.orthform = "up"
        self.invalidate_ip_cache()
        self.invalidate_rhs_cache()
        return self

    def orth(self, kind=None):
        if kind is None:
            return self
        key = str(kind).strip().lower()
        if key in {"left", "lr", "up"}:
            return self.orthLR()
        if key in {"right", "rl", "down"}:
            return self.orthRL()
        raise ValueError("kind must be one of {left, lr, up, right, rl, down}.")

    @classmethod
    def gaussian(
        cls,
        n_sites,
        k,
        chi,
        *,
        d=2,
        seed=None,
        rng=None,
        dtype=None,
        orth=False,
        normalize=False,
    ):
        n_sites = int(n_sites)
        k = int(k)
        chi = int(chi)
        d = int(d)

        if seed is not None and rng is not None:
            raise ValueError("Provide at most one of 'seed' or 'rng'.")

        if rng is not None:
            seed = rng.integers(0, 2**63, dtype=np.uint64)

        ss = np.random.SeedSequence(seed)
        child_ss = ss.spawn(k)

        cols = []
        for j in range(k):
            col_rng = np.random.default_rng(child_ss[j])
            m = MPS.rmps(n_sites, chi, d=d, rng=col_rng)

            if orth:
                m.orthR()
                m.orthL()

            if normalize:
                m.normalize()

            cols.append(m)

        trp = cls(cols, dtype=dtype, orthform="up")
        if dtype is not None:
            trp = trp.astype(dtype)

        trp *= float(1.0 / np.sqrt(k))
        return trp

    def __imul__(self, scalar):
        if not np.isscalar(scalar):
            raise TypeError("TRP can only be multiplied by a scalar")
        for j in range(self.k):
            self.cols[j] *= scalar
        self.dtype = np.result_type(self.dtype, np.asarray(scalar).dtype)
        self.orthform = None
        self.invalidate_ip_cache()
        self.invalidate_rhs_cache()
        return self

    def __mul__(self, scalar):
        if not np.isscalar(scalar):
            raise TypeError("TRP can only be multiplied by a scalar")
        new_cols = []
        for j in range(self.k):
            new_cols.append(self.cols[j] * scalar)
        out = TRP._from_parent(self, new_cols)
        out.dtype = np.result_type(self.dtype, np.asarray(scalar).dtype)
        out.orthform = None
        out.invalidate_ip_cache()
        out.invalidate_rhs_cache()
        return out

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __itruediv__(self, scalar):
        if not np.isscalar(scalar):
            raise TypeError("TRP can only be divided by a scalar")
        for j in range(self.k):
            self.cols[j] /= scalar
        self.dtype = np.result_type(self.dtype, np.asarray(scalar).dtype)
        self.orthform = None
        self.invalidate_ip_cache()
        self.invalidate_rhs_cache()
        return self

    def __truediv__(self, scalar):
        if not np.isscalar(scalar):
            raise TypeError("TRP can only be divided by a scalar")
        new_cols = []
        for j in range(self.k):
            new_cols.append(self.cols[j] / scalar)
        out = TRP._from_parent(self, new_cols)
        out.dtype = np.result_type(self.dtype, np.asarray(scalar).dtype)
        out.orthform = None
        out.invalidate_ip_cache()
        out.invalidate_rhs_cache()
        return out