import numpy as np
import copy
import math
import numpy.linalg as la

from ..linalg.orth import lq
from ..linalg.lra import truncated_svd
from .stopping import Cutoff, no_truncation,FixedDimension
from .tn1d import TN1D  
from .mps import MPS, ZpMPS
from .other.tensor_ops import flatten_left, flatten_right
# ======================================================================
"""
Matrix product operator class: Caltech Copyright Chris Camaño, Ethan N. Epperly
2023-2025
"""
# ======================================================================

class MPO(TN1D):
    __array_priority__ = 10_000

    def __init__(self, mpo, orthform="None", rounded=False, dtype=None, mv_alg="SRC"):
        super().__init__(mpo, orthform=orthform, rounded=rounded, dtype=dtype)
        self._tn_kind = "mpo"
        allowed = {"zipup", "SRC", "SRC_inc", "CTC", "DM", "fit"}
        if mv_alg not in allowed:
            raise ValueError(f"mv_alg must be in {sorted(allowed)}")
        self._mv_alg = mv_alg
        self.pivot_idx = None
        self.state = ["n"] * self.N

    # ======================================================================
    # Helpers
    # ======================================================================
    def set_mv(self, mv_alg: str):
        allowed = {"zipup", "SRC", "SRC_inc", "CTC", "DM", "fit"}
        if mv_alg not in allowed:
            raise ValueError(f"mv_alg must be in {sorted(allowed)}")
        self._mv_alg = mv_alg

    def _fast_clone(self):
        return MPO(
            [A.copy() for A in self.tensors],
            orthform=self.orthform,
            rounded=self.rounded,
            dtype=self.dtype,
            mv_alg=self._mv_alg,
        )

    def astype(self, dtype, copy_=True):
        return super().astype(np.dtype(dtype), copy=copy_)

    # ======================================================================
    # Algebraic ops
    # ======================================================================
    def add(self, mpo2, subtract=False, compress=False, stop=Cutoff(1e-14)):
        mpo = []
        out_dtype = np.result_type(self.dtype, mpo2.dtype)

        assert self[0].shape[0] == mpo2[0].shape[0]
        assert self[0].shape[2] == mpo2[0].shape[2]

        m1 = self[0].shape[1]
        m2 = mpo2[0].shape[1]
        new_left = np.zeros((self[0].shape[0], m1 + m2, self[0].shape[2]), dtype=out_dtype)
        new_left[:, 0:m1, :] = self[0].astype(out_dtype, copy=False)
        new_left[:, m1:, :] = mpo2[0].astype(out_dtype, copy=False)
        mpo.append(new_left)
        
        for i in range(1, self.N - 1):
            assert self[i].shape[1] == mpo2[i].shape[1]
            assert self[i].shape[3] == mpo2[i].shape[3]

            m1 = self[i].shape[0]
            n1 = self[i].shape[2]
            m2 = mpo2[i].shape[0]
            n2 = mpo2[i].shape[2]
            d = self[i].shape[1]
            d2 = self[i].shape[3]
            new_core = np.zeros((m1 + m2, d, n1 + n2, d2), dtype=out_dtype)
            new_core[0:m1, :, 0:n1, :] = self[i].astype(out_dtype, copy=False)
            new_core[m1:, :, n1:, :] = mpo2[i].astype(out_dtype, copy=False)
            mpo.append(new_core)

        assert self[-1].shape[1] == mpo2[-1].shape[1]
        assert self[-1].shape[2] == mpo2[-1].shape[2]
        m1 = self[-1].shape[0]
        m2 = mpo2[-1].shape[0]
        new_right = np.zeros((m1 + m2, self[-1].shape[1], self[-1].shape[2]), dtype=out_dtype)
        new_right[0:m1, :, :] = self[-1].astype(out_dtype, copy=False)
        new_right[m1:, :, :] = ((-1 if subtract else 1) * mpo2[-1]).astype(out_dtype, copy=False)
        mpo.append(new_right)

        out = MPO(mpo, dtype=out_dtype, mv_alg=self._mv_alg)
        out.orthform = "None"
        out.pivot_idx = None

        if compress:
            out.round(stop=stop)

        return out

    def __add__(self, other):
        if not isinstance(other, MPO):
            return NotImplemented
        return self.add(other, subtract=False, compress=False)

    def __sub__(self, other):
        if not isinstance(other, MPO):
            return NotImplemented
        return self.add(other, subtract=True, compress=False)

    def add_(self, other, subtract=False, compress=False, stop=Cutoff(1e-14)):
        return self._add_inplace_shared(
            other,
            subtract=subtract,
            compress=compress,
            stop=stop,
            extra_attrs=("_mv_alg",),
        )


    def sub_(self, other, compress=True, stop=Cutoff(1e-14)):
        return self.add_(other, subtract=True, compress=compress, stop=stop)
    
    def __iadd__(self, other):
        return self.add_(other, subtract=False, compress=False, stop=Cutoff(1e-14))
    
    def __isub__(self, other):
        return self.add_(other, subtract=True, compress=False, stop=Cutoff(1e-14))
    

    def add_hermitian(self, other, compress=False, stop=Cutoff(1e-14)):
        out = self.add(other, subtract=False, compress=False)
        out = 0.5 * (out.add(out.transpose(), subtract=False, compress=False))
        if compress:
            out.round(stop=stop)
            out = 0.5 * (out.add(out.transpose(), subtract=False, compress=False))
        return out

    def scale_(self, scalar):
        if not np.isscalar(scalar):
            raise TypeError("MPO can only be scaled by a scalar")

        out_dtype = np.result_type(self.dtype, np.asarray(scalar).dtype)
        if out_dtype != self.dtype:
            self.tensors = [t.astype(out_dtype, copy=False) for t in self.tensors]
            self.dtype = out_dtype

        if self.pivot_idx is None:
            self.orthR()
            self[-1] *= scalar
        else:
            self[self.pivot_idx] *= scalar
        self._invalidate_cache()
        return self

    def __imul__(self, alpha):
        return self.scale_(alpha)

    def __mul__(self, alpha):
        if not np.isscalar(alpha):
            return NotImplemented
        out = self._fast_clone()
        return out.scale_(alpha)

    def __rmul__(self, alpha):
        return self.__mul__(alpha)

    def div_(self, scalar):
        if not np.isscalar(scalar):
            raise TypeError("MPO can only be divided by a scalar")
        return self.scale_(1.0 / scalar)

    def __itruediv__(self, other):
        return self.div_(other)

    def __truediv__(self, other):
        if not np.isscalar(other):
            return NotImplemented
        out = self._fast_clone()
        return out.div_(other)

    def __rtruediv__(self, other):
        raise TypeError("Cannot divide a scalar by an MPO")

    def transpose(self):
        do_conj = np.issubdtype(self.dtype, np.complexfloating)
        cores = []
        for i, W in enumerate(self.tensors):
            if W.ndim == 3:
                Wt = W.transpose(2, 1, 0) if i == 0 else W.transpose(0, 2, 1)
            else:
                Wt = W.transpose(0, 3, 2, 1)
            if do_conj:
                Wt = np.conjugate(Wt)
            cores.append(Wt.copy())
        return MPO(cores, dtype=np.result_type(*[c.dtype for c in cores]), mv_alg=self._mv_alg)

    @property
    def T(self):
        return self.transpose()

    # ======================================================================
    # Orthogonalization @ Compression
    # ======================================================================

    def orthR(self):
        if "Left" == self.orthform:
            return
    
        # -------------------------
        # Left boundary
        # -------------------------
        tmp = flatten_left(self[0].transpose(0, 2, 1))                # (d1,d2,D1) -> (d1d2, D1)
        Q, R = la.qr(tmp, mode="reduced")                             # Q:(d1d2,r) R:(r,D1) with r=min(d1d2,D1)
        r = Q.shape[1]
        Q = Q.reshape(self[0].shape[0], self[0].shape[2], r)          # (d1,d2,r)
        self[0] = Q.transpose(0, 2, 1)                                # (d1,r,d2)
    
        # -------------------------
        # Middle tensors
        # -------------------------
        for i in range(1, self.N - 1):
            # absorb R into the left bond of core i
            A = np.einsum("ab,bi cj->ai cj", R, self[i])

            tmp = A.transpose(0, 1, 3, 2)                             # (Dl,d1,Dr,d2) -> (Dl,d1,d2,Dr)
            tmp = flatten_left(tmp)                                    # (Dl*d1*d2, Dr)
    
            Q, R = la.qr(tmp, mode="reduced")                         # Q:(Dl*d1*d2,r) R:(r,Dr)
            r = Q.shape[1]                                            # r = min(Dl*d^2, Dr)
    
            Q = Q.reshape(A.shape[0], A.shape[1], A.shape[3], r)      # (Dl,d1,d2,r)
            self[i] = Q.transpose(0, 1, 3, 2)                         # (Dl,d1,r,d2)
    
        # -------------------------
        # Right boundary
        # -------------------------
        self[-1] = np.einsum("ab,bij->aij", R, self[-1])              # (r,Dr)@(Dr,d,d)->(r,d,d)
        self.orthform = "Right"
        self.pivot_idx = -1
    
    def orthL(self):
        if "Right" == self.orthform:
            return
        
        #Right Boundary
        tmp = flatten_right(self[-1])                                                            # (D_1,d1,d2)-> (D_1,d1d2)
        L, Q = lq(tmp)                                                                           # L:(D1,D1) Q: (D1,d1d2) 
        self[-1] = np.reshape(Q, (Q.shape[0], self[-1].shape[1], self[-1].shape[2]))             #(D1,d1d2)-> (D1,d1,d2)
        
        #Middle Tensors
        for i in range(self.N - 2, 0, -1):
            X = self[i]
            D2, d1, D1, d2 = X.shape
            A = (X.transpose(0, 1, 3, 2).reshape(-1, D1) @ L)
            A = A.reshape(D2, d1, d2, L.shape[1]).transpose(0, 1, 3, 2)
            tmp = flatten_right(A)                                                              # (D2,d1,F,d2)->(D2,d1Fd2)
            L, Q = lq(tmp)                                                                       # L:(D2,D2) Q:(D2,d1D1d2)
            self[i] = np.reshape(Q, (Q.shape[0], A.shape[1], A.shape[2], A.shape[3]))            #(D2,d1D1d2)->(D2,d1,D1,d2)
        
        #Left Boundary
        self[0] = np.einsum("iDj,DE->iEj",self[0],L)
        self.orthform = "Left"
        self.pivot_idx = 0

    def round(self, stop=Cutoff(1e-14)):
        self.orthL()
        tmp = flatten_left(self[0].transpose(0, 2, 1))                                          # (d1,d2,D1)->(d1d2,D1)
        U, S, Vt = truncated_svd(tmp, stop=stop)
        U = U.reshape(self[0].shape[0],self[0].shape[2],U.shape[-1])                            #(d1d2,X)->(d1,d2,X)
        self[0] = U.transpose(0,2,1)                                                            #(d1,d2,X)->(d1,X,d2)

        for i in range(1, self.N - 1):
            K = S[:, None] * Vt
            tmp = flatten_right(self[i])                                                         # (D1,d1,D2,d2)->(D1,d1D2d2)
            A = K @ tmp                                                                          # (X,d1D2d2)
            A = A.reshape(A.shape[0], *self[i].shape[1:])                                        # (X,d1,D2,d2)
            tmp = A.transpose(0,1,3,2)                                                         #(X,d1,D2,d2)->(X,d1,d2,D2)
            tmp = flatten_left(tmp)                                                             #(X,d1,d2,D2)->(Xd1d2,D2)
            U, S, Vt = truncated_svd(tmp, stop=stop)
            U = U.reshape(K.shape[0],self[i].shape[1],self[i].shape[3],U.shape[-1])            #(Xd1d2,Y)->(X,d1,d2,Y)
            self[i] = U.transpose(0,1,3,2)                                                     #(X,d1,d2,Y)->(X,d1,Y,d2)

        K = np.diag(S) @ Vt
        self[-1] = np.einsum("kD,Ddl->kdl", K, self[-1]).astype(self.dtype, copy=False)
        self.orthform = "Left"
        self.pivot_idx = -1
        self.rounded = True

    
    # ======================================================================
    # Linear Algebra operations
    # ======================================================================
    # def mpo_mpo(self, other,compress=False, stop=Cutoff(1e-14)):
    #     N=self.N
    #     mpo_out = [None] * N
        
    #     # left boundary: 
    #     t1 = self._cache_get(
    #         ("mpo_mpo_left_mat", 0, self[0].shape),
    #         lambda: self[0].reshape(-1, self[0].shape[-1]),
    #     )                                                                                          #(d1,D1, d2)-> (d1D1, d2)
    #     t2 = other._cache_get(
    #         ("mpo_mpo_right_mat", 0, other[0].shape),
    #         lambda: other[0].reshape(other[0].shape[0], -1),
    #     )                                                                                          #(d2, E1,d3)-> (d2, E1d3)
    #     tmp = t1@t2                                                                               #(d1D1, E1d3)
    #     tmp = tmp.reshape(self[0].shape[0],self[0].shape[1],other[0].shape[0],
    #                       other[0].shape[1])                                                      #(d1D1, E1d3)-> (d1,D1,E1,d3)
    #     mpo_out[0] = tmp.reshape(self[0].shape[0],-1,self[0].shape[-1])                           #(d1,D1,E1,d3)-> (d1, D1E1, d3)
        
    #     # middle cores: 
    #     for i in range(1, N-1):
    #         t1 = self._cache_get(
    #             ("mpo_mpo_left_mat", i, self[i].shape),
    #             lambda i=i: self[i].reshape(-1, self[i].shape[-1]),
    #         )                                                                                      #(D1,d1,D2,d2)-> (D1d1D2,d2)
    #         t2 = other._cache_get(
    #             ("mpo_mpo_right_mat", i, other[i].shape),
    #             lambda i=i: other[i].transpose(1, 0, 2, 3).reshape(other[i].shape[1], -1),
    #         )                                                                                      #(d2,E1,E2,d3)->(d2,E1E2d3)
    #         tmp = t1@t2                                                                           #(D1d1D2,E1E2d3)
    #         tmp = tmp.reshape(self[i].shape[0],self[i].shape[1],self[i].shape[2],
    #                           other[i].shape[0],other[i].shape[2],other[i].shape[3])              #(D1d1D2,E1E2d3)->(D1,d1,D2,E1,E2,d3)
    #         tmp = tmp.transpose(0,3,1,2,4,5)                                                      #(D1,d1,D2,E1,E2,d3)->(D1,E1,d1,D2,E2,d3)
    #         mpo_out[i] = tmp.reshape(tmp.shape[0]*tmp.shape[1],
    #                                  tmp.shape[2],tmp.shape[3]*tmp.shape[4],tmp.shape[5])         #(D1,D2,d1,E1,E2,d3)->(D1,d1,D2,E1,E2,d3)
        
    #     #Right boundary
    #     t1 = self._cache_get(
    #         ("mpo_mpo_left_mat", N - 1, self[-1].shape),
    #         lambda: self[-1].reshape(-1, self[-1].shape[-1]),
    #     )                                                                                          #(D1,d1,d2)-> (D1d1,d2)
    #     t2 = other._cache_get(
    #         ("mpo_mpo_right_mat", N - 1, other[-1].shape),
    #         lambda: other[-1].transpose(1, 0, 2).reshape(other[-1].shape[1], -1),
    #     )                                                                                          #(d2,E1,d3)-> (d2,E1d3)
    #     tmp = t1 @ t2                                                                             # (D1d1,E1d3)
    #     tmp= tmp.reshape(self[-1].shape[0],self[-1].shape[1],other[-1].shape[0],
    #                      other[-1].shape[1])                                                      #(D1d1,E1d3)->(D1,d1,E1,d3)
    #     tmp = tmp.transpose(0,2,1,3)                                                              #(D,d1,E1,d3) - > (D,E1,d1,d3)
    #     mpo_out[-1] = tmp.reshape(-1,tmp.shape[2],tmp.shape[3])                                   #(D1,E1,d1,d3)->(D1E1,d1,d3)
    #     mpo_out= MPO(mpo_out)
    #     if compress:
    #         mpo_out.round(stop=stop)
    #     return mpo_out

    def mpo_mpo(self, other, compress=False, stop=Cutoff(1e-14)):
        """
        Multiply two MPOs with the following storage convention

            left core    -> (d_left, D_right, d_right)
            middle core  -> (D_left, d_left, D_right, d_right)
            right core   -> (D_left, d_left, d_right)

        The contracted index is always the last physical index of self
        with the first physical index of other.
        """
        N = self.N
        mpo_out = [None] * N

        # ------------------------------------------------------------------
        # Left boundary
        # self[0]   shape (d1, D1, d2)
        # other[0]  shape (d2, E1, d3)
        # output    shape (d1, D1*E1, d3)
        # ------------------------------------------------------------------
        d1, D1, d2 = self[0].shape
        d2_other, E1, d3 = other[0].shape
        if d2 != d2_other:
            raise ValueError(
                f"Left boundary physical mismatch in mpo_mpo: "
                f"self[0].shape = {self[0].shape}, other[0].shape = {other[0].shape}"
            )

        t1 = self._cache_get(
            ("mpo_mpo_left_mat", 0, self[0].shape),
            lambda: self[0].reshape(d1 * D1, d2),
        )
        t2 = other._cache_get(
            ("mpo_mpo_right_mat", 0, other[0].shape),
            lambda: other[0].reshape(d2_other, E1 * d3),
        )

        tmp = t1 @ t2
        tmp = tmp.reshape(d1, D1, E1, d3)
        mpo_out[0] = tmp.reshape(d1, D1 * E1, d3)

        # ------------------------------------------------------------------
        # Middle cores
        # self[i]   shape (D1, d1, D2, d2)
        # other[i]  shape (E1, d2, E2, d3)
        # output    shape (D1*E1, d1, D2*E2, d3)
        # ------------------------------------------------------------------
        for i in range(1, N - 1):
            D1, d1, D2, d2 = self[i].shape
            E1, d2_other, E2, d3 = other[i].shape
            if d2 != d2_other:
                raise ValueError(
                    f"Middle core physical mismatch at site {i} in mpo_mpo: "
                    f"self[{i}].shape = {self[i].shape}, other[{i}].shape = {other[i].shape}"
                )

            t1 = self._cache_get(
                ("mpo_mpo_left_mat", i, self[i].shape),
                lambda i=i: self[i].reshape(D1 * d1 * D2, d2),
            )
            t2 = other._cache_get(
                ("mpo_mpo_right_mat", i, other[i].shape),
                lambda i=i: other[i].transpose(1, 0, 2, 3).reshape(d2_other, E1 * E2 * d3),
            )

            tmp = t1 @ t2
            tmp = tmp.reshape(D1, d1, D2, E1, E2, d3)
            tmp = tmp.transpose(0, 3, 1, 2, 4, 5)
            mpo_out[i] = tmp.reshape(D1 * E1, d1, D2 * E2, d3)

        # ------------------------------------------------------------------
        # Right boundary
        # self[-1]   shape (D1, d1, d2)
        # other[-1]  shape (E1, d2, d3)
        # output     shape (D1*E1, d1, d3)
        # ------------------------------------------------------------------
        D1, d1, d2 = self[-1].shape
        E1, d2_other, d3 = other[-1].shape
        if d2 != d2_other:
            raise ValueError(
                f"Right boundary physical mismatch in mpo_mpo: "
                f"self[-1].shape = {self[-1].shape}, other[-1].shape = {other[-1].shape}"
            )

        t1 = self._cache_get(
            ("mpo_mpo_left_mat", N - 1, self[-1].shape),
            lambda: self[-1].reshape(D1 * d1, d2),
        )
        t2 = other._cache_get(
            ("mpo_mpo_right_mat", N - 1, other[-1].shape),
            lambda: other[-1].transpose(1, 0, 2).reshape(d2_other, E1 * d3),
        )

        tmp = t1 @ t2
        tmp = tmp.reshape(D1, d1, E1, d3)
        tmp = tmp.transpose(0, 2, 1, 3)
        mpo_out[-1] = tmp.reshape(D1 * E1, d1, d3)

        mpo_out = MPO(mpo_out)
        if compress:
            mpo_out.round(stop=stop)
        return mpo_out
    
    def __matmul__(self, other):
        if isinstance(other, MPO):
            return self.mpo_mpo(other, compress=True, stop=self.contraction_tol)

        if isinstance(other, MPS):
            return self._apply_mps(other)

        from .trp import TRP
        if isinstance(other, TRP):
            return other.apply_mpo_src(
                self,
                stop=self.contraction_tol,
                finalround=False,
                batch_cols=getattr(other, "_mpo_batch_size", 32),
                dtype=np.result_type(getattr(self, "dtype", np.float64), other.dtype),
            )

        return NotImplemented


    def _apply_mps(self, psi):
        alg = getattr(self, "_mv_alg", "zipup")

        def _src_finalround_from_stop(stop_rule):
            # Policy for SRC matvec:
            # - pure FixedDimension(...) => skip final rounding pass
            # - cutoff-driven rules       => keep final rounding pass
            if stop_rule is None:
                return True
            outputdim = getattr(stop_rule, "outputdim", None)
            cutoff = getattr(stop_rule, "cutoff", None)
            return not (outputdim is not None and cutoff is None)
    
        if alg == "zipup":
            from .contraction.zipup import zipup
            return zipup(self, psi, stop=self.contraction_tol,finalround=True)
    
        if alg == "SRC":
            from .contraction.src import SRC
            return SRC(
                self,
                psi,
                stop=self.contraction_tol,
                finalround=_src_finalround_from_stop(self.contraction_tol),
            )
    
        if alg == "CTC":
            from .contraction.mpo_mps import mpo_mps
            return mpo_mps(self, psi, stop=self.contraction_tol)
    
        if alg == "fit":
            from .contraction.fit import fit
            return fit(self, psi, stop=self.contraction_tol)
    
        if alg == "DM":
            from .contraction.density_matrix import density_matrix

            return density_matrix(self, psi, stop=self.contraction_tol,finalround=True)
    
        raise RuntimeError(f"Unknown mv_alg '{alg}'")

    def precompute_mv_cache(self, *, alg=None):
        """
        Precompute reusable MPO-side views used by repeated MPS matvec calls.

        This is safe to call repeatedly; values live in the TN1D LRU cache and
        are invalidated automatically when the MPO mutates through TN1D setters.
        """
        alg_eff = str(getattr(self, "_mv_alg", "zipup") if alg is None else alg)
        if alg_eff == "SRC":
            from .contraction.src_cacheview import SRCViewCache
            SRCViewCache.cache_H(self, assume_identity=False)
            return self

        if alg_eff == "CTC":
            # Left boundary reshape used in mpo_mps.
            self.cache(
                ("mpo_mps_left", 0, tuple(self[0].shape), np.dtype(self[0].dtype).str),
                lambda: self[0].reshape(self[0].shape[0] * self[0].shape[1], self[0].shape[2]),
                namespace="mpo_mps",
            )
            # Bulk reshapes used in mpo_mps.
            for i in range(1, self.N - 1):
                self.cache(
                    ("mpo_mps_bulk", i, tuple(self[i].shape), np.dtype(self[i].dtype).str),
                    lambda i=i: self[i].reshape(-1, self[i].shape[3]),
                    namespace="mpo_mps",
                )
            # Right boundary reshape used in mpo_mps.
            self.cache(
                ("mpo_mps_right", self.N - 1, tuple(self[-1].shape), np.dtype(self[-1].dtype).str),
                lambda: self[-1].reshape(self[-1].shape[0] * self[-1].shape[1], -1),
                namespace="mpo_mps",
            )
            return self

        return self
    def _result_dtype_for_mps(self, *mps_list):
        dtypes = [np.dtype(self.dtype)]
        for m in mps_list:
            for t in getattr(m, "tensors", []):
                if t is None:
                    continue
                dtypes.append(np.asarray(t).dtype)
        return np.result_type(*dtypes)

        
    def quadform(self, v, w=None):
        bra = v
        ket = v if w is None else w

        if len(self.tensors) < 2:
            raise ValueError("quadform requires an MPO with at least 2 sites")

        out_dtype = self._result_dtype_for_mps(bra, ket)
        do_conj = np.issubdtype(out_dtype, np.complexfloating)

        B0 = np.asarray(np.conjugate(bra.tensors[0]) if do_conj else bra.tensors[0], dtype=out_dtype, order="C")
        K0 = np.asarray(ket.tensors[0], dtype=out_dtype, order="C")
        W0 = np.asarray(self.tensors[0], dtype=out_dtype, order="C")

        W0_mat = W0.transpose(2, 1, 0).reshape(W0.shape[2], -1)                                       # (j, Dw*i)
        T0 = (K0.T @ W0_mat).reshape(K0.shape[1], W0.shape[1], W0.shape[0]).transpose(2, 1, 0)        # (i, Dw, Dk)
        E = (B0.T @ T0.reshape(B0.shape[0], -1)).reshape(B0.shape[1], W0.shape[1], K0.shape[1])       # (Db, Dw, Dk)

        for i in range(1, len(self.tensors) - 1):
            Bi = np.asarray(np.conjugate(bra.tensors[i]) if do_conj else bra.tensors[i], dtype=out_dtype, order="C")
            Ki = np.asarray(ket.tensors[i], dtype=out_dtype, order="C")
            Wi = np.asarray(self.tensors[i], dtype=out_dtype, order="C")

            Wi_mat = Wi.transpose(0, 2, 1, 3).reshape(Wi.shape[0] * Wi.shape[2] * Wi.shape[1], Wi.shape[3])          # (DwL*DwR*i, j)
            Ki_mat = Ki.transpose(1, 0, 2).reshape(Ki.shape[1], Ki.shape[0] * Ki.shape[2])                            # (j, DkL*DkR)
            T = (Wi_mat @ Ki_mat).reshape(Wi.shape[0], Wi.shape[2], Wi.shape[1], Ki.shape[0], Ki.shape[2]).transpose(0, 3, 1, 2, 4)  # (DwL, DkL, DwR, i, DkR)

            E_mat = E.transpose(1, 2, 0).reshape(E.shape[1] * E.shape[2], E.shape[0])                                 # (DwL*DkL, DbL)
            Bi_mat = Bi.reshape(Bi.shape[0], Bi.shape[1] * Bi.shape[2])                                                # (DbL, i*DbR)
            U = (E_mat @ Bi_mat).reshape(Wi.shape[0], Ki.shape[0], Bi.shape[1], Bi.shape[2])                          # (DwL, DkL, i, DbR)

            U_mat = U.transpose(3, 0, 1, 2).reshape(Bi.shape[2], Wi.shape[0] * Ki.shape[0] * Bi.shape[1])            # (DbR, DwL*DkL*i)
            T_mat = T.transpose(0, 1, 3, 2, 4).reshape(Wi.shape[0] * Ki.shape[0] * Wi.shape[1], Wi.shape[2] * Ki.shape[2])  # (DwL*DkL*i, DwR*DkR)
            E = (U_mat @ T_mat).reshape(Bi.shape[2], Wi.shape[2], Ki.shape[2])                                         # (DbR, DwR, DkR)

        BN = np.asarray(np.conjugate(bra.tensors[-1]) if do_conj else bra.tensors[-1], dtype=out_dtype, order="C")
        KN = np.asarray(ket.tensors[-1], dtype=out_dtype, order="C")
        WN = np.asarray(self.tensors[-1], dtype=out_dtype, order="C")

        WN_mat = WN.transpose(2, 0, 1).reshape(WN.shape[2], WN.shape[0] * WN.shape[1])                 # (j, Dw*i)
        Tn = (KN @ WN_mat).reshape(KN.shape[0], WN.shape[0], WN.shape[1]).transpose(1, 0, 2)           # (Dw, Dk, i)
        V = E.reshape(BN.shape[0], WN.shape[0] * KN.shape[0]) @ Tn.reshape(WN.shape[0] * KN.shape[0], WN.shape[1])   # (Db, i)

        return (V * BN).sum()                                                                           # scalar
                    
    # def trace(self):
    #     partial_traces = [None]*self.N
    #     partial_traces[0]= np.einsum("iDi->D",self[0])
    #     for i in range(1,self.N-1):
    #         partial_traces[i]= np.einsum("DiEi->DE",self[i])
    #     partial_traces[self.N-1] = np.einsum("Dii->D",self[-1])
    #     trace =partial_traces[0]
    #     for i in range(0,len(partial_traces)-1):
    #         trace=trace@partial_traces[i+1]
    #     return trace
    
    def trace(self):
        x = np.einsum("iDi->D", self[0])
        log_scale = 0.0

        s = np.max(np.abs(x))
        if not np.isfinite(s):
            raise FloatingPointError("Non-finite value encountered while initializing MPO trace.")
        if s == 0:
            return x.dtype.type(0)
        x = x / s
        log_scale += np.log(s)

        for i in range(1, self.N - 1):
            E = np.einsum("DiEi->DE", self[i])
            x = x @ E

            s = np.max(np.abs(x))
            if not np.isfinite(s):
                raise FloatingPointError(f"Non-finite value encountered at site {i}.")
            if s == 0:
                return x.dtype.type(0)

            x = x / s
            log_scale += np.log(s)

        x = x @ np.einsum("Dii->D", self[-1])

        if np.ndim(x) != 0 and np.size(x) != 1:
            raise ValueError(f"Trace contraction did not end in a scalar, got shape {np.shape(x)}")

        return np.asarray(x).reshape(()).item() * np.exp(log_scale)
    # ======================================================================
    # Diagnostics
    # ======================================================================
    def to_mps(self):
            from .mps import MPS
            super_cores, phys_pairs = [], []
            for i, W in enumerate(self.tensors):
                if W.ndim == 3:
                    if i == 0:        # left boundary: (s, D, s') -> (sigma, D)
                        s, D, sp = W.shape
                        super_cores.append(W.reshape(s * sp, D))
                        phys_pairs.append((s, sp))
                    else:             # right boundary: (D, s, s') -> (D, sigma)
                        D, s, sp = W.shape
                        super_cores.append(W.reshape(D, s * sp))
                        phys_pairs.append((s, sp))
                elif W.ndim == 4:     # middle: (Dl, s, Dr, s') -> (Dl, sigma, Dr)
                    Dl, s, Dr, sp = W.shape
                    super_cores.append(W.reshape(Dl, s * sp, Dr))
                    phys_pairs.append((s, sp))
                else:
                    raise ValueError("MPO cores must be rank-3 or rank-4.")
            out_dtype = np.result_type(*[np.asarray(c).dtype for c in super_cores])
            super_mps = MPS(super_cores, canform="None", dtype=out_dtype)
            return super_mps, phys_pairs
    
    def from_mps(self, super_mps, phys_pairs):
        super_cores = list(super_mps.tensors)
        L = len(super_cores)
        if L != len(phys_pairs):
            raise ValueError("Length mismatch between cores and phys_pairs.")
        U_left  = getattr(super_mps, "_U_sigma_left",  None)   
        U_right = getattr(super_mps, "_U_sigma_right", None)  
        mpo_cores = []
        for i, A in enumerate(super_cores):
            s, sp = phys_pairs[i]
            if i == 0:
                # (sigma, D) -> (s, D, s')
                if A.ndim != 2:
                    raise ValueError("Left super core must be (sigma, D).")
                sigma, D = A.shape
                if sigma != s * sp:
                    raise ValueError("Left fused size != s*s'.")
                if U_left is not None:
                    if U_left.shape != (sigma, sigma):
                        raise ValueError("Shape mismatch for _U_sigma_left.")
                    A = U_left.conj().T @ A                 
                mpo_cores.append(A.reshape(s, D, sp))
    
            elif i == L - 1:
                if A.ndim != 2:
                    raise ValueError("Right super core must be (D, sigma).")
                D, sigma = A.shape
                if sigma != s * sp:
                    raise ValueError("Right fused size != s*s'.")
                if U_right is not None:
                    if U_right.shape != (sigma, sigma):
                        raise ValueError("Shape mismatch for _U_sigma_right.")
                    A = A @ U_right.conj().T           
                mpo_cores.append(A.reshape(D, s, sp))
    
            else:
                if A.ndim != 3:
                    raise ValueError("Middle super core must be (Dl, sigma, Dr).")
                Dl, sigma, Dr = A.shape
                if sigma != s * sp:
                    raise ValueError(f"Middle fused size != s*s' at site {i}.")
                mpo_cores.append(A.reshape(Dl, s, Dr, sp))
    
        out_dtype = np.result_type(*[np.asarray(c).dtype for c in mpo_cores])
        if np.issubdtype(self.dtype, np.floating) and np.issubdtype(out_dtype, np.complexfloating):
            self.dtype = out_dtype
        else:
            self.dtype = np.result_type(self.dtype, out_dtype)
        self.tensors = [np.asarray(c, dtype=self.dtype, order="C") for c in mpo_cores]
    
        for attr in ("_U_sigma_left", "_U_sigma_right"):
            if hasattr(super_mps, attr):
                delattr(super_mps, attr)
                
    def view(self, **kwargs) -> None:
        try:
            from .other.view import MPOView
        except Exception as exc:
            raise ImportError(
                "MPO.view() requires optional plotting dependencies (plotly/matplotlib)."
            ) from exc
        if self.N <10:
            print("Warning you requested a plot of a large mpo this may take a moment....")
        MPOView(
            self,cmap="viridis",
            color_scale="raw",
            global_normalize=True,
            share_colorbar=True,
            gap_x=1.5,
            zoom=.3,
            mini_grid=True,
            mini_grid_mode="bbox",
            show_ticks=False,
            mini_grid_color="black",
            mini_grid_opacity=0.12,
            voxel_depth=0.8,
            transparent=True,
            background="rgba(0,0,0,0)",
            font_color="black",
            left_to_right=True
               ).show()
        
    # ======================================================================
    # Constructors
    # ======================================================================  
    @staticmethod
    def all_ones(n, m, d=2, d2=2, dtype=float):
        dtype = np.dtype(dtype)
    
        # local helper to ensure dtype
        def ones(shape):
            return np.ones(shape, dtype=dtype)
    
        if n < 2:
            raise ValueError("all_ones MPO requires n >= 2 to match (d,m,d2)/(m,d,d2) boundary shapes.")
    
        tensors = [ones((d, m, d2))]
        for _ in range(n - 2):
            tensors.append(ones((m, d, m, d2)))
        tensors.append(ones((m, d, d2)))
        return MPO(tensors, dtype=dtype)

    @staticmethod
    def eye(N,d=2, m=None, dtype=float):
        dtype = np.dtype(dtype)
        if m is None:
            m = 1
        tensors = []

        left_tensor = TN1D._zeros((d, m, d), dtype=dtype)
        for i in range(d):
            left_tensor[i, 0, i] = 1.0
        tensors.append(left_tensor)

        for _ in range(1, N - 1):
            middle_tensor = TN1D._zeros((m, d, m, d), dtype=dtype)
            for i in range(d):
                for j in range(m):
                    middle_tensor[j, i, j, i] = 1.0
            tensors.append(middle_tensor)

        right_tensor = TN1D._zeros((m, d, d), dtype=dtype)
        for i in range(d):
            right_tensor[0, i, i] = 1.0
        tensors.append(right_tensor)

        return MPO(tensors, dtype=dtype)
        
    @staticmethod 
    def rmpo(n, m, d=2, d2=2, dtype=float, random_tensor=None):
        dtype = np.dtype(dtype)
        if random_tensor is None:
            def rt(*shape):
                return TN1D._randn(shape, dtype)
        else:
            def rt(*shape):
                return np.asarray(random_tensor(*shape), dtype=dtype)

        tensors = [rt(d, m, d2)]
        for _ in range(n - 2):
            tensors.append(rt(m, d, m, d2))
        tensors.append(rt(m, d, d2))
        return MPO(tensors, dtype=dtype)

    @staticmethod
    def fully_random_mpo(n=10, dtype=float, random_tensor=None):
        dtype = np.dtype(dtype)
        rng = np.random
        if random_tensor is None:
            def rr(*shape):
                return TN1D._randn(shape, dtype)
        else:
            def rr(*shape):
                return np.asarray(random_tensor(*shape), dtype=dtype)

        randint = lambda: rng.randint(1, 7)
        bond_dim = randint()
        sites = [rr(randint(), bond_dim, randint())]
        for _ in range(n - 2):
            new_bond_dim = randint()
            sites.append(rr(bond_dim, randint(), new_bond_dim, randint()))
            bond_dim = new_bond_dim
        sites.append(rr(bond_dim, randint(), randint()))
        return MPO(sites, dtype=dtype)
        
    @staticmethod
    def random_incremental_mpo(N, d, seed, d2=None, dtype=np.complex128):
        dtype = np.dtype(dtype)
        if d2 is None:
            d2 = d
        rng = np.random.RandomState(seed)

        def rr(shape):
            if np.issubdtype(dtype, np.complexfloating):
                return (rng.randn(*shape) + 1j * rng.randn(*shape)).astype(dtype)
            return rng.randn(*shape).astype(dtype)

        mpo = [rr((d, seed, d2))]  # first tensor
        for i in range(0, N - 2):
            mpo.append(rr((seed + i, d, seed + i + 1, d2)))
        mpo.append(rr((seed + i + 1, d, d2)))
        return MPO(mpo, dtype=dtype)

