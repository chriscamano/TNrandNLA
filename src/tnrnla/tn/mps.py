import math
from contextlib import contextmanager
import numpy as np
import numpy.linalg as la

from tnrnla.linalg.orth import lq
from tnrnla.linalg.lra import truncated_svd
from .stopping import Cutoff
from .tn1d import TN1D
from .other.tensor_ops import lq_left, qr_right, ttm3


class MPS(TN1D):
    __array_priority__ = 10_000

    def __init__(self, mps, orthform="None", rounded=False, dtype=None, *, orthoform=None, canform=None):
        if orthoform is not None:
            orthform = orthoform
        if canform is not None:
            orthform = canform

        if orthform not in ("None", "Left", "Right", "Mixed"):
            orthform = "None"

        super().__init__(
            mps,
            canform=orthform,
            rounded=rounded,
            dtype=dtype,
            orthoform=orthform,
            orthform=orthform,
        )

        self._tn_kind = "mps"
        self.round_alg = "CTC"
        self.orthform = orthform

        if self.orthform == "Left":
            self.pivot_idx = 0
        elif self.orthform == "Right":
            self.pivot_idx = self.N - 1
        elif self.orthform == "Mixed":
            if self.N <= 2:
                self.pivot_idx = 0
                self.orthform = "Left"
            else:
                self.pivot_idx = self.N // 2
                if self.pivot_idx == 0:
                    self.pivot_idx = 1
                if self.pivot_idx == self.N - 1:
                    self.pivot_idx = self.N - 2
        else:
            self.pivot_idx = None

    def orth(self, direction="Right"):
        if self.orthform == "Left" and self.pivot_idx == 0:
            return
        if self.orthform == "Right" and self.pivot_idx == self.N - 1:
            return

        if direction == "Left":
            self.orthL()
        else:
            self.orthR()

    def add(self, mps2, subtract=False, compress=True, stop=Cutoff(1e-14)):
        mps = []
        out_dtype = np.result_type(self.dtype, mps2.dtype)

        assert self[0].shape[0] == mps2[0].shape[0]
        m1 = self[0].shape[1]
        m2 = mps2[0].shape[1]
        new_site = TN1D._zeros((self[0].shape[0], m1 + m2), dtype=out_dtype)
        new_site[:, 0:m1] = self[0].astype(out_dtype, copy=False)
        new_site[:, m1:] = mps2[0].astype(out_dtype, copy=False)
        mps.append(new_site)

        for i in range(1, self.N - 1):
            assert self[i].shape[1] == mps2[i].shape[1]
            m1 = self[i].shape[0]
            n1 = self[i].shape[2]
            m2 = mps2[i].shape[0]
            n2 = mps2[i].shape[2]
            d = self[i].shape[1]

            new_site = TN1D._zeros((m1 + m2, d, n1 + n2), dtype=out_dtype)
            new_site[0:m1, :, 0:n1] = self[i].astype(out_dtype, copy=False)
            new_site[m1:, :, n1:] = mps2[i].astype(out_dtype, copy=False)
            mps.append(new_site)

        assert self[-1].shape[1] == mps2[-1].shape[1]
        m1 = self[-1].shape[0]
        m2 = mps2[-1].shape[0]
        new_site = TN1D._zeros((m1 + m2, self[-1].shape[1]), dtype=out_dtype)
        new_site[0:m1, :] = self[-1].astype(out_dtype, copy=False)
        new_site[m1:, :] = ((-1 if subtract else 1) * mps2[-1]).astype(out_dtype, copy=False)
        mps.append(new_site)

        out = MPS(mps, orthform="None", dtype=out_dtype)
        out.orthform = "None"
        out.pivot_idx = None
        out.rounded = False

        if compress:
            # from tnrnla.tn.compression.round_src import src_rounding
            # out = src_rounding(out, stop=stop)
            out.round(stop=stop)

        return out

    def __add__(self, other):
        if not isinstance(other, MPS):
            return NotImplemented
        return self.add(other, subtract=False, compress=True, stop=Cutoff(1e-14))

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, MPS):
            return NotImplemented
        return self.add(other, subtract=True, compress=True, stop=Cutoff(1e-14))

    def add_(self, other, subtract=False, compress=True, stop=Cutoff(1e-14)):
        out = self._add_inplace_shared(
            other,
            subtract=subtract,
            compress=compress,
            stop=stop,
            extra_attrs=("round_alg",),
        )
        if out is self:
            self.orthform = "None"
            self.pivot_idx = None
        else:
            out.orthform = "None"
            out.pivot_idx = None
        return out

    def sub_(self, other, compress=True, stop=Cutoff(1e-14)):
        return self.add_(other, subtract=True, compress=compress, stop=stop)

    def __iadd__(self, other):
        return self.add_(other, subtract=False, compress=True, stop=Cutoff(1e-14))

    def __isub__(self, other):
        return self.add_(other, subtract=True, compress=True, stop=Cutoff(1e-14))

    @staticmethod
    def addall(mpss, subtract=False, compress=True, stop=Cutoff(1e-14)):
        if len(mpss) == 0:
            raise ValueError("addall requires at least one MPS")
        out = mpss[0]
        for mps in mpss[1:]:
            out = out.add(mps, subtract=subtract, compress=True, stop=stop)
        return out

    def scale_(self, scalar):
        if not np.isscalar(scalar):
            raise TypeError("MPS can only be multiplied by a scalar")

        out_dtype = np.result_type(self.dtype, np.asarray(scalar).dtype)
        if out_dtype != self.dtype:
            self.tensors = [t.astype(out_dtype, copy=False) for t in self.tensors]
            self.dtype = out_dtype

        self.tensors[0] *= scalar
        self.orthform = "None"
        self.pivot_idx = None
        return self

    def __imul__(self, other):
        return self.scale_(other)

    def __mul__(self, other):
        if not np.isscalar(other):
            raise TypeError("MPS can only be multiplied by a scalar")
        new = self.copy()
        return new.scale_(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def div_(self, scalar):
        if not np.isscalar(scalar):
            raise TypeError("MPS can only be divided by a scalar")
        return self.scale_(1.0 / scalar)

    def __itruediv__(self, other):
        return self.div_(other)

    def __truediv__(self, other):
        if not np.isscalar(other):
            raise TypeError("MPS can only be divided by a scalar")
        new = self.copy()
        return new.scale_(1.0 / other)

    def __rtruediv__(self, other):
        raise TypeError("Cannot divide a scalar by an MPS")

    def __matmul__(self, other):
        if isinstance(other, MPS):
            return self.inner_product(other)
        raise TypeError(f"Cannot multiply MPS by object of type {type(other)}")

    def orthL(self):
        if self.orthform == "Left" and self.pivot_idx == 0:
            return

        L, Q = lq(self[-1])
        self[-1] = Q
        for i in range(self.N - 2, 0, -1):
            tmp = ttm3(np.ascontiguousarray(self[i]), L.T, 2)
            L, self[i] = lq_left(tmp)
        self[0] = self[0] @ L

        for i in range(self.N):
            self[i] = np.ascontiguousarray(self[i])

        self.orthform = "Left"
        self.pivot_idx = 0

    def orthR(self):
        if self.orthform == "Right" and self.pivot_idx == self.N - 1:
            return

        Q, R = la.qr(self[0], mode="reduced")
        self[0] = Q
        for i in range(1, self.N - 1):
            self[i] = ttm3(self[i], R, 0)
            self[i], R = qr_right(self[i])
        self[-1] = R @ self[-1]

        self.orthform = "Right"
        self.pivot_idx = self.N - 1

    def move_pivot(self, target_idx):
        if target_idx < 0 or target_idx >= self.N:
            raise ValueError("target_idx out of range")

        if self.pivot_idx is None:
            self.orthR()

        if self.pivot_idx == target_idx:
            return

        if self.pivot_idx < target_idx:
            for k in range(self.pivot_idx, target_idx):
                if k == self.N - 2:
                    self[k], R = qr_right(self[k])
                    self[k + 1] = R @ self[k + 1]
                else:
                    if k == 0:
                        self[k], R = la.qr(self[k])
                    else:
                        self[k], R = qr_right(self[k])
                    self[k + 1] = ttm3(self[k + 1], R, 0)
        else:
            for k in range(self.pivot_idx, target_idx, -1):
                if k == 1:
                    L, self[k] = lq_left(self[k])
                    self[k - 1] = self[k - 1] @ L
                else:
                    if k == self.N - 1:
                        L, self[k] = lq_left(self[k])
                        self[k - 1] = self[k - 1] @ L
                    else:
                        L, self[k] = lq_left(self[k])
                        self[k - 1] = ttm3(self[k - 1], L.T, 2)

        self.pivot_idx = target_idx
        if target_idx == 0:
            self.orthform = "Left"
        elif target_idx == self.N - 1:
            self.orthform = "Right"
        else:
            self.orthform = "Mixed"

    def roundRL(self, stop=Cutoff(1e-14)):
        if not (self.orthform == "Right" and self.pivot_idx == self.N - 1):
            self.orthR()

        normY = la.norm(self[-1], "fro")
        if stop.cutoff is None:
            stop_eff = stop
        else:
            tau = stop.cutoff * normY / math.sqrt(max(1, self.N - 1))
            stop_eff = Cutoff(tau, maxdim=stop.maxdim)

        L, Q = lq(self[-1])
        U, s, Vt = truncated_svd(L, stop=stop_eff)
        self[-1] = Vt @ Q
        carry = (U * s[None, :]).T

        for i in range(self.N - 2, 0, -1):
            self[i] = ttm3(self[i], carry, 2)
            L, Q = lq_left(self[i])
            U, s, Vt = truncated_svd(L, stop=stop_eff)
            self[i] = ttm3(Q, Vt, 0)
            carry = (U * s[None, :]).T

        self[0] = self[0] @ carry.T
        self.orthform = "Left"
        self.pivot_idx = 0
        self.rounded = True

    def roundLR(self, stop=Cutoff(1e-14)):
        if not (self.orthform == "Left" and self.pivot_idx == 0):
            self.orthL()

        normY = la.norm(self[0], "fro")
        if stop.cutoff is None:
            stop_eff = stop
        else:
            tau = stop.cutoff * normY / math.sqrt(max(1, self.N - 1))
            stop_eff = Cutoff(tau, maxdim=stop.maxdim)

        Q, R = qr_right(self[0])
        U, s, Vt = truncated_svd(R, stop=stop_eff)
        self[0] = Q @ U
        carry = s[:, None] * Vt

        for i in range(1, self.N - 1):
            self[i] = ttm3(self[i], carry, 0)
            Q, R = qr_right(self[i])
            U, s, Vt = truncated_svd(R, stop=stop_eff)
            self[i] = Q @ U
            carry = s[:, None] * Vt

        self[-1] = carry @ self[-1]
        self.orthform = "Right"
        self.pivot_idx = self.N - 1
        self.rounded = True

    def round(self, **kwargs):
        if getattr(self, "rounded", False):
            return

        from .compression.round_daas import daas_round_blas

        DISPATCH = {"dass": daas_round_blas}
        alg = getattr(self, "round_alg", None)

        if alg == "CTC":
            stop = kwargs.get("stop", Cutoff(1e-14))

            if self.orthform == "Left":
                self.roundLR(stop=stop)
                return

            if self.orthform in ("Right", "None"):
                self.roundRL(stop=stop)
                return

            pivot = self.pivot_idx
            if pivot is not None:
                left_dist = pivot
                right_dist = (self.N - 1) - pivot
                if left_dist <= right_dist:
                    self.roundRL(stop=stop)
                else:
                    self.roundLR(stop=stop)
                return

        if alg in DISPATCH:
            DISPATCH[alg](self, **kwargs)
            self.orthform = "None"
            self.pivot_idx = None
            return

        raise ValueError(f"Unknown rounding method '{alg}'")

    def normalize(self):
        if self.orthform == "Left":
            self.orthL()
            self[0] /= np.linalg.norm(self[0], "fro")
        elif self.orthform == "Right":
            self.orthR()
            self[-1] /= np.linalg.norm(self[-1], "fro")
        else:
            self.orthR()
            self[-1] /= np.linalg.norm(self[-1], "fro")

    def self_inner_product(self):
        return self.inner_product(self)

    def inner_product(self, other):
        site = self[0].conj().T @ other[0]

        reshaped_mps_tensors = [
            other[k].reshape(other[k].shape[0], -1)
            for k in range(1, self.N - 1)
        ]
        self_bulk_conj = [
            self[k].conj().reshape(-1, self[k].shape[-1])
            for k in range(1, self.N - 1)
        ]

        for i in range(1, self.N - 1):
            temp = site @ reshaped_mps_tensors[i - 1]
            temp = temp.reshape(site.shape[0], other[i].shape[1], other[i].shape[2])
            temp_transposed = temp.transpose(2, 0, 1)
            temp_reshaped = temp_transposed.reshape(temp_transposed.shape[0], -1)
            self_reshaped = self_bulk_conj[i - 1]
            site = (temp_reshaped @ self_reshaped).T

        site = site @ other[-1]
        right_conj = self[-1].conj().ravel()
        inner_product = np.dot(site.ravel(), right_conj)
        return inner_product

    @staticmethod
    def _to_rank3_site(A, *, is_left=False, is_right=False):
        A = np.asarray(A)
        if is_left:
            if A.ndim != 2:
                raise ValueError("Left boundary tensor must be rank-2 with shape (d, D).")
            d, Dr = A.shape
            return A.reshape(1, d, Dr)
        if is_right:
            if A.ndim != 2:
                raise ValueError("Right boundary tensor must be rank-2 with shape (D, d).")
            Dl, d = A.shape
            return A.reshape(Dl, d, 1)
        if A.ndim != 3:
            raise ValueError("Bulk tensor must be rank-3 with shape (Dl, d, Dr).")
        return A

    def outer(self, bra=None, *, normalize=False, dtype=None):
        from .mpo import MPO

        ket = self
        if bra is None:
            bra = self

        if getattr(ket, "N", None) is None or getattr(bra, "N", None) is None:
            raise AttributeError("Both MPS must have attribute N.")
        if ket.N != bra.N:
            raise ValueError("ket and bra must have the same number of sites.")

        ket_tensors = [np.asarray(T, copy=False) for T in ket.tensors]
        bra_tensors = [np.asarray(T, copy=False) for T in bra.tensors]

        if dtype is None:
            dtype = np.result_type(*(t.dtype for t in ket_tensors), *(t.dtype for t in bra_tensors), np.float64)

        mpo_cores = []
        N = ket.N

        for i in range(N):
            Ak = self._to_rank3_site(
                ket_tensors[i],
                is_left=(i == 0),
                is_right=(i == N - 1),
            ).astype(dtype, copy=False)

            Ab = self._to_rank3_site(
                bra_tensors[i],
                is_left=(i == 0),
                is_right=(i == N - 1),
            ).astype(dtype, copy=False)

            Abc = np.conjugate(Ab)

            Dl_k, d_out, Dr_k = Ak.shape
            Dl_b, d_in, Dr_b = Abc.shape

            if d_out != d_in:
                raise ValueError(f"Physical dims differ at site {i}: {d_out} vs {d_in}")

            W = np.einsum("a d b, A l B -> aA d l bB", Ak, Abc)

            if i == 0:
                site = W.reshape(d_out, d_in, Dr_k * Dr_b).transpose(0, 2, 1)
                mpo_cores.append(site.astype(dtype, copy=False))
            elif i == N - 1:
                site = W.reshape(Dl_k * Dl_b, d_out, d_in)
                mpo_cores.append(site.astype(dtype, copy=False))
            else:
                site = W.reshape(Dl_k * Dl_b, d_out, d_in, Dr_k * Dr_b).transpose(0, 1, 3, 2)
                mpo_cores.append(site.astype(dtype, copy=False))

        out = MPO(mpo_cores, orthform="None", rounded=False, dtype=dtype)

        if normalize:
            if bra is not ket:
                raise ValueError("normalize=True only supported for bra is ket (density operator).")
            tr = ket.self_inner_product() if hasattr(ket, "self_inner_product") else ket.inner_product(ket)
            tr = np.asarray(tr).reshape(())
            if tr == 0:
                raise ValueError("Cannot normalize: <psi|psi> = 0.")
            mpo_cores[0] = mpo_cores[0] / tr
            out = MPO(mpo_cores, orthform="None", rounded=False, dtype=dtype)

        return out

    def view(self, **kwargs) -> None:
        try:
            from .other.view import MPSView
        except Exception as exc:
            raise ImportError(
                "MPS.view() requires optional plotting dependencies (plotly/matplotlib)."
            ) from exc

        if self.N > 10:
            print("Warning you requested a plot of a large mpo this may take a moment....")

        defaults = dict(
            cmap="viridis",
            color_scale="raw",
            global_normalize=False,
            share_colorbar=True,
            show_colorbar=False,
            gap_x=1.5,
            zoom=0.3,
            mini_grid=True,
            mini_grid_mode="full",
            show_ticks=False,
            mini_grid_color="black",
            mini_grid_opacity=0.22,
            voxel_depth=0.8,
            transparent=True,
            background="rgba(0,0,0,0)",
            font_color="black",
            left_to_right=True,
            vmin=None,
            vmax=None,
            cmap_vmin=0.0,
            cmap_vmax=1.0,
        )

        defaults.update(kwargs)
        MPSView(self, **defaults).show()

    @staticmethod
    def rmps(n, m, d=2, random_tensor=None, dtype=float, rng=None):
        dtype = np.dtype(dtype)
        chi = int(m)

        if rng is None:
            rng = np.random.default_rng()

        def rt(sigma, *shape):
            x = rng.standard_normal(size=shape)
            return (sigma * x).astype(dtype, copy=False)

        sigma_boundary = chi ** (-0.25)
        sigma_bulk = chi ** (-0.5)

        blocks = [rt(sigma_boundary, d, chi)]
        for _ in range(n - 2):
            blocks.append(rt(sigma_bulk, chi, d, chi))
        blocks.append(rt(sigma_boundary, chi, d))

        return MPS(blocks, dtype=dtype)


        def rademacher(size):
            return rng.choice(np.array([-1.0, 1.0], dtype=dtype), size=size)

        def ss_bulk_block():
            block = np.zeros((chi, d, chi), dtype=dtype)

            for i in range(d):
                for r in range(zeta):
                    local_cols = rng.integers(0, b, size=chi)
                    cols = r * b + local_cols
                    signs = rademacher(chi)
                    block[rows, i, cols] = bulk_scale * signs

            return block

        def ss_left_boundary():
            block = np.zeros((d, chi), dtype=dtype)

            for i in range(d):
                for r in range(zeta):
                    local_col = rng.integers(0, b)
                    col = r * b + local_col
                    sign = rademacher(())
                    block[i, col] = boundary_scale * sign

            return block

        def ss_right_boundary():
            block = np.zeros((chi, d), dtype=dtype)

            for i in range(d):
                local_cols = rng.integers(0, b, size=chi)
                signs = rademacher(chi)
                mask = local_cols == 0
                block[rows[mask], i] = boundary_scale * signs[mask]

            return block

        blocks = [ss_left_boundary()]

        for _ in range(n - 2):
            blocks.append(ss_bulk_block())

        blocks.append(ss_right_boundary())

        return MPS(blocks, dtype=dtype)
    def crmps(n, m, d=2, random_tensor=None, dtype=np.complex128, sigma=None, rng=None):
        n = int(n)
        chi = int(m)
        d = int(d)

        dt = np.dtype(dtype)
        if dt.kind != "c":
            dt = np.dtype(np.complex128)

        if sigma is None:
            sigma = chi ** (-(n - 1) / (2 * n))
        sigma = float(sigma)

        if rng is None:
            rng = np.random.default_rng()

        if random_tensor is None:
            def rt(*shape):
                x = rng.standard_normal(shape)
                y = rng.standard_normal(shape)
                z = (x + 1j * y) / np.sqrt(2.0)
                return (sigma * z).astype(dt, copy=False)
        else:
            def rt(*shape):
                return (sigma * np.asarray(random_tensor(*shape, rng=rng), dtype=dt))

        blocks = [rt(d, chi)]
        for _ in range(n - 2):
            blocks.append(rt(chi, d, chi))
        blocks.append(rt(chi, d))
        return MPS(blocks, dtype=dt)

    @staticmethod
    def ones(n, m, d=2, dtype=float):
        dtype = np.dtype(dtype)
        blocks = [np.ones((d, m), dtype=dtype)]
        for _ in range(n - 2):
            blocks.append(np.ones((m, d, m), dtype=dtype))
        blocks.append(np.ones((m, d), dtype=dtype))
        return MPS(blocks, dtype=dtype)

    @staticmethod
    def random_incremental_mps(n, d, seed, dtype=np.complex128):
        rng = np.random.RandomState(seed)
        dtype = np.dtype(dtype)

        def rr(shape):
            if np.issubdtype(dtype, np.complexfloating):
                return (rng.randn(*shape) + 1j * rng.randn(*shape)).astype(dtype)
            return rng.randn(*shape).astype(dtype)

        mps = [rr((d, seed))]
        last = seed

        for i in range(0, n - 2):
            mps.append(rr((last, d, last + 1)))
            last = last + 1

        mps.append(rr((last, d)))
        return MPS(mps, dtype=dtype)


class ZpMPS(MPS):
    """
    Z_p charge-conserving MPS wrapper.

    This class keeps standard dense tensor storage, but enforces an exact
    block-sparsity pattern by zeroing entries that violate the local charge
    rule:

        q_right == q_left + q_phys (mod p)

    where q_left/q_right are bond charges and q_phys are physical charges.
    """

    def __init__(
        self,
        mps,
        p=2,
        bond_charges=None,
        phys_charges=None,
        total_charge=0,
        enforce=True,
        **kwargs,
    ):
        super().__init__(mps, **kwargs)
        self._enforce_enabled = True
        self.p = int(p)
        if self.p < 2:
            raise ValueError("p must be >= 2")

        self.total_charge = int(total_charge) % self.p
        self.phys_charges = self._normalize_phys_charges(phys_charges)
        self.bond_charges = self._normalize_bond_charges(bond_charges)

        if enforce:
            self.enforce_symmetry()

    def copy(self):
        out = ZpMPS(
            [t.copy(order="K") for t in self.tensors],
            p=self.p,
            bond_charges=[bc.copy() for bc in self.bond_charges],
            phys_charges=self.phys_charges.copy(),
            total_charge=self.total_charge,
            enforce=False,
            orthform=self.orthform,
            rounded=self.rounded,
            dtype=self.dtype,
        )
        out._cache = self._cache.__class__(enabled=self._cache.enabled, max_entries=4096)
        if hasattr(self, "contraction_tol"):
            out.contraction_tol = self.contraction_tol
        if hasattr(self, "pivot_idx"):
            out.pivot_idx = getattr(self, "pivot_idx")
        if hasattr(self, "round_alg"):
            out.round_alg = getattr(self, "round_alg")
        return out

    def __setitem__(self, idx, val):
        super().__setitem__(idx, val)
        if not getattr(self, "_enforce_enabled", True):
            return
        site = int(idx) % self.N
        self._enforce_site(site)

    @contextmanager
    def _suspend_enforcement(self):
        prev = getattr(self, "_enforce_enabled", True)
        self._enforce_enabled = False
        try:
            yield
        finally:
            self._enforce_enabled = prev

    def orthL(self):
        # Block-safe sweep: run dense orthogonalization on current blocks,
        # then restore Zp sparsity pattern once per sweep.
        with self._suspend_enforcement():
            super().orthL()
        if getattr(self, "_enforce_enabled", True):
            self.enforce_symmetry()

    def orthR(self):
        # Block-safe sweep: run dense orthogonalization on current blocks,
        # then restore Zp sparsity pattern once per sweep.
        with self._suspend_enforcement():
            super().orthR()
        if getattr(self, "_enforce_enabled", True):
            self.enforce_symmetry()

    def roundRL(self, stop=Cutoff(1e-14)):
        # Block-safe sweep: avoid per-write masking during SVD propagation.
        with self._suspend_enforcement():
            super().roundRL(stop=stop)
        if getattr(self, "_enforce_enabled", True):
            self.enforce_symmetry()

    def roundLR(self, stop=Cutoff(1e-14)):
        # Block-safe sweep: avoid per-write masking during SVD propagation.
        with self._suspend_enforcement():
            super().roundLR(stop=stop)
        if getattr(self, "_enforce_enabled", True):
            self.enforce_symmetry()

    def add(self, mps2, subtract=False, compress=True, stop=Cutoff(1e-14)):
        out = super().add(mps2, subtract=subtract, compress=compress, stop=stop)
        merged_bond_charges = None
        if isinstance(mps2, ZpMPS) and int(mps2.p) == int(self.p):
            if np.array_equal(np.asarray(mps2.phys_charges), np.asarray(self.phys_charges)):
                merged_bond_charges = [None] * (self.N + 1)
                merged_bond_charges[0] = np.asarray(self.bond_charges[0]).copy()
                merged_bond_charges[-1] = np.asarray(self.bond_charges[-1]).copy()
                for i in range(1, self.N):
                    merged_bond_charges[i] = np.concatenate(
                        [np.asarray(self.bond_charges[i]), np.asarray(mps2.bond_charges[i])]
                    ) % self.p
        try:
            return ZpMPS.from_mps(
                out,
                p=self.p,
                bond_charges=merged_bond_charges,
                phys_charges=self.phys_charges.copy(),
                total_charge=self.total_charge,
                enforce=True,
            )
        except ValueError:
            # Compression can change bond dimensions; if inherited charge labels no
            # longer match, rebuild with auto-generated labels and re-enforce zeros.
            return ZpMPS.from_mps(
                out,
                p=self.p,
                bond_charges=None,
                phys_charges=self.phys_charges.copy(),
                total_charge=self.total_charge,
                enforce=True,
            )

    def __add__(self, other):
        if not isinstance(other, MPS):
            return NotImplemented
        return self.add(other, subtract=False, compress=False, stop=Cutoff(1e-14))

    def __sub__(self, other):
        if not isinstance(other, MPS):
            return NotImplemented
        return self.add(other, subtract=True, compress=False, stop=Cutoff(1e-14))

    def scale_(self, scalar):
        super().scale_(scalar)
        self.enforce_symmetry()
        return self

    def _normalize_phys_charges(self, phys_charges):
        d = int(self[0].shape[0])
        if phys_charges is None:
            q = np.arange(d, dtype=np.int64) % self.p
        else:
            q = np.asarray(phys_charges, dtype=np.int64).ravel() % self.p
            if q.size != d:
                raise ValueError(f"phys_charges length {q.size} does not match physical dim {d}")
        return q

    def _normalize_bond_charges(self, bond_charges):
        # Bond dims for open-boundary MPS:
        # b0 = 1, b1 = first right bond, ..., b_{N-1} = last left bond, bN = 1.
        dims = [1]
        dims.append(int(np.asarray(self[0]).shape[1]))
        for i in range(1, self.N - 1):
            dims.append(int(np.asarray(self[i]).shape[2]))
        dims.append(1)

        if bond_charges is None:
            out = []
            out.append(np.array([0], dtype=np.int64))
            for D in dims[1:-1]:
                out.append(np.arange(D, dtype=np.int64) % self.p)
            out.append(np.array([self.total_charge], dtype=np.int64))
            return out

        if len(bond_charges) != self.N + 1:
            raise ValueError(f"bond_charges must have length N+1={self.N + 1}")

        out = []
        for i, (q, D) in enumerate(zip(bond_charges, dims)):
            qv = np.asarray(q, dtype=np.int64).ravel() % self.p
            if qv.size != D:
                raise ValueError(f"bond_charges[{i}] length {qv.size} does not match bond dim {D}")
            out.append(qv)
        return out

    def _refresh_bond_charges_to_current_dims(self):
        dims = [1]
        dims.append(int(np.asarray(self[0]).shape[1]))
        for i in range(1, self.N - 1):
            dims.append(int(np.asarray(self[i]).shape[2]))
        dims.append(1)

        old = getattr(self, "bond_charges", None)
        if old is None or len(old) != self.N + 1:
            self.bond_charges = self._normalize_bond_charges(None)
            return

        out = []
        out.append(np.asarray(old[0], dtype=np.int64).ravel()[:1] % self.p)
        for i, D in enumerate(dims[1:-1], start=1):
            oi = np.asarray(old[i], dtype=np.int64).ravel() % self.p
            if oi.size == D:
                out.append(oi.copy())
            elif oi.size > D:
                out.append(oi[:D].copy())
            else:
                pad = (np.arange(D - oi.size, dtype=np.int64) + (oi[-1] if oi.size else 0)) % self.p
                out.append(np.concatenate([oi, pad]) % self.p)
        out.append(np.asarray([self.total_charge], dtype=np.int64))
        self.bond_charges = out

    def _site_mask(self, site):
        site = int(site) % self.N
        qL = self.bond_charges[site]
        qR = self.bond_charges[site + 1]
        qP = self.phys_charges

        if site == 0:
            # A[p, r]
            lhs = (qL[0] + qP[:, None]) % self.p
            return lhs == qR[None, :]
        if site == self.N - 1:
            # A[l, p]
            lhs = (qL[:, None] + qP[None, :]) % self.p
            return lhs == qR[0]
        # A[l, p, r]
        lhs = (qL[:, None, None] + qP[None, :, None]) % self.p
        return lhs == qR[None, None, :]

    def _enforce_site(self, site):
        site = int(site) % self.N
        A = np.asarray(self[site])
        mask = self._site_mask(site)
        if A.shape != mask.shape:
            raise ValueError(
                f"Site {site} mask shape {mask.shape} does not match tensor shape {A.shape}"
            )
        A = np.where(mask, A, np.zeros((), dtype=A.dtype))
        self.tensors[site] = np.ascontiguousarray(A)
        self._invalidate_cache()

    def enforce_symmetry(self):
        self._refresh_bond_charges_to_current_dims()
        for i in range(self.N):
            self._enforce_site(i)
        return self

    @classmethod
    def from_mps(
        cls,
        mps,
        *,
        p=2,
        bond_charges=None,
        phys_charges=None,
        total_charge=0,
        enforce=True,
    ):
        return cls(
            [np.asarray(t).copy() for t in mps.tensors],
            p=p,
            bond_charges=bond_charges,
            phys_charges=phys_charges,
            total_charge=total_charge,
            enforce=enforce,
            orthform=getattr(mps, "orthform", "None"),
            rounded=getattr(mps, "rounded", False),
            dtype=getattr(mps, "dtype", None),
        )

    @staticmethod
    def rmps(n, m, d=2, p=2, total_charge=0, phys_charges=None, dtype=float, rng=None):
        base = MPS.rmps(n=n, m=m, d=d, dtype=dtype, rng=rng)
        return ZpMPS.from_mps(
            base,
            p=p,
            bond_charges=None,
            phys_charges=phys_charges,
            total_charge=total_charge,
            enforce=True,
        )
