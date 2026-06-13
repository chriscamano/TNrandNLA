# ==================================== tn1d.py ====================================
import json
from pathlib import Path

import numpy as np
import numpy.linalg as la
import copy
from .stopping import Cutoff
from .other.tensor_cache import tensorCache


class TN1D:
    def __init__(self, tensors, canform="None", rounded=False, dtype=None, orthoform=None, orthform=None):
        inferred = np.result_type(*[np.asarray(t).dtype for t in tensors])
        if dtype is None:
            # Library default: keep real-valued networks in float64 unless explicitly overridden.
            # If any tensor is complex, default to complex128.
            
            if np.issubdtype(inferred, np.complexfloating):
                self.dtype = np.dtype(np.complex128)
            else:
                self.dtype = np.dtype(np.float64)
        else:
            requested = np.dtype(dtype)
            if np.issubdtype(requested, np.floating) and np.issubdtype(inferred, np.complexfloating):
                self.dtype = inferred  # promote to complex to avoid discarding imag part
            else:
                self.dtype = requested

        self.tensors = [np.asarray(t, dtype=self.dtype) for t in tensors]
        self.N = len(self.tensors)
        self.contraction_tol = Cutoff(1e-14)
        self._cache = tensorCache(enabled=True, max_entries=4096)
        if orthform is not None:
            self._orthoform = orthform
        elif orthoform is not None:
            self._orthoform = orthoform
        else:
            self._orthoform = canform
        self.rounded = rounded

    def _kind(self):
        kind = getattr(self, "_tn_kind", None)
        if kind in {"mps", "mpo"}:
            return kind
        if self.N == 0:
            raise ValueError("Cannot infer tensor-network kind from empty tensor list.")
        nd = self[0].ndim
        if nd == 2:
            return "mps"
        if nd == 3:
            return "mpo"
        raise ValueError("Unable to infer tensor-network kind; set self._tn_kind to 'mps' or 'mpo'.")

    @property
    def orthoform(self):
        return self._orthoform

    @orthoform.setter
    def orthoform(self, value):
        self._orthoform = value

    @property
    def orthform(self):
        return self._orthoform

    @orthform.setter
    def orthform(self, value):
        self._orthoform = value

    @property
    def canform(self):
        return self._orthoform

    @canform.setter
    def canform(self, value):
        self._orthoform = value

    # -------------------- container plumbing --------------------
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.tensors[idx]

    def __setitem__(self, idx, val):
        arr = np.asarray(val)
        if np.iscomplexobj(arr) and not np.issubdtype(self.dtype, np.complexfloating):
            self.dtype = np.complex64 if self.dtype == np.float32 else np.complex128
            self.tensors = [np.asarray(t, dtype=self.dtype) for t in self.tensors]
        self.tensors[idx] = np.asarray(arr, dtype=self.dtype)
        self._invalidate_cache()


    def copy(self):
        tensors = [t.copy(order="K") for t in self.tensors]
        out = self.__class__(tensors, orthform=self.orthform, rounded=self.rounded, dtype=self.dtype)
        out._cache = tensorCache(enabled=self._cache.enabled, max_entries=4096)
        if hasattr(self, "contraction_tol"):
            out.contraction_tol = self.contraction_tol
        if hasattr(self, "pivot_idx"):
            out.pivot_idx = getattr(self, "pivot_idx")
        if hasattr(self, "round_alg"):
            out.round_alg = getattr(self, "round_alg")
        return out

    def save_npz(self, path, *, compressed=True):
        """
        Save this tensor network to an `.npz` file without using pickle.
        """
        path = Path(path)
        metadata = {
            "kind": self._kind(),
            "n_tensors": int(self.N),
            "orthform": getattr(self, "orthform", "None"),
            "rounded": bool(getattr(self, "rounded", False)),
            "dtype": np.dtype(self.dtype).str,
            "pivot_idx": getattr(self, "pivot_idx", None),
        }
        if hasattr(self, "round_alg"):
            metadata["round_alg"] = self.round_alg
        if hasattr(self, "_mv_alg"):
            metadata["mv_alg"] = self._mv_alg
        if hasattr(self, "state"):
            metadata["state"] = list(self.state)

        arrays = {
            "metadata": np.array(json.dumps(metadata), dtype=np.str_),
        }
        for idx, tensor in enumerate(self.tensors):
            arrays[f"tensor_{idx}"] = np.asarray(tensor)

        saver = np.savez_compressed if compressed else np.savez
        saver(path, **arrays)
        return path

    @classmethod
    def load_npz(cls, path):
        """
        Load an MPS/MPO saved with `save_npz`.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            n_tensors = int(metadata["n_tensors"])
            tensors = [np.array(data[f"tensor_{idx}"], copy=True) for idx in range(n_tensors)]

        kind = metadata.get("kind")
        target_cls = cls
        if cls is TN1D:
            if kind == "mps":
                from .mps import MPS

                target_cls = MPS
            elif kind == "mpo":
                from .mpo import MPO

                target_cls = MPO
            else:
                raise ValueError(f"Unsupported tensor-network kind '{kind}' in {path}.")
        else:
            expected_kind = getattr(cls, "__name__", "").lower()
            if kind is not None and kind != expected_kind:
                raise ValueError(
                    f"File {path} contains kind '{kind}', cannot load it as {cls.__name__}."
                )

        dtype = np.dtype(metadata["dtype"])
        common_kwargs = {
            "orthform": metadata.get("orthform", "None"),
            "rounded": bool(metadata.get("rounded", False)),
            "dtype": dtype,
        }

        if kind == "mpo":
            out = target_cls(tensors, mv_alg=metadata.get("mv_alg", "SRC"), **common_kwargs)
            out.state = list(metadata.get("state", getattr(out, "state", ["n"] * out.N)))
        else:
            out = target_cls(tensors, **common_kwargs)
            if "round_alg" in metadata and hasattr(out, "round_alg"):
                out.round_alg = metadata["round_alg"]

        if hasattr(out, "pivot_idx"):
            out.pivot_idx = metadata.get("pivot_idx", getattr(out, "pivot_idx", None))
        return out

    # -------------------- cache utilities --------------------
    def enable_cache(self, enabled=True):
        self._cache.set_enabled(enabled)
        return self

    def disable_cache(self):
        return self.enable_cache(False)

    def _invalidate_cache(self):
        self._cache.clear()
        # SRC caches are stored directly on the TN object (__dict__) with keys
        # like ("src_H", ...) and ("src_psi", ...). They depend on tensor shapes,
        # so drop them whenever the network is mutated.
        try:
            keys_to_drop = []
            for k in self.__dict__.keys():
                if isinstance(k, tuple) and len(k) > 0 and isinstance(k[0], str) and k[0].startswith("src_"):
                    keys_to_drop.append(k)
            for k in keys_to_drop:
                self.__dict__.pop(k, None)
        except Exception:
            pass

    def _cache_get(self, key, builder):
        return self._cache.get(key, builder)

    def cache(self, key, builder, *, namespace="tn1d"):
        """
        Namespaced cache accessor for tensor-derived auxiliary data.

        This is a thin wrapper around the internal LRU cache and is intended for
        reusable reshape/transpose/materialization helpers in higher-level
        routines (for example Lanczos/SRC workspaces).
        """
        return self._cache_get((str(namespace), key), builder)

    def _adopt_from(self, other, *, extra_attrs=()):
        """
        Overwrite this instance in-place with state from another TN1D-like object.
        Subclasses pass extra_attrs for class-specific metadata (e.g. round_alg, _mv_alg).
        """
        self.tensors = other.tensors
        self.dtype = other.dtype
        self.orthform = getattr(other, "orthform", self.orthform)
        self.rounded = getattr(other, "rounded", self.rounded)
        if hasattr(self, "pivot_idx") or hasattr(other, "pivot_idx"):
            self.pivot_idx = getattr(other, "pivot_idx", getattr(self, "pivot_idx", None))
        if hasattr(other, "contraction_tol"):
            self.contraction_tol = other.contraction_tol
        for name in extra_attrs:
            if hasattr(other, name):
                setattr(self, name, getattr(other, name))
        self._invalidate_cache()
        return self

    def _add_inplace_shared(
        self,
        other,
        *,
        subtract=False,
        compress=False,
        stop=Cutoff(1e-14),
        extra_attrs=(),
    ):
        """
        Shared in-place add/sub skeleton used by MPS and MPO.
        """
        new = self.add(other, subtract=subtract, compress=compress, stop=stop)
        return self._adopt_from(new, extra_attrs=extra_attrs)
    # -------------------- dtype utilities --------------------
    @staticmethod
    def _zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def _randn(shape, dtype, rng=None):
        """
        RNG-compatible Gaussian sampler.
        If rng is None, falls back to np.random.randn for backwards compatibility.
        """
        # if rng is None:
        #     # original behavior
        #     if np.issubdtype(np.dtype(dtype), np.complexfloating):
        #         return (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        #     return 
    
        # # RNG-controlled behavior
        # if np.issubdtype(np.dtype(dtype), np.complexfloating):
        #     re = rng.standard_normal(size=shape)
        #     im = rng.standard_normal(size=shape)
        #     return (re + 1j * im).astype(dtype)
        #     ### missing root 2 
    
        return np.random.randn(*shape).astype(dtype)

    def astype(self, dtype, copy=True):
        """
        Return a new instance cast to dtype (or self if copy=False and dtype matches).
        Subclasses can wrap this to keep a different parameter name (e.g., copy_).
        """
        dtype = np.dtype(dtype)
        if dtype == self.dtype and not copy:
            return self
        tensors = [t.astype(dtype, copy=True) for t in self.tensors]
        return self.__class__(tensors, orthform=self.orthform, rounded=self.rounded, dtype=dtype)


    # -------------------- algebra shared bits --------------------
    def dagger(self):
        conj = [np.conj(t) for t in self.tensors]
        return self.__class__(conj, orthform=self.orthform, rounded=self.rounded, dtype=self.dtype)
    @property
    def T(self):
        out = self.dagger()
        out._is_dagger = True
        return out

    def normalize(self):
        """
        Divide the boundary tensor by its Frobenius norm, mirroring norm().
        """
        if hasattr(self, "orth"):
            self.orth()
        elif hasattr(self, "orthLR"):
            self.orthLR()
        else:
            raise AttributeError("Object must implement orth() or orthLR() to normalize.")
        if self.orthoform == "Right":
            nrm = la.norm(self[-1], "fro")
            if nrm != 0:
                self[-1] /= nrm
        else:
            nrm = la.norm(self[0], "fro")
            if nrm != 0:
                self[0] /= nrm

    # -------------------- simple bond utilities --------------------
    def bond_size(self, i):
        """
        Left boundary bond for i==0; otherwise right bond of site i.
        Assumes cores in (left-bond, physical, right-bond) order for 3-way cores.
        """
        if i == 0:
            return self[0].shape[1]
        return self[i].shape[2]

    def max_bond_dim(self):
        """
        Maximum internal bond across the chain (first site right bond, last site left bond).
        """
        maxbond = 0
        for s, t in enumerate(self.tensors):
            if s == 0:
                maxbond = t.shape[1]
            elif s == self.N - 1:
                maxbond = max(maxbond, t.shape[0])
            else:
                maxbond = max(maxbond, t.shape[0], t.shape[2])
        return maxbond

    def max_bond(self):
        return self.max_bond_dim()

    # -------------------- memory / bond QoL --------------------
    def memory_bytes(self):
        total = 0
        for t in self.tensors:
            total += int(np.asarray(t).nbytes)
        return int(total)

    def memory_gb(self):
        return float(self.memory_bytes()) / (1024.0**3)

    def all_bonds(self):
        bonds = []
        for i, t in enumerate(self.tensors):
            if i == 0:
                bonds.append(t.shape[1])
            elif i == self.N - 1:
                bonds.append(t.shape[0])
            else:
                bonds.append(t.shape[2])
        return bonds

    def show(self, max_width=None, show_bonds=True):
        kind = self._kind()
        if kind == "mps":
            from .other.show import show_mps

            show_mps(self, max_width=max_width, show_bonds=show_bonds)
            return
        from .other.show import show_mpo

        show_mpo(self, max_width=max_width)

    def to_dense(self, kind=None): # TODO: both of these are kind of slow but we can optimize them later if needed
        k = self._kind() if kind is None else kind
        if k == "mps":
            result = self[0]
            for i in range(1, self.N):
                if i == 1:
                    result = np.tensordot(result, self[i], axes=([1], [0]))
                else:
                    result = np.tensordot(result, self[i], axes=([-1], [0]))
            return result.reshape(-1)

        if k == "mpo":
            d = self[0].shape[0]
            if self.N == 2:
                return np.einsum("iDj,Dkl->ikjl", self[0], self[1]).reshape(d**2, d**2)

            size = d**2
            tmp = np.einsum("arb,rARB->aAbBR", self[0], self[1]).reshape(
                size, size, self[1].shape[2]
            )
            for i in range(2, self.N - 1):
                size *= d
                tmp = np.einsum("abr,rARB->aAbBR", tmp, self[i]).reshape(
                    size, size, self[i].shape[2]
                )
            return np.einsum("abr,rAB->aAbB", tmp, self[-1]).reshape(size * d, size * d)

        raise ValueError(f"Unknown kind '{k}'. Expected 'mps' or 'mpo'.")

    def orth(self, kind=None):
        k = self._kind() if kind is None else kind
        if k == "mps":
            if self.orthoform in {"Left", "Right"}:
                return
            pivot = getattr(self, "pivot_idx", None)
            if pivot is None:
                self.orthL()
                return
            dist_left = pivot
            dist_right = (self.N - 1) - pivot
            if dist_left <= dist_right:
                self.orthR()
            else:
                self.orthL()
            return

        if k == "mpo":
            if self.orthform in {"Left", "Right"}:
                return
            self.orthLR()
            return

        raise ValueError(f"Unknown kind '{k}'. Expected 'mps' or 'mpo'.")

    def norm(self, kind=None):
        k = self._kind() if kind is None else kind
        if k == "mps":
            self.orth()
            if self.orthoform == "Right":
                return np.linalg.norm(self[-1], "fro")
            return np.linalg.norm(self[0], "fro")

        if k == "mpo":
            if self.orthform in {"None", "Left"}:
                self.orthL()
                # After orthL(), pivot is at self[0]; all others are right-isometries
                tmp = self[0].reshape(-1, self[0].shape[2])
                return float(la.norm(tmp, "fro"))
            if self.orthform == "Right":
                # After orthR(), pivot is at self[-1]; all others are left-isometries
                tmp = self[-1].reshape(-1, self[-1].shape[2])
                return float(la.norm(tmp, "fro"))
            if self.orthform == "Mixed":
                print("chris is lazy error: norm for mixed mpo not implemented")
                return None

        raise ValueError(f"Unknown kind '{k}'. Expected 'mps' or 'mpo'.")

    def is_left_orth(self, kind=None):
        k = self._kind() if kind is None else kind
        if k == "mps":
            for i in range(0, self.N - 1):
                tensor = self[i]
                if i == 0:
                    reshaped_tensor = tensor
                else:
                    reshaped_tensor = tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])
                product = np.dot(reshaped_tensor.conj().T, reshaped_tensor)
                identity = np.eye(product.shape[0], dtype=product.dtype)
                if not np.allclose(product, identity):
                    print(f"Tensor at index {i} is not right-orthogonal.")
                    raise AssertionError("Orthogonality test failed")
            print("All tensors are left-orthogonal.")
            return

        if k == "mpo":
            for i in reversed(range(self.N - 1)):
                W = self[i]
                if W.ndim == 4:
                    tmp = W.transpose(0, 1, 3, 2)
                    Wmat = tmp.reshape(-1, tmp.shape[-1])
                elif i == 0:
                    tmp = W.transpose(0, 2, 1)
                    Wmat = tmp.reshape(-1, tmp.shape[-1])
                else:
                    continue

                product = Wmat.conj().T @ Wmat
                identity = np.eye(product.shape[0], dtype=product.dtype)
                if not np.allclose(product, identity):
                    print(f"Tensor at index {i+1} is NOT right orthogonal.")
                    raise AssertionError("left-orthogonality test failed")
            print("All tensors left-orthogonal.")
            return

        raise ValueError(f"Unknown kind '{k}'. Expected 'mps' or 'mpo'.")

    def is_right_orth(self, kind=None):
        k = self._kind() if kind is None else kind
        if k == "mps":
            for i in reversed(range(2, self.N)):
                tensor = self[i]
                if i == self.N - 1:
                    reshaped_tensor = tensor
                else:
                    reshaped_tensor = tensor.reshape(tensor.shape[0], self[i].shape[1] * self[i].shape[2])
                product = np.dot(reshaped_tensor, reshaped_tensor.conj().T)
                identity = np.eye(product.shape[0], dtype=product.dtype)
                if not np.allclose(product, identity):
                    print(f"Tensor at index {i} is not left-orthogonal.")
                    raise AssertionError("Orthogonality test failed")
            print("All tensors are right-orthogonal.")
            return

        if k == "mpo":
            for i in range(1, self.N):
                W = self[i]
                if W.ndim == 4:
                    tmp = W.transpose(1, 2, 3, 0)
                    Wmat = tmp.reshape(-1, tmp.shape[-1])
                elif i == self.N - 1:
                    Wmat = W.reshape(W.shape[0], -1).T
                else:
                    continue

                product = Wmat.conj().T @ Wmat
                identity = np.eye(product.shape[0], dtype=product.dtype)
                if not np.allclose(product, identity):
                    print(f"Tensor at index {i} is NOT left orthogonal.")
                    raise AssertionError("right-orthogonality test failed")
            print("All tensors right-orthogonal.")
            return

        raise ValueError(f"Unknown kind '{k}'. Expected 'mps' or 'mpo'.")

    # -------------------- convenience --------------------
    def size(self):
        return self.N
        
    def print(self):
        for i, t in enumerate(self.tensors, start=1):
            print(f"Core {i}: shape {t.shape}")
