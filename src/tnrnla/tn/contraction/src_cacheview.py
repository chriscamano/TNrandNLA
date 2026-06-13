
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..other.tensor_ops import flatten_right, c_reshape


@dataclass(slots=True)
class TermViews:
    identity_simple: bool
    V: int

    H_view: list[np.ndarray]
    H0_flat: np.ndarray | None
    H_env_T_erl: list[np.ndarray] | None
    env_shapes: list[tuple[int, int, int, int]]
    H_sk: list[np.ndarray] | None
    H_cap: list[np.ndarray | None]

    psi0: np.ndarray
    psi_env: list[np.ndarray]
    psi_sk: list[np.ndarray]
    psi_cap: list[np.ndarray | None]


class SRCViewCache:
    @staticmethod
    def infer_phys_dim(H) -> int:
        n = int(H.N)
        for j in range(min(n, 8)):
            T = H[j]
            if getattr(T, "ndim", 0) == 4:
                return int(T.shape[1])
        T0 = H[0]
        return int(T0.shape[0])

    @staticmethod
    def mpo_views(H):
        n = int(H.N)
        V = int(SRCViewCache.infer_phys_dim(H))

        views = [None] * n
        for j in range(n):
            T = np.asarray(H[j])
            if T.ndim == 4:
                views[j] = T
                continue
            if T.ndim != 3:
                raise ValueError(f"Unsupported MPO tensor rank at site {j}: ndim={T.ndim}")

            if j == 0:
                if T.shape[0] != V or T.shape[2] != V:
                    raise ValueError(f"Left boundary expected shape (V, D, V), got {T.shape}")
                views[j] = T
            elif j == n - 1:
                if T.shape[1] != V or T.shape[2] != V:
                    raise ValueError(f"Right boundary expected shape (D, V, V), got {T.shape}")
                views[j] = T
            else:
                raise ValueError(f"Internal MPO tensor at site {j} must be rank 4, got rank 3 with shape {T.shape}")

        return views, V

    @staticmethod
    def _obj_dict(obj) -> dict | None:
        try:
            return obj.__dict__  # type: ignore[attr-defined]
        except Exception:
            return None

    @staticmethod
    def cache_H(H, *, assume_identity: bool = False) -> dict:
        H_dict = SRCViewCache._obj_dict(H)
        key = ("src_H", bool(assume_identity))
        if H_dict is not None:
            cached = H_dict.get(key)
            if cached is not None:
                return cached

        n = int(H.N)
        H_view, V = SRCViewCache.mpo_views(H)
        identity_simple = bool(assume_identity)

        H0_flat = None if identity_simple else flatten_right(H_view[0])

        H_env_T_erl = None
        if not identity_simple:
            H_env_T_erl = []
            for k in range(1, n - 1):
                Hk = np.asarray(H_view[k])
                HkT = np.ascontiguousarray(Hk.transpose(1, 2, 3, 0))
                H_env_T_erl.append(c_reshape(HkT, HkT.shape[0], -1))

        env_shapes = [
            (
                H_view[k].shape[0],
                H_view[k].shape[2],
                H_view[k].shape[3],
                H_view[k].shape[2] * H_view[k].shape[3],
            )
            for k in range(1, n - 1)
        ]

        H_sk = None
        if not identity_simple:
            H_sk = []
            for j in range(1, n - 1):
                Hj = np.asarray(H_view[j])
                HjT = np.ascontiguousarray(Hj.transpose(1, 2, 0, 3))
                H_sk.append(np.ascontiguousarray(c_reshape(HjT, -1, Hj.shape[0] * Hj.shape[3])))

        H_cap: list[np.ndarray | None] = [None] * n
        if not identity_simple:
            for j in range(1, n - 1):
                Hj = np.asarray(H_view[j])
                HjT = np.ascontiguousarray(Hj.transpose(1, 2, 0, 3))
                H_cap[j] = np.ascontiguousarray(
                    c_reshape(HjT, Hj.shape[1] * Hj.shape[2], Hj.shape[0] * Hj.shape[3])
                )

        out = {
            "identity_simple": identity_simple,
            "V": int(V),
            "H_view": H_view,
            "H0_flat": H0_flat,
            "H_env_T_erl": H_env_T_erl,
            "env_shapes": env_shapes,
            "H_sk": H_sk,
            "H_cap": H_cap,
        }

        if H_dict is not None:
            H_dict[key] = out
        return out

    @staticmethod
    def cache_psi(psi, *, V: int) -> dict:
        V = int(V)
        d = SRCViewCache._obj_dict(psi)
        if d is not None:
            cached = d.get(("src_psi", V))
            if cached is not None:
                return cached

        n = int(psi.N)
        psi0 = np.ascontiguousarray(psi[0])

        psi_env: list[np.ndarray] = []
        for k in range(1, n - 1):
            pk = np.asarray(psi[k])
            pkT = np.ascontiguousarray(pk.transpose(1, 0, 2))
            psi_env.append(c_reshape(pkT, pkT.shape[0] * pkT.shape[1], pkT.shape[2]))

        psi_sk = []
        for j in range(1, n - 1):
            pj = np.asarray(psi[j])
            psi_sk.append(np.ascontiguousarray(c_reshape(pj, pj.shape[0], -1)))

        psi_cap: list[np.ndarray | None] = [None] * n
        for j in range(1, n - 1):
            pj = np.asarray(psi[j])
            pjT = np.ascontiguousarray(pj.transpose(2, 1, 0))
            psi_cap[j] = c_reshape(pjT, pjT.shape[0] * pjT.shape[1], pjT.shape[2])

        out = {"psi0": psi0, "psi_env": psi_env, "psi_sk": psi_sk, "psi_cap": psi_cap}

        if d is not None:
            d[("src_psi", V)] = out
        return out

    @staticmethod
    def term_views(H, psi, *, assume_identity: bool = False) -> TermViews:
        hc = SRCViewCache.cache_H(H, assume_identity=assume_identity)
        pc = SRCViewCache.cache_psi(psi, V=int(hc["V"]))

        return TermViews(
            identity_simple=bool(hc["identity_simple"]),
            V=int(hc["V"]),
            H_view=hc["H_view"],
            H0_flat=hc["H0_flat"],
            H_env_T_erl=hc["H_env_T_erl"],
            env_shapes=hc["env_shapes"],
            H_sk=hc["H_sk"],
            H_cap=hc["H_cap"],
            psi0=pc["psi0"],
            psi_env=pc["psi_env"],
            psi_sk=pc["psi_sk"],
            psi_cap=pc["psi_cap"],
        )

    @staticmethod
    def build(H_list, psi_list, *, identity_flags=None) -> tuple[list[TermViews], int]:
        t = int(len(H_list))
        if t <= 0:
            raise ValueError("H_list must be non-empty")
        if int(len(psi_list)) != t:
            raise ValueError("H_list and psi_list must have the same length")

        if identity_flags is None:
            identity_flags = [False] * t
        if int(len(identity_flags)) != t:
            raise ValueError("identity_flags must have length len(H_list)")

        caches: list[TermViews] = []
        V0 = None
        for i in range(t):
            tv = SRCViewCache.term_views(H_list[i], psi_list[i], assume_identity=bool(identity_flags[i]))
            caches.append(tv)
            if V0 is None:
                V0 = int(tv.V)
            elif int(tv.V) != int(V0):
                raise ValueError("All MPOs must share the same inferred physical dimension V")

        return caches, int(V0 if V0 is not None else 0)



