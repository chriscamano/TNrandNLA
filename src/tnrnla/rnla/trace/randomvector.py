import numpy as np


class rvec:
    def __init__(
        self,
        d,
        mode="gaussian",
        seed=None,
        *,
        ll=None,
        kr_base_mode=None,
    ):
        self.d = int(d)
        if self.d < 1:
            raise ValueError("d must be >= 1")
        self.mode = str(mode).lower().strip()
        self.ll = None if ll is None else int(ll)
        self.kr_base_mode = None if kr_base_mode is None else str(kr_base_mode).lower().strip()
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    def sample(self, k=1):
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        if k == 1:
            return self._sample_one()
        return np.column_stack([self._sample_one() for _ in range(k)])

    def _sample_one(self):
        if self.mode in ("kr", "kron", "kronecker"):
            if self.ll is None or self.ll < 1:
                raise ValueError("KR mode requires ll >= 1")
            base_mode = self.kr_base_mode if self.kr_base_mode is not None else "gaussian"
            v = self._sample_one_base(base_mode)
            for _ in range(self.ll - 1):
                v = np.kron(v, self._sample_one_base(base_mode))
            return v
        return self._sample_one_base(self.mode)

    def _sample_one_base(self, mode):
        d = self.d
        rng = self.rng

        if mode in ("gaussian", "normal"):
            return rng.standard_normal(d).astype(np.float64, copy=False)

        if mode in ("uniform", "real_uniform", "unif"):
            a = np.sqrt(3.0)
            return rng.uniform(-a, a, size=d).astype(np.float64, copy=False)

        if mode in ("rademacher", "rade"):
            return (2.0 * rng.integers(0, 2, size=d, dtype=np.int8).astype(np.float64) - 1.0)

        if mode in ("sphere", "spherical"):
            v = rng.standard_normal(d).astype(np.float64, copy=False)
            nrm = float(np.linalg.norm(v))
            if nrm == 0.0:
                v[0] = 1.0
                nrm = 1.0
            return v * (np.sqrt(float(d)) / nrm)

        if mode in ("complex_gaussian", "cgaussian", "cnormal"):
            z1 = rng.standard_normal(d)
            z2 = rng.standard_normal(d)
            return ((z1 + 1j * z2) / np.sqrt(2.0)).astype(np.complex128, copy=False)

        if mode in ("complex_uniform", "cunif", "cuniform"):
            a = np.sqrt(3.0)
            z1 = rng.uniform(-a, a, size=d)
            z2 = rng.uniform(-a, a, size=d)
            return ((z1 + 1j * z2) / np.sqrt(2.0)).astype(np.complex128, copy=False)

        if mode in ("complex_rademacher", "crademacher", "crade", "complex_radehamcher", "complex_rade"):
            r1 = 2.0 * rng.integers(0, 2, size=d) - 1.0
            r2 = 2.0 * rng.integers(0, 2, size=d) - 1.0
            return ((r1 + 1j * r2) / np.sqrt(2.0)).astype(np.complex128, copy=False)

        if mode in ("steinhaus", "unitcircle"):
            theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
            return np.exp(1j * theta).astype(np.complex128, copy=False)

        if mode in ("complex_sphere", "csphere", "cspherical"):
            z1 = rng.standard_normal(d)
            z2 = rng.standard_normal(d)
            v = (z1 + 1j * z2) / np.sqrt(2.0)
            nrm = float(np.linalg.norm(v))
            if nrm == 0.0:
                v[0] = 1.0 + 0.0j
                nrm = 1.0
            return (v * (np.sqrt(float(d)) / nrm)).astype(np.complex128, copy=False)

        raise ValueError(f"Unknown probe mode: {mode!r}")
