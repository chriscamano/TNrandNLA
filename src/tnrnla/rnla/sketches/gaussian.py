import numpy as np


class Gaussian:
    """Dense Gaussian sketch S in R^{d x m} with y = S @ x."""

    def __init__(self, d, m, *, seed=None, normalize=True, dtype=np.float64):
        self.d = int(d)
        self.m = int(m)
        if self.d < 1 or self.m < 1:
            raise ValueError("d and m must be >= 1")
        self.normalize = bool(normalize)
        self.dtype = np.dtype(dtype)

        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._S = rng.standard_normal((self.d, self.m)).astype(self.dtype, copy=False)
        if self.normalize:
            self._S *= 1.0 / np.sqrt(float(self.d))

    @property
    def matrix(self):
        return self._S

    def apply(self, x):
        X = np.asarray(x)
        if X.ndim == 1:
            if X.shape[0] != self.m:
                raise ValueError("x has incompatible dimension")
            return self._S @ X
        if X.ndim == 2:
            if X.shape[0] != self.m:
                raise ValueError("x has incompatible leading dimension")
            return self._S @ X
        raise ValueError("x must be 1D or 2D")

    def __matmul__(self, x):
        return self.apply(x)
