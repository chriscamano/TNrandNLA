import numpy as np

try:
    from scipy.fft import dct as _scipy_dct
except Exception:  # pragma: no cover
    _scipy_dct = None


def _is_power_of_two(n):
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


def _fwht_rows(X):
    """In-place unnormalized Walsh-Hadamard transform along axis 0."""
    Y = np.array(X, copy=True)
    n = int(Y.shape[0])
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            a = Y[i : i + h, ...]
            b = Y[i + h : i + 2 * h, ...]
            Y[i : i + h, ...] = a + b
            Y[i + h : i + 2 * h, ...] = a - b
        h *= 2
    return Y


class SRTT:
    """Subsampled Randomized Trigonometric Transform sketch.

    Applies `rounds` of random sign flips + transform (`dct`, `fft`, or `wht`),
    then row-subsamples to `d` rows and rescales by sqrt(m / d).
    """

    def __init__(self, d, m, rounds=1, *, transform="dct", seed=None):
        self.d = int(d)
        self.m = int(m)
        self.rounds = int(rounds)
        self.transform = str(transform).lower().strip()
        if self.d < 1 or self.m < 1:
            raise ValueError("d and m must be >= 1")
        if self.d > self.m:
            raise ValueError("d must be <= m for subsampling")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.transform not in ("dct", "fft", "wht"):
            raise ValueError("transform must be one of: 'dct', 'fft', 'wht'")
        if self.transform == "dct" and _scipy_dct is None:
            raise ImportError("scipy is required for transform='dct'")
        if self.transform == "wht" and not _is_power_of_two(self.m):
            raise ValueError("transform='wht' requires m to be a power of 2")

        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.signs = 2.0 * rng.integers(0, 2, size=(self.m, self.rounds), dtype=np.int8).astype(np.float64) - 1.0
        self.idx = rng.choice(self.m, size=self.d, replace=False)
        self.idx.sort()

    def _transform(self, X):
        if self.transform == "dct":
            return _scipy_dct(X, axis=0, norm="ortho")
        if self.transform == "fft":
            return np.fft.fft(X, axis=0, norm="ortho")
        # wht
        return _fwht_rows(X) / np.sqrt(float(self.m))

    def apply(self, x):
        X = np.asarray(x)
        squeeze = False
        if X.ndim == 1:
            if X.shape[0] != self.m:
                raise ValueError("x has incompatible dimension")
            X = X[:, None]
            squeeze = True
        elif X.ndim == 2:
            if X.shape[0] != self.m:
                raise ValueError("x has incompatible leading dimension")
        else:
            raise ValueError("x must be 1D or 2D")

        Y = X
        for i in range(self.rounds):
            Y = self._transform(self.signs[:, i : i + 1] * Y)

        Y = Y[self.idx, :]
        Y = np.sqrt(float(self.m) / float(self.d)) * Y
        return Y[:, 0] if squeeze else Y

    def __matmul__(self, x):
        return self.apply(x)


srtt = SRTT
