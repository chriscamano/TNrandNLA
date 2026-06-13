# src/tnrnla/linalg/__init__.py

from __future__ import annotations

from . import error as error
from . import expm as expm
from . import incrementalqr as incrementalqr
from . import krlyov as krlyov
from . import lra as lra
from . import orth as orth
from . import utils as utils

from .error import (
    relerr,
    nuc_sym,
    rel_fro,
    rel_nuclear,
    rel_trace,
    rel_fro_dense_over_mps,
    rel_nuc_dense_over_mps,
)
from .expm import  lan_exp, lanczos_fun
from .incrementalqr import IncrementalQR
from .krlyov import arnoldi
from .lra import truncated_eig, truncated_svd
from .orth import lq
from .utils import (
    sym,
    hermitian_sym,
    product_trace,
    cnormc_np,
    diagprod_np,
    sqcolnorms_np,
    sqrownorms_np,
    _inv_conj_transpose,
    _as_real_if_close,
    supfind,
    chol_shift_psd,
    safe_cholesky_upper_psd,
    pinv_hermitian,
)
# Provide a correctly spelled alias while keeping backward compatibility.
krylov = krlyov

__all__ = [
    "error",
    "orth",
    "expm",
    "krlyov",
    "krylov",
    "lra",
    "utils",
    "incrementalqr",
    "relerr",
    "sym",
    "hermitian_sym",
    "nuc_sym",
    "rel_fro",
    "rel_nuclear",
    "rel_trace",
    "rel_fro_dense_over_mps",
    "rel_nuc_dense_over_mps",
    "lq",
    "product_trace",
    "cnormc_np",
    "diagprod_np",
    "sqcolnorms_np",
    "sqrownorms_np",
    "_inv_conj_transpose",
    "_as_real_if_close",
    "supfind",
    "chol_shift_psd",
    "safe_cholesky_upper_psd",
    "pinv_hermitian",
    "truncated_svd",
    "truncated_eig",
    "lan_exp",
    "lanczos_fun",
    "arnoldi",
    "IncrementalQR",
]
