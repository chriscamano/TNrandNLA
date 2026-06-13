from .xnystrace import xnystrace
from .common import ensure_oracle_and_dim


def xnystrace_resphere(
    matvec_oracle,
    num_queries,
    dimension=-1,
    *,
    vec_type="gaussian",
    seed=None,
    ridge_H=None,
    ridge_G=None,
    eps_denom=1e-12,
    alpha_clip=1e8,
    chol_jitter0=0.0,
):
    _, n = ensure_oracle_and_dim(matvec_oracle, dimension)

    tr, est, _ = xnystrace(
        matvec_oracle,
        n=int(n),
        s=int(num_queries),
        vec_type=vec_type,
        seed=seed,
        ridge_H=ridge_H,
        ridge_G=ridge_G,
        eps_denom=eps_denom,
        alpha_clip=alpha_clip,
        chol_jitter0=chol_jitter0,
        resphere=True,
    )
    return tr, est
