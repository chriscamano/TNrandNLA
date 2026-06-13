from .xtrace import xtrace


def xtrace_resphere(matvec_oracle, num_queries, dimension=-1, *, vec_type="gaussian", seed=None):
    return xtrace(
        matvec_oracle,
        num_queries,
        dimension=dimension,
        vec_type=vec_type,
        seed=seed,
        resphere=True,
    )
