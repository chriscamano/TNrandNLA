import numpy as np
from tnrnla.linalg.incrementalqr.incrementalqr import IncrementalQR
from ..mps import MPS
from ..stopping import Cutoff
from ..other.tensor_ops import flatten_left, batch_rightmul
from .src_cacheview import SRCViewCache


"""
Sucessive Randomzied Compression (SRC): 
This file contains BLAS3 and einsum level implementations of SRC from the paper Successive randomized compression: A randomized algorithm for the compressed MPO-MPS product
https://arxiv.org/abs/2504.06475. SRC uses randomized low rank approximation the compressed MPO-MPS product and dynamically determines 
bond dimensions using a leave-one-out estimator.

NOTE: This method uses an optimized implementation that passes a cache of tensor reshapes for optimized performance in other algorithms 
NOTE: This method may run slower when the stopping condition is of type Cutoff if the MKL version of incrementalQR is not used. 
Author: Chris Camaño Circa: 2023-2026
"""

class EnvStore:
    __slots__ = ("n", "buf", "cap", "m", "pstart", "pZ")

    def __init__(self, n: int, init_cap: int = 0):
        self.n = int(n)
        self.buf = [None] * max(0, self.n - 1)
        self.cap = [0] * max(0, self.n - 1)
        self.m = 0
        self.pstart = 0
        self.pZ = 0
        if init_cap > 0:
            init_cap = int(init_cap)
            for k in range(self.n - 1):
                self.cap[k] = init_cap

    def __len__(self):
        return self.m

    def _ensure(self, k: int, need: int, tail_shape, dtype, filled: int):
        b = self.buf[k]
        if b is None:
            c = self.cap[k]
            if c <= 0:
                c = 64
            if c < need:
                c = need
            self.cap[k] = int(c)
            self.buf[k] = np.empty((c,) + tail_shape, dtype=dtype, order="C")
            return
    
        c = self.cap[k]
        if need <= c:
            return
    
        nc = c + (c >> 1) + 1
        if nc < need:
            nc = need
        nb = np.empty((nc,) + b.shape[1:], dtype=b.dtype, order="C")
    
        filled = int(filled)
        if filled:
            nb[:filled] = b[:filled]
    
        self.buf[k] = nb
        self.cap[k] = int(nc)

    def append_batch(self, k: int, arr: np.ndarray):
        arr = np.asarray(arr)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
    
        Z = int(arr.shape[0])
    
        if k == 0:
            old_m = self.m
            self.pstart = old_m
            self.pZ = Z
            need = old_m + Z
    
            self._ensure(k, need, arr.shape[1:], arr.dtype, filled=old_m)
            self.buf[k][old_m:need] = arr
            self.m = need
            return
    
        if Z != self.pZ:
            raise ValueError("batch size mismatch")
    
        need = self.pstart + Z
        self._ensure(k, need, arr.shape[1:], arr.dtype, filled=self.pstart)
        self.buf[k][self.pstart:need] = arr

    def get(self, k: int, a: int, b: int):
        bk = self.buf[k]
        if bk is None:
            raise ValueError("env buffer not initialized")
        return bk[a:b]


def make_envs(tv, envs: EnvStore, j: int, sketchdim_now: int, block_env: int = 16, rng=None) -> None:
    start = len(envs)
    if start >= sketchdim_now:
        return
    if tv.identity_simple:
        raise ValueError("identity_simple=True not supported")

    V = int(tv.V)
    H0 = tv.H_view[0]
    H0_s1 = int(H0.shape[1])
    H0_s2 = int(H0.shape[2])

    H0_flat = tv.H0_flat
    if H0_flat is None:
        raise ValueError("missing H0_flat")

    H_env = tv.H_env_T_erl
    if H_env is None:
        raise ValueError("missing H_env_T_erl")

    psi0 = tv.psi0
    psi_env = tv.psi_env
    env_shapes = tv.env_shapes

    if rng is None:
        randn2 = np.random.randn
        def draw(Z, V):  # (Z,V) C-contig
            return randn2(Z, V)
    else:
        stdn = rng.standard_normal
        def draw(Z, V):  # (Z,V) C-contig
            return stdn((Z, V))

    remaining = sketchdim_now - start
    while remaining > 0:
        Z = min(block_env, remaining)

        g0 = draw(Z, V)
        r0 = g0 @ H0_flat
        prev = (r0.reshape(Z, H0_s1, H0_s2) @ psi0)
        if not prev.flags["C_CONTIGUOUS"]:
            prev = np.ascontiguousarray(prev)
        envs.append_batch(0, prev)

        for k in range(1, j):
            L, E, R, ER = env_shapes[k - 1]

            gk = draw(Z, V)
            vec = gk @ H_env[k - 1]

            t2T = vec.reshape(Z, ER, L)
            t3 = (t2T @ prev).reshape(Z, E, R, prev.shape[2])

            prev = batch_rightmul(t3.reshape(Z, E, R * prev.shape[2]), psi_env[k - 1])
            if not prev.flags["C_CONTIGUOUS"]:
                prev = np.ascontiguousarray(prev)

            envs.append_batch(k, prev)

        remaining -= Z



def make_sketch(
    sketch: np.ndarray,
    tv,
    psi,
    envs: EnvStore,
    done_sketches: int,
    sketchdim_now: int,
    j: int,
    cap_flat: np.ndarray | None,
) -> None:
    Z = sketchdim_now - done_sketches
    if Z <= 0:
        return
    all_env = envs.get(j - 1, done_sketches, sketchdim_now)
    Hj = tv.H_view[j]
    psij = psi[j]
    if j == psi.N - 1:
      
        temp = all_env @ psij
        Hmat = Hj.reshape(Hj.shape[1], -1)
        sketch[:, done_sketches:sketchdim_now] = Hmat @ temp.reshape(Z, -1).T
        return
    H_sk = tv.H_sk
    if H_sk is None:
        raise ValueError("missing H_sk")
    psi_sk = tv.psi_sk
    temp = all_env @ psi_sk[j - 1]
    temp = temp.reshape(Z, -1, psij.shape[1], psij.shape[2])
    temp_DdY = temp.reshape(Z, temp.shape[1] * temp.shape[2], temp.shape[3])
    C = H_sk[j - 1][None, :, :] @ temp_DdY
    t2 = C.reshape(Z, Hj.shape[1], Hj.shape[2], temp_DdY.shape[2])
    t2d = t2.reshape(Z * Hj.shape[1], Hj.shape[2] * temp_DdY.shape[2])
    if cap_flat is None:
        raise ValueError("cap_flat required")
    blk2d = t2d @ cap_flat
    blk = blk2d.reshape(Z, Hj.shape[1], cap_flat.shape[1]).transpose(1, 2, 0)
    sketch[:, :, done_sketches:sketchdim_now] = blk


def build_cap(tv, psi, j: int, Q: np.ndarray, cap, *, is_boundary: bool, Hj_r=None, psi_r=None):
    Hj = tv.H_view[j]
    psij = psi[j]
    if is_boundary:
        Hj_mat = Hj.transpose(1, 0, 2).reshape(Hj.shape[1], -1)
        temp = (np.conj(Q).T @ Hj_mat).reshape(Q.shape[1], Hj.shape[0], Hj.shape[2])
        return (temp.reshape(-1, temp.shape[-1]) @ psij.reshape(psij.shape[0], -1).T).reshape(
            temp.shape[0], temp.shape[1], psij.shape[0]
        )
    if cap is None or Hj_r is None or psi_r is None:
        raise ValueError("cap/Hj_r/psi_r required")
    cap_r = cap.reshape(cap.shape[0], Hj.shape[2] * psij.shape[2])
    Qr = np.conj(Q).transpose(0, 2, 1).reshape(Q.shape[0] * Q.shape[2], cap.shape[0])
    tmp = (Qr @ cap_r).reshape(Q.shape[0], Q.shape[2], Hj.shape[2], psij.shape[2])
    left1 = np.ascontiguousarray(tmp.transpose(1, 3, 0, 2)).reshape(Q.shape[2] * psij.shape[2], Q.shape[0] * Hj.shape[2])
    mid = (left1 @ Hj_r).reshape(Q.shape[2], psij.shape[2], Hj.shape[0], Hj.shape[3])
    left2 = np.ascontiguousarray(mid.transpose(0, 2, 1, 3)).reshape(Q.shape[2] * Hj.shape[0], psij.shape[2] * Hj.shape[3])
    return (left2 @ psi_r).reshape(Q.shape[2], Hj.shape[0], psij.shape[0])



def SRC(
    H,
    psi,
    stop=None,
    sketchdim: int = None,
    sketchincrement: int = None,
    finalround=None,
    accuracychecks: bool = False,
    cpp: bool = True,
    dtype=np.float64,
    rng=None,
    seed=None,
):
    if seed is not None:
        rng = np.random.default_rng(int(seed))
    elif rng is None:
        rng = np.random.default_rng()

    if stop is None:
        stop = Cutoff(1e-8)

    n = H.N
    if n != psi.N:
        raise ValueError("lengths of MPO and MPS do not match")
    if n == 1:
        raise NotImplementedError("n=1 not implemented")

    maxdim = stop.maxdim
    outputdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff

    if outputdim is None:
        if maxdim is None:
            if maxlinkdim is None:
                raise ValueError("maxlinkdim not available")
            maxdim = maxlinkdim(H) * maxlinkdim(psi)
        mindim = max(mindim, 1)
        if sketchdim is None:
            sketchdim = max(4, min(maxdim, 32))
        if sketchincrement is None:
            sketchincrement = sketchdim
    else:
        sketchdim = sketchincrement = outputdim
        maxdim = mindim = outputdim

    tv = SRCViewCache.term_views(H, psi, assume_identity=False)

    if tv.H0_flat is not None and not tv.H0_flat.flags["C_CONTIGUOUS"]:
        tv.H0_flat = np.ascontiguousarray(tv.H0_flat)

    if tv.H_env_T_erl is not None:
        for k in range(len(tv.H_env_T_erl)):
            a = tv.H_env_T_erl[k]
            if not a.flags["C_CONTIGUOUS"]:
                tv.H_env_T_erl[k] = np.ascontiguousarray(a)

    if tv.H_sk is not None:
        for k in range(len(tv.H_sk)):
            a = tv.H_sk[k]
            if not a.flags["C_CONTIGUOUS"]:
                tv.H_sk[k] = np.ascontiguousarray(a)

    if tv.H_cap is not None:
        for k in range(1, n - 1):
            a = tv.H_cap[k]
            if a is not None and not a.flags["C_CONTIGUOUS"]:
                tv.H_cap[k] = np.ascontiguousarray(a)

    base_dtype = np.dtype(dtype)
    sketch_dtype = np.result_type(np.asarray(tv.H_view[0]).dtype, np.asarray(psi[0]).dtype, base_dtype)

    envs = EnvStore(n)
    cap = None
    cap_dim = 1
    V = int(tv.V)
    psi_out = [None] * n

    for j in reversed(range(1, n)):
        is_boundary = (j == n - 1)
        Hj = tv.H_view[j]
        psij = psi[j]

        if is_boundary:
            prod_bonds = Hj.shape[0] * psij.shape[0]
        else:
            prod_bonds = max(Hj.shape[0] * psij.shape[0], Hj.shape[2] * psij.shape[2])

        maxdim_j = min(prod_bonds, maxdim, V * cap_dim)
        mindim_j = min(mindim, maxdim_j)
        sketchdim_j = max(min(sketchdim, maxdim_j), mindim_j)

        if is_boundary:
            sketch = np.empty((V, maxdim_j), dtype=sketch_dtype, order="C")
        else:
            sketch = np.empty((V, cap_dim, maxdim_j), dtype=sketch_dtype, order="C")

        sketch_sq = 0.0
        done_sketches = 0
        qr = None

        cap_flat = Hj_r = psi_r = None
        if not is_boundary:
            cap_flat = flatten_left(cap.transpose(1, 2, 0))
            Hj_r = tv.H_cap[j]
            psi_r = tv.psi_cap[j]

        while True:
            make_envs(tv, envs, j, sketchdim_j, block_env=64, rng=rng)
            make_sketch(sketch, tv, psi, envs, done_sketches, sketchdim_j, j, cap_flat)

            new_blk = sketch[..., done_sketches:sketchdim_j]
            sketch_sq += float(np.vdot(new_blk, new_blk).real)

            if is_boundary:
                qr = IncrementalQR(sketch[:, :sketchdim_j].copy(order="F")) if qr is None else (qr.append(new_blk) or qr)
            else:
                flat0 = flatten_left(sketch[:, :, :sketchdim_j]).copy(order="F")
                qr = IncrementalQR(flat0) if qr is None else (qr.append(flatten_left(new_blk)) or qr)

            done_sketches = sketchdim_j

            if outputdim is not None or sketchdim_j == maxdim_j:
                done = True
            else:
                err_est = qr.error_estimate()
                norm_est = (sketch_sq**0.5) / (sketchdim_j**0.5)
                done = (err_est <= cutoff * norm_est) and (sketchdim_j >= mindim_j)

            if done:
                Q = qr.get_q()
                if is_boundary:
                    psi_out[j] = Q.transpose(1, 0)
                    cap = build_cap(tv, psi, j, Q, cap, is_boundary=True)
                else:
                    cap_dim_prev = cap_dim
                    Q3 = Q.reshape((V, cap_dim_prev, sketchdim_j))
                    psi_out[j] = Q3.transpose(2, 0, 1)
                    cap = build_cap(tv, psi, j, Q3, cap, is_boundary=False, Hj_r=Hj_r, psi_r=psi_r)

                cap_dim = sketchdim_j
                # if accuracychecks and check_randomized_apply is not None:
                #     check_randomized_apply(
                #         H, psi, cap, psi_out, j + 1, verbose=False,
                #         cutoff=((100 * cutoff) if (maxdim is None) else np.Inf),
                #     )
                break

            sketchdim_j = min(maxdim_j, sketchdim_j + sketchincrement)

    temp = np.einsum("ijk,lk", cap, psi[0])
    psi_out[0] = np.einsum("ijk,ljk->il", tv.H_view[0], temp)

    out = MPS(psi_out, orthform="Left")
    if finalround:
        out.round(stop=stop)
    #out.orthR() 
    #out.orthL()
    return out


#  #Random contraction (einsums only)


def src_einsum(
    H,
    psi,
    stop=Cutoff(1e-6),
    sketchdim=1,
    sketchincrement=1,
    finalround=None,
    accuracychecks=False,
):
    # ===========================================================================
    # 1. Initialization
    # ===========================================================================
    n = len(H)

    if n != psi.N:
        raise ValueError("lengths of MPO and MPS do not match")

    if n == 1:
        raise NotImplementedError("MPO-MPS product for n=1 has not been implemented yet")
        # return [H[0] * psi[0]]

    psi_out = psi.N * [None]

    maxdim = stop.maxdim
    outputdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff

    if outputdim is None: # Adaptive determination of sketch dimension
        if maxdim is None:
            maxdim = maxlinkdim(H) * maxlinkdim(psi)
        mindim = max(mindim, 1)
    else:
        maxdim = outputdim
        mindim = outputdim
        sketchdim = outputdim
    
    envs = []
    cap = None
    cap_dim = 1

    visible_dim = H[0].shape[0]

    for j in reversed(range(1, n)):
        # ===========================================================================
        # 2. Dimension Handling
        # ===========================================================================
        if j == n - 1:
            prod_bond_dims = (
                H[j].shape[0] * psi[j].shape[0]
            )  # product of left bond
        else:
            prod_bond_dims = max(
                (H[j].shape[0] * psi[j].shape[0]),
                (H[j].shape[2] * psi[j].shape[2]),
            )  # either left or right

        current_maxdim = min(prod_bond_dims, maxdim, visible_dim * cap_dim)
        current_mindim = min(mindim, current_maxdim)
        current_sketchdim = max(min(sketchdim, current_maxdim), current_mindim)

        sketches_complete = 0

        if j == n - 1:
            sketch = np.zeros((visible_dim, current_sketchdim),dtype='complex')
        else:
            sketch = np.zeros((visible_dim, cap_dim, current_sketchdim),dtype='complex')

        while True:
            # ===========================================================================
            # 3. env formation
            # ===========================================================================                
            for idx in range(len(envs), current_sketchdim):
                env = []
                x = np.random.randn(visible_dim) # consider precompute
                temp= np.einsum('ijk,i->jk',H[0],x)
                env.append(temp @ psi[0])
                for k in range(1, j):
                    x = np.random.randn(visible_dim) # consider precompute
                    temp = np.einsum('ijkl,j->ikl', H[k], x)
                    temp = np.einsum('ijk,il->jkl', temp, env[k - 1])
                    temp = np.einsum('ijk,kjl->il', temp, psi[k])
                    env.append(temp)
                envs.append(env)

            # ===========================================================================
            # 4. sketch  formation
            # ===========================================================================
            for idx in range(sketches_complete, current_sketchdim):
                if j == n - 1:
                    # C[n-2] (Hψ)[n-1]
                    temp = envs[idx][j - 1] @ psi[j]
                    temp = np.einsum('ijk,ik->j',H[j],temp)
                    sketch[:, idx] = temp
                else:
                    # C[j-1] (Hψ)[j] S[j+1]
                    temp=np.einsum('ij,jkl->ikl',envs[idx][j - 1],psi[j])
                    temp=np.einsum('ijlk,ikn->jln',H[j], temp)
                    temp = np.einsum('ijk,ljk->il',temp,cap)
                    sketch[:, :, idx] = temp


            sketches_complete = current_sketchdim

            # ===========================================================================
            # 5. QR decomposition
            # ===========================================================================
            if j == n - 1:
                Q, R = np.linalg.qr(sketch)
            else:
                temp = sketch.reshape(cap_dim * visible_dim, current_sketchdim)
                Q, R = np.linalg.qr(temp)
                Q = Q.reshape((visible_dim, cap_dim, current_sketchdim))
            
            if not (outputdim is None):
                done = True
            else:
                # ===========================================================================
                # 7. error calc
                # ===========================================================================
                done = False
                if current_sketchdim == current_maxdim:
                    done = True
                else:
                    norm_est = np.linalg.norm(sketch) / np.sqrt(current_sketchdim)
                    G = np.linalg.inv(R.T)

                    err_est = np.sqrt(sum(np.linalg.norm(G, axis=0) ** (-2)) / current_sketchdim)
                    done = err_est <= cutoff * norm_est and current_sketchdim >= current_mindim
            
            # print("error estimation", err_est)
            # print("norm estimation",norm_est)
            # print("Validity of condition 1:",(err_est <= cutoff * norm_est and current_sketchdim >= current_mindim))
            # print("Validity of condition 2:",(current_sketchdim == current_maxdim))
            
            # ===========================================================================
            # 9. cap construction
            # ===========================================================================
            if done:
                if j == n - 1:
                    psi_out[j] = Q.transpose(1, 0)
                else:
                    psi_out[j] = Q.transpose(2, 0, 1)
                cap_dim = current_sketchdim

                if j == n - 1:
                    temp=np.einsum("ij,kil->jkl",np.conj(Q), H[j])
                    cap =np.einsum("ijk,lk->ijl",temp, psi[j])
                else:
                    #temp = np.einsum('ij,kjl->jkl',np.conj(Q),H[j])

                    temp = np.tensordot(np.conj(Q), cap, axes=(1, 0))
                    temp = np.einsum("ijkl,mikn->jlmn",temp, H[j])
                    cap = np.einsum("ijkl,mlj->ikm",temp, psi[j])
                if accuracychecks:
                    check_randomized_apply(H, psi, cap, psi_out, j+1, verbose=False, cutoff=((100*cutoff) if (maxdim is None) else np.Inf))
                break

            # ===========================================================================
            # 10. sketch correction
            # ===========================================================================
            current_sketchdim = min(current_maxdim, current_sketchdim + sketchincrement)
            old_sketch = sketch

            if j == n - 1:
                sketch = np.zeros(
                    (
                        old_sketch.shape[0],
                        current_sketchdim,
                    ),dtype='complex'
                )
                sketch[:, : old_sketch.shape[1]] = old_sketch
            else:
               
                sketch = np.zeros(
                    (
                        old_sketch.shape[0],
                        old_sketch.shape[1],
                        current_sketchdim,
                    ),dtype='complex'
                )
                sketch[:, :, : old_sketch.shape[2]] = old_sketch

    temp = np.einsum('ijk,lk',cap, psi[0])
    psi_out[0] = np.einsum('ijk,ljk->il',H[0],temp)
    psi_out= MPS(psi_out, orthform="Left")
    if not (finalround is None):
        psi_out.round(stop = finalround)
    return psi_out
