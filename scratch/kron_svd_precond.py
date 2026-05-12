"""
Variant of create_kronecker_precond that uses best rank-1 SVD / HOOI of the
weight tensor ws_nd for per-axis spectral factors, instead of the origin-pinned
slice used by the current efgpnd implementation.

Motivation: for isotropic Matérn, S(ξ) ∝ (c + |ξ|²)^(−ν−d/2) is not a product
of 1D factors. The origin-pinned slice has 12–34% relative Frobenius error vs
the full ws tensor, and the *best* rank-1 approximation has 8–16% error.
SVD gives the best rank-1 exactly (d=2); HOOI gives it in ~3 iterations (d=3).

Exposes: create_kronecker_svd_precond(...) with the same signature as
create_kronecker_precond(...).
"""

from __future__ import annotations
import torch


def best_rank1_factors(W: torch.Tensor, *, n_iter: int = 5):
    """
    Best rank-1 approximation of a d-way tensor W as c * ⊗_k α_k, where α_k are
    unit vectors. For d=2 this is exact via SVD; for d≥3 we run HOOI for n_iter
    sweeps (it's monotone in Frobenius norm). Returns (α_list, c) such that
    ⊗_k α_k * c ≈ W. Also returns the relative Frobenius error.
    """
    d = W.ndim
    m = W.shape[0]
    assert all(s == m for s in W.shape), f"expected cube-shaped tensor, got {W.shape}"
    w_norm = torch.linalg.norm(W)
    if w_norm.item() == 0.0:
        alphas = [torch.zeros(m, dtype=W.dtype, device=W.device) for _ in range(d)]
        alphas[0][0] = 1.0
        return alphas, torch.tensor(0.0, dtype=W.dtype, device=W.device), 0.0

    if d == 1:
        c = w_norm.to(W.dtype)
        return [W / c], c, 0.0

    # Init each α_k as the top-1 left singular vector of the mode-k unfolding.
    alphas = []
    for k in range(d):
        # Move axis k to front and flatten the rest.
        Wk = torch.movedim(W, k, 0).reshape(m, -1)
        U, _, _ = torch.linalg.svd(Wk, full_matrices=False)
        alphas.append(U[:, 0].contiguous())

    # HOOI sweeps: for each axis k, contract W with all other α_j to get g_k;
    # renormalize α_k = g_k / ||g_k||.
    for _ in range(n_iter):
        for k in range(d):
            # Contract all axes except k with their respective α_j^conj.
            Wx = W
            # Multiply-reduce each non-k axis.  Do it in reverse so axis indices
            # stay stable as we collapse.
            for j in reversed(range(d)):
                if j == k:
                    continue
                Wx = torch.tensordot(Wx, alphas[j].conj(), dims=([j], [0]))
            # Wx is now a vector of length m along axis k.
            g = Wx
            nrm = torch.linalg.norm(g)
            if nrm.item() > 0:
                alphas[k] = g / nrm

    # Compute scalar c = W ×_1 α_1^H ×_2 ... ×_d α_d^H
    c = W
    for j in reversed(range(d)):
        c = torch.tensordot(c, alphas[j].conj(), dims=([j], [0]))
    # c is now a scalar tensor

    # Relative Frobenius error of the best rank-1 approximation.
    # ||W - c ⊗ α||² = ||W||² - |c|²  (since ⊗α is unit Frobenius)
    err = torch.sqrt(torch.clamp(w_norm**2 - c.abs()**2, min=0.0)) / w_norm
    return alphas, c, float(err.item())


def create_kronecker_svd_precond(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                                 device=None, cdtype=torch.complex128,
                                 rdtype=torch.float64, hooi_iter: int = 5):
    """
    Same contract as efgpnd.create_kronecker_precond, except the per-axis
    spectral factors D_k are drawn from the best rank-1 factorization of the
    weight tensor ws_nd = ws.view(m, m, ..., m), via SVD (d=2) or HOOI (d≥3).

    For product kernels (SE, product-Matérn), ws is rank-1 exactly and the
    result matches the origin-pinned version to machine precision.
    For isotropic Matérn, the rank-1 error drops from ~20–34% (origin-pin) to
    the best possible rank-1 residual (~8–16%).

    Returns M_inv(v) callable.
    """
    m = int(mtot_1d)
    M_total = m ** d
    assert ws.numel() == M_total, f"ws size {ws.numel()} != m^d = {M_total}"
    if device is None:
        device = ws.device

    if torch.is_tensor(sigmasq_scalar):
        sigsq = float(sigmasq_scalar.detach().real.item())
    else:
        sigsq = float(sigmasq_scalar)

    ws_nd = ws.view(*(m,) * d).to(cdtype)

    # --- 1D factors from best rank-1 approximation of ws_nd
    alphas, c_w, rank1_err = best_rank1_factors(ws_nd, n_iter=hooi_iter)
    # Fold the scalar c_w equally into each axis: D_k = c_w^(1/d) α_k.
    # Guard against c_w near zero.
    if c_w.abs().item() == 0.0:
        c_w = c_w + torch.finfo(rdtype).tiny
    c_root = c_w ** (1.0 / d)
    Ds = [(c_root * a).to(cdtype) for a in alphas]

    # --- 1D convolution vectors (unchanged: origin-pin marginal of v_kernel)
    v_shape = v_kernel.shape
    if any(s != v_shape[0] for s in v_shape):
        raise ValueError(f"Expected isotropic v_kernel shape, got {v_shape}")
    L = v_shape[0]
    ctr_v = (L - 1) // 2

    v_ctr_val = v_kernel[(ctr_v,) * d].to(cdtype)
    if v_ctr_val.abs().item() == 0.0:
        v_ctr_val = v_ctr_val + torch.finfo(rdtype).tiny
    v_norm = v_ctr_val ** ((d - 1) / d)

    Vs = []
    Lams = []
    for k in range(d):
        slc = [ctr_v] * d
        slc[k] = slice(None)
        v_1d = (v_kernel[tuple(slc)].to(cdtype)) / v_norm
        a = torch.arange(m, device=device)
        T_k = v_1d[(a[:, None] - a[None, :]) + ctr_v]
        H_k = Ds[k][:, None] * T_k * Ds[k][None, :]
        H_k = 0.5 * (H_k + H_k.conj().T)
        Lam_k, V_k = torch.linalg.eigh(H_k)
        Lams.append(Lam_k.to(rdtype).clamp_min(0.0))
        Vs.append(V_k.to(cdtype))

    lam_prod = torch.ones(*(m,) * d, device=device, dtype=rdtype)
    for k in range(d):
        shape = [1] * d
        shape[k] = m
        lam_prod = lam_prod * Lams[k].view(shape)
    diag_inv = (1.0 / (lam_prod + sigsq)).to(cdtype)

    def _apply_kron(t, mats, hermitian):
        nd = t.ndim
        for k in range(d):
            ax = nd - d + k
            t = torch.movedim(t, ax, -1)
            if hermitian:
                t = t @ mats[k].conj()
            else:
                t = t @ mats[k].transpose(-1, -2)
            t = torch.movedim(t, -1, ax)
        return t

    def M_inv(v):
        is_batch = v.ndim > 1
        if is_batch:
            B = v.shape[0]
            t = v.to(cdtype).reshape(B, *(m,) * d)
        else:
            t = v.to(cdtype).reshape(*(m,) * d)
        t = _apply_kron(t, Vs, hermitian=True)
        t = t * diag_inv
        t = _apply_kron(t, Vs, hermitian=False)
        if is_batch:
            return t.reshape(B, -1)
        return t.reshape(-1)

    M_inv.rank1_err = rank1_err
    return M_inv
