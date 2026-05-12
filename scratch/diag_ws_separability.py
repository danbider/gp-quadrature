"""
Diagnose: how well does the spectral weight vector ws factor as a rank-1 tensor
product (which is what the Kron preconditioner assumes) for SE vs Matern?

For a truly separable spectral density, rank-1 error is 0 (to finufft precision).
For an isotropic Matern, S(ξ) ∝ (a + |ξ|²)^(-p) does NOT factor, so rank-1 error
> 0 and Kron's core assumption is wrong even on product data.
"""

from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from kernels.squared_exponential import SquaredExponential
from kernels.matern import Matern
from utils.kernels import get_xis

DTYPE = torch.float64
CDTYPE = torch.complex128
torch.set_default_dtype(DTYPE)


def _ws_on_grid(kernel, d, L, eps=1e-3):
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L,
                              use_integral=True, l2scaled=False)
    xis_1d = xis_1d.to(DTYPE)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).view(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(CDTYPE) * (h ** d))
    return ws.view(*(mtot,) * d), mtot


def rank1_error(ws_nd):
    """Frobenius error of the best rank-1 outer-product approximation of ws_nd."""
    d = ws_nd.ndim
    m = ws_nd.shape[0]
    # Use the origin-pinned factorization (what Kron actually uses).
    ctr = m // 2
    ws_ctr = ws_nd[(ctr,) * d]
    norm = ws_ctr ** ((d - 1) / d)
    Ds = []
    for k in range(d):
        slc = [ctr] * d
        slc[k] = slice(None)
        Ds.append(ws_nd[tuple(slc)] / norm)  # (m,)
    # reconstruct ws_rank1 = ⊗_k Ds[k]  with shape (m,)*d
    shape0 = [1] * d
    shape0[0] = m
    rec = Ds[0].view(*shape0).clone()
    for k in range(1, d):
        shape = [1] * d
        shape[k] = m
        rec = rec * Ds[k].view(*shape)
    rec = rec.expand(*(m,) * d).contiguous()
    err = torch.linalg.norm(ws_nd - rec) / torch.linalg.norm(ws_nd)
    # Also report best-possible rank-1 via SVD along axis 0 (unfolds multi-way
    # tensor; this is a lower bound on rank-1 but not true multi-way rank-1).
    mat = ws_nd.reshape(m, -1)
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    best_rank1_lb = (S[1:].real.pow(2).sum().sqrt() / S.real.pow(2).sum().sqrt())
    return float(err.real.item()), float(best_rank1_lb.item())


cases = [
    ("SE", SquaredExponential),
    ("Matern-1.5", lambda d: Matern(dimension=d, nu=1.5)),
    ("Matern-2.5", lambda d: Matern(dimension=d, nu=2.5)),
]

print(f"{'kernel':<15} {'d':>2} {'ls':>5} {'m':>5} {'origin-pin err':>14} {'SVD-1 err':>12}")
for name, mkker in cases:
    for d in (2, 3):
        for ls in (0.25 if d == 2 else 0.4,):
            if name == "SE":
                k = SquaredExponential(dimension=d)
            else:
                k = mkker(d)
            k.set_hyper("lengthscale", ls)
            k.set_hyper("variance", 1.0)
            L = 2.0 * 2  # same L as the bench (range of square/disk)
            ws_nd, m = _ws_on_grid(k, d, L)
            err, lb = rank1_error(ws_nd)
            print(f"{name:<15} {d:>2} {ls:>5} {m:>5} {err:>14.3e} {lb:>12.3e}")
