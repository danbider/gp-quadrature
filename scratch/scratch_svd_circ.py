"""
SVD-circulant preconditioner — user's Idea #3.

Motivation: plain Kronecker preconditioner uses the rank-1 axis slices of
v_kernel (T_k = slice of v with other axes pinned at origin). That is
EXACT when the BTTB kernel Toeplitz T factorises as ⊗_k T_k, which holds on
tensor-grid data but fails on non-product measures (SST with continents,
uniform in disk, ...).

Idea: replace rank-1 slices with a rank-R SVD of v_kernel, then use a
CIRCULANT approximation of each 1D Toeplitz factor so the whole operator is
diagonal in (F⊗F) basis. Per-apply cost is O(M log M) regardless of R.

Difference from scratch_kron_sok.py (Idea #2):
  - SoK-exact:  R separate Kron ops, exact eigendecomp per 1D Toeplitz
                (O(R d m^3) setup, O(R d M m) per apply).
  - SVD-circ:   R SVD terms collapsed into ONE diagonal in F⊗F via Chan
                circulant (O(R d m^2) setup, O(d M log m) per apply).

The cost of the circulant approximation is a bias: Chan's circulant is the
closest circulant in Frobenius. If T is "close to circulant" (mild boundary),
the loss is small and R≥2 picks up the non-product correction cheaply.

For D_k T_k D_k (Hermitian but NOT Toeplitz), Chan's circulant is computed
from the averaged wrapped-diagonals of the m×m matrix.

d=2 only for this prototype.

Run: ~/myenv/bin/python -u scratch/scratch_svd_circ.py
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import efgpnd as efgp_mod
from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential

torch.set_default_dtype(torch.float64)
DT = torch.float64
CDT = torch.complex128

_ORIG = efgp_mod.create_kronecker_precond
_state = {"installed": False, "last_info": ""}


def _factor_ws(ws, m, d, cdtype):
    """Rank-1 factor ws → d one-dimensional slices through the origin.

    Exact for product kernels (SE, product-Matérn). Same routine as used by
    the existing Kronecker preconditioner, so we reuse its D_k factors.
    """
    ws_nd = ws.view(*(m,) * d).to(cdtype)
    ctr = m // 2
    idx_ctr = (ctr,) * d
    ws_ctr = ws_nd[idx_ctr]
    if ws_ctr.abs().item() == 0.0:
        ws_ctr = ws_ctr + torch.finfo(torch.float64).tiny
    norm = ws_ctr ** ((d - 1) / d)
    Ds = []
    for k in range(d):
        slc = [ctr] * d
        slc[k] = slice(None)
        Ds.append((ws_nd[tuple(slc)] / norm).to(cdtype))
    return Ds


def _chan_circulant_eigs_DTD(d_k, sym, ctr_v, m):
    """Chan circulant eigenvalues of D T D where D=diag(d_k), T Toeplitz.

    T[i,j] = sym[(i-j) + ctr_v], i,j ∈ [0, m).
    The Chan circulant has first column
        c[l] = (1/m) * Σ_{i: (i-j) ≡ l mod m} (D T D)[i, j]
    and its eigenvalues are DFT(c).

    For (D T D)[i, j] = d[i] sym[i-j+ctr_v] d[j]:
        c[l] = (1/m) * ( sym[ctr_v + l] * α_plus[l]  +  sym[ctr_v + l - m] * α_minus[l] )
    where
        α_plus[l]  = Σ_{i=l}^{m-1} d[i] d[i-l]       (standard autocorrelation, no conj)
        α_minus[l] = Σ_{i=0}^{l-1} d[i] d[i-l+m]    (wrap-around correction)

    Complexity: O(m^2) per axis per rank. For m ≤ 1000 this is sub-millisecond.
    """
    # Compute α_plus, α_minus via direct sums (m ≤ 1000 is fine)
    # d_k: (m,) complex. For complex d, the product is without conjugation.
    alpha_plus = torch.empty(m, dtype=d_k.dtype, device=d_k.device)
    alpha_minus = torch.empty(m, dtype=d_k.dtype, device=d_k.device)
    # Vectorized: build (m, m) matrix of d[i] * d[j] and index by offset
    # autocorrelation at lag l: Σ_{i=l}^{m-1} d[i] d[i-l]
    dd = d_k[:, None] * d_k[None, :]  # (m, m)  dd[i,j] = d[i]*d[j]
    for l in range(m):
        # α_plus[l]: j = i - l,  i in [l, m-1],  j in [0, m-1-l]
        # This picks entries on the l-th lower diagonal of dd
        alpha_plus[l] = torch.diagonal(dd, offset=-l).sum()
        if l == 0:
            alpha_minus[l] = 0.0
        else:
            # α_minus[l]: j = i - l + m, i in [0, l-1], j in [m-l, m-1]
            # This picks entries on the (m-l)-th upper diagonal of dd (only l entries)
            alpha_minus[l] = torch.diagonal(dd, offset=(m - l)).sum()

    # sym is the 1D symbol of length L = 4m_conv+1 with center ctr_v.
    # sym[ctr_v + l]: valid for l ∈ [-(m-1), m-1]  (ctr_v ≥ m-1 since L=2m-1 at minimum).
    L = sym.shape[0]
    idx_pos = ctr_v + torch.arange(m, device=sym.device)          # ctr_v + l for l=0..m-1
    idx_neg = ctr_v + torch.arange(m, device=sym.device) - m      # ctr_v + l - m for l=0..m-1
    # idx_neg for l=0 is ctr_v - m = -1 in a size-L array; but α_minus[0]=0 kills that term.
    # Protect against out-of-bounds by clamping when α_minus is zero.
    idx_neg = idx_neg.clamp_min(0)
    sym_pos = sym[idx_pos]
    sym_neg = sym[idx_neg]

    c = (sym_pos * alpha_plus + sym_neg * alpha_minus) / m
    # DFT (eigenvalues of circulant with first column c)
    lam = torch.fft.fft(c)
    return lam  # (m,) complex


def build_svd_circ_precond(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                           *, R, device=None, cdtype=CDT, rdtype=torch.float64):
    """SVD+Chan-circulant preconditioner for d=2."""
    assert d == 2, "svd_circ prototype is d=2 only"
    m = int(mtot_1d)
    if device is None:
        device = ws.device
    sigsq = float(sigmasq_scalar.detach().real.item()
                  if torch.is_tensor(sigmasq_scalar)
                  else sigmasq_scalar)

    # 1D ws factors — reuse the same rank-1 factorisation as Kron precond.
    Ds = _factor_ws(ws, m, d, cdtype)

    # SVD of v_kernel matrix (L, L)
    V = v_kernel.to(cdtype)
    L = V.shape[0]
    ctr_v = (L - 1) // 2
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    R_use = min(R, S.numel())

    # Accumulate rank-R diagonal Λ[i,j] = Σ_r λ_1^(r)[i] λ_2^(r)[j]
    Lambda = torch.zeros((m, m), dtype=cdtype, device=device)
    sigs = []
    for r in range(R_use):
        s_r = S[r].to(cdtype)
        u_r = U[:, r]
        w_r = Vh[r, :].conj()

        # Absorb σ_r into axis-0 symbol; axis-1 uses conj(w_r).
        sym_1 = s_r * u_r     # (L,)
        sym_2 = w_r.conj()    # (L,)

        lam_1 = _chan_circulant_eigs_DTD(Ds[0], sym_1, ctr_v, m)
        lam_2 = _chan_circulant_eigs_DTD(Ds[1], sym_2, ctr_v, m)
        Lambda = Lambda + lam_1[:, None] * lam_2[None, :]
        sigs.append(float(S[r].detach().real.item()))

    Lambda_r = Lambda.real.clamp_min(0.0)
    diag_inv = (1.0 / (Lambda_r + sigsq)).to(cdtype)

    _state["last_info"] = (
        f"[SVDcirc R={R_use}] σ/σ_1: "
        f"{['{:.3g}'.format(s / sigs[0]) for s in sigs[:min(8, R_use)]]}  "
        f"Λ range=[{Lambda_r.min().item():.3g}, {Lambda_r.max().item():.3g}]  "
        f"σ²={sigsq:.3g}"
    )

    def M_inv(v):
        is_batch = v.ndim > 1
        if is_batch:
            B = v.shape[0]
            t = v.to(cdtype).reshape(B, m, m)
            vhat = torch.fft.fft2(t)
            vhat = vhat * diag_inv[None, :, :]
            y = torch.fft.ifft2(vhat)
            return y.reshape(B, m * m)
        else:
            t = v.to(cdtype).reshape(m, m)
            vhat = torch.fft.fft2(t)
            vhat = vhat * diag_inv
            y = torch.fft.ifft2(vhat)
            return y.reshape(m * m)

    return M_inv


def install_svd_circ(R):
    def patched(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                device=None, cdtype=CDT, rdtype=torch.float64):
        if d != 2:
            return _ORIG(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                         device=device, cdtype=cdtype, rdtype=rdtype)
        return build_svd_circ_precond(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                                      R=R, device=device, cdtype=cdtype, rdtype=rdtype)
    efgp_mod.create_kronecker_precond = patched
    _state["installed"] = True


def install_sok(R, weights="uniform"):
    """SoK-exact variant: R Kron ops with EXACT eigendecomp per 1D Toeplitz
    (from scratch_kron_sok.py). Higher setup cost (O(R d m^3)) but no
    circulant-approximation bias."""
    from scratch_kron_sok import build_sok_precond

    def patched(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                device=None, cdtype=CDT, rdtype=torch.float64):
        if d != 2:
            return _ORIG(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                         device=device, cdtype=cdtype, rdtype=rdtype)
        return build_sok_precond(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                                 R=R, weights=weights, device=device,
                                 cdtype=cdtype, rdtype=rdtype)
    efgp_mod.create_kronecker_precond = patched
    _state["installed"] = True


def uninstall():
    efgp_mod.create_kronecker_precond = _ORIG
    _state["installed"] = False
    _state["last_info"] = ""


# ---------- data distributions ----------

def _standardize(y):
    y = y.to(DT)
    return (y - y.mean()) / (y.std() + 1e-12)


def data_square(n, seed=0):
    """Uniform on [0, 1]^2 (product measure)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g, dtype=DT)
    return x


def data_disk(n, seed=0):
    """Uniform on unit disk, translated to [0,1]^2. Strongly non-product."""
    g = torch.Generator().manual_seed(seed)
    pts = []
    while sum(p.shape[0] for p in pts) < n:
        batch = torch.rand(2 * n, 2, generator=g, dtype=DT) * 2 - 1  # [-1,1]^2
        keep = (batch.norm(dim=1) <= 1.0)
        pts.append(batch[keep])
    x = torch.cat(pts, dim=0)[:n]
    x = 0.5 * (x + 1.0)  # map [-1,1]^2 → [0,1]^2
    return x


def data_annulus(n, seed=0, r_inner=0.3, r_outer=1.0):
    """Uniform on annulus. Also non-product, but with a hole."""
    g = torch.Generator().manual_seed(seed)
    pts = []
    while sum(p.shape[0] for p in pts) < n:
        batch = torch.rand(3 * n, 2, generator=g, dtype=DT) * 2 - 1
        r = batch.norm(dim=1)
        keep = (r >= r_inner) & (r <= r_outer)
        pts.append(batch[keep])
    x = torch.cat(pts, dim=0)[:n]
    x = 0.5 * (x + 1.0)
    return x


def make_y(x, ls, seed=1):
    """Simple random target (no true GP prior — we only benchmark CG/grad)."""
    g = torch.Generator().manual_seed(seed)
    y = 0.5 * torch.randn(x.shape[0], generator=g, dtype=DT)
    return _standardize(y)


# ---------- one benchmark shot ----------

def one_shot(x, y, precond_kind, *, ls, var, sig2,
             eps=1e-3, cg_tol=1e-4, cg_max=3000):
    d = x.shape[1]
    kernel = SquaredExponential(dimension=d, init_lengthscale=ls,
                                init_variance=var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=sig2, eps=eps,
                   estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond_kind,
                         "max_cg_iterations": cg_max})
    # warmup
    model.compute_gradients(trace_samples=1, cg_tol=cg_tol, noise_floor=1e-5)
    t0 = time.perf_counter()
    model.compute_gradients(trace_samples=1, cg_tol=cg_tol, noise_floor=1e-5)
    dt = time.perf_counter() - t0
    s = model.last_gradient_stats
    return dt, s.get('trace_cg_iters'), s.get('mean_cg_iters'), s.get('feature_count')


def run_dataset(name, x, y, configs):
    print(f"\n=== {name}: N={x.shape[0]:,}, d={x.shape[1]} ===", flush=True)
    for ls, var, sig2 in configs:
        print(f"\n  (ℓ={ls}, σ_f²={var}, σ²={sig2})", flush=True)

        uninstall()
        dt, it_t, it_m, M = one_shot(x, y, "jacobi", ls=ls, var=var, sig2=sig2)
        print(f"    Jacobi                : {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}  M={M}",
              flush=True)

        dt, it_t, it_m, _ = one_shot(x, y, "kronecker", ls=ls, var=var, sig2=sig2)
        print(f"    plain Kron            : {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}",
              flush=True)

        for R in [1, 2, 4, 8]:
            install_svd_circ(R=R)
            try:
                dt, it_t, it_m, _ = one_shot(x, y, "kronecker",
                                             ls=ls, var=var, sig2=sig2)
                print(f"    SVDcirc(R={R:<2d})        : "
                      f"{dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}",
                      flush=True)
            except Exception as e:
                print(f"    SVDcirc(R={R}): FAILED {type(e).__name__}: {e}", flush=True)
            finally:
                uninstall()

        # SoK-exact: eigendecomp per 1D Toeplitz (no circulant bias)
        for R in [1, 2, 4, 8]:
            install_sok(R=R, weights="uniform")
            try:
                dt, it_t, it_m, _ = one_shot(x, y, "kronecker",
                                             ls=ls, var=var, sig2=sig2)
                print(f"    SoK-exact(R={R:<2d})      : "
                      f"{dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}",
                      flush=True)
            except Exception as e:
                print(f"    SoK-exact(R={R}): FAILED {type(e).__name__}: {e}", flush=True)
            finally:
                uninstall()


if __name__ == "__main__":
    print("SVD-circulant preconditioner (Idea 3)\n", flush=True)

    CONFIGS = [
        (0.05, 1.0, 1e-2),
        (0.02, 1.0, 1e-2),
    ]
    N = 50_000

    # 1) Product measure (square). Expect Kron to dominate everything.
    x = data_square(N, seed=0)
    y = make_y(x, ls=0.05)
    run_dataset("Uniform [0,1]^2 (product)", x, y, CONFIGS)

    # 2) Disk (non-product): Kron is known to struggle here.
    x = data_disk(N, seed=0)
    y = make_y(x, ls=0.05)
    run_dataset("Uniform disk (non-product)", x, y, CONFIGS)

    # 3) Annulus (non-product, with hole).
    x = data_annulus(N, seed=0)
    y = make_y(x, ls=0.05)
    run_dataset("Uniform annulus (non-product)", x, y, CONFIGS)
