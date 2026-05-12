"""
Iteration-count sweep: Kronecker vs Jacobi vs Nystrom preconditioners.

Runs PCG on A = D T D + sigma^2 I with several (N, ell, sigmasq, x_kind)
configurations at SE in d=2.  Quadrature eps=1e-4; CG tol=1e-4.

Run:  ~/myenv/bin/python scratch/scratch_kronecker_sweep.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from kernels.squared_exponential import SquaredExponential
from utils.kernels import get_xis
from efgpnd import (
    ToeplitzND,
    compute_convolution_vector_vectorized_dD,
    create_A_mean,
    create_jacobi_precond,
    create_nystrom_precond,
    create_kronecker_precond,
)

torch.set_default_dtype(torch.float64)
CDTYPE = torch.complex128
RDTYPE = torch.float64


def sample_x(kind: str, N: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        return torch.rand(N, d, generator=g, dtype=RDTYPE) - 0.5
    if kind == "gauss":
        return (0.1 * torch.randn(N, d, generator=g, dtype=RDTYPE)).clamp(-0.499, 0.499)
    if kind == "two-cluster":
        n1 = N // 2
        c1 = 0.05 * torch.randn(n1, d, generator=g, dtype=RDTYPE) + torch.tensor([-0.3] * d)
        c2 = 0.05 * torch.randn(N - n1, d, generator=g, dtype=RDTYPE) + torch.tensor([0.3] * d)
        return torch.cat([c1, c2], 0).clamp(-0.499, 0.499)
    if kind == "grid":
        m = int(round(N ** (1.0 / d)))
        g1 = torch.linspace(-0.49, 0.49, m, dtype=RDTYPE)
        gs = torch.meshgrid(*(g1 for _ in range(d)), indexing="ij")
        return torch.stack(gs, -1).reshape(-1, d)
    if kind == "jitter-grid":
        m = int(round(N ** (1.0 / d)))
        g1 = torch.linspace(-0.49, 0.49, m, dtype=RDTYPE)
        gs = torch.meshgrid(*(g1 for _ in range(d)), indexing="ij")
        x = torch.stack(gs, -1).reshape(-1, d)
        dx = (0.3 / m) * torch.randn(x.shape, generator=g, dtype=RDTYPE)
        return (x + dx).clamp(-0.499, 0.499)
    raise ValueError(kind)


def build(N, x_kind, ell, sigmasq, d=2, eps=1e-4):
    x = sample_x(x_kind, N, d)
    kernel = SquaredExponential(dimension=d, init_lengthscale=ell, init_variance=1.0)
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=torch.tensor(1.0),
                              use_integral=True, l2scaled=False)
    gs = torch.meshgrid(*(xis_1d for _ in range(d)), indexing='ij')
    xis = torch.stack(gs, -1).view(-1, d)
    h_t = h if torch.is_tensor(h) else torch.tensor(h, dtype=RDTYPE)
    ws = torch.sqrt(kernel.spectral_density(xis).to(CDTYPE) * h_t ** d).reshape(-1)
    m_conv = (mtot - 1) // 2
    h_f = float(h_t.item()) if torch.is_tensor(h_t) else float(h_t)
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h_f).to(CDTYPE)
    toep = ToeplitzND(v_kernel, force_pow2=True)
    ctr = tuple(int(c) for c in ((torch.tensor(v_kernel.shape) - 1) // 2))
    T00 = v_kernel[ctr].real.item()
    return dict(ws=ws, v_kernel=v_kernel, toep=toep, sigmasq=float(sigmasq),
                T_00=T00, mtot=mtot, M=mtot ** d, d=d)


def pcg(A, b, Minv=None, tol=1e-4, max_iter=2000):
    x = torch.zeros_like(b); r = b.clone()
    z = Minv(r) if Minv else r.clone()
    p = z.clone(); rz = torch.vdot(r, z).real
    bn = torch.linalg.norm(b).real.clamp_min(1e-30); rel = torch.tensor(1.0)
    for it in range(max_iter):
        Ap = A(p); pAp = torch.vdot(p, Ap).real + 1e-30
        a = rz / pAp; x = x + a * p; r = r - a * Ap
        rel = torch.linalg.norm(r).real / bn
        if rel < tol: return it + 1, rel.item()
        z = Minv(r) if Minv else r
        rzn = torch.vdot(r, z).real
        p = z + (rzn / (rz + 1e-30)) * p; rz = rzn
    return max_iter, rel.item()


def run_case(prob, tol, max_iter, nystrom_k=100):
    ws = prob["ws"]; toep = prob["toep"]; s2 = prob["sigmasq"]
    A = create_A_mean(ws, toep, s2, CDTYPE)
    torch.manual_seed(0)
    b = ws * (torch.randn(prob["M"], dtype=RDTYPE) + 1j * torch.randn(prob["M"], dtype=RDTYPE)).to(CDTYPE)
    out = {}
    # none
    it, rel = pcg(A, b, None, tol=tol, max_iter=max_iter)
    out["none"] = (it, rel)
    # jacobi
    Mj = create_jacobi_precond(ws, s2, diag_scale=prob["T_00"])
    it, rel = pcg(A, b, Mj, tol=tol, max_iter=max_iter)
    out["jacobi"] = (it, rel)
    # kronecker
    Mk = create_kronecker_precond(ws, prob["v_kernel"], s2, d=prob["d"],
                                  mtot_1d=prob["mtot"], cdtype=CDTYPE, rdtype=RDTYPE)
    it, rel = pcg(A, b, Mk, tol=tol, max_iter=max_iter)
    out["kron"] = (it, rel)
    # nystrom
    if nystrom_k + 10 < prob["M"]:
        Mn = create_nystrom_precond(A, M=prob["M"], sigmasq_scalar=s2,
                                    rank=nystrom_k, oversample=10, seed=0,
                                    cdtype=CDTYPE, rdtype=RDTYPE)
        it, rel = pcg(A, b, Mn, tol=tol, max_iter=max_iter)
        out[f"nys{nystrom_k}"] = (it, rel)
    return out


def fmt(it, rel, tol):
    star = "" if rel < tol * 1.01 else "*"
    return f"{it}{star}"


if __name__ == "__main__":
    TOL = 1e-4; MAX_IT = 2000

    # Regimes: (tag, N, x_kind, ell, sigmasq)
    cases = [
        # --- typical: moderate ell, moderate noise, uniform-ish data ---
        ("typical-easy",      20_000, "uniform",     0.10, 1e-2),
        ("typical",           20_000, "uniform",     0.05, 1e-4),
        ("typical-large-N",  100_000, "uniform",     0.05, 1e-4),
        ("jitter-grid",       10_000, "jitter-grid", 0.05, 1e-4),
        # --- short lengthscale (more ill-conditioned) ---
        ("short-ell",         20_000, "uniform",     0.03, 1e-4),
        # --- low noise (ill-conditioned) ---
        ("low-noise",         20_000, "uniform",     0.05, 1e-6),
        # --- high noise (easy) ---
        ("high-noise",        20_000, "uniform",     0.05, 1e-2),
        # --- separable-friendly grid ---
        ("grid",              10_000, "grid",        0.05, 1e-4),
        # --- Kronecker-adversarial (clustered data) ---
        ("two-cluster",       20_000, "two-cluster", 0.05, 1e-4),
        ("gauss-blob",        20_000, "gauss",       0.05, 1e-4),
    ]

    print(f"\nPCG iters to rel-res < {TOL}  (* = did not converge in {MAX_IT} iters)")
    print(f"{'case':<18s} {'N':>7s} {'xkind':<12s} {'ell':>5s} {'s2':>7s} "
          f"{'mtot':>5s} {'M':>7s}   "
          f"{'none':>6s} {'jacobi':>7s} {'kron':>6s} {'nys100':>7s}")
    print("-" * 110, flush=True)

    rows = []
    for tag, N, xk, ell, s2 in cases:
        t0 = time.perf_counter()
        prob = build(N, xk, ell, s2)
        res = run_case(prob, tol=TOL, max_iter=MAX_IT)
        dt = time.perf_counter() - t0
        rows.append((tag, N, xk, ell, s2, prob["mtot"], prob["M"], res, dt))
        # print row immediately
        tag_, N_, xk_, ell_, s2_, mtot, M, res_, dt_ = rows[-1]
        n_i, n_rel = res.get("none", (0, 0))
        j_i, j_rel = res.get("jacobi", (0, 0))
        k_i, k_rel = res.get("kron", (0, 0))
        ny = res.get("nys100", (None, None))
        none_s = fmt(n_i, n_rel, TOL)
        jac_s  = fmt(j_i, j_rel, TOL)
        kron_s = fmt(k_i, k_rel, TOL)
        nys_s  = fmt(ny[0], ny[1], TOL) if ny[0] is not None else "-"
        print(f"{tag:<18s} {N:>7d} {xk:<12s} {ell:>5.2f} {s2:>7.0e} "
              f"{mtot:>5d} {M:>7d}   "
              f"{none_s:>6s} {jac_s:>7s} {kron_s:>6s} {nys_s:>7s}",
              flush=True)
