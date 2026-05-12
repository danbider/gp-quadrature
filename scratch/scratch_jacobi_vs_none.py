"""
When does Jacobi help vs hurt CG on the EFGP mean system?

System:  A = D T D + sigma^2 I,  D = diag(ws),  T symmetric Hermitian Toeplitz.
Jacobi:  M = diag(A) = T_00 * D^2 + sigma^2 I.

Sweep over:
  - N        (number of data points)
  - x-dist   (uniform / jittered-grid / gaussian-concentrated / two-cluster / one-sided)
  - ell      (lengthscale)
  - sigmasq  (noise)

For each config, record: mtot, CG iters and wallclock for both preconds,
and a "non-uniformity" proxy:  off_ratio = max_{k>0} |t_k| / t_0  (computed from
v_kernel).  Small off_ratio = grid-like; large off_ratio = clustered.

Run:
  ~/myenv/bin/python scratch/scratch_jacobi_vs_none.py
"""

from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from kernels.squared_exponential import SquaredExponential
from utils.kernels import get_xis
from efgpnd import ToeplitzND, compute_convolution_vector_vectorized_dD

torch.set_default_dtype(torch.float64)
CDTYPE = torch.complex128
RDTYPE = torch.float64


# ---------------------------------------------------------------------------
# x samplers.  All produce points in (-0.5, 0.5).
# ---------------------------------------------------------------------------
def sample_x(kind: str, N: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        x = torch.rand(N, generator=g, dtype=RDTYPE) - 0.5
    elif kind == "jittered-grid":
        # grid + small jitter: very grid-like -> very low off_ratio
        grid = torch.linspace(-0.499, 0.499, N, dtype=RDTYPE)
        jit = 0.1 * (1.0 / N) * (torch.rand(N, generator=g, dtype=RDTYPE) - 0.5)
        x = (grid + jit).clamp(-0.499, 0.499)
    elif kind == "gauss-0.15":
        x = 0.15 * torch.randn(N, generator=g, dtype=RDTYPE)
        x = x.clamp(-0.499, 0.499)
    elif kind == "gauss-0.05":
        x = 0.05 * torch.randn(N, generator=g, dtype=RDTYPE)
        x = x.clamp(-0.499, 0.499)
    elif kind == "two-cluster":
        n1 = N // 2
        n2 = N - n1
        x1 = 0.02 * torch.randn(n1, generator=g, dtype=RDTYPE) - 0.3
        x2 = 0.02 * torch.randn(n2, generator=g, dtype=RDTYPE) + 0.3
        x = torch.cat([x1, x2]).clamp(-0.499, 0.499)
    elif kind == "one-sided":
        # all points on left half
        x = 0.5 * torch.rand(N, generator=g, dtype=RDTYPE) - 0.5
    else:
        raise ValueError(kind)
    return x.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Problem build (1D)
# ---------------------------------------------------------------------------
def build(N: int, x_kind: str, ell: float, sigmasq: float, eps: float = 1e-6):
    x = sample_x(x_kind, N)
    kernel = SquaredExponential(dimension=1, init_lengthscale=ell, init_variance=1.0)
    xis_1d, h, mtot = get_xis(kernel, eps=eps, L=1.0, use_integral=True,
                              l2scaled=False, dtype=RDTYPE)
    xis = xis_1d.reshape(-1, 1)
    h_t = h if torch.is_tensor(h) else torch.tensor(h, dtype=RDTYPE)
    ws = torch.sqrt(kernel.spectral_density(xis).to(CDTYPE) * h_t).reshape(-1).to(CDTYPE)
    D = ws.real.to(RDTYPE)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, float(h)).to(CDTYPE)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)

    t_col = v_kernel[2 * m_conv: 2 * m_conv + mtot].clone()
    T_00 = t_col[0].real.item()

    # Non-uniformity proxy: max off-diagonal / diagonal of the Toeplitz symbol.
    off_ratio = (t_col[1:].abs().max() / T_00).item() if mtot > 1 else 0.0
    # Also: how many t_k have |t_k| > 0.5 * T_00  (rough "coherent modes" count)
    n_coherent = int((t_col.abs() > 0.5 * T_00).sum().item()) - 1  # exclude k=0

    return dict(x=x, D=D, toeplitz=toeplitz, T_00=T_00, sigmasq=sigmasq,
                mtot=mtot, t_col=t_col, off_ratio=off_ratio, n_coherent=n_coherent)


# ---------------------------------------------------------------------------
# Minimal PCG with counters
# ---------------------------------------------------------------------------
def pcg(A_apply, b, M_inv=None, tol=1e-6, max_iter=5000):
    x = torch.zeros_like(b)
    r = b.clone()
    z = M_inv(r) if M_inv is not None else r.clone()
    p = z.clone()
    rz = torch.vdot(r, z).real
    b_norm = torch.linalg.norm(b).real.clamp_min(1e-30)
    for it in range(max_iter):
        Ap = A_apply(p)
        pAp = torch.vdot(p, Ap).real + 1e-30
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rel = torch.linalg.norm(r).real / b_norm
        if rel < tol:
            return x, it + 1, rel.item()
        z = M_inv(r) if M_inv is not None else r
        rz_new = torch.vdot(r, z).real
        beta = rz_new / (rz + 1e-30)
        p = z + beta * p
        rz = rz_new
    return x, max_iter, rel.item()


def run(prob, precond: str, tol=1e-6, max_iter=5000):
    D = prob["D"]
    Dc = D.to(CDTYPE)
    toep = prob["toeplitz"]
    sigmasq = prob["sigmasq"]

    def A_apply(v):
        return Dc * toep(Dc * v) + sigmasq * v

    torch.manual_seed(0)
    b = (torch.randn(prob["mtot"], dtype=RDTYPE)
         + 1j * torch.randn(prob["mtot"], dtype=RDTYPE)).to(CDTYPE)
    ws = prob["D"].to(CDTYPE)
    b = ws * b  # shape of rhs matching practice

    if precond == "none":
        M_inv = None
    elif precond == "jacobi":
        diag = prob["T_00"] * D.pow(2) + sigmasq
        diag_c = diag.to(CDTYPE)
        M_inv = lambda r: r / diag_c
    else:
        raise ValueError(precond)

    # Warm up once (JIT / FFT plan), then time.
    pcg(A_apply, b, M_inv=M_inv, tol=tol, max_iter=5)
    t0 = time.perf_counter()
    _, iters, rel = pcg(A_apply, b, M_inv=M_inv, tol=tol, max_iter=max_iter)
    dt = time.perf_counter() - t0
    return iters, dt, rel


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep():
    Ns = [5_000, 20_000, 100_000]
    x_kinds = ["jittered-grid", "uniform", "gauss-0.15", "gauss-0.05",
               "one-sided", "two-cluster"]
    ells = [0.1, 0.03, 0.01]
    s2s = [1e-2, 1e-4, 1e-6]

    header = (f"{'x-dist':<15s} {'N':>7s} {'ell':>6s} {'s2':>7s} "
              f"{'mtot':>5s} {'off_rat':>8s} {'ncoh':>5s} "
              f"{'none_it':>7s} {'jac_it':>7s} {'it_ratio':>9s} "
              f"{'none_s':>8s} {'jac_s':>8s} {'t_ratio':>8s} "
              f"{'jac_res':>9s}")
    print(header)
    print("-" * len(header))

    rows = []
    for xk in x_kinds:
        for N in Ns:
            for ell in ells:
                for s2 in s2s:
                    try:
                        p = build(N, xk, ell, s2)
                    except Exception as e:
                        continue
                    # cap to avoid 30-min runs
                    max_it = max(4000, 10 * p["mtot"])
                    it_n, t_n, _ = run(p, "none", max_iter=max_it)
                    it_j, t_j, rel_j = run(p, "jacobi", max_iter=max_it)
                    it_ratio = it_j / max(it_n, 1)
                    t_ratio = t_j / max(t_n, 1e-9)
                    row = dict(xkind=xk, N=N, ell=ell, s2=s2, mtot=p["mtot"],
                               off_ratio=p["off_ratio"], n_coh=p["n_coherent"],
                               it_none=it_n, it_jac=it_j, t_none=t_n, t_jac=t_j,
                               it_ratio=it_ratio, t_ratio=t_ratio, rel_j=rel_j)
                    rows.append(row)
                    print(f"{xk:<15s} {N:>7d} {ell:>6.3f} {s2:>7.0e} "
                          f"{p['mtot']:>5d} {p['off_ratio']:>8.3f} {p['n_coherent']:>5d} "
                          f"{it_n:>7d} {it_j:>7d} {it_ratio:>9.2f} "
                          f"{t_n:>8.3f} {t_j:>8.3f} {t_ratio:>8.2f} "
                          f"{rel_j:>9.2e}")
    return rows


if __name__ == "__main__":
    rows = sweep()
    # Save as JSON for later analysis
    import json
    out = os.path.join(os.path.dirname(__file__), "jacobi_vs_none_sweep.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out}")

    # Quick summary: when does jacobi hurt (it_ratio > 1.2)?
    hurts = [r for r in rows if r["it_ratio"] > 1.2]
    helps = [r for r in rows if r["it_ratio"] < 0.8]
    print(f"\n{'='*70}")
    print(f"Jacobi HURTS (iters jacobi / iters none > 1.2): {len(hurts)} / {len(rows)} configs")
    for r in hurts[:15]:
        print(f"  {r['xkind']:<15s} N={r['N']} ell={r['ell']} s2={r['s2']:.0e} "
              f"off_ratio={r['off_ratio']:.2f}  it_ratio={r['it_ratio']:.2f}  "
              f"t_ratio={r['t_ratio']:.2f}")
    print(f"\nJacobi HELPS (iters jacobi / iters none < 0.8): {len(helps)} / {len(rows)} configs")
    for r in helps[:10]:
        print(f"  {r['xkind']:<15s} N={r['N']} ell={r['ell']} s2={r['s2']:.0e} "
              f"off_ratio={r['off_ratio']:.2f}  it_ratio={r['it_ratio']:.2f}  "
              f"t_ratio={r['t_ratio']:.2f}")
