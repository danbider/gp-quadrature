"""
Kronecker vs Jacobi vs Nystrom preconditioners for the EFGP mean system.

System:  A = D T D + sigma^2 I  acting on (mtot**d,)-vectors.
Three preconditioners:

  jacobi    : M = diag(|ws|^2 * T[0] + sigma^2).  O(1) setup, cheap apply.
  nystrom   : rank-k randomized low-rank + diagonal-shift (Frangella-Tropp-Udell).
              Setup = (k+p) full A-matvecs + O(M k^2).  Apply = O(M k).
  kronecker : EXACT preconditioner for (⊗_k H_k) + sigma^2 I, where
              H_k = D_k T_k D_k with D_k the 1D factor of a separable ws
              and T_k the 1D Toeplitz obtained by pinning other axes to zero.
              Setup = d dense Hermitian eigendecompositions, O(d * m^3).
              Apply = O(d * M * m), same order as d 1D dense mat-vecs per mode.

Run:  ~/myenv/bin/python scratch/scratch_kronecker_vs_others.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import torch

from kernels.squared_exponential import SquaredExponential
from kernels.matern import Matern
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


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def sample_x(kind: str, N: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        x = torch.rand(N, d, generator=g, dtype=RDTYPE) - 0.5
    elif kind == "gauss":
        x = 0.1 * torch.randn(N, d, generator=g, dtype=RDTYPE)
        x = x.clamp(-0.499, 0.499)
    elif kind == "two-cluster":
        n1 = N // 2
        n2 = N - n1
        centers = torch.tensor([-0.3] * d), torch.tensor([0.3] * d)
        x1 = 0.05 * torch.randn(n1, d, generator=g, dtype=RDTYPE) + centers[0]
        x2 = 0.05 * torch.randn(n2, d, generator=g, dtype=RDTYPE) + centers[1]
        x = torch.cat([x1, x2], dim=0).clamp(-0.499, 0.499)
    elif kind == "grid":
        m = int(round(N ** (1.0 / d)))
        g1 = torch.linspace(-0.49, 0.49, m, dtype=RDTYPE)
        grids = torch.meshgrid(*(g1 for _ in range(d)), indexing="ij")
        x = torch.stack(grids, dim=-1).reshape(-1, d)
    else:
        raise ValueError(kind)
    return x


def build(N, x_kind, ell, sigmasq, d, eps=1e-4, kernel_name="SE"):
    x = sample_x(x_kind, N, d)
    if kernel_name == "SE":
        kernel = SquaredExponential(dimension=d, init_lengthscale=ell, init_variance=1.0)
    elif kernel_name == "matern32":
        kernel = Matern(dimension=d, nu=1.5, init_lengthscale=ell, init_variance=1.0)
    else:
        raise ValueError(kernel_name)
    L = torch.tensor(1.0)
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True,
                              l2scaled=False)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing='ij')
    xis = torch.stack(grids, dim=-1).view(-1, d)
    h_t = h if torch.is_tensor(h) else torch.tensor(h, dtype=RDTYPE)
    ws = torch.sqrt(kernel.spectral_density(xis).to(CDTYPE) * h_t ** d).reshape(-1)

    m_conv = (mtot - 1) // 2
    h_float = float(h_t.item()) if torch.is_tensor(h_t) else float(h_t)
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h_float).to(CDTYPE)
    toep = ToeplitzND(v_kernel, force_pow2=True)
    v_shape = v_kernel.shape
    ctr = tuple((torch.tensor(v_shape) - 1) // 2)
    ctr = tuple(int(c) for c in ctr)
    T_00 = v_kernel[ctr].real.item()

    return dict(
        x=x, ws=ws, v_kernel=v_kernel, toep=toep, sigmasq=float(sigmasq),
        T_00=T_00, mtot=mtot, d=d, M=mtot ** d, kernel=kernel,
    )


# ---------------------------------------------------------------------------
# PCG
# ---------------------------------------------------------------------------
def pcg(A_apply, b, M_inv=None, tol=1e-6, max_iter=5000):
    x = torch.zeros_like(b)
    r = b.clone()
    z = M_inv(r) if M_inv is not None else r.clone()
    p = z.clone()
    rz = torch.vdot(r, z).real
    b_norm = torch.linalg.norm(b).real.clamp_min(1e-30)
    rel = torch.tensor(1.0)
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


def run_one(prob, label, M_inv, tol=1e-6, max_iter=5000):
    ws = prob["ws"]
    toep = prob["toep"]
    sigmasq = prob["sigmasq"]
    M_total = prob["M"]
    A_apply = create_A_mean(ws, toep, sigmasq, CDTYPE)

    torch.manual_seed(0)
    b = (torch.randn(M_total, dtype=RDTYPE) + 1j * torch.randn(M_total, dtype=RDTYPE)).to(CDTYPE)
    b = ws * b  # RHS in the natural range of D T D + s^2 I

    # warmup
    pcg(A_apply, b, M_inv=M_inv, tol=tol, max_iter=3)
    t0 = time.perf_counter()
    _, iters, rel = pcg(A_apply, b, M_inv=M_inv, tol=tol, max_iter=max_iter)
    dt = time.perf_counter() - t0
    return dict(label=label, iters=iters, time=dt, rel=rel)


def benchmark(tag, N, x_kind, ell, sigmasq, d=2, tol=1e-4, max_iter=2000,
              nystrom_ranks=(30, 100), kernel_name="SE"):
    prob = build(N, x_kind, ell, sigmasq, d, kernel_name=kernel_name)
    print(f"\n{tag}: d={d}, kernel={kernel_name}, x={x_kind}, N={N}, "
          f"ell={ell}, s2={sigmasq}, mtot={prob['mtot']}, M={prob['M']}")

    ws = prob["ws"]
    toep = prob["toep"]
    sigmasq_t = prob["sigmasq"]
    A_apply = create_A_mean(ws, toep, sigmasq_t, CDTYPE)

    results = []
    results.append(run_one(prob, "none", M_inv=None, tol=tol, max_iter=max_iter))

    t0 = time.perf_counter()
    Minv_jac = create_jacobi_precond(ws, sigmasq_t, diag_scale=prob["T_00"])
    t_setup = time.perf_counter() - t0
    r = run_one(prob, "jacobi", M_inv=Minv_jac, tol=tol, max_iter=max_iter)
    r["setup_s"] = t_setup
    results.append(r)

    t0 = time.perf_counter()
    Minv_kron = create_kronecker_precond(
        ws, prob["v_kernel"], sigmasq_t, d=prob["d"], mtot_1d=prob["mtot"],
        cdtype=CDTYPE, rdtype=RDTYPE,
    )
    t_setup = time.perf_counter() - t0
    r = run_one(prob, "kronecker", M_inv=Minv_kron, tol=tol, max_iter=max_iter)
    r["setup_s"] = t_setup
    results.append(r)

    for k in nystrom_ranks:
        if k + 10 >= prob["M"]:
            continue
        t0 = time.perf_counter()
        Minv_nys = create_nystrom_precond(
            A_apply, M=prob["M"], sigmasq_scalar=sigmasq_t,
            rank=k, oversample=10, seed=0,
            cdtype=CDTYPE, rdtype=RDTYPE,
        )
        t_setup = time.perf_counter() - t0
        r = run_one(prob, f"nystrom(k={k})", M_inv=Minv_nys, tol=tol, max_iter=max_iter)
        r["setup_s"] = t_setup
        results.append(r)

    print(f"  {'label':<18s} {'iters':>6s} {'setup_s':>9s} {'solve_s':>9s} "
          f"{'total_s':>9s} {'rel':>10s}")
    for r in results:
        setup_s = r.get("setup_s", 0.0)
        print(f"  {r['label']:<18s} {r['iters']:>6d} {setup_s:>9.3f} "
              f"{r['time']:>9.3f} {setup_s + r['time']:>9.3f} {r['rel']:>10.2e}")
    return results


if __name__ == "__main__":
    # --- 1D sanity: Kronecker degenerates to exact eigendecomposition ---
    benchmark("1D-sanity",           N=5_000,  x_kind="uniform",     ell=0.05, sigmasq=1e-4, d=1)

    # --- 2D SE, varying difficulty ---
    benchmark("2D-uniform",          N=20_000, x_kind="uniform",     ell=0.05, sigmasq=1e-4, d=2)
    benchmark("2D-two-cluster",      N=20_000, x_kind="two-cluster", ell=0.05, sigmasq=1e-4, d=2)
    benchmark("2D-gauss",            N=20_000, x_kind="gauss",       ell=0.05, sigmasq=1e-4, d=2)
    benchmark("2D-grid (separable)", N=10_000, x_kind="grid",        ell=0.05, sigmasq=1e-4, d=2)

    # --- Matern32: non-separable spectral density, Kronecker is approximate ---
    benchmark("2D-matern32-uniform", N=20_000, x_kind="uniform",     ell=0.05, sigmasq=1e-4,
              d=2, kernel_name="matern32")
