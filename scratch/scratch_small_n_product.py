"""
Small-N product-measure sweep: does Kron still help when CLT hasn't kicked in?

Product measures tested:
  - uniform on [0,1]^d       (iid product)
  - iid Gaussian blob         (isotropic => product of 1D Gaussians)

N varies from 200 up to 20k. d=2 so we see "small N per axis" regime.
We compare Kron vs Jacobi CG iter counts + wallclock.

Run: ~/myenv/bin/python -u scratch/scratch_small_n_product.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.optim import Adam

from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential
from vanilla_gp_sampling import sample_gp_rff

torch.set_default_dtype(torch.float64)
DT = torch.float64


def sample_x(kind, n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        return torch.rand(n, d, generator=g, dtype=DT)
    if kind == "gauss":
        return (0.15 * torch.randn(n, d, generator=g, dtype=DT) + 0.5).clamp(0.01, 0.99)
    raise ValueError(kind)


def make_data(kind, n, d, true_ls=0.05, true_var=1.0, true_noise=0.01, seed=1):
    torch.manual_seed(seed)
    x = sample_x(kind, n, d, seed)
    f = sample_gp_rff(x, length_scale=true_ls, variance=true_var,
                      num_features=2000, seed=0)
    torch.manual_seed(seed)
    y = f + torch.sqrt(torch.tensor(true_noise)) * torch.randn(f.numel(), dtype=DT)
    return x, y


def run(x, y, d, precond, *, max_iters=10, eps=1e-3, cg_tol=1e-4, lr=0.3,
        init_ls=0.3, init_var=0.5, init_sigmasq=0.3, J=1, noise_floor=1e-5,
        cg_max=500):
    kernel = SquaredExponential(dimension=d, init_lengthscale=init_ls,
                                init_variance=init_var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=init_sigmasq, eps=eps,
                   estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond,
                         "max_cg_iterations": cg_max})
    opt = Adam(model.parameters(), lr=lr)
    mean_cg, trace_cg = [], []
    t0 = time.perf_counter()
    for it in range(max_iters):
        opt.zero_grad()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
        opt.step()
        s = model.last_gradient_stats
        mean_cg.append(s.get('mean_cg_iters'))
        trace_cg.append(s.get('trace_cg_iters'))
    total = time.perf_counter() - t0
    return dict(total=total, mean_cg=mean_cg, trace_cg=trace_cg,
                mtot=model.last_gradient_stats.get('mtot'),
                M=model.last_gradient_stats.get('feature_count'))


def summarize(xs):
    xs = [v for v in xs if v is not None]
    if not xs: return (float('nan'),) * 3
    s = sorted(xs); k = len(s)
    med = s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])
    return min(xs), med, max(xs)


cases = []
for kind in ["uniform", "gauss"]:
    for n in [200, 500, 1000, 2000, 5000, 20000]:
        cases.append((f"{kind:<8s} n={n:>6d}", kind, n, 2))


if __name__ == "__main__":
    print("Small-N product-measure sweep: Kron vs Jacobi (10 Adam steps, d=2)",
          flush=True)
    print("eps=1e-3, cg_tol=1e-4, init ℓ=0.3, σ_f²=0.5, σ_n²=0.3\n", flush=True)
    print(f"{'case':<22s}  {'precond':<9s}  {'total(s)':>8s}  "
          f"{'cg-mean (min-med-max)':<22s}  "
          f"{'cg-trace (min-med-max)':<22s}  {'ratio':>5s}", flush=True)
    print("-" * 110, flush=True)
    for name, kind, n, d in cases:
        x, y = make_data(kind, n, d)
        kron_total = None
        for precond in ["kronecker", "jacobi"]:
            try:
                r = run(x, y, d, precond)
                m = summarize(r['mean_cg']); t = summarize(r['trace_cg'])
                if precond == "kronecker": kron_total = r['total']
                ratio = (r['total'] / kron_total) if kron_total else float('nan')
                print(f"{name:<22s}  {precond:<9s}  {r['total']:8.2f}  "
                      f"{m[0]:>4d}-{m[1]:>5.1f}-{m[2]:<4d}        "
                      f"{t[0]:>4d}-{t[1]:>5.1f}-{t[2]:<4d}        "
                      f"{ratio:5.2f}x", flush=True)
            except Exception as e:
                print(f"{name:<22s}  {precond:<9s}  FAILED: {type(e).__name__}: {e}",
                      flush=True)
        print(flush=True)
