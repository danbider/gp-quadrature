"""
How does tightening CG tol affect Kron vs Jacobi?

Theory: for a well-conditioned preconditioned system (kron),
  iters ~ sqrt(kappa) * log(1/tol)
  → halving tol ≈ +const iters (small slope in kappa^{1/2}).
For a loosely-conditioned one (jacobi), kappa is huge, so the slope is large.

We sweep cg_tol ∈ {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10}
and measure: CG iters, wallclock, and the resulting grad-norm for the final
Adam step (sanity).

Run: ~/myenv/bin/python -u scratch/scratch_cg_tol_sweep.py
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


def make_data(n, d=2, true_ls=0.05, true_var=1.0, true_noise=0.01, seed=1):
    torch.manual_seed(seed)
    x = torch.rand(n, d, dtype=DT)
    f = sample_gp_rff(x, length_scale=true_ls, variance=true_var,
                      num_features=2000, seed=0)
    torch.manual_seed(seed)
    y = f + torch.sqrt(torch.tensor(true_noise)) * torch.randn(f.numel(), dtype=DT)
    return x, y


def run(x, y, d, precond, cg_tol, *, max_iters=10, eps=1e-3, lr=0.3,
        init_ls=0.3, init_var=0.5, init_sigmasq=0.3, J=1, noise_floor=1e-5,
        cg_max=2000):
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
    return dict(total=total, mean_cg=mean_cg, trace_cg=trace_cg)


def summarize(xs):
    xs = [v for v in xs if v is not None]
    if not xs: return (float('nan'),) * 3
    s = sorted(xs); k = len(s)
    med = s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])
    return min(xs), med, max(xs)


cases = [("uniform n=20k, d=2", 20_000, 2),
         ("uniform n=100k, d=2", 100_000, 2)]

tols = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]


if __name__ == "__main__":
    print("CG tol sweep: Kron vs Jacobi (10 Adam steps, d=2, eps=1e-3)",
          flush=True)
    print("init ℓ=0.3, σ_f²=0.5, σ_n²=0.3; cg_max=2000\n", flush=True)
    for name, n, d in cases:
        print(f"=== {name} ===", flush=True)
        x, y = make_data(n, d)
        print(f"{'precond':<9s}  {'cg_tol':<8s}  {'wall(s)':>7s}  "
              f"{'cg-mean(min-med-max)':<22s}  "
              f"{'cg-trace(min-med-max)':<22s}", flush=True)
        print("-" * 88, flush=True)
        for precond in ["kronecker", "jacobi"]:
            for tol in tols:
                try:
                    r = run(x, y, d, precond, tol)
                    m = summarize(r['mean_cg']); t = summarize(r['trace_cg'])
                    print(f"{precond:<9s}  {tol:<8.0e}  {r['total']:7.2f}  "
                          f"{m[0]:>4d}-{m[1]:>5.1f}-{m[2]:<4d}        "
                          f"{t[0]:>4d}-{t[1]:>5.1f}-{t[2]:<4d}",
                          flush=True)
                except Exception as e:
                    print(f"{precond:<9s}  {tol:<8.0e}  FAILED: {type(e).__name__}: {e}",
                          flush=True)
            print(flush=True)
