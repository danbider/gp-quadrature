"""
Does Kron's flat tol-response survive as N grows?

For each N in {20k, 100k, 500k}, sweep cg_tol ∈ {1e-2, 1e-4, 1e-6, 1e-8}.
Report CG iters (mean path) + wallclock.

Theory predictions:
  Kron: M^{-1}A ≈ I + O(1/√N). kappa_{precond} should stay O(1) and maybe
        IMPROVE with N.  → iter count stays flat, slope in log(1/tol) stays small.
  Jacobi: kappa(diag(A)^{-1} A) scales with N (DTD has O(N) entries,
        diag does not capture off-diagonal). → slope in log(1/tol) grows with N.

Run: ~/myenv/bin/python -u scratch/scratch_cg_tol_vs_n.py
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
        cg_max=3000):
    kernel = SquaredExponential(dimension=d, init_lengthscale=init_ls,
                                init_variance=init_var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=init_sigmasq, eps=eps,
                   estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond,
                         "max_cg_iterations": cg_max})
    opt = Adam(model.parameters(), lr=lr)
    mean_cg = []
    t0 = time.perf_counter()
    for it in range(max_iters):
        opt.zero_grad()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
        opt.step()
        s = model.last_gradient_stats
        mean_cg.append(s.get('mean_cg_iters'))
    total = time.perf_counter() - t0
    return dict(total=total, mean_cg=mean_cg,
                M=model.last_gradient_stats.get('feature_count'))


def med(xs):
    xs = [v for v in xs if v is not None]
    if not xs: return float('nan')
    s = sorted(xs); k = len(s)
    return s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])


ns = [20_000, 100_000, 500_000]
tols = [1e-2, 1e-4, 1e-6, 1e-8]


if __name__ == "__main__":
    print("CG tol vs N: Kron vs Jacobi (10 Adam steps, d=2, eps=1e-3)\n",
          flush=True)
    for n in ns:
        print(f"=== N = {n:,} ===", flush=True)
        x, y = make_data(n, d=2)
        M = None
        print(f"{'precond':<9s}  {'tol':<6s}  {'wall(s)':>8s}  "
              f"{'cg-med':>7s}", flush=True)
        print("-" * 42, flush=True)
        for precond in ["kronecker", "jacobi"]:
            for tol in tols:
                try:
                    r = run(x, y, 2, precond, tol)
                    if M is None: M = r['M']
                    print(f"{precond:<9s}  {tol:<6.0e}  {r['total']:8.2f}  "
                          f"{med(r['mean_cg']):>7.1f}", flush=True)
                except Exception as e:
                    print(f"{precond:<9s}  {tol:<6.0e}  FAILED: {e}", flush=True)
            print(flush=True)
        print(f"  (feature count M={M})\n", flush=True)
