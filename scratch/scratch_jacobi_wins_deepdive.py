"""
Deep-dive the two regimes where Jacobi beat Kron in scratch_hard_regimes.py:
  (A) short-ℓ d=2:    M=961,   m=31
  (B) HARD CORNER:    M=67081, m=259

Sweep cg_tol to see:
  - does Kron's iter count stay flat while Jacobi's grows?  (if yes, Kron
    eventually wins at tight tol, because jacobi iter grows faster)
  - at what tol does the crossover happen?

Also report setup cost vs apply cost breakdown for Kron.

Run: ~/myenv/bin/python -u scratch/scratch_jacobi_wins_deepdive.py
"""
from __future__ import annotations
import sys, os, time, math
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


def run(x, y, d, precond, cg_tol, *, max_iters=3, eps=1e-3, lr=0.3,
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


cases = [
    ("(A) short-ℓ d=2",     50_000, 2, 1e-3, 0.02, 0.3,  0.5),
    ("(B) HARD CORNER d=2", 50_000, 2, 1e-3, 0.02, 1e-4, 0.5),
]
tols = [1e-3, 1e-4, 1e-6, 1e-8]


if __name__ == "__main__":
    print("Jacobi-wins deep-dive: sweep cg_tol (3 Adam steps, d=2, eps=1e-3)\n",
          flush=True)
    for name, n, d, eps, init_ls, init_sig, init_var in cases:
        print(f"=== {name}  (init ℓ={init_ls}, σ²={init_sig}) ===", flush=True)
        x, y = make_data(n, d=d)
        first_M = None
        for precond in ["kronecker", "jacobi"]:
            for tol in tols:
                try:
                    r = run(x, y, d, precond, tol, eps=eps,
                            init_ls=init_ls, init_sigmasq=init_sig,
                            init_var=init_var)
                    if first_M is None:
                        first_M = r['M']
                        m = int(round(first_M ** (1.0/d)))
                        print(f"  (M={first_M}, m={m}, m/log2(m)={m/math.log2(m):.1f})",
                              flush=True)
                        print(f"  {'precond':<9s}  {'tol':<6s}  {'wall(s)':>7s}  "
                              f"{'cg-med':>6s}", flush=True)
                        print(f"  {'-'*40}", flush=True)
                    print(f"  {precond:<9s}  {tol:<6.0e}  {r['total']:7.2f}  "
                          f"{med(r['mean_cg']):>6.1f}", flush=True)
                except Exception as e:
                    print(f"  {precond:<9s}  {tol:<6.0e}  FAILED: {e}", flush=True)
            print(flush=True)
        print(flush=True)
