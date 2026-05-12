"""
EFGP "hard" hyperparameter regimes — where CG gets ugly.

Hard = large M, bad conditioning, or both.
  - Short ℓ:         m grows, tail weights vanish, kappa blows up
  - Low σ²:          no regularization floor, kappa ~ kappa(DTD)
  - Tight eps:       larger m for quadrature → larger M
  - Higher d (d=3):  M = m^d explodes; kron apply overhead m/log m bites

Compare kron vs jacobi on each corner. 5 Adam steps, cg_max=1000.

Run: ~/myenv/bin/python -u scratch/scratch_hard_regimes.py
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


def run(x, y, d, precond, *, max_iters=5, eps=1e-3, cg_tol=1e-4, lr=0.3,
        init_ls=0.3, init_var=0.5, init_sigmasq=0.3, J=1, noise_floor=1e-5,
        cg_max=1000):
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
                mtot=model.last_gradient_stats.get('mtot'),
                M=model.last_gradient_stats.get('feature_count'))


def med(xs):
    xs = [v for v in xs if v is not None]
    if not xs: return float('nan')
    s = sorted(xs); k = len(s)
    return s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])


# (name, n, d, eps, init_ls, init_sigmasq, init_var)
# init_* are what we use at run time; also pass true_ls/true_noise via data
cases = [
    # baseline for reference
    ("baseline        d=2, ℓ=0.05, σ²=0.1",  50_000, 2, 1e-3, 0.3, 0.3, 0.5),

    # short lengthscale
    ("short-ℓ         d=2, ℓ=0.01, σ²=0.1",  50_000, 2, 1e-3, 0.02, 0.3, 0.5),
    ("very-short-ℓ    d=2, ℓ=0.005, σ²=0.1", 50_000, 2, 1e-3, 0.01, 0.3, 0.5),

    # low noise
    ("low-σ²          d=2, ℓ=0.05, σ²=1e-3", 50_000, 2, 1e-3, 0.3, 1e-3, 0.5),
    ("very-low-σ²     d=2, ℓ=0.05, σ²=1e-5", 50_000, 2, 1e-3, 0.3, 1e-5, 0.5),

    # the corner: short-ℓ and low-σ² together
    ("HARD CORNER     d=2, ℓ=0.01, σ²=1e-4", 50_000, 2, 1e-3, 0.02, 1e-4, 0.5),

    # tight quadrature
    ("tight-eps       d=2, eps=1e-5",        50_000, 2, 1e-5, 0.3, 0.3, 0.5),
    ("tight+short-ℓ   d=2, ℓ=0.02, eps=1e-5",50_000, 2, 1e-5, 0.05, 0.3, 0.5),

    # d=3
    ("d=3             ℓ=0.15, σ²=0.1",       40_000, 3, 1e-3, 0.3, 0.3, 0.5),
    ("d=3 short-ℓ     ℓ=0.05, σ²=0.1",       40_000, 3, 1e-3, 0.1, 0.3, 0.5),
]


if __name__ == "__main__":
    print("Hard-regime sweep (5 Adam steps, cg_tol=1e-4, cg_max=1000)\n",
          flush=True)
    print(f"{'case':<42s}  {'M':>6s}  {'precond':<9s}  "
          f"{'wall(s)':>7s}  {'cg-med':>6s}", flush=True)
    print("-" * 82, flush=True)
    for name, n, d, eps, init_ls, init_sig, init_var in cases:
        x, y = make_data(n, d=d)
        first_M = None
        for precond in ["kronecker", "jacobi"]:
            try:
                r = run(x, y, d, precond, eps=eps, init_ls=init_ls,
                        init_sigmasq=init_sig, init_var=init_var)
                if first_M is None: first_M = r['M']
                cg = med(r['mean_cg'])
                print(f"{name:<42s}  {r['M']:>6d}  {precond:<9s}  "
                      f"{r['total']:7.2f}  {cg:>6.1f}", flush=True)
            except Exception as e:
                print(f"{name:<42s}  {'?':>6s}  {precond:<9s}  FAILED: {type(e).__name__}: {e}",
                      flush=True)
        print(flush=True)
