"""
Fix (ℓ, σ², σ_f²) at the N=50k Jacobi-winning point and sweep N.

Predict: at m=157, crossover happens around N ~ m² ≈ 25k.
  N < 25k:  Jacobi wins  (Kron CF factorization error dominates)
  N > 25k:  Kron wins  (CF error shrinks below m/log(m) threshold)

Run: ~/myenv/bin/python -u scratch/scratch_sweep_n_at_hard.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
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


def time_grad(x, y, d, precond, *, ls, var, sig2, K=4, warmup=1,
              eps=1e-3, cg_tol=1e-4, J=1, noise_floor=1e-5, cg_max=3000):
    kernel = SquaredExponential(dimension=d, init_lengthscale=ls,
                                init_variance=var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=sig2, eps=eps,
                   estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond,
                         "max_cg_iterations": cg_max})
    for _ in range(warmup):
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
    t_list = []; trace_iters = []
    for _ in range(K):
        t0 = time.perf_counter()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
        t_list.append(time.perf_counter() - t0)
        s = model.last_gradient_stats
        trace_iters.append(s.get('trace_cg_iters'))
    return dict(times=t_list, trace_iters=trace_iters,
                M=model.last_gradient_stats.get('feature_count'))


def mean(xs):
    xs = [v for v in xs if v is not None]
    return sum(xs) / len(xs) if xs else float('nan')


LS, VAR, SIG2 = 0.01, 0.5, 1e-4
n_grid = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000]


if __name__ == "__main__":
    print(f"Sweep N at pinned ℓ={LS}, σ_f²={VAR}, σ²={SIG2} (d=2, eps=1e-3, cg_tol=1e-4)",
          flush=True)
    print(f"K=4 trials + 1 warmup\n", flush=True)
    print(f"{'N':>8s} {'M':>6s} {'m':>4s}  "
          f"{'kron':<20s}  {'jacobi':<20s}  {'jac/kron':>10s}",
          flush=True)
    print("-" * 80, flush=True)
    for n in n_grid:
        x, y = make_data(n, d=2)
        ran = {}; M_seen = None
        for precond in ["kronecker", "jacobi"]:
            try:
                r = time_grad(x, y, 2, precond, ls=LS, var=VAR, sig2=SIG2)
                ran[precond] = r; M_seen = r['M']
            except Exception as e:
                print(f"  {precond} FAILED at N={n}: {e}", flush=True)
        if "kronecker" in ran and "jacobi" in ran:
            rk, rj = ran["kronecker"], ran["jacobi"]
            tk = sum(rk['times']) / len(rk['times'])
            tj = sum(rj['times']) / len(rj['times'])
            ck = mean(rk['trace_iters']); cj = mean(rj['trace_iters'])
            m_dim = int(round(M_seen ** 0.5))
            kstr = f"{tk:.2f}s/{ck:.1f}it"
            jstr = f"{tj:.2f}s/{cj:.1f}it"
            winner = "KRON" if tk < tj else "JAC "
            print(f"{n:>8d} {M_seen:>6d} {m_dim:>4d}  "
                  f"{kstr:<20s}  {jstr:<20s}  "
                  f"{tj/tk:>6.2f}x {winner}", flush=True)
