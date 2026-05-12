"""
Fixed ℓ=0.01 at N=250k, sweep σ² and σ_f². Pinned hypers (no Adam drift).

At N=50k this was the only regime where Jacobi beat Kron (ℓ=0.01, σ²=1e-4).
Question: does N=250k tilt the balance back to Kron?
  - pro-Kron: T ≈ T_1⊗T_2 tightens (CF error shrinks as 1/√N)
  - pro-Jac:  raw κ(A) grows with N, same-size m so same Kron overhead

Run: ~/myenv/bin/python -u scratch/scratch_large_n_short_l.py
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


def mean_iter(xs):
    xs = [v for v in xs if v is not None]
    return sum(xs) / len(xs) if xs else float('nan')


sig2_grid = [1e-2, 1e-4, 1e-6]
var_grid  = [0.1, 0.5, 2.0]


if __name__ == "__main__":
    N = 250_000
    LS = 0.01
    print(f"N={N:,}, ℓ={LS} (fixed), d=2, eps=1e-3, cg_tol=1e-4", flush=True)
    print(f"K=4 trials + 1 warmup, pinned hypers (no Adam)\n", flush=True)
    print("Building data...", flush=True); sys.stdout.flush()
    t0 = time.perf_counter()
    x, y = make_data(N, d=2)
    print(f"  data built in {time.perf_counter()-t0:.1f}s\n", flush=True)

    print(f"{'σ_f²':<5s} {'σ²':<8s} {'M':>6s}  "
          f"{'kron':<22s}  {'jacobi':<22s}  {'jac/kron':>8s}",
          flush=True)
    print("-" * 92, flush=True)
    for var in var_grid:
        for sig2 in sig2_grid:
            ran = {}
            M_seen = None
            for precond in ["kronecker", "jacobi"]:
                try:
                    r = time_grad(x, y, 2, precond, ls=LS, var=var, sig2=sig2)
                    ran[precond] = r
                    M_seen = r['M']
                except Exception as e:
                    print(f"  {precond} FAILED: {e}", flush=True)
            if "kronecker" in ran and "jacobi" in ran:
                rk = ran["kronecker"]; rj = ran["jacobi"]
                tk = sum(rk['times']) / len(rk['times'])
                tj = sum(rj['times']) / len(rj['times'])
                ck = mean_iter(rk['trace_iters'])
                cj = mean_iter(rj['trace_iters'])
                kstr = f"{tk:.2f}s/{ck:.1f}it"
                jstr = f"{tj:.2f}s/{cj:.1f}it"
                winner = "KRON" if tk < tj else "JAC "
                print(f"{var:<5.1f} {sig2:<8.0e} {M_seen:>6d}  "
                      f"{kstr:<22s}  {jstr:<22s}  {tj/tk:>6.2f}x {winner}",
                      flush=True)
