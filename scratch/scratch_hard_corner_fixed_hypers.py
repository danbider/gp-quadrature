"""
HARD CORNER with hyperparameters PINNED — no Adam drift between Kron/Jacobi.

Issue with prior benchmark: each precond ran its own 5-step Adam trajectory, so
by step 2+ they were at different (ℓ, σ², σ_f²), i.e. different M and different
conditioning. The reported wallclock combined both preconditioner and
"different-problem" effects.

Fix: skip Adam entirely. Fix (ℓ, σ², σ_f²) and call compute_gradients() K times
at exactly that point. Both Kron and Jacobi see the identical system.

Sweep ℓ across {0.05, 0.02, 0.01} at σ²=1e-4 to map the crossover.

Run: ~/myenv/bin/python -u scratch/scratch_hard_corner_fixed_hypers.py
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


def time_grad_at_fixed_hypers(x, y, d, precond, *, ls, var, sig2,
                              K=4, warmup=1, eps=1e-3, cg_tol=1e-4,
                              J=1, noise_floor=1e-5, cg_max=3000, seed=0):
    kernel = SquaredExponential(dimension=d, init_lengthscale=ls,
                                init_variance=var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=sig2, eps=eps,
                   estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond,
                         "max_cg_iterations": cg_max})
    for _ in range(warmup):
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
    t_list = []
    mean_iters, trace_iters = [], []
    for k in range(K):
        t0 = time.perf_counter()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
        t_list.append(time.perf_counter() - t0)
        s = model.last_gradient_stats
        mean_iters.append(s.get('mean_cg_iters'))
        trace_iters.append(s.get('trace_cg_iters'))
    return dict(times=t_list,
                mean_iters=mean_iters,
                trace_iters=trace_iters,
                M=model.last_gradient_stats.get('feature_count'))


def fmt_stats(xs):
    xs = [v for v in xs if v is not None]
    if not xs: return "nan"
    return f"{min(xs):.0f}/{sum(xs)/len(xs):.1f}/{max(xs):.0f}"


hyper_cases = [
    ("ℓ=0.05, σ²=1e-4",  0.05, 0.5, 1e-4),
    ("ℓ=0.02, σ²=1e-4",  0.02, 0.5, 1e-4),
    ("ℓ=0.01, σ²=1e-4",  0.01, 0.5, 1e-4),
    ("ℓ=0.02, σ²=1e-2",  0.02, 0.5, 1e-2),
    ("ℓ=0.02, σ²=1e-6",  0.02, 0.5, 1e-6),
]


if __name__ == "__main__":
    print("Hard-corner with PINNED hypers: same (ℓ, σ², σ_f²) for both preconds",
          flush=True)
    print("N=50k, d=2, eps=1e-3, cg_tol=1e-4, K=4 trials + 1 warmup\n", flush=True)
    x, y = make_data(50_000, d=2)
    for label, ls, var, sig2 in hyper_cases:
        print(f"=== {label} ===", flush=True)
        ran = {}
        for precond in ["kronecker", "jacobi"]:
            try:
                r = time_grad_at_fixed_hypers(x, y, 2, precond,
                                              ls=ls, var=var, sig2=sig2)
                ran[precond] = r
                t_med = sorted(r['times'])[len(r['times']) // 2]
                t_mean = sum(r['times']) / len(r['times'])
                print(f"  M={r['M']:<6d}  {precond:<9s}  "
                      f"per-grad: mean={t_mean:.3f}s med={t_med:.3f}s  "
                      f"mean-CG(min/avg/max)={fmt_stats(r['mean_iters'])}  "
                      f"trace-CG={fmt_stats(r['trace_iters'])}",
                      flush=True)
            except Exception as e:
                print(f"  {precond:<9s}  FAILED: {type(e).__name__}: {e}",
                      flush=True)
        if "kronecker" in ran and "jacobi" in ran:
            tk = sum(ran["kronecker"]['times']) / len(ran["kronecker"]['times'])
            tj = sum(ran["jacobi"]['times']) / len(ran["jacobi"]['times'])
            winner = "KRON" if tk < tj else "JAC"
            print(f"  -> {winner} wins  (jac/kron = {tj/tk:.2f}x)", flush=True)
        print(flush=True)
