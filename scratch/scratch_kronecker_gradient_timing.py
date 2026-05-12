"""
End-to-end gradient-step timing: jacobi vs kronecker vs nystrom
for the *realistic* EFGP hyperparameter-learning setup from
scratch/hyperparameter_comparison.ipynb.

Synthetic data: y = GP(SE, ls=0.05, var=1) + noise(0.01), d=2, n ∈ {20k, 100k}.
Starts from the same init hypers used in the notebook.  We time:

  (a) FIRST gradient step (cold: solve is far from initialization).
  (b) A full sequence of Adam steps (warm: init hypers converging to truth).

Each preconditioner gets the same model.compute_gradients(...) path so
per-iteration PCG counts and wall-clock reflect how things actually run.

Run:  ~/myenv/bin/python -u scratch/scratch_kronecker_gradient_timing.py
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


def make_data(n: int, d: int = 2, true_ls=0.05, true_var=1.0, true_noise=0.01, seed=42):
    torch.manual_seed(seed)
    x = torch.rand(n, d, dtype=DT)
    y = sample_gp_rff(x, length_scale=true_ls, variance=true_var,
                     num_features=2000, seed=seed)
    torch.manual_seed(seed + 1)
    y = y + torch.sqrt(torch.tensor(true_noise)) * torch.randn(n, dtype=DT)
    return x, y


def fit_time(x, y, *, d, precond, init_ls, init_var, init_noise,
             eps, cg_tol, max_iters, lr, J=1, noise_floor=1e-5, label=""):
    kernel = SquaredExponential(dimension=d, init_lengthscale=init_ls,
                                init_variance=init_var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=init_noise, eps=eps,
                   estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond})
    opt = Adam(model.parameters(), lr=lr)

    # ---- warm-up step (JIT, kernel caches): time separately ----
    t0 = time.perf_counter()
    opt.zero_grad()
    model.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=noise_floor)
    opt.step()
    t_first = time.perf_counter() - t0
    stats = model.last_gradient_stats
    first_mean_iters = stats.get('mean_cg_iters')
    first_trace_iters = stats.get('trace_cg_iters')
    mtot = stats.get('mtot'); Mfeat = stats.get('feature_count')

    # ---- remaining steps ----
    iter_times = []
    iter_mean_cg = []
    iter_trace_cg = []
    for it in range(max_iters - 1):
        t1 = time.perf_counter()
        opt.zero_grad()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
        opt.step()
        iter_times.append(time.perf_counter() - t1)
        s = model.last_gradient_stats
        iter_mean_cg.append(s.get('mean_cg_iters'))
        iter_trace_cg.append(s.get('trace_cg_iters'))

    total = t_first + sum(iter_times)
    ls_final = model.kernel.get_hyper('lengthscale')
    var_final = model.kernel.get_hyper('variance')
    s2_final = model._gp_params.sig2.item()

    return dict(
        label=label, precond=precond, total=total, t_first=t_first,
        iter_times=iter_times, iter_mean_cg=iter_mean_cg,
        iter_trace_cg=iter_trace_cg,
        first_mean_iters=first_mean_iters, first_trace_iters=first_trace_iters,
        mtot=mtot, Mfeat=Mfeat,
        ls=ls_final, var=var_final, s2=s2_final,
    )


def median(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def report(r):
    mt = median(r['iter_times']) if r['iter_times'] else float('nan')
    mc = median([x for x in r['iter_mean_cg'] if x is not None]) if r['iter_mean_cg'] else float('nan')
    tc = median([x for x in r['iter_trace_cg'] if x is not None]) if r['iter_trace_cg'] else float('nan')
    print(f"  {r['precond']:<10s}  total={r['total']:7.2f}s  "
          f"first={r['t_first']:6.2f}s (mean-cg={r['first_mean_iters']}, trace-cg={r['first_trace_iters']})  "
          f"med/iter={mt:6.3f}s  med-cg(mean/trace)={mc}/{tc}  "
          f"final (ℓ,σ_f²,σ_n²)=({r['ls']:.3f},{r['var']:.3f},{r['s2']:.3g})",
          flush=True)


if __name__ == "__main__":
    # Match hyperparameter_comparison.ipynb
    init_ls = 0.3
    init_var = 0.5
    init_noise = 0.3
    max_iters = 10         # keep short for timing; enough to see warm-phase behaviour
    lr = 0.1
    J = 1
    cg_tol = 1e-4
    eps = 1e-4

    configs = [
        dict(n=20_000, d=2, label="n=20k, d=2"),
        dict(n=100_000, d=2, label="n=100k, d=2"),
    ]

    for cfg in configs:
        print(f"\n=== {cfg['label']}  eps={eps:.0e}  cg_tol={cg_tol:.0e}  "
              f"max_iters={max_iters}  init(ℓ,σ_f²,σ_n²)=({init_ls},{init_var},{init_noise}) ===",
              flush=True)
        x, y = make_data(cfg['n'], d=cfg['d'])

        for precond in ["jacobi", "kronecker", "nystrom"]:
            try:
                r = fit_time(x, y, d=cfg['d'], precond=precond,
                             init_ls=init_ls, init_var=init_var, init_noise=init_noise,
                             eps=eps, cg_tol=cg_tol, max_iters=max_iters, lr=lr, J=J,
                             label=cfg['label'])
                report(r)
            except Exception as e:
                print(f"  {precond:<10s}  FAILED: {type(e).__name__}: {e}", flush=True)
