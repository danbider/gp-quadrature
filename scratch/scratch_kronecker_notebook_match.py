"""
Replicate scratch/hyperparameter_comparison.ipynb (big-N EFGP run) with
each of the three preconditioners and compare total wall-clock.

Config (exact match to notebook cells 3, 5, 18-20):
  d=2, n=250_000, true_ls=0.05, true_var=1.0, true_noise=0.01
  init_ls=0.3, init_var=0.5, init_noise(sigmasq)=0.3
  EPSILON=1e-3, cg_tol=1e-4, max_iters=50, lr=0.3, J=1, noise_floor=1e-5

Run: ~/myenv/bin/python -u scratch/scratch_kronecker_notebook_match.py
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

# --- Data: match notebook exactly ---
true_ls, true_var, true_noise = 0.05, 1.0, 0.01
d = 2
n_big = 250_000
n_feat_rff = 2000

torch.manual_seed(1)
x_big = torch.rand(n_big, d, dtype=DT)
f_big = sample_gp_rff(x_big, length_scale=true_ls, variance=true_var,
                     num_features=n_feat_rff, seed=0)
torch.manual_seed(1)
y_big = f_big + torch.sqrt(torch.tensor(true_noise, dtype=DT)) * torch.randn(n_big, dtype=DT)
print(f"Data: n={n_big}, d={d}, y std={y_big.std():.3f}", flush=True)

# --- Training config: match notebook ---
init_ls, init_var, init_sigmasq = 0.3, 0.5, 0.3
EPSILON = 1e-3
cg_tol = 1e-4
max_iters = 50
lr = 0.3
J = 1
noise_floor = 1e-5


def run(precond: str):
    kernel = SquaredExponential(dimension=d, init_lengthscale=init_ls,
                                init_variance=init_var)
    model = EFGPND(x_big, y_big, kernel=kernel, sigmasq=init_sigmasq,
                   eps=EPSILON, estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond})
    opt = Adam(model.parameters(), lr=lr)

    log_iters = []
    cg_mean_log = []
    cg_trace_log = []

    t0 = time.perf_counter()
    for it in range(max_iters):
        opt.zero_grad()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=noise_floor)
        opt.step()
        s = model.last_gradient_stats
        cg_mean_log.append(s.get('mean_cg_iters'))
        cg_trace_log.append(s.get('trace_cg_iters'))
        log_iters.append(time.perf_counter() - t0)
        if it % 10 == 0:
            ls = model.kernel.get_hyper('lengthscale')
            var = model.kernel.get_hyper('variance')
            sig = model._gp_params.sig2.item()
            print(f"  [{precond:<10s}] it={it:>3}  l={ls:.4g}  sf2={var:.4g}  "
                  f"sn2={sig:.4g}  cg(m/t)={s.get('mean_cg_iters')}/"
                  f"{s.get('trace_cg_iters')}  elapsed={log_iters[-1]:.1f}s",
                  flush=True)
    total = time.perf_counter() - t0
    ls = model.kernel.get_hyper('lengthscale')
    var = model.kernel.get_hyper('variance')
    sig = model._gp_params.sig2.item()
    return dict(precond=precond, total=total,
                cg_mean=cg_mean_log, cg_trace=cg_trace_log,
                ls=ls, var=var, sig=sig,
                mtot=model.last_gradient_stats.get('mtot'),
                Mfeat=model.last_gradient_stats.get('feature_count'))


results = []
for precond in ["jacobi", "kronecker", "nystrom"]:
    print(f"\n=== precond={precond} ===", flush=True)
    try:
        r = run(precond)
        results.append(r)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

print("\n" + "=" * 78)
print(f"Notebook-match run: n={n_big}, d={d}, eps={EPSILON:.0e}, cg_tol={cg_tol:.0e}, "
      f"max_iters={max_iters}")
print(f"init hypers: ℓ={init_ls}, σ_f²={init_var}, σ_n²={init_sigmasq}")
print(f"(mtot={results[0]['mtot'] if results else '?'}, M={results[0]['Mfeat'] if results else '?'})")
print("=" * 78)


def med(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return float('nan')
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


print(f"{'precond':<12s} {'total (s)':>10s} {'s/iter':>9s} "
      f"{'med cg-mean':>12s} {'med cg-trace':>13s}  "
      f"{'ℓ_final':>8s} {'σ_f²':>7s} {'σ_n²':>8s}")
print("-" * 95)
for r in results:
    s_per = r['total'] / max_iters
    print(f"{r['precond']:<12s} {r['total']:>10.2f} {s_per:>9.3f} "
          f"{med(r['cg_mean']):>12.1f} {med(r['cg_trace']):>13.1f}  "
          f"{r['ls']:>8.4f} {r['var']:>7.4f} {r['sig']:>8.4f}")

if len(results) >= 2:
    base = next((r for r in results if r['precond'] == 'jacobi'), results[0])
    print(f"\nSpeedup vs jacobi:")
    for r in results:
        if r is base: continue
        print(f"  {r['precond']:<10s}: {base['total']/r['total']:.2f}x")
