"""
CG-iter sweep: Kronecker vs Jacobi across realistic EFGP regimes.

Each case runs a short Adam trajectory from the notebook's init hypers and
records (mean-CG, trace-CG) iters per step for both preconditioners on the
same model/data/RNG.  We report per-step and aggregate stats so you can see
where Kron's advantage comes from (stiff middle vs warm tail).

Regimes (d=2 unless noted):
  - data geometry: uniform, jittered grid, 2-cluster, gauss blob, tensor grid
  - N: 20k, 100k, 250k
  - d: 2, 3
  - true-noise: 0.01 (default), 1e-4 (low)
  - lengthscale: 0.05 (default), 0.03 (short)

Run: ~/myenv/bin/python -u scratch/scratch_kron_vs_jacobi_regimes.py
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


def sample_x(kind: str, n: int, d: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        return torch.rand(n, d, generator=g, dtype=DT)
    if kind == "gauss":
        return (0.15 * torch.randn(n, d, generator=g, dtype=DT) + 0.5).clamp(0.01, 0.99)
    if kind == "two-cluster":
        n1 = n // 2
        c1 = 0.05 * torch.randn(n1, d, generator=g, dtype=DT) + 0.25
        c2 = 0.05 * torch.randn(n - n1, d, generator=g, dtype=DT) + 0.75
        return torch.cat([c1, c2], 0).clamp(0.01, 0.99)
    if kind == "grid":
        m = int(round(n ** (1.0 / d)))
        g1 = torch.linspace(0.01, 0.99, m, dtype=DT)
        gs = torch.meshgrid(*(g1 for _ in range(d)), indexing="ij")
        return torch.stack(gs, -1).reshape(-1, d)
    if kind == "jitter-grid":
        m = int(round(n ** (1.0 / d)))
        g1 = torch.linspace(0.01, 0.99, m, dtype=DT)
        gs = torch.meshgrid(*(g1 for _ in range(d)), indexing="ij")
        x = torch.stack(gs, -1).reshape(-1, d)
        dx = (0.3 / m) * torch.randn(x.shape, generator=g, dtype=DT)
        return (x + dx).clamp(0.01, 0.99)
    raise ValueError(kind)


def make_data(kind, n, d, true_ls=0.05, true_var=1.0, true_noise=0.01, seed=1):
    torch.manual_seed(seed)
    x = sample_x(kind, n, d, seed)
    f = sample_gp_rff(x, length_scale=true_ls, variance=true_var,
                     num_features=2000, seed=0)
    torch.manual_seed(seed)
    y = f + torch.sqrt(torch.tensor(true_noise)) * torch.randn(f.numel(), dtype=DT)
    return x, y


def run(x, y, d, precond, *, init_ls=0.3, init_var=0.5, init_sigmasq=0.3,
        eps=1e-3, cg_tol=1e-4, max_iters=30, lr=0.3, J=1, noise_floor=1e-5):
    kernel = SquaredExponential(dimension=d, init_lengthscale=init_ls,
                                init_variance=init_var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=init_sigmasq, eps=eps,
                   estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond})
    opt = Adam(model.parameters(), lr=lr)
    mean_cg, trace_cg, t_per = [], [], []
    t_total0 = time.perf_counter()
    for it in range(max_iters):
        t0 = time.perf_counter()
        opt.zero_grad()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
        opt.step()
        t_per.append(time.perf_counter() - t0)
        s = model.last_gradient_stats
        mean_cg.append(s.get('mean_cg_iters'))
        trace_cg.append(s.get('trace_cg_iters'))
    total = time.perf_counter() - t_total0
    return dict(total=total, t_per=t_per, mean_cg=mean_cg, trace_cg=trace_cg,
                mtot=model.last_gradient_stats.get('mtot'),
                M=model.last_gradient_stats.get('feature_count'))


def summarise(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float('nan'),) * 4
    xs_s = sorted(xs)
    n = len(xs_s)
    med = xs_s[n // 2] if n % 2 else 0.5 * (xs_s[n // 2 - 1] + xs_s[n // 2])
    return min(xs), med, max(xs), xs[0]  # min, median, max, first-step


def run_case(name, kind, n, d, *, true_ls=0.05, true_noise=0.01,
             init_ls=0.3, init_var=0.5, init_sigmasq=0.3, eps=1e-3):
    x, y = make_data(kind, n, d, true_ls=true_ls, true_noise=true_noise)
    out = {}
    for precond in ["kronecker", "jacobi"]:
        out[precond] = run(x, y, d, precond,
                           init_ls=init_ls, init_var=init_var,
                           init_sigmasq=init_sigmasq, eps=eps)
    row = dict(name=name, kind=kind, n=n, d=d, ls=true_ls, s2=true_noise,
               mtot=out['kronecker']['mtot'], M=out['kronecker']['M'])
    for pc in ["kronecker", "jacobi"]:
        r = out[pc]
        row[pc] = dict(
            total=r['total'],
            mean_cg=summarise(r['mean_cg']),
            trace_cg=summarise(r['trace_cg']),
            trace_all=r['trace_cg'],
            mean_all=r['mean_cg'],
        )
    return row


def fmt_range(s):
    # s = (min, med, max, first)
    return f"{s[0]:>3d}-{s[2]:<3d} (med {s[1]:>4.1f})"


def print_row(row):
    k = row['kronecker']; j = row['jacobi']
    speedup = j['total'] / k['total']
    print(f"  {row['name']:<22s} n={row['n']:>6d} d={row['d']} "
          f"kind={row['kind']:<11s} mtot={row['mtot']:>3d} M={row['M']:>6d}")
    print(f"    {'kronecker':<10s} total={k['total']:6.2f}s  "
          f"cg-mean {fmt_range(k['mean_cg'])}  cg-trace {fmt_range(k['trace_cg'])}")
    print(f"    {'jacobi':<10s} total={j['total']:6.2f}s  "
          f"cg-mean {fmt_range(j['mean_cg'])}  cg-trace {fmt_range(j['trace_cg'])}")
    print(f"    speedup (jacobi/kron) = {speedup:.2f}x")
    print(f"    trace-cg trajectory (kron):   {k['trace_all']}")
    print(f"    trace-cg trajectory (jacobi): {j['trace_all']}")
    print()


cases = [
    # name, kind, n, d, true_ls, true_noise
    ("A baseline 2D",       "uniform",     100_000, 2, 0.05, 1e-2),
    ("B bigger N",          "uniform",     250_000, 2, 0.05, 1e-2),
    ("C short lengthscale", "uniform",     100_000, 2, 0.03, 1e-2),
    ("D low noise",         "uniform",     100_000, 2, 0.05, 1e-4),
    ("E short ℓ + low σ²",  "uniform",     100_000, 2, 0.03, 1e-4),
    ("F jittered grid",     "jitter-grid", 100_000, 2, 0.05, 1e-2),
    ("G tensor grid",       "grid",        100_000, 2, 0.05, 1e-2),
    ("H two-cluster",       "two-cluster", 100_000, 2, 0.05, 1e-2),
    ("I gauss blob",        "gauss",       100_000, 2, 0.05, 1e-2),
    ("J 3D uniform",        "uniform",      30_000, 3, 0.10, 1e-2),
    ("K 3D short ℓ",        "uniform",      30_000, 3, 0.05, 1e-2),
]


if __name__ == "__main__":
    print(f"\nCG-iter sweep: Kron vs Jacobi, 30 Adam steps, eps=1e-3, cg_tol=1e-4", flush=True)
    print(f"(init ℓ=0.3, σ_f²=0.5, σ_n²=0.3; cg-mean/cg-trace = (min, med, max) across 30 steps)\n",
          flush=True)
    rows = []
    for args in cases:
        name, kind, n, d, true_ls, true_noise = args
        print(f"[running] {name}...", flush=True)
        try:
            row = run_case(name, kind, n, d,
                           true_ls=true_ls, true_noise=true_noise)
            rows.append(row)
            print_row(row)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
