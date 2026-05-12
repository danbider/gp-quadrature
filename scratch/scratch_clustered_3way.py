"""
3-way comparison (Kron vs Jacobi vs Nystrom) on clustered geometries.
Small-n / short trajectory so Nystrom doesn't stall.

Run: ~/myenv/bin/python -u scratch/scratch_clustered_3way.py
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


def sample_x(kind, n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "two-cluster":
        n1 = n // 2
        c1 = 0.05 * torch.randn(n1, d, generator=g, dtype=DT) + 0.25
        c2 = 0.05 * torch.randn(n - n1, d, generator=g, dtype=DT) + 0.75
        return torch.cat([c1, c2], 0).clamp(0.01, 0.99)
    if kind == "gauss":
        return (0.15 * torch.randn(n, d, generator=g, dtype=DT) + 0.5).clamp(0.01, 0.99)
    if kind == "four-cluster":
        n4 = n // 4
        centers = torch.tensor([[0.25, 0.25], [0.25, 0.75],
                                [0.75, 0.25], [0.75, 0.75]], dtype=DT)
        parts = []
        for i, c in enumerate(centers[:-1]):
            parts.append(0.04 * torch.randn(n4, d, generator=g, dtype=DT) + c)
        parts.append(0.04 * torch.randn(n - 3 * n4, d, generator=g, dtype=DT) + centers[-1])
        return torch.cat(parts, 0).clamp(0.01, 0.99)
    raise ValueError(kind)


def make_data(kind, n, d, true_ls=0.05, true_var=1.0, true_noise=0.01, seed=1):
    torch.manual_seed(seed)
    x = sample_x(kind, n, d, seed)
    f = sample_gp_rff(x, length_scale=true_ls, variance=true_var,
                      num_features=2000, seed=0)
    torch.manual_seed(seed)
    y = f + torch.sqrt(torch.tensor(true_noise)) * torch.randn(f.numel(), dtype=DT)
    return x, y


def run(x, y, d, opts, *, max_iters=10, lr=0.3, init_ls=0.3, init_var=0.5,
        init_sigmasq=0.3, eps=1e-3, cg_tol=1e-4, J=1, noise_floor=1e-5,
        cg_max=500):
    opts = dict(opts)
    opts.setdefault("max_cg_iterations", cg_max)
    kernel = SquaredExponential(dimension=d, init_lengthscale=init_ls,
                                init_variance=init_var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=init_sigmasq, eps=eps,
                   estimate_params=False, opts=opts)
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
    return dict(total=total, mean_cg=mean_cg, trace_cg=trace_cg,
                mtot=model.last_gradient_stats.get('mtot'),
                M=model.last_gradient_stats.get('feature_count'))


def label_of(opts):
    t = opts["mean_cg_preconditioner_type"]
    if t == "nystrom":
        return f"nystrom-r{opts.get('nystrom_rank', 30)}"
    return t


def summarize(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return (float('nan'),) * 3
    xs_s = sorted(xs); n = len(xs_s)
    med = xs_s[n // 2] if n % 2 else 0.5 * (xs_s[n // 2 - 1] + xs_s[n // 2])
    return min(xs), med, max(xs)


cases = [
    ("two-cluster, n=20k", "two-cluster", 20_000, 2),
    ("gauss blob,  n=20k", "gauss",       20_000, 2),
]

opts_list = [
    {"mean_cg_preconditioner_type": "jacobi"},
    {"mean_cg_preconditioner_type": "kronecker"},
    {"mean_cg_preconditioner_type": "nystrom", "nystrom_rank": 30},
    {"mean_cg_preconditioner_type": "nystrom", "nystrom_rank": 100},
    {"mean_cg_preconditioner_type": "nystrom", "nystrom_rank": 300},
]


if __name__ == "__main__":
    print("3-way on clustered data (10 Adam steps, eps=1e-3, cg_tol=1e-4, cg_max=500)",
          flush=True)
    print("init ℓ=0.3, σ_f²=0.5, σ_n²=0.3", flush=True)
    for name, kind, n, d in cases:
        print(f"\n=== {name} ===", flush=True)
        x, y = make_data(kind, n, d)
        kron_total = None
        for opts in opts_list:
            lbl = label_of(opts)
            try:
                r = run(x, y, d, opts)
                m = summarize(r['mean_cg']); t = summarize(r['trace_cg'])
                if lbl == "kronecker": kron_total = r['total']
                ratio = (r['total'] / kron_total) if kron_total else float('nan')
                print(f"  {lbl:<14s} total={r['total']:7.2f}s  "
                      f"cg-mean {m[0]:>4d}-{m[2]:<4d} (med {m[1]:>5.1f})  "
                      f"cg-trace {t[0]:>4d}-{t[2]:<4d} (med {t[1]:>5.1f})  "
                      f"this/kron={ratio:5.2f}x  (mtot={r['mtot']}, M={r['M']})",
                      flush=True)
            except Exception as e:
                print(f"  {lbl:<14s} FAILED: {type(e).__name__}: {e}", flush=True)
