"""
OISST SKI timing sweep: vary grid size on 2D OISST data at n=10,000.
Uses OISST notebook settings: cg_tol=1, max_cg=1000, trace_samples=1,
max_lanczos=1, toeplitz=True, memory_efficient=True, precond_size=10.
"""
import sys
sys.path.insert(0, 'oisst_experiment')

import torch
from load_oisst import load_oisst_torch
from utils.ski import fit_ski_gp

x, y = load_oisst_torch(n_sub=10_000, seed=0)
x_min, x_max = x.min(dim=0).values, x.max(dim=0).values
x = (x - x_min) / (x_max - x_min)
y_mean, y_std = y.mean(), y.std()
y = (y - y_mean) / y_std
x = x.to(dtype=torch.float32)
y = y.to(dtype=torch.float32)
print(f"n={x.shape[0]}, d={x.shape[1]}")

max_iters = 50

grid_sizes = [
    (32, 16),
    (48, 24),
    (64, 32),
    (96, 48),
    (128, 64),
    (256, 128),
]

print(f"\n{'grid':>12s}  {'total_pts':>9s}  {'time(s)':>8s}  {'ls':>8s}  {'os':>8s}  {'noise':>8s}")
print("-" * 70)

for gs in grid_sizes:
    res = fit_ski_gp(
        x, y, kernel='SE', max_iters=max_iters, lr=0.3,
        grid_size=gs,
        dtype=torch.float32, verbose=False,
        cg_tolerance=1,
        max_cg_iterations=1000,
        max_preconditioner_size=10,
        max_lanczos_quadrature_iterations=1,
        num_trace_samples=1,
        use_toeplitz=True,
        memory_efficient=True,
    )
    h = res['history']
    total = gs[0] * gs[1]
    print(f"{str(gs):>12s}  {total:>9d}  {res['fit_time_sec']:>8.2f}  "
          f"{h['lengthscale'][-1]:>8.4f}  {h['outputscale'][-1]:>8.4f}  {h['noise'][-1]:>8.4f}")
