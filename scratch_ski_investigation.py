"""
Lean SKI investigation: why is 1D so fast (~15s) vs 2D OISST (~100s)?
Only 10 iters each to get per-iter timings quickly.
"""
import sys
sys.path.insert(0, 'oisst_experiment')

import torch
from utils.ski import fit_ski_gp
from vanilla_gp_sampling import sample_gp_fast
from load_oisst import load_oisst_torch

torch.manual_seed(42)

# --- Datasets ---
x_1d = torch.rand(500, 1, dtype=torch.float64)
y_1d = sample_gp_fast(x_1d, length_scale=0.1, variance=1.0, noise_variance=1.0, num_samples=1).squeeze()

x_1d_10k = torch.rand(10_000, 1, dtype=torch.float64)
y_1d_10k = torch.randn(10_000, dtype=torch.float64)

x_2d, y_2d = load_oisst_torch(n_sub=10_000, seed=0)
xmin, xmax = x_2d.min(0).values, x_2d.max(0).values
x_2d = (x_2d - xmin) / (xmax - xmin)
y_2d = (y_2d - y_2d.mean()) / y_2d.std()

iters = 10
init = dict(init_lengthscale=0.5, init_outputscale=2.0, init_noise=0.5)

# Notebook settings (what comparison notebook uses)
nb = dict(cg_tolerance=1e-3, max_cg_iterations=100, use_toeplitz=True,
          num_trace_samples=2, max_lanczos_quadrature_iterations=10,
          memory_efficient=True, max_preconditioner_size=10)

# OISST settings
oi = dict(cg_tolerance=1, max_cg_iterations=1000, use_toeplitz=True,
          num_trace_samples=1, max_lanczos_quadrature_iterations=1,
          memory_efficient=True, max_preconditioner_size=10)

def run(x, y, grid_size, settings, label):
    res = fit_ski_gp(x, y, kernel='SE', max_iters=iters, lr=0.1,
                     grid_size=grid_size, dtype=torch.float32, verbose=False, **init, **settings)
    h = res['history']
    avg_fwd = sum(h['forward_sec']) / len(h['forward_sec'])
    avg_bwd = sum(h['backward_sec']) / len(h['backward_sec'])
    total = res['fit_time_sec']
    print(f"  {label:45s}  {total:>5.1f}s  fwd={avg_fwd:.3f}s  bwd={avg_bwd:.3f}s  per_iter={total/iters:.3f}s")

print(f"{'='*90}")
print("A) Same data (1D n=500), notebook vs OISST settings")
print(f"{'='*90}")
run(x_1d, y_1d, 500, nb, "1D n=500, grid=500, notebook settings")
run(x_1d, y_1d, 500, oi, "1D n=500, grid=500, OISST settings")

print(f"\n{'='*90}")
print("B) Effect of n: 1D n=500 vs n=10K (OISST settings)")
print(f"{'='*90}")
run(x_1d, y_1d, 500, oi, "1D n=500, grid=500")
run(x_1d_10k, y_1d_10k, 500, oi, "1D n=10K, grid=500")

print(f"\n{'='*90}")
print("C) 1D vs 2D at same n=10K (OISST settings)")
print(f"{'='*90}")
run(x_1d_10k, y_1d_10k, 500, oi, "1D n=10K, grid=500 (500 pts)")
run(x_2d, y_2d, (128, 64), oi, "2D OISST n=10K, grid=(128,64) (8192 pts)")
run(x_2d, y_2d, (22, 22), oi, "2D OISST n=10K, grid=(22,22) (~500 pts)")

print(f"\n{'='*90}")
print("D) 2D grid sweep (OISST settings, n=10K)")
print(f"{'='*90}")
for gs in [(22,22), (32,16), (64,32), (128,64), (256,128)]:
    run(x_2d, y_2d, gs, oi, f"2D OISST, grid={gs} ({gs[0]*gs[1]} pts)")
