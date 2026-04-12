"""
Posterior mean comparison: EFGP (eps=1e-4) vs SGPR (M=50).
4 panels: posterior for each method, error (pred - true) for each method.
"""
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import gpytorch
from torch.optim import Adam

from kernels.squared_exponential import SquaredExponential
from efgpnd import EFGPND
from utils.sgpr import fit_sgpr
from vanilla_gp_sampling import sample_gp_rff

dtype = torch.float64

# --- Ground truth data (matches notebook) ---
n = 100_000
d = 2
true_ls = 0.05
true_var = 1.0
true_noise = 0.01

torch.manual_seed(1)
x = torch.rand(n, d, dtype=dtype)
f = sample_gp_rff(x, length_scale=true_ls, variance=true_var, num_features=5000, seed=0)
torch.manual_seed(1)
y = f + torch.sqrt(torch.tensor(true_noise, dtype=dtype)) * torch.randn(n, dtype=dtype)
print(f"Data: n={n}, d={d}, y std={y.std():.3f}")

# --- Shared config ---
init_ls, init_var, init_noise = 0.3, 1.0, 0.3
max_iters = 50
lr = 0.5
noise_floor = 1e-5

# --- Test grid ---
n_test_per_dim = 50
g1 = torch.linspace(0, 1, n_test_per_dim, dtype=dtype)
g2 = torch.linspace(0, 1, n_test_per_dim, dtype=dtype)
grid = torch.meshgrid(g1, g2, indexing='ij')
x_test = torch.stack([g.flatten() for g in grid], dim=-1)
f_test_true = sample_gp_rff(x_test, length_scale=true_ls, variance=true_var,
                             num_features=5000, seed=0)
print(f"Test grid: {x_test.shape[0]} points")

# ============================================================
# 1. EFGP  eps=1e-4
# ============================================================
print("\n--- EFGP eps=1e-4 ---")
eps = 1e-4
cg_tol = 1e-5

kernel_efgp = SquaredExponential(dimension=d, init_lengthscale=init_ls, init_variance=init_var)
model_efgp = EFGPND(x, y, kernel=kernel_efgp, sigmasq=init_noise, eps=eps, estimate_params=False)
opt_efgp = Adam(model_efgp.parameters(), lr=lr)

t0 = time.time()
for it in range(max_iters):
    opt_efgp.zero_grad()
    model_efgp.compute_gradients(trace_samples=1, cg_tol=cg_tol, noise_floor=noise_floor)
    opt_efgp.step()
    if it % 10 == 0:
        ls = model_efgp.kernel.get_hyper('lengthscale')
        var = model_efgp.kernel.get_hyper('variance')
        sig = model_efgp._gp_params.sig2.item()
        print(f"  iter {it:>3}  l={ls:.4g}  sf2={var:.4g}  sn2={sig:.4g}")
efgp_fit_time = time.time() - t0
ls = model_efgp.kernel.get_hyper('lengthscale')
var = model_efgp.kernel.get_hyper('variance')
sig = model_efgp._gp_params.sig2.item()
print(f"  Final: l={ls:.4g}, sf2={var:.4g}, sn2={sig:.4g}  ({efgp_fit_time:.1f}s)")

t0 = time.time()
efgp_mean, _ = model_efgp.predict(x_test, return_variance=False)
efgp_pred_time = time.time() - t0
efgp_mean = efgp_mean.detach()
efgp_mse = float(((efgp_mean - f_test_true) ** 2).mean())
print(f"  Predict: {efgp_pred_time:.1f}s  MSE={efgp_mse:.4e}")

# ============================================================
# 2. SGPR  M=50
# ============================================================
print("\n--- SGPR M=50 ---")
sgpr_res = fit_sgpr(
    x, y, kernel='SE', num_inducing=50,
    max_iters=max_iters, lr=lr / 5,
    init_lengthscale=init_ls, init_outputscale=init_var, init_noise=init_noise,
    dtype=dtype, verbose=False,
)
h = sgpr_res['history']
print(f"  Final: ls={h['lengthscale'][-1]:.4f}, os={h['outputscale'][-1]:.4f}, "
      f"noise={h['noise'][-1]:.4f}  ({sgpr_res['fit_time_sec']:.1f}s)")

sgpr_res['model'].eval()
sgpr_res['likelihood'].eval()
with torch.no_grad():
    sgpr_pred = sgpr_res['likelihood'](sgpr_res['model'](x_test))
    sgpr_mean = sgpr_pred.mean.to(dtype=dtype)
sgpr_mse = float(((sgpr_mean - f_test_true) ** 2).mean())
print(f"  MSE={sgpr_mse:.4e}")

# ============================================================
# Figure: Ground truth, EFGP, SGPR M=50
# ============================================================
shape = (n_test_per_dim, n_test_per_dim)
extent = [0, 1, 0, 1]

f_true_grid = f_test_true.reshape(shape).numpy()
efgp_grid = efgp_mean.reshape(shape).numpy()
sgpr_grid = sgpr_mean.detach().reshape(shape).numpy()

vmin = min(f_true_grid.min(), efgp_grid.min(), sgpr_grid.min())
vmax = max(f_true_grid.max(), efgp_grid.max(), sgpr_grid.max())

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

im0 = axes[0].imshow(f_true_grid.T, origin='lower', extent=extent,
                      vmin=vmin, vmax=vmax, cmap='RdBu_r', aspect='equal')
axes[0].set_title('Ground Truth', fontsize=13)

im1 = axes[1].imshow(efgp_grid.T, origin='lower', extent=extent,
                      vmin=vmin, vmax=vmax, cmap='RdBu_r', aspect='equal')
axes[1].set_title(r'EFGP  $\varepsilon = 10^{-4}$', fontsize=13)

im2 = axes[2].imshow(sgpr_grid.T, origin='lower', extent=extent,
                      vmin=vmin, vmax=vmax, cmap='RdBu_r', aspect='equal')
axes[2].set_title(r'SGPR  $M = 50$', fontsize=13)

cb = fig.colorbar(im0, ax=axes.tolist(), shrink=0.9, pad=0.02)
cb.set_label(r'$f(\mathbf{x})$', fontsize=12)

for ax in axes:
    ax.set_xlabel(r'$x_1$', fontsize=11)
    ax.set_ylabel(r'$x_2$', fontsize=11)

fig.suptitle(
    f'Posterior Mean  ($n = {n:,}$,  $d = {d}$)',
    fontsize=14,
)
plt.savefig('posterior_comparison.png', dpi=180, bbox_inches='tight')
print("\nSaved posterior_comparison.png")
plt.show()
