"""
Compare M-step gradients: joint solver vs Schur complement.
Single call, same data, same delta.
"""
import sys
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hierarchical"))

from test_blockcd_convergence import make_synthetic_data
from hierarchical.pg_hierarchical import (
    _build_hierarchical_spectral_state,
    _compute_hierarchical_mstep_gradient,
)
from hierarchical.pg_hierarchical_blockcd import (
    _build_local_spectral_state,
    _compute_schur_mstep_gradient,
    _sample_rademacher,
)
from kernels import SquaredExponential
from utils.kernels import get_xis

# Small synthetic data
X, y, locations, f_true, g_true, h_true = make_synthetic_data(n_per_loc=200, n_locations=5)
n = len(y)

X_t = torch.tensor(X, dtype=torch.float64)
y_t = torch.tensor(y, dtype=torch.float64)
loc_t = torch.tensor(locations, dtype=torch.long)

kernel_g = SquaredExponential(dimension=1, init_lengthscale=0.3, init_variance=1.5)
kernel_h = SquaredExponential(dimension=1, init_lengthscale=0.1, init_variance=0.5)
r = 5.0

kappa = 0.5 * (y_t - r)
pg_b = y_t + r
delta = 0.25 * pg_b

rdtype = torch.float64
cdtype = torch.complex128
dev = torch.device("cpu")

# --- Build joint spectral state ---
joint_spec = _build_hierarchical_spectral_state(
    X_t, loc_t, kernel_g, kernel_h,
    spectral_eps=1e-4, trunc_eps=1e-4, nufft_eps=1e-7,
    rdtype=rdtype, cdtype=cdtype, device=dev,
)

# --- Build block CD spectral states ---
d = X_t.shape[1]
x0 = X_t.min(dim=0).values
x1 = X_t.max(dim=0).values
L_domain = (x1 - x0).max()

xis_g_1d, h_g, mtot_g = get_xis(kernel_g, eps=1e-4, L=L_domain, use_integral=True, l2scaled=False, trunc_eps=1e-4)
xis_h_1d, h_h, mtot_h = get_xis(kernel_h, eps=1e-4, L=L_domain, use_integral=True, l2scaled=False, trunc_eps=1e-4)
if mtot_h >= mtot_g:
    xis_1d, h, mtot = xis_h_1d, h_h, mtot_h
else:
    xis_1d, h, mtot = xis_g_1d, h_g, mtot_g

spec_g = _build_local_spectral_state(
    X_t, kernel_g, xis_1d=xis_1d, h=h, mtot=mtot,
    spectral_eps=1e-4, nufft_eps=1e-7,
    rdtype=rdtype, cdtype=cdtype, device=dev,
)

unique_locs = torch.unique(loc_t)
loc_indices = []
specs_h = []
for loc in unique_locs:
    idx = torch.where(loc_t == loc)[0]
    loc_indices.append(idx)
    spec_ell = _build_local_spectral_state(
        X_t[idx], kernel_h, xis_1d=xis_1d, h=h, mtot=mtot,
        spectral_eps=1e-4, nufft_eps=1e-7,
        rdtype=rdtype, cdtype=cdtype, device=dev,
    )
    specs_h.append(spec_ell)

print(f"n={n}, L={len(specs_h)}, m={spec_g.ws.numel()}")

# --- Joint gradient ---
print("\n--- Joint M-step gradient ---")
joint_out = _compute_hierarchical_mstep_gradient(
    kappa, delta, joint_spec,
    n_probes=30, cg_tol=1e-8, seed=42,
)
print(f"grad_g = {joint_out['grad_g'].real}")
print(f"grad_h = {joint_out['grad_h'].real}")
print(f"CG iters: {joint_out['cg_iters']}")

# --- Schur gradient with different inner tolerances ---
for inner_tol in [1e-6, 1e-8, 1e-10]:
    print(f"\n--- Schur M-step gradient (inner_tol={inner_tol:.0e}) ---")
    schur_out = _compute_schur_mstep_gradient(
        kappa, delta, spec_g, specs_h, loc_indices,
        n_probes=30, cg_tol=1e-8, inner_cg_tol=inner_tol, seed=42,
    )
    print(f"grad_g = {schur_out['grad_g'].real}")
    print(f"grad_h = {schur_out['grad_h'].real}")
    print(f"CG iters: {schur_out['cg_iters']}")

    diff_g = (schur_out['grad_g'] - joint_out['grad_g']).abs().real
    diff_h = (schur_out['grad_h'] - joint_out['grad_h']).abs().real
    print(f"Diff grad_g: {diff_g}")
    print(f"Diff grad_h: {diff_h}")
