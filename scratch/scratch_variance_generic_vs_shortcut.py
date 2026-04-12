"""
Test: does the generic Hutchinson path (same as lengthscale) give correct
variance gradients?  Compare to the shortcut currently in efgpnd.

The generic path for any kernel hyperparameter θ computes:
  term1[θ] = (1/T) Σ_t z_t^T K^{-1} (dK/dθ) K^{-1} z_t    (Hutchinson)
  term2[θ] = α^T (dK/dθ) α

For variance specifically, dK/dvar = K/var (since K = var * K_unit),
so this should work but may have different variance characteristics
than the algebraic shortcut.
"""

import torch
import numpy as np
from kernels.squared_exponential import SquaredExponential
from efgpnd import EFGPND, _cmplx, NUFFT, ToeplitzND
from efgpnd import compute_convolution_vector_vectorized_dD, create_A_mean
from utils.gradient_tests import compute_gradients_vanilla
from utils.kernels import get_xis
from cg import ConjugateGradients

torch.manual_seed(0)
dtype = torch.float64
n, d = 200, 1

from vanilla_gp_sampling import sample_gp_fast
x = torch.rand(n, d, dtype=dtype)
y = sample_gp_fast(x, length_scale=0.15, variance=1.0, noise_variance=0.5, num_samples=1).squeeze()
ls_val, var_val, noise_val = 0.5, 2.0, 0.5

# --- Exact vanilla gradient ---
kernel_v = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
sigmasq_v = torch.tensor(noise_val, dtype=dtype)
grad_vanilla = compute_gradients_vanilla(x, y, sigmasq_v, kernel_v, "SquaredExponential")
print(f"Vanilla exact:  ls={grad_vanilla[0]:.8f}  var={grad_vanilla[1]:.8f}  noise={grad_vanilla[2]:.8f}")

# --- Build spectral machinery ---
eps = 1e-6
kernel_e = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
cdtype = _cmplx(dtype)
L = (x.max(dim=0).values - x.min(dim=0).values).max()
xis_1d, h, mtot = get_xis(kernel_obj=kernel_e, eps=eps, L=L, use_integral=True, l2scaled=False)
grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing='ij')
xis = torch.stack(grids, dim=-1).view(-1, d)
ws = torch.sqrt(kernel_e.spectral_density(xis).to(dtype=cdtype) * h**d)
M = ws.numel()
Dprime = (h**d * kernel_e.spectral_grad(xis)).to(cdtype)
print(f"M={M}, n={n}")

OUT = (mtot,) * d
xcen = torch.zeros(d, dtype=dtype)
nufft_op = NUFFT(x, xcen, h, 1e-14, cdtype=cdtype)
fadj = lambda v: nufft_op.type1(v, out_shape=OUT).reshape(v.shape[0], -1) if v.ndim == 2 else nufft_op.type1(v, out_shape=OUT).reshape(-1)
fwd = lambda fk: nufft_op.type2(fk, out_shape=OUT)

m_conv = (mtot - 1) // 2
v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
toeplitz = ToeplitzND(v_kernel, force_pow2=True)
A_apply = create_A_mean(ws, toeplitz, sigmasq_v, cdtype)

# --- Solve for alpha (shared across all seeds) ---
Fy = fadj(y).reshape(-1)
rhs_mean = ws * Fy
cg = ConjugateGradients(A_apply, rhs_mean, torch.zeros_like(rhs_mean), tol=1e-14, early_stopping=True)
beta = cg.solve()

z = fwd((ws * beta).unsqueeze(0)).squeeze()
alpha = (y.to(dtype=cdtype) - z) / sigmasq_v
fadj_alpha = (Fy - toeplitz(ws * beta)) / sigmasq_v

# --- term2 (deterministic, same for both paths) ---
# Lengthscale
term2_ls = torch.vdot(fadj_alpha, Dprime[:, 0] * fadj_alpha).real
# Variance: shortcut
alpha_norm = torch.vdot(alpha, alpha).real
y_alpha = torch.vdot(y.to(dtype=cdtype), alpha).real
term2_var_shortcut = (y_alpha - noise_val * alpha_norm) / var_val
# Variance: generic (same formula as ls but with Dprime[:, 1])
term2_var_generic = torch.vdot(fadj_alpha, Dprime[:, 1] * fadj_alpha).real
# Noise
term2_noise = alpha_norm

print(f"\nterm2 comparison:")
print(f"  var shortcut: {term2_var_shortcut.item():.8f}")
print(f"  var generic:  {term2_var_generic.item():.8f}")
print(f"  difference:   {abs(term2_var_shortcut.item() - term2_var_generic.item()):.2e}")

# --- Monte Carlo term1 across seeds ---
T = 200
num_seeds = 200

shortcut_grads = []
generic_grads = []
ls_grads = []

for seed in range(num_seeds):
    torch.manual_seed(seed + 100)

    # --- Generic path for lengthscale AND variance ---
    Z = torch.empty((T, n), dtype=dtype).bernoulli_(0.5).mul_(2).sub_(1).to(cdtype)
    fadjZ = fadj(Z).reshape(T, -1)

    # Lengthscale: generic Hutchinson
    DpFZ_ls = Dprime[:, 0] * fadjZ
    rhs_ls = fwd(DpFZ_ls)
    B_ls = ws * toeplitz(DpFZ_ls)
    cg_ls = ConjugateGradients(A_apply, B_ls, torch.zeros_like(B_ls), tol=1e-14, early_stopping=True)
    Beta_ls = cg_ls.solve()
    Alpha_ls = (rhs_ls - fwd(ws * Beta_ls)) / sigmasq_v
    term1_ls = (Z.conj() * Alpha_ls).sum(dim=1).real.mean()

    # Variance: GENERIC Hutchinson (using Dprime[:, 1])
    DpFZ_var = Dprime[:, 1] * fadjZ
    rhs_var = fwd(DpFZ_var)
    B_var = ws * toeplitz(DpFZ_var)
    cg_var = ConjugateGradients(A_apply, B_var, torch.zeros_like(B_var), tol=1e-14, early_stopping=True)
    Beta_var = cg_var.solve()
    Alpha_var = (rhs_var - fwd(ws * Beta_var)) / sigmasq_v
    term1_var_generic = (Z.conj() * Alpha_var).sum(dim=1).real.mean()

    # Variance: SHORTCUT (uses noise trace)
    V = torch.empty((T, M), dtype=dtype).bernoulli_(0.5).mul_(2).sub_(1).to(cdtype)
    B_noise = ws * toeplitz(ws * V)
    cg_noise = ConjugateGradients(A_apply, B_noise, torch.zeros_like(B_noise), tol=1e-14, early_stopping=True)
    Beta_noise = cg_noise.solve()
    term1_noise = (n / noise_val) - (V.conj() * Beta_noise).sum(dim=1).real.mean() / noise_val
    term1_var_shortcut = (n - noise_val * term1_noise) / var_val

    grad_ls = 0.5 * (term1_ls - term2_ls)
    grad_var_generic = 0.5 * (term1_var_generic - term2_var_generic)
    grad_var_shortcut = 0.5 * (term1_var_shortcut - term2_var_shortcut)

    ls_grads.append(grad_ls.item())
    generic_grads.append(grad_var_generic.item())
    shortcut_grads.append(grad_var_shortcut.item())

ls_grads = np.array(ls_grads)
generic_grads = np.array(generic_grads)
shortcut_grads = np.array(shortcut_grads)

true_ls = grad_vanilla[0].item()
true_var = grad_vanilla[1].item()

print(f"\n{'='*70}")
print(f"Results over {num_seeds} seeds, T={T} probes each")
print(f"True vanilla gradient: ls={true_ls:.8f}  var={true_var:.8f}")
print(f"{'='*70}")

for name, arr, true_val in [
    ("ls (generic)", ls_grads, true_ls),
    ("var (shortcut)", shortcut_grads, true_var),
    ("var (generic)", generic_grads, true_var),
]:
    mean = arr.mean()
    std = arr.std()
    bias = mean - true_val
    stderr = std / np.sqrt(num_seeds)
    print(f"  {name:16s}: mean={mean:.6f}  std={std:.4f}  bias={bias:+.6f}  |bias|/stderr={abs(bias)/stderr:.1f}σ")

print(f"\nKey question: does generic var have similar bias/variance to generic ls?")
print(f"  ls std / var_generic std = {ls_grads.std() / generic_grads.std():.2f}")
print(f"  ls std / var_shortcut std = {ls_grads.std() / shortcut_grads.std():.2f}")
