"""
Scratch: verify that the CG + NUFFT + Toeplitz pipeline recovers exact
gradients when given enough trace samples and tight CG tolerance.

At M=17, Hutchinson with T=M probe vectors (identity matrix) should give
EXACT trace. And CG on an M×M system should converge in ≤M iterations.
So any remaining error isolates bugs in the pipeline itself.
"""

import torch
from kernels.squared_exponential import SquaredExponential
from efgpnd import EFGPND, _cmplx, NUFFT, ToeplitzND
from efgpnd import compute_convolution_vector_vectorized_dD, create_A_mean
from utils.gradient_tests import compute_gradients_vanilla, compute_gradients_truncated
from utils.kernels import get_xis
from cg import ConjugateGradients

torch.manual_seed(0)
dtype = torch.float64
n, d = 80, 1
from vanilla_gp_sampling import sample_gp_fast
x = torch.rand(n, d, dtype=dtype)
y = sample_gp_fast(x, length_scale=0.15, variance=1.0, noise_variance=0.5, num_samples=1).squeeze()
ls_val, var_val, noise_val = 0.3, 1.5, 0.8

# --- Reference: exact gradient ---
kernel_v = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
sigmasq_v = torch.tensor(noise_val, dtype=dtype)
grad_vanilla = compute_gradients_vanilla(x, y, sigmasq_v, kernel_v, "SquaredExponential")
print(f"Vanilla d(NLL)/d(pos): ls={grad_vanilla[0]:.8f}, var={grad_vanilla[1]:.8f}, noise={grad_vanilla[2]:.8f}")

# --- Reference: truncated (exact inv of approx kernel) ---
eps = 1e-6
kernel_t = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
grad_trunc = compute_gradients_truncated(x, y, sigmasq_v, kernel_t, eps)
print(f"Truncated (eps={eps}): ls={grad_trunc[0]:.8f}, var={grad_trunc[1]:.8f}, noise={grad_trunc[2]:.8f}")

# --- Build EFGP spectral machinery manually ---
print(f"\n{'='*70}")
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

# --- Step 1: Check CG solve for alpha ---
Fy = fadj(y).reshape(-1)
rhs_mean = ws * Fy
cg = ConjugateGradients(A_apply, rhs_mean, torch.zeros_like(rhs_mean), tol=1e-14, early_stopping=True)
beta = cg.solve()
print(f"\nAlpha CG iters: {cg.iters_completed}")

# Check residual
resid = A_apply(beta.unsqueeze(0)).squeeze() - rhs_mean
print(f"Alpha CG residual: {resid.abs().max().item():.2e}")

# Build A_mean dense for comparison
A_dense = torch.zeros(M, M, dtype=cdtype)
for j in range(M):
    e_j = torch.zeros(1, M, dtype=cdtype)
    e_j[0, j] = 1.0
    A_dense[:, j] = A_apply(e_j).squeeze()

beta_exact = torch.linalg.solve(A_dense, rhs_mean)
print(f"Beta CG vs exact: max diff = {(beta - beta_exact).abs().max().item():.2e}")

# --- Step 2: Check trace with identity probes (should be EXACT) ---
# Use standard basis as probes: tr(A^{-1}) = sum_j e_j^T A^{-1} e_j
I_M = torch.eye(M, dtype=cdtype)
cg_trace_exact = ConjugateGradients(A_apply, I_M, torch.zeros_like(I_M), tol=1e-14, early_stopping=True)
Ainv_cols = cg_trace_exact.solve()
print(f"\nTrace CG (identity probes, M={M}): iters={cg_trace_exact.iters_completed}")
tr_Ainv_cg = torch.diag(Ainv_cols).real.sum().item()
tr_Ainv_exact = torch.trace(torch.linalg.inv(A_dense)).real.item()
print(f"tr(A^-1) via CG identity probes: {tr_Ainv_cg:.8f}")
print(f"tr(A^-1) via dense inv:          {tr_Ainv_exact:.8f}")
print(f"diff: {abs(tr_Ainv_cg - tr_Ainv_exact):.2e}")

# --- Step 3: Reproduce the FULL gradient using CG but with exact trace ---
z = fwd((ws * beta).unsqueeze(0)).squeeze()
alpha = (y.to(dtype=cdtype) - z) / sigmasq_v

# term2
fadj_alpha = (Fy - toeplitz(ws * beta)) / sigmasq_v
term2_ls = torch.vdot(fadj_alpha, Dprime[:, 0] * fadj_alpha).real
alpha_norm = torch.vdot(alpha, alpha).real
y_alpha = torch.vdot(y.to(dtype=cdtype), alpha).real
term2_var = (y_alpha - noise_val * alpha_norm) / var_val
term2_noise = alpha_norm

# term1: use identity probes (exact trace)
# Lengthscale: tr(K^{-1} dK/dls) via feature space
# We need: for each basis vector e_j, compute e_j^T A^{-1} (Dprime[:,0] * e_j)
# This is sum_j (A^{-1} e_j)[j] * Dprime[j,0]  -- but that's not quite right.
# Actually tr(C^{-1} dK/dls) in feature space needs the full derivation.
# Let's just use Hutchinson with identity for the noise trace and do ls separately.

# For noise: tr(K^{-1}) = (n-M)/sig2 + tr(A^{-1})
term1_noise = (n - M) / noise_val + tr_Ainv_cg
term1_var = (M - noise_val * tr_Ainv_cg) / var_val

# For ls: reproduce the current data-space Hutchinson but with identity probes in data space
# Actually let's just use tons of random probes for ls since it converges fine
T = 5000
torch.manual_seed(42)
Z = torch.empty((T, n), dtype=dtype).bernoulli_(0.5).mul_(2).sub_(1).to(cdtype)
fadjZ = fadj(Z).reshape(T, -1)
DpFZ = Dprime[:, 0] * fadjZ
rhs_ls = fwd(DpFZ).reshape(T, -1)
B_ls = ws * toeplitz(DpFZ).reshape(T, -1)

cg_ls = ConjugateGradients(A_apply, B_ls, torch.zeros_like(B_ls), tol=1e-14, early_stopping=True)
Beta_ls = cg_ls.solve()
fwdBeta_ls = fwd(ws * Beta_ls)
Alpha_ls = (rhs_ls - fwdBeta_ls) / sigmasq_v
term1_ls = (Z.conj() * Alpha_ls).sum(dim=1).real.mean()

grad_ls = 0.5 * (term1_ls - term2_ls)
grad_var = 0.5 * (term1_var - term2_var)
grad_noise = 0.5 * (term1_noise - term2_noise)

print(f"\n{'='*70}")
print(f"Gradient comparison (CG pipeline, exact trace for noise/var)")
print(f"{'='*70}")
print(f"{'':16s} {'vanilla':>12s} {'truncated':>12s} {'CG+exact_tr':>12s}")
for i, name in enumerate(['d/d(ls)', 'd/d(var)', 'd/d(noise)']):
    cg_val = [grad_ls, grad_var, grad_noise][i]
    print(f"  {name:14s} {grad_vanilla[i].item():>12.8f} {grad_trunc[i].item():>12.8f} {cg_val.item():>12.8f}")

print(f"\nRelative errors vs vanilla:")
for i, name in enumerate(['d/d(ls)', 'd/d(var)', 'd/d(noise)']):
    cg_val = [grad_ls, grad_var, grad_noise][i]
    err_trunc = abs(grad_trunc[i].item() - grad_vanilla[i].item()) / max(abs(grad_vanilla[i].item()), 1e-12)
    err_cg = abs(cg_val.item() - grad_vanilla[i].item()) / max(abs(grad_vanilla[i].item()), 1e-12)
    print(f"  {name:14s} truncated={err_trunc:.2e}  CG={err_cg:.2e}")

# --- Step 4: Now check what happens through efgpnd_gradient_batched ---
print(f"\n{'='*70}")
print(f"Through efgpnd_gradient_batched (the actual code path)")
print(f"{'='*70}")
kernel_efgp = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
model = EFGPND(x, y, kernel=kernel_efgp, sigmasq=noise_val, eps=eps, estimate_params=False)
for T_test in [10, 100, 1000, 5000]:
    errs = []
    for seed in range(5):
        torch.manual_seed(seed)
        model._gp_params.raw.grad = None
        raw_grad = model.compute_gradients(
            trace_samples=T_test, cg_tol=1e-14, apply_gradients=False, noise_floor=None
        )
        pos = model._gp_params.pos.detach()
        grad_pos = raw_grad.detach() / pos
        errs.append([(grad_pos[i].item() - grad_vanilla[i].item()) / max(abs(grad_vanilla[i].item()), 1e-12) for i in range(3)])

    import numpy as np
    errs = np.array(errs)
    print(f"  T={T_test:>5d}  ls_err={np.mean(np.abs(errs[:,0])):.2e}±{np.std(errs[:,0]):.2e}  "
          f"var_err={np.mean(np.abs(errs[:,1])):.2e}±{np.std(errs[:,1]):.2e}  "
          f"noise_err={np.mean(np.abs(errs[:,2])):.2e}±{np.std(errs[:,2]):.2e}")
