"""
Test: proposed single-NUFFT Term 2 is identical to the current path,
      for all three hypers (lengthscale, variance, noise).

Current path:
  F*y = NUFFT(y)                   (NUFFT #1)
  beta_0 = A_mean^{-1} D F*y
  alpha = (y - F D beta_0) / s2    (NUFFT #2 for F D beta_0)
  F*alpha = (F*y - T D beta_0)/s2  (Toeplitz)
  T2(ell)    = <F*alpha, D'_ell F*alpha>
  T2(sig_f2) = (y^T alpha - s2 ||alpha||^2) / sig_f2
  T2(s2)     = ||alpha||^2

Proposed path (no NUFFT #2):
  F*y = NUFFT(y)                   (NUFFT #1, same)
  beta_0 = A_mean^{-1} D F*y       (same)
  F*alpha = (F*y - T D beta_0)/s2  (Toeplitz, same)
  c = Re{beta_0^* D F*y}           (M-space dot)
  T2(ell)    = <F*alpha, D'_ell F*alpha>
  T2(sig_f2) = ||D F*alpha||^2 / sig_f2            (generic D' path)
  T2(s2)     = s2^{-4} (||y||^2 - c - s2 ||beta_0||^2)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from kernels.squared_exponential import SquaredExponential
from efgpnd import (
    _cmplx, NUFFT, ToeplitzND,
    compute_convolution_vector_vectorized_dD,
    create_A_mean, create_jacobi_precond,
)
from utils.kernels import get_xis
from cg import ConjugateGradients
from vanilla_gp_sampling import sample_gp_fast


torch.manual_seed(0)
dtype = torch.float64
cdtype = _cmplx(dtype)

d = 1
n = 4000
ls_val, var_val, noise_val = 0.08, 1.3, 0.17   # pick non-trivial values
eps = 1e-6

x = torch.rand(n, d, dtype=dtype)
y = sample_gp_fast(x, length_scale=ls_val, variance=var_val,
                   noise_variance=noise_val, num_samples=1).squeeze().to(dtype)

kernel = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
sigmasq = torch.tensor(noise_val, dtype=dtype)

L = (x.max(dim=0).values - x.min(dim=0).values).max()
xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L,
                          use_integral=True, l2scaled=False)
grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
xis = torch.stack(grids, dim=-1).view(-1, d)
ws = torch.sqrt(kernel.spectral_density(xis).to(cdtype) * h**d)
Dprime_all = (h**d * kernel.spectral_grad(xis)).to(cdtype)
Dp_ell = Dprime_all[:, kernel.hypers.index("lengthscale")]
Dp_var = Dprime_all[:, kernel.hypers.index("variance")]
M = ws.numel()

OUT = (mtot,) * d
xcen = torch.zeros(d, dtype=dtype)
nufft_op = NUFFT(x, xcen, h, 1e-12, cdtype=cdtype)
def fadj(v):
    out = nufft_op.type1(v, out_shape=OUT)
    return out.reshape(-1) if v.ndim == 1 else out.reshape(v.shape[0], -1)
def fwd(fk):
    return nufft_op.type2(fk.reshape(OUT), out_shape=OUT)

m_conv = (mtot - 1) // 2
v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(cdtype)
toeplitz = ToeplitzND(v_kernel, force_pow2=True)
A_apply = create_A_mean(ws, toeplitz, sigmasq, cdtype)
center_tuple = tuple(((torch.tensor(v_kernel.shape) - 1) // 2).tolist())
jacobi = create_jacobi_precond(ws, sigmasq, diag_scale=v_kernel[center_tuple].real)

# ---- shared solve ----
Fy = fadj(y.to(cdtype))
rhs = ws * Fy
cg = ConjugateGradients(A_apply, rhs, torch.zeros_like(rhs),
                        tol=1e-12, M_inv_apply=jacobi, early_stopping=True)
beta_0 = cg.solve()
print(f"n={n}, M={M}, CG iters={cg.iters_completed}")

# ============ CURRENT PATH ============
# alpha = (y - F D beta_0) / s2   <-- uses NUFFT #2
Dbeta = ws * beta_0
z = fwd(Dbeta)                                    # the NUFFT we want to kill
alpha_N = (y.to(cdtype) - z) / sigmasq            # N-vector

Fstar_alpha_current = (Fy - toeplitz(Dbeta)) / sigmasq

T2_ell_current = torch.vdot(Fstar_alpha_current, Dp_ell * Fstar_alpha_current).real
y_alpha = torch.vdot(y.to(cdtype), alpha_N).real
alpha_norm_current = torch.vdot(alpha_N, alpha_N).real

variance_scalar = torch.tensor(var_val, dtype=dtype)
T2_var_current   = (y_alpha - sigmasq * alpha_norm_current) / variance_scalar
T2_noise_current = alpha_norm_current

# ============ PROPOSED PATH (no NUFFT #2) ============
Fstar_alpha_prop = (Fy - toeplitz(Dbeta)) / sigmasq   # identical to current

# lengthscale: same formula
T2_ell_prop = torch.vdot(Fstar_alpha_prop, Dp_ell * Fstar_alpha_prop).real

# variance: generic D' path with D'_var = D^2 / sig_f^2
# (check: spectral_grad's variance column should equal ws*ws.conj() / var_val)
Dp_var_expected = (ws * ws.conj()) / variance_scalar
print(f"variance D' matches D^2/sig_f^2: "
      f"max|diff|={(Dp_var - Dp_var_expected).abs().max().item():.3e}")
T2_var_prop = torch.vdot(Fstar_alpha_prop, Dp_var * Fstar_alpha_prop).real

# noise: scalar Woodbury identity
y_norm2 = (y.to(cdtype) * y.to(cdtype).conj()).sum().real
c_scalar = torch.vdot(beta_0, ws * Fy).real        # Re{beta_0^* D Fy}
beta_norm2 = torch.vdot(beta_0, beta_0).real
T2_noise_prop = (y_norm2 - c_scalar - sigmasq * beta_norm2) / (sigmasq ** 2)

# ============ DENSE GROUND TRUTH ============
F_dense = torch.zeros((n, M), dtype=cdtype)
I_M = torch.eye(M, dtype=cdtype)
for k in range(M):
    F_dense[:, k] = fwd(I_M[k])
D2_diag = ws * ws.conj()
K_tilde = (F_dense * D2_diag) @ F_dense.conj().T + sigmasq * torch.eye(n, dtype=cdtype)
alpha_dense = torch.linalg.solve(K_tilde, y.to(cdtype))
# T2 dense: y^T K^{-1} dK/dtheta K^{-1} y = <alpha, dK/dtheta alpha>
# for each theta
dK_ell = (F_dense * Dp_ell) @ F_dense.conj().T
dK_var = (F_dense * Dp_var) @ F_dense.conj().T            # equals K_approx / sig_f^2
dK_s2  = torch.eye(n, dtype=cdtype)
T2_ell_dense   = (alpha_dense.conj() @ dK_ell @ alpha_dense).real
T2_var_dense   = (alpha_dense.conj() @ dK_var @ alpha_dense).real
T2_noise_dense = (alpha_dense.conj() @ alpha_dense).real

# ============ REPORT ============
def fmt(label, cur, prop, dense):
    d_cp = abs(cur - prop) / max(abs(cur), 1e-30)
    d_pd = abs(prop - dense) / max(abs(dense), 1e-30)
    print(f"  {label:<14s} current={cur:+.10e}  proposed={prop:+.10e}  dense={dense:+.10e}")
    print(f"  {'':<14s}  |cur-prop|/cur   = {d_cp:.2e}   (should be ~CG tol)")
    print(f"  {'':<14s}  |prop-dense|/dense = {d_pd:.2e}   (approx error of EFGP)")
    return d_cp

print("\n--- Term 2 comparison ---")
d_ell = fmt("lengthscale",
            T2_ell_current.item(), T2_ell_prop.item(), T2_ell_dense.item())
d_var = fmt("variance",
            T2_var_current.item(), T2_var_prop.item(), T2_var_dense.item())
d_noi = fmt("noise",
            T2_noise_current.item(), T2_noise_prop.item(), T2_noise_dense.item())

tol = 1e-9
status = "PASS" if max(d_ell, d_var, d_noi) < tol else "FAIL"
print(f"\nMax |current-proposed|/|current| = {max(d_ell, d_var, d_noi):.2e}   [{status} at tol {tol}]")
