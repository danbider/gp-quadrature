"""
Does EFGP's CG+Hutchinson pipeline match a DENSE solve on the same
approximate (truncated spectral) kernel?

This isolates CG/Hutchinson errors from spectral truncation errors.
We build the approximate kernel K_approx = F*D^2*F^T, form it densely,
compute exact gradients via autograd on K_approx, and compare against
EFGP's stochastic gradient.

If these match: the pipeline is correct, any remaining gap vs vanilla
is just spectral truncation.
If these don't match: there's a bug in the CG/Hutchinson/Toeplitz pipeline.
"""

import torch
from kernels.squared_exponential import SquaredExponential
from efgpnd import EFGPND, _cmplx, NUFFT, ToeplitzND
from efgpnd import compute_convolution_vector_vectorized_dD, create_A_mean
from utils.gradient_tests import compute_gradients_vanilla, compute_gradients_truncated
from utils.kernels import get_xis
from vanilla_gp_sampling import sample_gp_fast
import numpy as np

torch.manual_seed(42)
dtype = torch.float64
n, d = 500, 1
x = torch.rand(n, d, dtype=dtype)
y = sample_gp_fast(x, length_scale=0.1, variance=1.0, noise_variance=1.0, num_samples=1).squeeze()

EPSILON = 1e-8

# Test at multiple points along the trajectory
test_points = [
    (0.5, 2.0, 0.5),     # init
    (0.35, 3.0, 0.7),    # early
    (0.24, 4.0, 0.9),    # mid
    (0.17, 6.0, 1.1),    # near overshoot peak
    (0.13, 7.5, 1.1),    # at overshoot peak
    (0.13, 2.3, 1.03),   # near convergence
    (0.10, 1.0, 1.0),    # at truth
]

print(f"{'ls':>5s} {'var':>5s} {'nse':>5s}  |  "
      f"{'':>8s} {'vanilla':>10s} {'trunc':>10s} {'EFGP(200s)':>10s} {'dense_approx':>12s}")

for ls_val, var_val, noise_val in test_points:
    sigmasq = torch.tensor(noise_val, dtype=dtype)

    # 1. Vanilla (exact, full kernel)
    kernel_v = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
    grad_vanilla = compute_gradients_vanilla(x, y, sigmasq, kernel_v, "SquaredExponential")

    # 2. Truncated (exact inverse of approx kernel, via utils)
    kernel_t = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
    grad_trunc = compute_gradients_truncated(x, y, sigmasq, kernel_t, EPSILON)

    # 3. Dense solve on the approximate kernel (build K_approx explicitly)
    kernel_d = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
    cdtype = _cmplx(dtype)
    L = (x.max(dim=0).values - x.min(dim=0).values).max()
    xis_1d, h, mtot = get_xis(kernel_obj=kernel_d, eps=EPSILON, L=L, use_integral=True, l2scaled=False)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing='ij')
    xis = torch.stack(grids, dim=-1).view(-1, d)

    # Make these require grad for autograd
    ls_param = torch.tensor(ls_val, dtype=dtype, requires_grad=True)
    var_param = torch.tensor(var_val, dtype=dtype, requires_grad=True)
    noise_param = torch.tensor(noise_val, dtype=dtype, requires_grad=True)

    # Build spectral density with grad tracking
    two_pi_sq = (2 * torch.pi) ** 2
    xid_norm_sq = (xis ** 2).sum(dim=-1)
    s_w = var_param * (ls_param * torch.sqrt(torch.tensor(2 * torch.pi, dtype=dtype))) ** d * \
          torch.exp(-0.5 * two_pi_sq * ls_param ** 2 * xid_norm_sq)
    ws = torch.sqrt(s_w * h ** d).to(cdtype)

    # Build F matrix (NUFFT type-2 as dense matrix)
    # F[i, j] = exp(2*pi*i * x[i] . (xcen + h*xi[j]))  -- but actually we need
    # the exact same NUFFT that efgpnd uses. Let's just build K_approx = F' diag(|w|^2) F
    # where F is the type-2 NUFFT matrix
    OUT = (mtot,) * d
    xcen = torch.zeros(d, dtype=dtype)
    nufft_op = NUFFT(x, xcen, h, 1e-14, cdtype=cdtype)

    # Build F densely: F[i,j] such that F @ coeff = type2(coeff)
    M = ws.numel()
    F_mat = torch.zeros(n, M, dtype=cdtype)
    for j in range(M):
        e_j = torch.zeros(1, M, dtype=cdtype)
        e_j[0, j] = 1.0
        F_mat[:, j] = nufft_op.type2(e_j, out_shape=OUT).squeeze()

    # K_approx = F diag(|w|^2) F^* + sigma^2 I
    W2 = (ws.abs() ** 2).to(cdtype)
    # Need to keep var_param in the computation graph
    # ws^2 = s_w * h^d, and s_w contains var_param
    W2_grad = (s_w * h ** d).to(cdtype)
    K_approx = F_mat @ torch.diag(W2_grad) @ F_mat.conj().T + noise_param * torch.eye(n, dtype=cdtype)

    # NLL = 0.5 * (y^T K^{-1} y + log|K|) + const
    K_real = K_approx.real  # should be real-valued
    L_chol = torch.linalg.cholesky(K_real)
    alpha_dense = torch.cholesky_solve(y.unsqueeze(1), L_chol).squeeze()
    logdet = 2 * L_chol.diagonal().log().sum()
    nll = 0.5 * (y @ alpha_dense + logdet)

    nll.backward()
    grad_dense = torch.tensor([ls_param.grad.item(), var_param.grad.item(), noise_param.grad.item()])

    # 4. EFGP stochastic (average over seeds)
    num_seeds = 200
    efgp_grads = []
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        kernel_e = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
        model_e = EFGPND(x, y, kernel=kernel_e, sigmasq=noise_val, eps=EPSILON, estimate_params=False)
        raw_grad = model_e.compute_gradients(trace_samples=50, cg_tol=1e-8, apply_gradients=False, noise_floor=None)
        pos = model_e._gp_params.pos.detach()
        grad_pos = raw_grad.detach() / pos
        efgp_grads.append(grad_pos.numpy())
    efgp_grads = np.array(efgp_grads)
    efgp_mean = efgp_grads.mean(axis=0)

    print(f"\n{ls_val:>5.2f} {var_val:>5.1f} {noise_val:>5.1f}")
    for j, name in enumerate(['ls', 'var', 'noise']):
        v = grad_vanilla[j].item()
        t = grad_trunc[j].item()
        e = efgp_mean[j]
        da = grad_dense[j].item()
        # relative error of EFGP vs dense_approx
        rel_err = abs(e - da) / max(abs(da), 1e-12)
        print(f"  {name:>5s}:  vanilla={v:>+10.4f}  trunc={t:>+10.4f}  "
              f"efgp={e:>+10.4f}  dense_approx={da:>+10.4f}  "
              f"|efgp-dense|/|dense|={rel_err:.2e}")
