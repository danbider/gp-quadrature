"""
Check whether the EFGP variance gradient is biased at multiple points
along the optimization trajectory — specifically at the parameter values
where the variance overshoots (var ~ 3, 5, 7).
"""

import torch
import numpy as np
from kernels.squared_exponential import SquaredExponential
from efgpnd import EFGPND
from utils.gradient_tests import compute_gradients_vanilla
from vanilla_gp_sampling import sample_gp_fast

torch.manual_seed(42)
dtype = torch.float64
n, d = 500, 1
x = torch.rand(n, d, dtype=dtype)
y = sample_gp_fast(x, length_scale=0.1, variance=1.0, noise_variance=1.0, num_samples=1).squeeze()

EPSILON = 1e-8
J = 50
cg_tol = 1e-8

# Test at several points along the trajectory where the overshoot happens
test_points = [
    (0.5, 2.0, 0.5),    # init
    (0.35, 3.0, 0.7),   # early
    (0.24, 4.0, 0.9),   # mid
    (0.17, 6.0, 1.1),   # approaching peak
    (0.13, 7.5, 1.1),   # near peak overshoot
    (0.13, 5.0, 1.05),  # coming back down
    (0.10, 1.0, 1.0),   # near truth
]

num_seeds = 200
T = 50  # match what we use in training

for ls_val, var_val, noise_val in test_points:
    # Exact vanilla gradient
    kernel_v = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
    sigmasq_v = torch.tensor(noise_val, dtype=dtype)
    grad_vanilla = compute_gradients_vanilla(x, y, sigmasq_v, kernel_v, "SquaredExponential")
    true_var_grad = grad_vanilla[1].item()

    # EFGP gradient over many seeds
    efgp_var_grads = []
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        kernel_e = SquaredExponential(dimension=d, init_lengthscale=ls_val, init_variance=var_val)
        model = EFGPND(x, y, kernel=kernel_e, sigmasq=noise_val, eps=EPSILON, estimate_params=False)
        model._gp_params.raw.grad = None
        raw_grad = model.compute_gradients(trace_samples=T, cg_tol=cg_tol, apply_gradients=False, noise_floor=None)
        pos = model._gp_params.pos.detach()
        grad_pos = raw_grad.detach() / pos  # convert raw grad to pos-space
        efgp_var_grads.append(grad_pos[1].item())

    arr = np.array(efgp_var_grads)
    bias = arr.mean() - true_var_grad
    stderr = arr.std() / np.sqrt(num_seeds)

    print(f"ls={ls_val:.2f} var={var_val:.1f} noise={noise_val:.1f}  |  "
          f"true={true_var_grad:+.4f}  efgp_mean={arr.mean():+.4f}  "
          f"bias={bias:+.4f}  std={arr.std():.4f}  |bias|/se={abs(bias)/stderr:.1f}σ")
