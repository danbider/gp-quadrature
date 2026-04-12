"""
Direct test: does switching EFGP from exp to softplus parameterization
(+ 1/n normalization) make the trajectory match GPyTorch ExactGP exactly?

Three variants, all plain SGD:
1. GPyTorch ExactGP (softplus param, 1/n normalized) — ground truth trajectory
2. EFGP with exp param, lr/n — current approach
3. EFGP with softplus param + 1/n — manual update loop, same lr as GPyTorch

For variant 3, after each compute_gradients call we:
  - Extract pos-space gradient: pos_grad = raw_grad / pos
  - Divide by n (to match GPyTorch's 1/n normalization)
  - Maintain a softplus-raw parameter: raw_sp = log(exp(pos) - 1)
  - Compute softplus raw gradient: sp_grad = pos_grad/n * sigmoid(raw_sp)
  - SGD step: raw_sp -= lr * sp_grad
  - Convert back: pos = softplus(raw_sp) = log(1 + exp(raw_sp))
  - Write pos back into EFGP's internal parameters
"""

import torch
import torch.nn.functional as F
import gpytorch
from kernels.squared_exponential import SquaredExponential
from efgpnd import EFGPND
from vanilla_gp_sampling import sample_gp_fast

torch.manual_seed(42)
dtype = torch.float64
n, d = 500, 1
x = torch.rand(n, d, dtype=dtype)
y = sample_gp_fast(x, length_scale=0.1, variance=1.0, noise_variance=1.0, num_samples=1).squeeze()

init_ls, init_var, init_noise = 0.5, 2.0, 0.5
max_iters = 500
lr = 0.1
EPSILON = 1e-8
J = 50
cg_tol = 1e-8

# ---- 1. GPyTorch ExactGP + SGD ----
class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))

lik = gpytorch.likelihoods.GaussianLikelihood().to(dtype=dtype)
gp_model = _ExactGPModel(x, y, lik).to(dtype=dtype)
with torch.no_grad():
    gp_model.base_kernel.lengthscale = init_ls
    gp_model.covar_module.outputscale = init_var
    lik.noise = init_noise
gp_model.train(); lik.train()
opt_gp = torch.optim.SGD(gp_model.parameters(), lr=lr)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp_model)

gp_hist = {'ls': [], 'var': [], 'noise': []}
for i in range(max_iters):
    opt_gp.zero_grad()
    loss = -mll(gp_model(x), y)
    loss.backward()
    opt_gp.step()
    gp_hist['ls'].append(float(gp_model.base_kernel.lengthscale.detach().mean()))
    gp_hist['var'].append(float(gp_model.covar_module.outputscale.detach()))
    gp_hist['noise'].append(float(lik.noise.detach()))
print("GPyTorch done")

# ---- 2. EFGP + SGD with exp param, lr/n ----
kernel_e = SquaredExponential(dimension=d, init_lengthscale=init_ls, init_variance=init_var)
model_e = EFGPND(x, y, kernel=kernel_e, sigmasq=init_noise, eps=EPSILON, estimate_params=False)
opt_e = torch.optim.SGD(model_e.parameters(), lr=lr/n)

exp_hist = {'ls': [], 'var': [], 'noise': []}
for i in range(max_iters):
    opt_e.zero_grad()
    model_e.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=1e-5)
    opt_e.step()
    exp_hist['ls'].append(model_e.kernel.get_hyper('lengthscale'))
    exp_hist['var'].append(model_e.kernel.get_hyper('variance'))
    exp_hist['noise'].append(model_e._gp_params.sig2.item())
    if i % 100 == 0:
        print(f"  EFGP-exp iter {i}")
print("EFGP-exp done")

# ---- 3. EFGP + manual softplus param + 1/n, same lr as GPyTorch ----
kernel_sp = SquaredExponential(dimension=d, init_lengthscale=init_ls, init_variance=init_var)
model_sp = EFGPND(x, y, kernel=kernel_sp, sigmasq=init_noise, eps=EPSILON, estimate_params=False)

# Initialize softplus-raw parameters from current pos values
pos = model_sp._gp_params.pos.detach().clone()
raw_sp = torch.log(torch.exp(pos) - 1)  # softplus_inv

sp_hist = {'ls': [], 'var': [], 'noise': []}
for i in range(max_iters):
    # Compute EFGP gradients (sets raw.grad in exp-space)
    model_sp._gp_params.raw.grad = None
    raw_grad = model_sp.compute_gradients(
        trace_samples=J, cg_tol=cg_tol, apply_gradients=False, noise_floor=1e-5
    )

    # Extract pos-space gradient: pos_grad = raw_grad / pos (undo exp chain rule)
    pos = model_sp._gp_params.pos.detach().clone()
    pos_grad = raw_grad.detach() / pos

    # Divide by n (match GPyTorch's 1/n normalization)
    pos_grad_normalized = pos_grad / n

    # Compute softplus-raw gradient: d(NLL)/d(raw_sp) = d(NLL)/d(pos) * sigmoid(raw_sp)
    sp_raw_grad = pos_grad_normalized * torch.sigmoid(raw_sp)

    # SGD step in softplus-raw space
    with torch.no_grad():
        raw_sp = raw_sp - lr * sp_raw_grad

    # Convert back to pos-space: pos = softplus(raw_sp)
    pos_new = F.softplus(raw_sp)

    # Write back into EFGP's internal parameters (log-space raw)
    with torch.no_grad():
        model_sp._gp_params.raw.copy_(torch.log(pos_new))

    sp_hist['ls'].append(pos_new[0].item())
    sp_hist['var'].append(pos_new[1].item())
    sp_hist['noise'].append(pos_new[2].item())
    if i % 100 == 0:
        print(f"  EFGP-softplus iter {i}")
print("EFGP-softplus done")

# ---- Print comparison ----
print(f"\n{'='*90}")
print(f"Trajectory comparison (SGD lr={lr}, n={n})")
print(f"{'='*90}")
print(f"{'iter':>4s}  {'GPy ls':>7s} {'Eexp ls':>7s} {'Esp ls':>7s}  |  "
      f"{'GPy var':>7s} {'Eexp var':>8s} {'Esp var':>8s}  |  "
      f"{'GPy nse':>7s} {'Eexp nse':>8s} {'Esp nse':>8s}")
for i in [0, 9, 19, 49, 99, 199, 299, 399, 499]:
    if i < max_iters:
        print(f"{i+1:>4d}  "
              f"{gp_hist['ls'][i]:>7.4f} {exp_hist['ls'][i]:>7.4f} {sp_hist['ls'][i]:>7.4f}  |  "
              f"{gp_hist['var'][i]:>7.4f} {exp_hist['var'][i]:>8.4f} {sp_hist['var'][i]:>8.4f}  |  "
              f"{gp_hist['noise'][i]:>7.4f} {exp_hist['noise'][i]:>8.4f} {sp_hist['noise'][i]:>8.4f}")
