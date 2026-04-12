"""
Compare EFGP vs GPyTorch ExactGP: raw gradient magnitudes and SGD trajectories.
"""
import torch, time
from kernels.squared_exponential import SquaredExponential
from efgpnd import EFGPND
from vanilla_gp_sampling import sample_gp_fast
import gpytorch

torch.manual_seed(42)
dtype = torch.float64
n, d = 500, 1
x = torch.rand(n, d, dtype=dtype)
y = sample_gp_fast(x, length_scale=0.1, variance=1.0, noise_variance=1.0, num_samples=1).squeeze()

init_ls, init_var, init_noise = 0.5, 2.0, 0.5
EPSILON = 1e-8

# ---- Step 1: Compare raw gradient magnitudes ----
print("="*70)
print("Raw gradient magnitudes at init point")
print("="*70)

# GPyTorch
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
loss = -gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp_model)(gp_model(x), y)
loss.backward()

gpy_grads = {}
for name, p in list(gp_model.named_parameters()) + list(lik.named_parameters()):
    if p.grad is not None:
        gpy_grads[name] = p.grad.item()
        print(f"  GPyTorch {name:40s} raw_grad = {p.grad.item():+.6f}")

# EFGP
kernel_e = SquaredExponential(dimension=d, init_lengthscale=init_ls, init_variance=init_var)
model_e = EFGPND(x, y, kernel=kernel_e, sigmasq=init_noise, eps=EPSILON, estimate_params=False)
model_e._gp_params.raw.grad = None
raw_grad = model_e.compute_gradients(trace_samples=50, cg_tol=1e-8, apply_gradients=False, noise_floor=None)
print(f"\n  EFGP raw_grad = {raw_grad.tolist()}")
print(f"  EFGP params: {model_e._gp_params.raw.data.tolist()} (log-space)")

# Compute the ratio
# GPyTorch raw params are softplus-inverse, EFGP raw params are log
# GPyTorch also normalizes by 1/n
# The key raw grads to compare: lengthscale, outputscale/variance, noise
print(f"\n  GPyTorch normalizes by 1/n={n}, so multiply by n to get unnormalized:")
for name, g in gpy_grads.items():
    print(f"    {name:40s} unnorm_grad = {g*n:+.6f}")

print(f"\n  EFGP grad / GPyTorch_unnorm_grad ratio would show transform difference")

# ---- Step 2: SGD with matched lr ----
print(f"\n{'='*70}")
print(f"SGD trajectories")
print(f"{'='*70}")

max_iters = 60
gpy_lr = 0.01  # small enough for GPyTorch SGD to work

# GPyTorch SGD
lik2 = gpytorch.likelihoods.GaussianLikelihood().to(dtype=dtype)
gp2 = _ExactGPModel(x, y, lik2).to(dtype=dtype)
with torch.no_grad():
    gp2.base_kernel.lengthscale = init_ls
    gp2.covar_module.outputscale = init_var
    lik2.noise = init_noise
gp2.train(); lik2.train()
opt_gp = torch.optim.SGD(gp2.parameters(), lr=gpy_lr)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik2, gp2)

gp_hist = {'ls': [], 'var': [], 'noise': []}
for i in range(max_iters):
    opt_gp.zero_grad()
    loss = -mll(gp2(x), y)
    loss.backward()
    opt_gp.step()
    gp_hist['ls'].append(float(gp2.base_kernel.lengthscale.detach().mean()))
    gp_hist['var'].append(float(gp2.covar_module.outputscale.detach()))
    gp_hist['noise'].append(float(lik2.noise.detach()))

# EFGP SGD with lr/n to match GPyTorch's 1/n normalization
efgp_lr = gpy_lr / n
kernel_e3 = SquaredExponential(dimension=d, init_lengthscale=init_ls, init_variance=init_var)
model_e3 = EFGPND(x, y, kernel=kernel_e3, sigmasq=init_noise, eps=EPSILON, estimate_params=False)
opt_e3 = torch.optim.SGD(model_e3.parameters(), lr=efgp_lr)

efgp_hist = {'ls': [], 'var': [], 'noise': []}
for i in range(max_iters):
    opt_e3.zero_grad()
    model_e3.compute_gradients(trace_samples=50, cg_tol=1e-8, noise_floor=1e-5)
    opt_e3.step()
    efgp_hist['ls'].append(model_e3.kernel.get_hyper('lengthscale'))
    efgp_hist['var'].append(model_e3.kernel.get_hyper('variance'))
    efgp_hist['noise'].append(model_e3._gp_params.sig2.item())
    if i % 10 == 0:
        print(f"  EFGP iter {i}")

print(f"\nTrajectory (GPyTorch lr={gpy_lr}, EFGP lr={efgp_lr:.6f}):")
print(f"{'iter':>4s}  {'GPy ls':>8s} {'EFGP ls':>8s}  |  {'GPy var':>8s} {'EFGP var':>9s}  |  {'GPy nse':>8s} {'EFGP nse':>9s}")
for i in [0, 4, 9, 19, 29, 39, 49, 59]:
    if i < max_iters:
        print(f"{i+1:>4d}  {gp_hist['ls'][i]:>8.4f} {efgp_hist['ls'][i]:>8.4f}"
              f"  |  {gp_hist['var'][i]:>8.4f} {efgp_hist['var'][i]:>9.4f}"
              f"  |  {gp_hist['noise'][i]:>8.4f} {efgp_hist['noise'][i]:>9.4f}")
