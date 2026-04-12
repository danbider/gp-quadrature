"""
Generate learning curve figure: Vanilla GP vs EFGP-softplus vs EFGP-exp.
All use Adam with the same lr. Saves figure.
"""
import torch
import gpytorch
import matplotlib.pyplot as plt
from kernels.squared_exponential import SquaredExponential
from efgpnd import EFGPND
from vanilla_gp_sampling import sample_gp_fast

torch.manual_seed(42)
dtype = torch.float64
n, d = 500, 1
x = torch.rand(n, d, dtype=dtype)
y = sample_gp_fast(x, length_scale=0.1, variance=1.0, noise_variance=1.0, num_samples=1).squeeze()

init_ls, init_var, init_noise = 0.5, 2.0, 0.5
max_iters = 200
lr = 0.1
EPSILON = 1e-8
J = 50
cg_tol = 1e-8

# ---- 1. Vanilla GP + Adam ----
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
gp = _ExactGPModel(x, y, lik).to(dtype=dtype)
with torch.no_grad():
    gp.base_kernel.lengthscale = init_ls
    gp.covar_module.outputscale = init_var
    lik.noise = init_noise
gp.train(); lik.train()
opt_gp = torch.optim.Adam(gp.parameters(), lr=lr)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp)
gp_hist = {'ls': [], 'var': [], 'noise': []}
for i in range(max_iters):
    opt_gp.zero_grad()
    (-mll(gp(x), y)).backward()
    opt_gp.step()
    gp_hist['ls'].append(float(gp.base_kernel.lengthscale.detach().mean()))
    gp_hist['var'].append(float(gp.covar_module.outputscale.detach()))
    gp_hist['noise'].append(float(lik.noise.detach()))
print("Vanilla done")

# ---- 2. EFGP softplus (default) ----
kernel_sp = SquaredExponential(dimension=d, init_lengthscale=init_ls, init_variance=init_var)
model_sp = EFGPND(x, y, kernel=kernel_sp, sigmasq=init_noise, eps=EPSILON, estimate_params=False)
opt_sp = torch.optim.Adam(model_sp.parameters(), lr=lr)
sp_hist = {'ls': [], 'var': [], 'noise': []}
for i in range(max_iters):
    opt_sp.zero_grad()
    model_sp.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=1e-5)
    opt_sp.step()
    sp_hist['ls'].append(model_sp.kernel.get_hyper('lengthscale'))
    sp_hist['var'].append(model_sp.kernel.get_hyper('variance'))
    sp_hist['noise'].append(model_sp._gp_params.sig2.item())
print("EFGP-softplus done")

# ---- 3. EFGP exp (legacy) ----
kernel_exp = SquaredExponential(dimension=d, init_lengthscale=init_ls, init_variance=init_var)
model_exp = EFGPND(x, y, kernel=kernel_exp, sigmasq=init_noise, eps=EPSILON,
                   estimate_params=False, param_transform='exp')
opt_exp = torch.optim.Adam(model_exp.parameters(), lr=lr)
exp_hist = {'ls': [], 'var': [], 'noise': []}
for i in range(max_iters):
    opt_exp.zero_grad()
    model_exp.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=1e-5)
    opt_exp.step()
    exp_hist['ls'].append(model_exp.kernel.get_hyper('lengthscale'))
    exp_hist['var'].append(model_exp.kernel.get_hyper('variance'))
    exp_hist['noise'].append(model_exp._gp_params.sig2.item())
print("EFGP-exp done")

# ---- Plot ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
iters = list(range(max_iters))
specs = [('Lengthscale', 'ls', 0.1), ('Variance', 'var', 1.0), ('Noise', 'noise', 1.0)]

for ax, (title, key, true_val) in zip(axes, specs):
    ax.plot(iters, gp_hist[key], 'g-', lw=2, label='Vanilla GP')
    ax.plot(iters, sp_hist[key], 'r--', lw=2, label='EFGP (softplus + 1/n)')
    ax.plot(iters, exp_hist[key], 'b-', lw=1.5, alpha=0.7, label='EFGP (exp, no 1/n)')
    ax.axhline(true_val, color='gray', ls=':', lw=1.5, label='True')
    ax.set_xlabel('Iteration')
    ax.set_title(title)
    ax.legend(fontsize=8)

fig.suptitle(f'Adam lr={lr}: softplus+1/n makes EFGP match vanilla GP (n={n})', fontsize=13)
fig.tight_layout()
fig.savefig('learning_curves_parameterization.png', dpi=150, bbox_inches='tight')
print("Saved learning_curves_parameterization.png")
plt.close()
