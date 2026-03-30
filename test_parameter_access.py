import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.squared_exponential import SquaredExponential
from kernels.kernel_params import GPParams
from efgpnd import EFGPND

# Create some test data
x = torch.linspace(0, 10, 100).reshape(-1, 1)
y = torch.sin(x) + 0.1 * torch.randn_like(x)

# Create kernel and initialize model
kernel = SquaredExponential(dimension=1, lengthscale=1.0, variance=1.0)
model = EFGPND(x, y, kernel, sigmasq=0.1, eps=1e-2)

# Test direct parameter access
print("Initial parameters:")
print(f"Lengthscale: {kernel.get_hyper('lengthscale'):.4f}")
print(f"Variance: {kernel.get_hyper('variance'):.4f}")
print(f"Sigmasq: {model.sigmasq.item():.4f}")

# Create optimizer
optimizer = torch.optim.Adam([model._gp_params.raw], lr=0.1)
model.register_optimizer(optimizer)

# Verify initial parameters in GPParams
print("\nInitial GPParams values:")
print(f"Raw parameters: {model._gp_params.raw.tolist()}")
print(f"Positive parameters: {model._gp_params.pos.tolist()}")

# Update parameters and verify changes
model._gp_params.raw.data[0] += 0.5  # Increase lengthscale
model._gp_params.raw.data[1] -= 0.2  # Decrease variance
model._gp_params.raw.data[2] += 0.3  # Increase sigmasq

# Call sync_parameters to update cache
model._update_param_cache()

print("\nAfter direct update:")
print(f"Lengthscale: {kernel.get_hyper('lengthscale'):.4f}")
print(f"Variance: {kernel.get_hyper('variance'):.4f}")
print(f"Sigmasq: {model.sigmasq.item():.4f}")

# Test spectral density computation using dynamic parameters
xid = torch.linspace(-5, 5, 10).reshape(-1, 1)
sd = kernel.spectral_density(xid)
print(f"\nSpectral density (first 3 values): {sd[:3].tolist()}")

print("\nTest complete!") 