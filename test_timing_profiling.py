import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.kernels import get_xis
from kernels.squared_exponential import SquaredExponential
from kernels.matern import Matern
import math
import pytorch_finufft.functional as pff
# import sys
# sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature/Tests and Sanity Checks/')
from efgpnd import EFGPND
import warnings
# warnings.filterwarnings("ignore", message=".*disabling cuda.*")


# --- Parameters ---
n = 1_000_000  # Number of points
d = 2  # Dimensionality of the input space
dtype = torch.float64  # Use float64 for numerical stability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
true_noise_variance = 0.2
# --- Generate Input Points ---
# Generate random points in d-dimensional space from -1 to 1
x = torch.rand(n, d, dtype=dtype, device=device) * 2 - 1

# --- Define the true function ---
def true_function(x):
    """
    A 2D function to generate synthetic data
    x: tensor of shape [n, 2]
    """
    # Example of a non-linear function with some interesting features
    return torch.sin(3 * x[:, 0]) * torch.cos(4 * x[:, 1]) + \
           0.5 * torch.exp(-((x[:, 0] - 0.3)**2 + (x[:, 1] + 0.3)**2) / 0.3) + \
           0.7 * torch.sin(2 * torch.pi * (x[:, 0]**2 + x[:, 1]**2))

# --- Generate target values with noise ---
# Compute true function values
f_true = true_function(x)

# Add Gaussian noise
noise = torch.randn(n, dtype=dtype, device=device) * math.sqrt(true_noise_variance)
y = f_true + noise

print(f"Generated {n} points with shape {x.shape}")
print(f"Using device: {device}")

# # --- Visualize a subset of the data ---
# if d == 2:
#     plt.figure(figsize=(10, 8))
    
#     # Plot the first 5000 points (or fewer if n is smaller)
#     subset_size = min(5000, n)
    
#     # Scatter plot colored by function value
#     plt.subplot(1, 2, 1)
#     sc = plt.scatter(x[:subset_size, 0].cpu(), x[:subset_size, 1].cpu(), 
#                      c=f_true[:subset_size].cpu(), cmap='viridis', s=5, alpha=0.7)
#     plt.colorbar(sc, label='True function value')
#     plt.title('True function values')
#     plt.xlabel('x₁')
#     plt.ylabel('x₂')
    
#     # Scatter plot colored by noisy observations
#     plt.subplot(1, 2, 2)
#     sc = plt.scatter(x[:subset_size, 0].cpu(), x[:subset_size, 1].cpu(), 
#                      c=y[:subset_size].cpu(), cmap='viridis', s=5, alpha=0.7)
#     plt.colorbar(sc, label='Noisy observation')
#     plt.title('Noisy observations')
#     plt.xlabel('x₁')
#     plt.ylabel('x₂')
    
#     plt.tight_layout()
#     plt.show()
from efgpnd import efgpnd_gradient_batched
x0 = x.min(dim=0).values
x1 = x.max(dim=0).values
eps = 1e-4
EPSILON = 1e-4
sigmasq = 0.1
from torch.optim import Adam
max_iters = 50
J = 10

# Initialize training log
training_log = {
    'iter': [],
    'lengthscale': [],
    'variance': [],
    'sigmasq': [],
}

model = EFGPND(x, y, kernel='SquaredExponential', eps=EPSILON)
optimizer = Adam(model.parameters(), lr=0.1)
# can be reasonable to increase J and decrease cg_tol as we get closer to convergence
for it in range(max_iters):
    optimizer.zero_grad()

    if it > max_iters*0.8:
        t1 = time.time()
        model.compute_gradients(trace_samples=J)
        t2 = time.time()
        print(f"Time taken: {t2-t1} seconds")
    else:
        t1 = time.time()
        model.compute_gradients(trace_samples=5,cg_tol = 1e-3)

        t2 = time.time()
        print(f"Time taken: {t2-t1} seconds")
    optimizer.step() 



##### Tracking 




    # Record current hyperparameters in the log
    lengthscale = model.kernel.get_hyper('lengthscale')
    variance = model.kernel.get_hyper('variance')
    sigmasq = model._gp_params.sig2.item()
    training_log['iter'].append(it)
    training_log['lengthscale'].append(lengthscale)
    training_log['variance'].append(variance)
    training_log['sigmasq'].append(sigmasq)

    if it % 10 == 0:
        print(f"[ε={EPSILON} | J={J}] iter {it:>3}  "
              f"ℓ={lengthscale:.4g}  "
              f"σ_f²={variance:.4g}  σ_n²={sigmasq:.4g}")

print(f'Final hyperparams: ℓ={lengthscale:.4g}, σ_f²={variance:.4g}, σ_n²={sigmasq:.4g}')

# t2 = time.time()
# print(f"Time taken: {t2-t1} seconds")
history = training_log
lengthscale_history = history['lengthscale']
variance_history = history['variance']
sigmasq_history = history['sigmasq']