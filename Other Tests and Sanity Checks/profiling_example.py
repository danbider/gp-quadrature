#!/usr/bin/env python
# Profiling example script for EFGPND gradient computation
# This example demonstrates both direct profiling of the gradient function and
# profiling within the optimize_hyperparameters method.

import torch
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import sys 
import os
import gpytorch
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature/')
from efgpnd import efgpnd_gradient_batched, EFGPND
from kernels.squared_exponential import SquaredExponential

def generate_synthetic_data(N=100000, d=2, seed=42):
    """Generate synthetic data for GP regression"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Inputs
    x = torch.rand(N, d)
    
    # Function to generate outputs (with noise)
    def f(x):
        return torch.sin(2*np.pi*x[:, 0]) * torch.cos(2*np.pi*x[:, 1]) + 0.1*torch.randn(x.shape[0])
    
    # Outputs
    y = f(x)
    
    return x, y

def direct_gradient_profiling():
    """Test direct profiling of the gradient computation function"""
    print("\n" + "="*60)
    print("DIRECT GRADIENT PROFILING")
    print("="*60)
    
    # # Generate synthetic data
    # N =1_000_000  # number of data points
    # d = 2    # dimensionality
    # x, y = generate_synthetic_data(N, d, seed=123)
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 10_000_000
    N = n
    d = 2
    true_noise_variance = 0.2
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
    
    # Create SquaredExponential kernel instance
    kernel = SquaredExponential(dimension=d, init_lengthscale=0.05, init_variance=3)
    # Set parameters for the gradient computation
    sigmasq = 0.2  # noise variance
    eps = 1e-4     # quadrature accuracy parameter
    cg_tol = eps
    trace_samples = 5  # number of samples for Hutchinson trace estimation
    
    # Data bounds
    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values
    
    print(f"Dataset: {N} points in {d} dimensions")
    print(f"Kernel: SquaredExponential with lengthscale={kernel.lengthscale}, variance={kernel.variance}")
    print(f"Noise variance: {sigmasq}")
    
    # Enable deterministic algorithms for consistent results
    # torch.manual_seed(123)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(123)
    
    # # Run without profiling first for comparison
    # print("\nRunning gradient computation without profiling...")
    # start_time = time.time()
    # grad_no_profile = efgpnd_gradient_batched(
    #     x, y, sigmasq, kernel, eps, trace_samples, x0, x1,
    #     do_profiling=False
    # )
    # no_profile_time = time.time() - start_time
    # print(f"Gradient without profiling completed in {no_profile_time:.4f} seconds")
    # print(f"Gradient result: {grad_no_profile}")
    
    # Reset random seed to get identical results
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    
    # Run with profiling enabled
    print("\nRunning gradient computation with profiling enabled...")
    model = EFGPND(x, y, kernel, sigmasq, eps,estimate_params=False)
    start_time = time.time()
    t1 = time.time()

    # grad_with_profile = efgpnd_gradient_batched(
    #     x, y, sigmasq, kernel, eps, trace_samples, x0, x1,
    #     do_profiling=True, nufft_eps=0.1*eps,compute_log_marginal=False
    # )
    model.compute_gradients(do_profiling=True, trace_samples=5)
    t2 = time.time()
    profile_time = t2 - t1
    print(f"Gradient with profiling completed in {profile_time:.4f} seconds")

    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

            # SKI requires a grid size hyperparameter. This util can help with that. Here we are using a grid that has the same number of points as the training data (a ratio of 1.0). Performance can be sensitive to this parameter, so you may want to adjust it for your own problem on a validation set.
            grid_size = gpytorch.utils.grid.choose_grid_size(train_x,0.1)
            grid_size = 100

            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.GridInterpolationKernel(
                    gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=2
                )
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(x.to(dtype=torch.float32), y.to(dtype=torch.float32), likelihood)

    # Initialize hyperparameters to match EFGPND model
    model.covar_module.base_kernel.base_kernel.lengthscale = kernel.lengthscale
    model.covar_module.outputscale = kernel.variance
    likelihood.noise = sigmasq

    smoke_test = ('CI' in os.environ)
    training_iterations = 2 if smoke_test else 100

    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iterations = 1

    best_loss = float('inf')
    best_model_state = None

    # Initialize training log dictionary
    ski_log = {
        'iteration': [],
        'loss': [],
        'lengthscale': [],
        'noise': [],
        'outputscale': []
    }

    print("--- Starting SKI Gradient Profiling ---")
    for i in range(training_iterations):
        t1 = time.time()
        output = model(x.to(dtype=torch.float32))
        loss = -mll(output, y.to(dtype=torch.float32))
        loss.backward()
        t2 = time.time()
        print(f"Iteration {i+1} completed in {t2 - t1:.4f} seconds")




    # print(f"Gradient result: {grad_with_profile}")
    
    # Verify that results are identical (within numerical precision)
    # if torch.allclose(grad_no_profile, grad_with_profile, rtol=1e-5, atol=1e-5):
    #     print("\nResults with and without profiling match ✓")
    # else:
    #     print("\nWARNING: Results with and without profiling differ!")
    #     print(f"  No profile: {grad_no_profile}")
    #     print(f"  With profile: {grad_with_profile}")
    #     print(f"  Max difference: {torch.max(torch.abs(grad_no_profile - grad_with_profile))}")
        
    # print(f"Profiling overhead: {profile_time - no_profile_time:.4f} seconds")


# def optimization_with_profiling():
#     """Test profiling within the optimize_hyperparameters method"""
#     print("\n" + "="*60)
#     print("OPTIMIZATION WITH PROFILING")
#     print("="*60)
    
#     # Generate synthetic data
#     N = 150
#     d = 2
#     x, y = generate_synthetic_data(N, d)
    
#     # Create SquaredExponential kernel instance
#     kernel = SquaredExponential(dimension=d, lengthscale=0.5, variance=1.2)
    
#     # Set parameters for GP
#     sigmasq = 0.2
#     eps = 1e-2
    
#     print(f"Dataset: {N} points in {d} dimensions")
#     print(f"Initial kernel: SquaredExponential with lengthscale={kernel.lengthscale}, variance={kernel.variance}")
#     print(f"Initial noise variance: {sigmasq}")
    
#     # Create EFGPND model
#     model = EFGPND(x, y, kernel, sigmasq, eps)
    
#     # First run without profiling
#     print("\nRunning optimization without profiling...")
#     start_time = time.time()
#     model.optimize_hyperparameters(
#         epsilon_values=[eps],
#         trace_samples_values=[5],
#         max_iters=3,  # Small number of iterations for demonstration
#         profile_gradient=False
#     )
#     no_profile_time = time.time() - start_time
    
#     print(f"\nOptimization without profiling completed in {no_profile_time:.4f} seconds")
#     print(f"Final hyperparameters: lengthscale={model.kernel.lengthscale:.4f}, variance={model.kernel.variance:.4f}, noise={model.sigmasq.item():.4f}")
    
#     # Reset model
#     model = EFGPND(x, y, kernel, sigmasq, eps)
    
#     # Run with profiling enabled (first iteration only)
#     print("\nRunning optimization with profiling (first iteration only)...")
#     start_time = time.time()
#     model.optimize_hyperparameters(
#         epsilon_values=[eps],
#         trace_samples_values=[5],
#         max_iters=3,  # Small number of iterations for demonstration
#         profile_gradient=True,
#         profile_first_iter=True,
#         profile_last_iter=False
#     )
#     profile_time = time.time() - start_time
    
#     print(f"\nOptimization with profiling completed in {profile_time:.4f} seconds")
#     print(f"Final hyperparameters: lengthscale={model.kernel.lengthscale:.4f}, variance={model.kernel.variance:.4f}, noise={model.sigmasq.item():.4f}")


if __name__ == "__main__":
    print("\nEFGPND Profiling Example")
    print("========================\n")
    
    # Demonstrate direct gradient profiling
    direct_gradient_profiling()
    

    
    # # Demonstrate profiling during optimization
    # optimization_with_profiling()
    
    print("\nProfiling example completed successfully!") 