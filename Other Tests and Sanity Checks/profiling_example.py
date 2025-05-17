#!/usr/bin/env python
# Profiling example script for EFGPND gradient computation
# This example demonstrates both direct profiling of the gradient function and
# profiling within the optimize_hyperparameters method.

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import sys 
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
    
    # Generate synthetic data
    N =1_000_000  # number of data points
    d = 2    # dimensionality
    x, y = generate_synthetic_data(N, d, seed=42)
    
    # Create SquaredExponential kernel instance
    kernel = SquaredExponential(dimension=d, lengthscale=0.5, variance=1.0)
    
    # Set parameters for the gradient computation
    sigmasq = 0.1  # noise variance
    eps = 1e-4     # quadrature accuracy parameter
    trace_samples = 10  # number of samples for Hutchinson trace estimation
    
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
    start_time = time.time()
    grad_with_profile = efgpnd_gradient_batched(
        x, y, sigmasq, kernel, eps, trace_samples, x0, x1,
        do_profiling=True, nufft_eps=0.1*eps
    )
    profile_time = time.time() - start_time
    print(f"Gradient with profiling completed in {profile_time:.4f} seconds")
    print(f"Gradient result: {grad_with_profile}")
    
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