import torch
import math
import numpy as np
import time
from typing import Callable, Optional, Tuple, Union

def mean_func_zero(x: torch.Tensor) -> torch.Tensor:
    """
    Mean function that returns zero vector.
    
    Args:
        x: Input points tensor of shape (n, d)
        
    Returns:
        Tensor of zeros with shape (n,)
    """
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

def squared_exponential_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: float = 1.0,
    variance: float = 1.0
) -> torch.Tensor:
    """
    Squared Exponential kernel (RBF) for multi-dimensional inputs.
    
    Args:
        x1: First set of points, tensor of shape (n, d)
        x2: Second set of points, tensor of shape (m, d)
        length_scale: Kernel length scale parameter
        variance: Kernel variance parameter
        
    Returns:
        Kernel matrix of shape (n, m)
    """
    # Compute pairwise squared Euclidean distances
    sum_sq_x1 = torch.sum(x1**2, dim=1, keepdim=True)  # Shape (n, 1)
    sum_sq_x2 = torch.sum(x2**2, dim=1)                # Shape (m,)
    cross_term = torch.matmul(x1, x2.T)                # Shape (n, m)

    # sq_dist shape: (n, 1) + (m,) - 2 * (n, m) -> broadcasting -> (n, m)
    sq_dist = sum_sq_x1 + sum_sq_x2 - 2 * cross_term

    # Ensure squared distances are non-negative
    sq_dist = torch.clamp(sq_dist, min=0.0)

    # Compute the Squared Exponential kernel
    cov = variance * torch.exp(-0.5 * sq_dist / length_scale**2)
    return cov

def matern_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: float = 1.0,
    variance: float = 1.0,
    nu: float = 1.5
) -> torch.Tensor:
    """
    Matérn kernel for multi-dimensional inputs.
    
    Args:
        x1: First set of points, tensor of shape (n, d)
        x2: Second set of points, tensor of shape (m, d)
        length_scale: Kernel length scale parameter
        variance: Kernel variance parameter
        nu: Smoothness parameter (0.5, 1.5, or 2.5 supported)
        
    Returns:
        Kernel matrix of shape (n, m)
    """
    # Compute pairwise Euclidean distances
    sum_sq_x1 = torch.sum(x1**2, dim=1, keepdim=True)  # Shape (n, 1)
    sum_sq_x2 = torch.sum(x2**2, dim=1)                # Shape (m,)
    cross_term = torch.matmul(x1, x2.T)                # Shape (n, m)
    
    sq_dist = sum_sq_x1 + sum_sq_x2 - 2 * cross_term
    sq_dist = torch.clamp(sq_dist, min=0.0)
    dist = torch.sqrt(sq_dist)
    
    # Scaled distance
    scaled_dist = dist / length_scale
    
    # Calculate kernel based on nu value
    if nu == 0.5:  # Matérn 1/2
        K = variance * torch.exp(-scaled_dist)
    elif nu == 1.5:  # Matérn 3/2
        K = variance * (1 + math.sqrt(3) * scaled_dist) * torch.exp(-math.sqrt(3) * scaled_dist)
    elif nu == 2.5:  # Matérn 5/2
        K = variance * (1 + math.sqrt(5) * scaled_dist + 5 * scaled_dist**2 / 3) * torch.exp(-math.sqrt(5) * scaled_dist)
    else:
        raise ValueError(f"Matérn kernel with nu={nu} not supported. Use nu=0.5, 1.5, or 2.5")
    
    return K

def sample_gp_fast(
    x: torch.Tensor,
    mean_func: Callable[[torch.Tensor], torch.Tensor] = mean_func_zero,
    kernel_func: Callable = squared_exponential_kernel,
    num_samples: int = 1,
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise_variance: float = 0.1,
    kernel_params: Optional[dict] = None,
) -> torch.Tensor:
    """
    Efficiently sample from a Gaussian Process by pre-computing the Cholesky decomposition.
    
    Args:
        x: Input points tensor of shape (n, d)
        mean_func: Mean function, takes x and returns tensor of shape (n,)
        kernel_func: Kernel function
        num_samples: Number of samples to draw
        length_scale: Kernel length scale parameter
        variance: Kernel variance parameter
        noise_variance: Observation noise variance
        kernel_params: Additional kernel parameters (e.g., nu for Matérn)
        
    Returns:
        Tensor of samples with shape (n, num_samples) if num_samples > 1 
        or (n,) if num_samples == 1
    """
    n_points = x.shape[0]
    mean = mean_func(x)  # Shape (n_points,)
    
    # Handle additional kernel parameters if provided
    if kernel_params is None:
        kernel_params = {}
    
    # Compute covariance matrix
    K = kernel_func(x, x, length_scale=length_scale, variance=variance, **kernel_params)
    
    # Add noise to diagonal
    K_noisy = K + noise_variance * torch.eye(n_points, dtype=x.dtype, device=x.device)
    
    # Compute Cholesky decomposition with error handling
    try:
        L = torch.linalg.cholesky(K_noisy)
    except torch.linalg.LinAlgError:
        # Add jitter if Cholesky fails
        jitter = 1e-6 * torch.eye(n_points, dtype=x.dtype, device=x.device)
        try:
            L = torch.linalg.cholesky(K_noisy + jitter)
            print("Cholesky succeeded after adding jitter.")
        except torch.linalg.LinAlgError as e:
            print(f"Cholesky decomposition failed even with jitter: {e}")
            raise RuntimeError("Could not compute Cholesky decomposition.") from e
    
    # Draw standard normal samples
    Z = torch.randn(n_points, num_samples, dtype=x.dtype, device=x.device)
    
    # Compute samples
    samples = mean.unsqueeze(1) + L @ Z  # (n_points, num_samples)
    
    # If only 1 sample, flatten the result
    if num_samples == 1:
        samples = samples.squeeze(1)
    
    return samples

def sample_gp_matern(
    x: torch.Tensor,
    nu: float = 1.5,
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise_variance: float = 0.1,
    num_samples: int = 1,
    mean_func: Callable[[torch.Tensor], torch.Tensor] = mean_func_zero
) -> torch.Tensor:
    """
    Convenience function to sample from a Gaussian Process with a Matérn kernel.
    
    Args:
        x: Input points tensor of shape (n, d)
        nu: Smoothness parameter (0.5, 1.5, or 2.5)
        length_scale: Kernel length scale parameter
        variance: Kernel variance parameter
        noise_variance: Observation noise variance
        num_samples: Number of samples to draw
        mean_func: Mean function
        
    Returns:
        Tensor of samples with shape (n, num_samples) if num_samples > 1 
        or (n,) if num_samples == 1
    """
    # Map nu values to common names for clarity in errors
    nu_names = {0.5: "1/2", 1.5: "3/2", 2.5: "5/2"}
    if nu not in nu_names:
        raise ValueError(f"Matérn kernel with nu={nu} not supported. Use nu=0.5, 1.5, or 2.5")
    
    return sample_gp_fast(
        x=x,
        mean_func=mean_func,
        kernel_func=matern_kernel,
        num_samples=num_samples,
        length_scale=length_scale,
        variance=variance,
        noise_variance=noise_variance,
        kernel_params={"nu": nu}
    )

def test_gp_sampling():
    """
    Test the GP sampling functionality with different kernels.
    """
    # Set parameters
    n = 200  # Number of points
    d = 2    # Dimensionality
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Testing GP sampling with {n} points in {d} dimensions on {device}")
    
    # Generate input points
    x = torch.rand(n, d, dtype=dtype, device=device) * 2 - 1
    
    # Sample from squared exponential kernel
    print("\nTesting squared exponential kernel sampling...")
    start_time = time.time()
    samples_se = sample_gp_fast(
        x,
        length_scale=0.3,
        variance=1.0,
        noise_variance=0.05,
        num_samples=1
    )
    print(f"Generated SE sample of shape {samples_se.shape} in {time.time() - start_time:.4f} seconds")
    
    # Sample from Matérn kernel with different nu values
    for nu in [0.5, 1.5, 2.5]:
        print(f"\nTesting Matérn {nu} kernel sampling...")
        start_time = time.time()
        samples_matern = sample_gp_matern(
            x,
            nu=nu,
            length_scale=0.3,
            variance=1.0,
            noise_variance=0.05,
            num_samples=1
        )
        print(f"Generated Matérn {nu} sample of shape {samples_matern.shape} in {time.time() - start_time:.4f} seconds")
    
    # Multiple samples test
    print("\nTesting multiple samples...")
    start_time = time.time()
    multi_samples = sample_gp_matern(
        x,
        nu=1.5,
        length_scale=0.3,
        variance=1.0,
        noise_variance=0.05,
        num_samples=3
    )
    print(f"Generated {multi_samples.shape[1]} Matérn samples of shape {multi_samples.shape} in {time.time() - start_time:.4f} seconds")
    
    print("\nAll tests completed successfully!")
    return True

if __name__ == "__main__":
    test_gp_sampling() 