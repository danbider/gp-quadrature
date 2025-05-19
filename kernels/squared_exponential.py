import math
import torch
from typing import List, Tuple
from pydantic import Field

from .kernel import Kernel

class SquaredExponential(Kernel):
    """
    Squared-exponential kernel function and Fourier transform, general dim.
    """
    dimension: int = Field(..., ge=1)  # dimension, integer greater or equal to 1
    lengthscale: float = Field(..., gt=0)   # lengthscale, positive float
    variance: float = Field(1.0, gt=0) # variance, positive float
    num_hypers: int = Field(3, frozen=True)  # number of hyperparameters
    hypers: List[str] = Field(default_factory=lambda: ['lengthscale', 'variance'], frozen=True)

    def kernel(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel of the Squared-exponential kernel at distance distance.
        Args:
            distance: distance, tensor of shape (n, d)
        Returns:
            kernel, tensor of shape (n,)
        """
        ls = self.get_hyper('lengthscale')
        var = self.get_hyper('variance')
        return var * torch.exp(-0.5 * (distance/ls)**2)

    def spectral_density(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the spectral density of the Squared-exponential kernel at frequency xid.
        Args:
            xid: frequency, tensor of shape (n, d)
        Returns:
            spectral density, tensor of shape (n,)
        """
        # Ensure xid is 2D
        if xid.ndim == 1:
            xid = xid.unsqueeze(-1)
        ls = self.get_hyper('lengthscale')
        var = self.get_hyper('variance')
        return var * (2*math.pi*ls**2)**(self.dimension/2) * torch.exp(-2*math.pi**2*ls**2 * torch.sum(xid**2, dim=-1))

    def spectral_grad(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the spectral density S(xid) with respect to the hyperparameters
        θ = (lengthscale, variance). The gradients are given by:
        
            ∂S/∂σ² = S(xid) / σ²,
            ∂S/∂ℓ  = S(xid) * (dimension/ℓ - 4π²ℓ*Σxid_i²)
        
        Args:
            xid: frequency, tensor of shape (n,) or (n, d)
            
        Returns:
            A tensor of shape (n, 2) where each row contains the gradients 
            [∂S/∂lengthscale, ∂S/∂variance] evaluated at the corresponding frequency.
        """
        # Ensure xid is 2D
        if xid.ndim == 1:
            xid = xid.unsqueeze(-1)
            
        # Compute the spectral density S(xid)
        S = self.spectral_density(xid)
        
        ls = self.get_hyper('lengthscale')
        var = self.get_hyper('variance')
        
        # Gradient with respect to variance: ∂S/∂variance = S / variance
        dS_dvariance = S / var
        
        # Compute sum of squared frequencies across dimensions
        xid_squared_sum = torch.sum(xid**2, dim=-1)
        
        # Gradient with respect to lengthscale:
        # ∂S/∂lengthscale = S * (dimension/lengthscale - 4π² * lengthscale * Σxid_i²)
        dS_dlengthscale = S * ((self.dimension / ls) - 4 * math.pi**2 * ls * xid_squared_sum)
        
        # Stack the gradients so that the last dimension contains [dS/dlengthscale, dS/dvariance]
        grad = torch.stack([dS_dlengthscale, dS_dvariance], dim=-1) # in the same order as hypers 
        return grad
        
    def log_marginal(self, x: torch.Tensor, y: torch.Tensor, sigmasq: float) -> float:
        """
        Compute the log marginal likelihood of the Squared-exponential kernel.
        Args:
            x: input tensor of shape (n, d)
            y: output tensor of shape (n,)
            sigmasq: noise variance
        Returns:
            log marginal likelihood, float
        """
        # Ensure x is 2D
        if x.ndim == 1 and self.dimension > 1:
            x = x.unsqueeze(-1)
            
        # Compute kernel matrix with noise
        K = self.kernel_matrix(x, x) + sigmasq * torch.eye(x.shape[0], device=x.device)
        
        # Compute Cholesky decomposition
        L = torch.linalg.cholesky(K)
        
        # Solve system for alpha
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y))
        
        # Compute log determinant term
        logdet = 2 * torch.sum(torch.log(torch.diagonal(L)))
        
        # Return log marginal likelihood
        return -0.5 * (torch.dot(y, alpha) + logdet + x.shape[0] * math.log(2 * math.pi))
    
    def estimate_hyperparameters(self, x: torch.Tensor, y: torch.Tensor, K: int = 1000) -> Tuple[float, float, float]:
        """
        Estimate initial hyperparameters for SquaredExponential kernel based on data characteristics.
        
        Args:
            x: Input features tensor of shape (n, d)
            y: Target values tensor of shape (n,)
            K: Sample size for estimation (default: 1000)
        
        Returns:
            Tuple containing:
                - lengthscale: Estimated length scale
                - variance: Estimated signal variance
                - noise_var: Estimated noise variance (10% of y variance)
        """
        # Get a random sample of x of size K
        if x.shape[0] > K:
            random_indices = torch.randperm(x.shape[0])[:K]
            x_sample = x[random_indices]
            y_sample = y[random_indices]
        else:
            x_sample = x
            y_sample = y
        
        # Calculate pairwise distances between all points in x_sample
        pairwise_distances = torch.cdist(x_sample, x_sample)

        # Set diagonal elements to infinity to exclude self-distances
        mask = torch.eye(x_sample.shape[0], device=x_sample.device).bool()
        pairwise_distances[mask] = float('inf')

        # Calculate the minimum distance for each point
        min_distances = torch.min(pairwise_distances, dim=1).values

        # Calculate the median of these minimum distances
        median_distance = torch.median(min_distances)

        # Estimate lengthscale as half the median distance between nearest neighbors
        lengthscale = 10*median_distance.item()

        # Calculate the variance of y_sample
        y_var = torch.var(y_sample).item()

        # Estimate signal variance as 90% of y variance
        signal_var = 0.9 * y_var
        
        # Estimate noise variance as 10% of y variance
        noise_var = 0.1 * y_var
        
        return lengthscale, signal_var, noise_var
    
