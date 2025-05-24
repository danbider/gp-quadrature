import math
import torch
import numpy as np
from typing import List, Tuple, Optional
from pydantic import Field

from .kernel import Kernel

class SquaredExponential(Kernel):
    """
    Squared Exponential (RBF) kernel.
    
    k(r) = variance * exp(-0.5 * r^2 / lengthscale^2)
    
    where r is the Euclidean distance between inputs.
    """
    
    # Define parameter names for this kernel
    hypers: List[str] = Field(default=['lengthscale', 'variance'], frozen=True)
    num_hypers: int = Field(default=3, frozen=True)  # includes noise variance
    
    # Initial hyperparameter values (used during initialization)
    init_lengthscale: float = Field(default=float('nan'), ge=1e-6)
    init_variance: float = Field(default=float('nan'), ge=1e-6)
    
    @property
    def lengthscale(self) -> float:
        """Property that uses get_hyper to retrieve the lengthscale."""
        return self.get_hyper('lengthscale')
    
    @lengthscale.setter
    def lengthscale(self, value: float) -> None:
        """Setter for lengthscale property that uses set_hyper."""
        self.set_hyper('lengthscale', value)
    
    @property
    def variance(self) -> float:
        """Property that uses get_hyper to retrieve the variance."""
        return self.get_hyper('variance')
    
    @variance.setter
    def variance(self, value: float) -> None:
        """Setter for variance property that uses set_hyper."""
        self.set_hyper('variance', value)
    
    def kernel(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Compute the SE kernel at the given distances.
        
        Args:
            distance: Tensor of distances of shape (n,) or (n, m)
            
        Returns:
            Kernel values of same shape as distance
        """
        # Get current parameter values from GPParams
        lengthscale = self.get_hyper('lengthscale')
        variance = self.get_hyper('variance')
        
        # Compute kernel
        scaled_dist = distance / lengthscale
        return variance * torch.exp(-0.5 * scaled_dist**2)
    
    def spectral_density(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the spectral density at the frequency xid.
        
        The spectral density of the SE kernel is:
        S(w) = (2π * lengthscale^2)^(d/2) * variance * exp(-2π^2 * lengthscale^2 * |w|^2)
        
        Args:
            xid: Frequency tensor of shape (n, d) or (n,)
            
        Returns:
            Spectral density of shape (n,)
        """
        # Get parameters
        lengthscale = self.get_hyper('lengthscale')
        variance = self.get_hyper('variance')
        
        # Ensure xid is 2D
        if xid.ndim == 1:
            xid = xid.unsqueeze(-1)
            
        # Compute squared norm along last dimension
        xid_norm_sq = torch.sum(xid**2, dim=-1)
        
        # Compute spectral density
        two_pi_sq = (2 * np.pi)**2
        prefactor = ((2 * np.pi) * lengthscale**2)**(self.dimension/2) * variance
        return prefactor * torch.exp(-two_pi_sq * lengthscale**2 * xid_norm_sq / 2)
    
    def spectral_grad(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the spectral density with respect to hyperparameters.
        
        Args:
            xid: Frequency tensor of shape (n, d) or (n,)
            
        Returns:
            Gradient tensor of shape (n, 2) - [∂S/∂l, ∂S/∂σ²]
        """
        # Get parameters
        lengthscale = self.get_hyper('lengthscale')
        variance = self.get_hyper('variance')
        
        # Ensure xid is 2D
        if xid.ndim == 1:
            xid = xid.unsqueeze(-1)
            
        # Compute squared norm along last dimension
        xid_norm_sq = torch.sum(xid**2, dim=-1)
        
        # Compute base spectral density
        two_pi_sq = (2 * np.pi)**2
        prefactor = ((2 * np.pi) * lengthscale**2)**(self.dimension/2) * variance
        s_w = prefactor * torch.exp(-two_pi_sq * lengthscale**2 * xid_norm_sq / 2)
        
        # Gradients
        dl = s_w * (self.dimension / lengthscale - two_pi_sq * lengthscale * xid_norm_sq)
        dv = s_w / variance
        
        return torch.stack([dl, dv], dim=-1)
    
    def log_marginal(self, x: torch.Tensor, y: torch.Tensor, sigmasq: float) -> float:
        """
        Compute the log marginal likelihood for the SE kernel.
        
        Args:
            x: Input tensor of shape (n, d) or (n,)
            y: Target tensor of shape (n,)
            sigmasq: Noise variance
            
        Returns:
            Log marginal likelihood value
        """
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.unsqueeze(-1)
            
        n = x.shape[0]
        
        # Compute kernel matrix
        K = self.kernel_matrix(x, x)
        
        # Add noise variance to diagonal
        K_noise = K + sigmasq * torch.eye(n, device=K.device)
        
        # Compute log marginal likelihood
        # log p(y|X) = -0.5 * (y^T K_noise^-1 y + log |K_noise| + n log(2π))
        try:
            L = torch.linalg.cholesky(K_noise)
            alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
            
            # Compute terms
            data_fit = 0.5 * torch.sum(y * alpha)
            complexity = torch.sum(torch.log(torch.diag(L)))
            constant = 0.5 * n * np.log(2 * np.pi)
            
            return -(data_fit + complexity + constant).item()
        except RuntimeError:
            # Fallback if Cholesky decomposition fails
            return float('-inf')
    
    def estimate_hyperparameters(self, x: torch.Tensor, y: torch.Tensor, K: int = 1000) -> Tuple[float, float, float]:
        """
        Estimate reasonable initial hyperparameters based on data characteristics.
        
        For the SE kernel:
        - lengthscale: Use median distance between randomly selected points
        - variance: Set to variance of target values
        - noise_var: Set to a small fraction of target variance
        
        Args:
            x: Input features tensor of shape (n, d)
            y: Target values tensor of shape (n,)
            K: Number of random points to sample for lengthscale estimation
            
        Returns:
            Tuple of (lengthscale, variance, noise_var)
        """
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.unsqueeze(-1)
            
        n, d = x.shape
        
        # Compute target variance for signal variance estimate
        y_var = torch.var(y).item()
        
        # Estimate lengthscale using median distance heuristic
        if n > K:
            # Randomly sample K points if dataset is large
            idx = torch.randperm(n)[:K]
            x_sample = x[idx]
        else:
            x_sample = x
            
        # Compute pairwise distances
        dists = torch.cdist(x_sample, x_sample)
        
        # Compute median of non-zero distances
        mask = dists > 0
        median_dist = torch.median(dists[mask]).item()

            
        # Set lengthscale to median distance
        lengthscale = 0.5*median_dist
        
        # Set variance to target variance
        variance = y_var
        
        # Set noise variance to small fraction of target variance
        noise_var = 0.01 * y_var
        
        return lengthscale, variance, noise_var
    
