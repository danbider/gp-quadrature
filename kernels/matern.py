import math
import torch
import numpy as np
from typing import List, Tuple, Optional
from pydantic import Field, field_validator

from .kernel import Kernel

class Matern(Kernel):
    """
    Matérn kernel.
    
    k(r) = variance * 2^(1-nu) / Gamma(nu) * (sqrt(2*nu) * r / lengthscale)^nu * K_nu(sqrt(2*nu) * r / lengthscale)
    
    where:
    - r is the Euclidean distance between inputs
    - K_nu is the modified Bessel function of the second kind
    - nu controls the smoothness
    
    When nu = 1/2, the Matérn kernel becomes the exponential kernel.
    When nu = infinity, it becomes the squared exponential kernel.
    Common values are nu = 1/2, 3/2, 5/2.
    """
    # Define parameter names for this kernel
    hypers: List[str] = Field(default=['lengthscale', 'variance'], frozen=True)
    num_hypers: int = Field(default=3, frozen=True)  # includes noise variance
    
    # Initial hyperparameter values (used during initialization)
    init_lengthscale: float = Field(default=float('nan'), ge=1e-6)
    init_variance: float = Field(default=float('nan'), ge=1e-6)
    nu: float = Field(default=2.5, ge=0.1, le=10.0, description="Smoothness parameter")
    
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
        Compute the kernel of the Matern kernel at distance distance. The formula depends on the nu parameter. 
        Args:
            distance: distance, tensor of shape (n,) or (n, m)
        Returns:
            kernel, tensor of shape (n,) or (n, m)
        """
        # Handle scalar or array inputs
        scaled_dist = torch.abs(distance) / self.lengthscale

        if self.nu == 0.5:
            return self.variance * torch.exp(-scaled_dist)
        elif self.nu == 1.5:
            return self.variance * (1 + math.sqrt(3) * scaled_dist) * torch.exp(-math.sqrt(3) * scaled_dist)
        elif self.nu == 2.5:
            return self.variance * (1 + math.sqrt(5) * scaled_dist + 5 * scaled_dist**2 / 3) * torch.exp(-math.sqrt(5) * scaled_dist)

            
        # For other nu values, use a general implementation
        else:
            # For small distances, avoid numerical issues
            eps = 1e-8
            scaled_dist = torch.sqrt(2.0 * self.nu) * distance / lengthscale
            scaled_dist = torch.clamp(scaled_dist, min=eps)
            
            # We use the bessel_kv function for the modified Bessel function
            try:
                from scipy import special
                bessel = special.kv(self.nu, scaled_dist.cpu().numpy())
                bessel = torch.tensor(bessel, device=distance.device, dtype=distance.dtype)
            except ImportError:
                # Fallback to approximation for common nu values
                raise RuntimeError("scipy is required for Matérn kernel with custom nu values")
                
            # Compute prefactor terms
            gamma_term = torch.tensor(special.gamma(self.nu), device=distance.device, dtype=distance.dtype)
            prefactor = variance * 2**(1-self.nu) / gamma_term
            
            # Final kernel value
            K = prefactor * scaled_dist**self.nu * bessel
            
            # Handle numerical instabilities
            K = torch.nan_to_num(K, nan=variance, posinf=variance, neginf=0.0)
            
            return K
    
    def spectral_density(self, xid: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the spectral density of the Matern kernel at frequency xid.
        Mathematical formula (for d>=1):

        .. math::

            \hat{k}(xid) = \frac{\sigma^2}{l^{2\nu} (2\sqrt{\pi})^{d} \Gamma(\nu) \Gamma(\nu + d/2)} (2\nu/l^2 + 4\pi^2 xid^2)^{-\nu - d/2}

        Args:
            xid: frequency, tensor of shape (n,) or (n, d)
        Returns:
            spectral density, tensor of shape (n,)
        """
        # Ensure xid is 2D for consistent handling
        if xid.ndim == 1:
            xid = xid.unsqueeze(-1)

        # Calculate the squared norm of frequencies
        xid_squared_sum = torch.sum(xid**2, dim=-1)

        scaling = ((2 * math.sqrt(math.pi))**self.dimension * math.gamma(self.nu + self.dimension/2) * (2*self.nu)**self.nu / 
                   (math.gamma(self.nu) * self.lengthscale**(2*self.nu)))
        return self.variance * scaling * (2*self.nu/self.lengthscale**2 + (4*math.pi**2) * xid_squared_sum)**(-(self.nu + self.dimension/2))

    def spectral_grad(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the spectral density with respect to the hyperparameters
        θ = (lengthscale, variance).
        
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
        
        # Gradient with respect to variance: ∂S/∂variance = S / variance
        dS_dvariance = S / self.variance
        
        # Compute sum of squared frequencies across dimensions
        xid_squared_sum = torch.sum(xid**2, dim=-1)
        
        # For the Matern kernel, the spectral density is proportional to:
        # S(xid) ∝ (2*nu/l^2 + 4*pi^2*|xid|^2)^(-(nu+d/2))
        # and includes a scaling factor with l^(-2*nu)
        
        # Define the denominator term
        denominator = 2*self.nu/self.lengthscale**2 + (4*math.pi**2) * xid_squared_sum
        
        # The gradient has two components from the chain rule:
        # 1. From l^(-2*nu): -2*nu/l * S
        # 2. From (...)^(-(nu+d/2)): (-(nu+d/2)) * (-4*nu/l^3) / denominator * S
        
        # Combined gradient with respect to lengthscale
        power = -(self.nu + self.dimension/2)
        exponent_grad = power * (-4*self.nu/self.lengthscale**3) / denominator
        dS_dlengthscale = S * (-2*self.nu/self.lengthscale + exponent_grad)
        
        # Stack the gradients
        grad = torch.stack([dS_dlengthscale, dS_dvariance], dim=-1)
        return grad
    
    def log_marginal(self, x: torch.Tensor, y: torch.Tensor, sigmasq: float) -> float:
        """
        Compute the log marginal likelihood for the Matérn kernel.
        
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
        
        For the Matérn kernel:
        - lengthscale: Use median distance between randomly selected points
        - variance: Set to variance of target values
        - noise_var: Set to a small fraction of target variance
        - nu is kept at initialization value
        
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
        if mask.sum() > 0:
            median_dist = torch.median(dists[mask]).item()
        else:
            # Fallback if all distances are zero
            median_dist = 1.0
            
        # Set lengthscale to median distance
        lengthscale = median_dist
        
        # Set variance to target variance
        variance = y_var
        
        # Set noise variance to small fraction of target variance
        noise_var = 0.2 * y_var
        
        return lengthscale, variance, noise_var