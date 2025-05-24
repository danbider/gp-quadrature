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
    hypers: List[str] = Field(default=['lengthscale', 'variance', 'nu'], frozen=True)
    num_hypers: int = Field(default=4, frozen=True)  # includes noise variance
    
    # Initial hyperparameter values (used during initialization)
    init_lengthscale: float = Field(default=float('nan'), ge=1e-6)
    init_variance: float = Field(default=float('nan'), ge=1e-6)
    init_nu: float = Field(default=2.5, ge=0.1, le=10.0, description="Smoothness parameter")
    
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
    
    @property
    def nu(self) -> float:
        """Property that uses get_hyper to retrieve the nu parameter."""
        return self.get_hyper('nu')
    
    @nu.setter
    def nu(self, value: float) -> None:
        """Setter for nu property that uses set_hyper."""
        self.set_hyper('nu', value)
    
    def kernel(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Compute the Matérn kernel at the given distances.
        
        Args:
            distance: Tensor of distances of shape (n,) or (n, m)
            
        Returns:
            Kernel values of same shape as distance
        """
        # Get current parameter values from GPParams
        lengthscale = self.get_hyper('lengthscale')
        variance = self.get_hyper('variance')
        nu = self.get_hyper('nu')
        
        # Handle nu = 0.5 (exponential kernel)
        if abs(nu - 0.5) < 1e-6:
            scaled_dist = torch.sqrt(2.0) * distance / lengthscale
            K = variance * torch.exp(-scaled_dist)
            return K
            
        # Handle nu = 1.5
        elif abs(nu - 1.5) < 1e-6:
            scaled_dist = torch.sqrt(3.0) * distance / lengthscale
            K = variance * (1.0 + scaled_dist) * torch.exp(-scaled_dist)
            return K
            
        # Handle nu = 2.5
        elif abs(nu - 2.5) < 1e-6:
            scaled_dist = torch.sqrt(5.0) * distance / lengthscale
            K = variance * (1.0 + scaled_dist + scaled_dist**2/3.0) * torch.exp(-scaled_dist)
            return K
            
        # For other nu values, use a general implementation
        else:
            # For small distances, avoid numerical issues
            eps = 1e-8
            scaled_dist = torch.sqrt(2.0 * nu) * distance / lengthscale
            scaled_dist = torch.clamp(scaled_dist, min=eps)
            
            # We use the bessel_kv function for the modified Bessel function
            try:
                from scipy import special
                bessel = special.kv(nu, scaled_dist.cpu().numpy())
                bessel = torch.tensor(bessel, device=distance.device, dtype=distance.dtype)
            except ImportError:
                # Fallback to approximation for common nu values
                raise RuntimeError("scipy is required for Matérn kernel with custom nu values")
                
            # Compute prefactor terms
            gamma_term = torch.tensor(special.gamma(nu), device=distance.device, dtype=distance.dtype)
            prefactor = variance * 2**(1-nu) / gamma_term
            
            # Final kernel value
            K = prefactor * scaled_dist**nu * bessel
            
            # Handle numerical instabilities
            K = torch.nan_to_num(K, nan=variance, posinf=variance, neginf=0.0)
            
            return K
    
    def spectral_density(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the spectral density at the frequency xid.
        
        The spectral density of the Matérn kernel is:
        S(w) = variance * Γ(nu + d/2) / (Γ(nu) * π^(d/2)) * 
               (2*nu/lengthscale^2 + 4π^2*|w|^2)^(-(nu+d/2))
               
        Args:
            xid: Frequency tensor of shape (n, d) or (n,)
            
        Returns:
            Spectral density of shape (n,)
        """
        # Get parameters
        lengthscale = self.get_hyper('lengthscale')
        variance = self.get_hyper('variance')
        nu = self.get_hyper('nu')
        d = self.dimension
        
        # Ensure xid is 2D
        if xid.ndim == 1:
            xid = xid.unsqueeze(-1)
            
        # Compute squared norm along last dimension
        xid_norm_sq = torch.sum(xid**2, dim=-1)
        
        # Compute terms for spectral density
        from scipy import special
        gamma_nu = special.gamma(nu)
        gamma_nud2 = special.gamma(nu + d/2)
        pi_d2 = np.pi**(d/2)
        
        # Compute denominator term
        denom = (2 * nu) / lengthscale**2 + 4 * np.pi**2 * xid_norm_sq
        
        # Compute spectral density
        spec_density = variance * gamma_nud2 / (gamma_nu * pi_d2) * denom**(-(nu + d/2))
        
        return spec_density
    
    def spectral_grad(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the spectral density with respect to hyperparameters.
        
        Args:
            xid: Frequency tensor of shape (n, d) or (n,)
            
        Returns:
            Gradient tensor of shape (n, 3) - [∂S/∂l, ∂S/∂σ², ∂S/∂ν]
        """
        # Get parameters
        lengthscale = self.get_hyper('lengthscale')
        variance = self.get_hyper('variance')
        nu = self.get_hyper('nu')
        d = self.dimension
        
        # Ensure xid is 2D
        if xid.ndim == 1:
            xid = xid.unsqueeze(-1)
            
        # Compute squared norm along last dimension
        xid_norm_sq = torch.sum(xid**2, dim=-1)
        
        # Compute base spectral density
        s_w = self.spectral_density(xid)
        
        # Compute the denominator
        denom = (2 * nu) / lengthscale**2 + 4 * np.pi**2 * xid_norm_sq
        
        # Gradients
        # For lengthscale: ∂S/∂l = S(w) * (2*nu*(nu+d/2)) / (l^3 * denom)
        dl = s_w * (4 * nu * (nu + d/2)) / (lengthscale**3 * denom)
        
        # For variance: ∂S/∂σ² = S(w) / σ²
        dv = s_w / variance
        
        # For nu: This is complex due to Gamma function derivatives
        # Approximation: ∂S/∂ν ≈ S(w) * (log(denom) - ψ(ν+d/2) + ψ(ν))
        from scipy import special
        digamma_nu = special.digamma(nu)
        digamma_nud2 = special.digamma(nu + d/2)
        dnu = s_w * (np.log(denom) - digamma_nud2 + digamma_nu)
        
        return torch.stack([dl, dv, dnu], dim=-1)
    
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
        noise_var = 0.01 * y_var
        
        return lengthscale, variance, noise_var