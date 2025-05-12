import math
import torch
from typing import List
from pydantic import BaseModel, Field, field_validator

class Matern(BaseModel):
    """
    Matérn kernel with different smoothness hyperparameters, specified by the name.
    Currently supports Matérn 1/2, 3/2, and 5/2.
    """
    dimension: int = Field(..., ge=1)  # dimension, integer greater or equal to 1
    name: str = Field(..., pattern='^(matern12|matern32|matern52)$')
    lengthscale: float = Field(..., gt=0)   # lengthscale, positive float
    variance: float = Field(1.0, gt=0) # variance, positive float
    nu: float = Field(None)       # smoothness parameter, will be set in model_post_init
    num_hypers: int = Field(3, frozen=True)  # number of hyperparameters: lengthscale, variance, noise variance 
    hypers: List[str] = Field(default_factory=lambda: ['lengthscale', 'variance'], frozen=True)

    class ConfigDict:
        arbitrary_types_allowed = True

    def model_post_init(self, __context):
        """
        Set the smoothness parameter nu based on the name after initialization.
        """
        nu_values = {
            'matern12': 0.5,
            'matern32': 1.5,
            'matern52': 2.5
        }
        self.nu = nu_values[self.name]

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

        if self.name == 'matern12':
            return self.variance * torch.exp(-scaled_dist)
        elif self.name == 'matern32':
            return self.variance * (1 + math.sqrt(3) * scaled_dist) * torch.exp(-math.sqrt(3) * scaled_dist)
        elif self.name == 'matern52':
            return self.variance * (1 + math.sqrt(5) * scaled_dist + 5 * scaled_dist**2 / 3) * torch.exp(-math.sqrt(5) * scaled_dist)

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

    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix K(x, y) for the Matern kernel.
        
        Args:
            x: first input tensor of shape (n, d) or (n,)
            y: second input tensor of shape (m, d) or (m,)
            
        Returns:
            kernel matrix K(x, y), tensor of shape (n, m)
        """
        # Ensure inputs are 2D
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
            
        # Compute pairwise distances between points
        dist = torch.cdist(x, y)
        return self.kernel(dist)
    
    def log_marginal(self, x: torch.Tensor, y: torch.Tensor, sigmasq: float) -> float:
        """
        Compute the log marginal likelihood of the Matern kernel.
        
        Args:
            x: input tensor of shape (n, d) or (n,)
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
    
    def get_hyper(self, name: str) -> float:
        """
        Get hyperparameter value by name.
        """
        return getattr(self, name)
    
    def set_hyper(self, name: str, value: float) -> None:
        """
        Set hyperparameter value by name.
        """
        setattr(self, name, value)
        
    def iter_hypers(self):
        """
        Iterate through hyperparameters and their values.
        
        Returns:
            Iterator of (name, value) tuples
        """
        for name in self.hypers:
            yield name, getattr(self, name)