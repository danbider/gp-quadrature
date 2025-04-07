import math
import torch

from pydantic import BaseModel, Field

class SquaredExponential(BaseModel):
    """
    Squared-exponential kernel function and Fourier transform, general dim.
    """
    dimension: int = Field(..., ge=1)  # dimension, integer greater or equal to 1
    lengthscale: float = Field(..., gt=0)   # lengthscale, positive float
    variance: float = Field(1.0, gt=0) # variance, positive float

    class ConfigDict:
        arbitrary_types_allowed = True

    def kernel(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel of the Squared-exponential kernel at distance distance.
        Args:
            distance: distance, tensor of shape (n, d)
        Returns:
            kernel, tensor of shape (n,)
        """
        return self.variance * torch.exp(-0.5 * (distance/self.lengthscale)**2)

    def spectral_density(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the spectral density of the Squared-exponential kernel at frequency xid.
        Args:
            xid: frequency, tensor of shape (n, d)
        Returns:
            spectral density, tensor of shape (n,)
        """
        return self.variance * (2*math.pi*self.lengthscale**2)**(self.dimension/2) * torch.exp(-2*math.pi**2*self.lengthscale**2 * xid**2)

    def spectral_grad(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the spectral density S(xid) with respect to the hyperparameters
        θ = (lengthscale, variance). The gradients are given by:
        
            ∂S/∂σ² = S(xid) / σ²,
            ∂S/∂ℓ  = S(xid) * (dimension/ℓ - 4π²ℓ*xid²)
        
        Args:
            xid: frequency, tensor of shape (n,) or (n, d)
            
        Returns:
            A tensor of shape (n, 2) where each row contains the gradients 
            [∂S/∂lengthscale, ∂S/∂variance] evaluated at the corresponding frequency.
        """
        # Compute the spectral density S(xid)
        S = self.spectral_density(xid)
        
        # Gradient with respect to variance: ∂S/∂variance = S / variance
        dS_dvariance = S / self.variance
        
        # Gradient with respect to lengthscale:
        # ∂S/∂lengthscale = S * (dimension/lengthscale - 4π² * lengthscale * xid²)
        dS_dlengthscale = S * ((self.dimension / self.lengthscale) - 4 * math.pi**2 * self.lengthscale * xid**2)
        # dS_dsigma = torch.zeros_like(dS_dlengthscale)
        # Stack the gradients so that the last dimension contains [dS/dlengthscale, dS/dvariance]
        grad = torch.stack([dS_dlengthscale, dS_dvariance], dim=-1)
        return grad