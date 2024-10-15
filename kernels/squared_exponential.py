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
