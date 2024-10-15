import math
import torch

from pydantic import BaseModel, Field, field_validator
import math
import torch

from pydantic import BaseModel, Field

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
        TODO: check 
        Compute the kernel of the Matern kernel at distance distance. The formula depends on the nu parameter. 
        Args:
            distance: distance, tensor of shape (n, d)
        Returns:
            kernel, tensor of shape (n,)
        """
        if self.name == 'matern12':
            return self.variance * torch.exp(-torch.abs(distance) / self.lengthscale)
        elif self.name == 'matern32':
            return self.variance * (1 + math.sqrt(3) * torch.abs(distance) / self.lengthscale) * torch.exp(-math.sqrt(3) * torch.abs(distance) / self.lengthscale)
        elif self.name == 'matern52':
            return self.variance * (1 + math.sqrt(5) * torch.abs(distance) / self.lengthscale + 5 * torch.abs(distance)**2 / (3 * self.lengthscale**2)) * torch.exp(-math.sqrt(5) * torch.abs(distance) / self.lengthscale)

    def spectral_density(self, xid: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the spectral density of the Matern kernel at frequency xid.
        Mathematical formula (for d=1):

        .. math::

            \hat{k}(xid) = \frac{\sigma^2}{l^{2\nu} (2\sqrt{\pi})^{d} \Gamma(\nu) \Gamma(\nu + d/2)} (2\nu/l^2 + 4\pi^2 xid^2)^{-\nu - d/2}

        Args:
            xid: frequency, tensor of shape (n, d)
        Returns:
            spectral density, tensor of shape (n,)
        """
        scaling = ((2 * math.sqrt(math.pi))**self.dimension * math.gamma(self.nu + self.dimension/2) * (2*self.nu)**self.nu / 
                   (math.gamma(self.nu) * self.lengthscale**(2*self.nu)))
        return self.variance * scaling * (2*self.nu/self.lengthscale**2 + (4*math.pi**2) * xid**2)**(-(self.nu + self.dimension/2))
