import math
import torch
from typing import List, Iterator, Tuple
from pydantic import BaseModel, Field

class Kernel(BaseModel):
    """
    Base class for all kernel functions.
    
    Defines the common interface and functionality that all kernels should implement.
    
    Subclasses must implement:
    - kernel(distance): Compute kernel values at given distances
    - spectral_density(xid): Compute spectral density at given frequencies
    - spectral_grad(xid): Compute gradient of spectral density w.r.t. hyperparameters
    - log_marginal(x, y, sigmasq): Compute log marginal likelihood
    """
    dimension: int = Field(..., ge=1)  # dimension, integer greater or equal to 1
    lengthscale: float = Field(..., gt=0)  # lengthscale, positive float
    variance: float = Field(1.0, gt=0)  # variance, positive float
    hypers: List[str] = Field(default_factory=lambda: ['lengthscale', 'variance'], frozen=True)
    num_hypers: int = Field(3, frozen=True)  # number of hyperparameters (including noise variance)
    
    class ConfigDict:
        arbitrary_types_allowed = True
    
    def kernel(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value at given distances.
        
        Args:
            distance: distance tensor of shape (n,) or (n, m)
        Returns:
            kernel values tensor of shape (n,) or (n, m)
        """
        raise NotImplementedError("Subclasses must implement kernel()")
    
    def spectral_density(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the spectral density of the kernel at frequency xid.
        
        Args:
            xid: frequency tensor of shape (n,) or (n, d)
        Returns:
            spectral density tensor of shape (n,)
        """
        raise NotImplementedError("Subclasses must implement spectral_density()")
    
    def spectral_grad(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the spectral density with respect to hyperparameters.
        
        Args:
            xid: frequency tensor of shape (n,) or (n, d)
        Returns:
            gradient tensor of shape (n, num_params)
        """
        raise NotImplementedError("Subclasses must implement spectral_grad()")
    
    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix K(x, y).
        
        Args:
            x: first input tensor of shape (n, d) or (n,)
            y: second input tensor of shape (m, d) or (m,)
        Returns:
            kernel matrix of shape (n, m)
        """
        # Ensure inputs are 2D
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
            
        # Compute pairwise distances
        dist = torch.cdist(x, y)
        return self.kernel(dist)
    
    def log_marginal(self, x: torch.Tensor, y: torch.Tensor, sigmasq: float) -> float:
        """
        Compute the log marginal likelihood.
        
        Subclasses should implement this method for their specific kernel.
        
        Args:
            x: input tensor of shape (n, d) or (n,)
            y: output tensor of shape (n,)
            sigmasq: noise variance
        Returns:
            log marginal likelihood value
        """
        raise NotImplementedError("Subclasses must implement log_marginal()")
    
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
    
    def iter_hypers(self) -> Iterator[Tuple[str, float]]:
        """
        Iterate through hyperparameters and their values.
        
        Returns:
            Iterator of (name, value) tuples
        """
        for name in self.hypers:
            yield name, getattr(self, name)
    
    def model_copy(self) -> 'Kernel':
        """
        Create a copy of the kernel with the same parameters.
        
        Returns:
            A new kernel instance with the same parameters
        """
        return self.__class__(**self.model_dump()) 
        
    def estimate_hyperparameters(self, x: torch.Tensor, y: torch.Tensor, K: int = 1000) -> Tuple[float, float, float]:
        """
        Estimate initial hyperparameters for GP model based on data characteristics.
        
        This is a base implementation that should be overridden by specific kernel classes
        to provide kernel-specific hyperparameter estimation strategies.
        
        Args:
            x: Input features tensor of shape (n, d)
            y: Target values tensor of shape (n,)
            K: Sample size for estimation (default: 1000)
            
        Returns:
            Tuple containing:
                - lengthscale: Estimated length scale
                - variance: Estimated signal variance
                - noise_var: Estimated noise variance
        """
        raise NotImplementedError("Subclasses should implement their own hyperparameter estimation strategy") 