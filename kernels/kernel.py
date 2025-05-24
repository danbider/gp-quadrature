import math
import torch
from typing import List, Iterator, Tuple, Optional, Any
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
    
    Hyperparameter handling:
    - Define the list of your kernel's hyperparameter names in the 'hypers' field
    - Define initial values using init_* fields (e.g., init_lengthscale, init_variance)
    - Access parameters through get_hyper() and set_hyper() methods
    
    Example:
    ```python
    class CustomKernel(Kernel):
        # Define hyperparameter names
        hypers: List[str] = Field(default=['param1', 'param2'], frozen=True)
        num_hypers: int = Field(default=3, frozen=True)  # includes noise variance
        
        # Define initial values
        init_param1: float = Field(default=1.0)
        init_param2: float = Field(default=0.5)
        
        # Add properties for backward compatibility
        @property
        def param1(self) -> float:
            return self.get_hyper('param1')
        
        @param1.setter
        def param1(self, value: float) -> None:
            self.set_hyper('param1', value)
        
        @property
        def param2(self) -> float:
            return self.get_hyper('param2')
        
        @param2.setter
        def param2(self, value: float) -> None:
            self.set_hyper('param2', value)
        
        def kernel(self, distance):
            param1 = self.get_hyper('param1')  # or self.param1
            param2 = self.get_hyper('param2')  # or self.param2
            # Implement kernel logic using param1 and param2
    ```
    """
    dimension: int = Field(..., ge=1)  # dimension, integer greater or equal to 1
    hypers: List[str] = Field(..., description="Names of hyperparameters for this kernel")
    num_hypers: int = Field(..., description="Number of hyperparameters (including noise variance)")
    _gp_params_ref: Optional[object] = None  # Reference to GPParams for direct parameter access
    
    class ConfigDict:
        arbitrary_types_allowed = True
        populate_by_name = True  # Allows field initialization using either alias or field name
    
    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook called after model is fully initialized.
        Always creates a GPParams instance for this kernel.
        
        This method is automatically called by Pydantic after all fields are initialized.
        """
        if self._gp_params_ref is None:
            # Lazily import to avoid circular imports
            from .kernel_params import GPParams
            
            # Get initialization values from init_* fields
            obj_dict = vars(self)
            
            # Extract initial parameter values
            param_values = {}
            for name in self.hypers:
                init_field = f"init_{name}"
                if init_field in obj_dict:
                    param_values[name] = float(obj_dict[init_field])
                else:
                    # Hyperparameter declared but no corresponding init_* field
                    raise ValueError(f"Hyperparameter '{name}' declared in hypers list but no corresponding '{init_field}' field found")
            
            # Store params in internal dictionary - this is used by iter_hypers
            # when creating the GPParams instance
            self._params_dict = param_values.copy()
            
            # Create GPParams with these values (default noise variance of 0.1)
            GPParams(kernel=self, init_sig2=0.1)
    
    def get_hyper(self, name: str) -> float:
        """
        Get hyperparameter value by name from GPParams reference.
        
        After initialization, parameters are always accessed from GPParams.
        """
        # We should always have _gp_params_ref after initialization
        if self._gp_params_ref is None:
            # If we somehow don't have GPParams (should never happen in normal use),
            # use _params_dict as fallback
            if hasattr(self, '_params_dict') and name in self._params_dict:
                return self._params_dict[name]
            else:
                raise RuntimeError(f"No GPParams reference available and unknown parameter: {name}")
        
        # Normal path: get from GPParams
        if name in self._gp_params_ref.hypers_names:
            idx = self._gp_params_ref.hypers_names.index(name)
            return float(self._gp_params_ref.pos[idx].item())
        
        # Parameter not found in GPParams
        raise ValueError(f"Unknown hyperparameter: {name}")
    
    def set_hyper(self, name: str, value: float) -> None:
        """
        Set hyperparameter value by name in GPParams reference.
        """
        if name not in self.hypers:
            raise ValueError(f"Unknown hyperparameter: {name}")
        
        # Update _params_dict for completeness
        if not hasattr(self, '_params_dict'):
            self._params_dict = {}
        self._params_dict[name] = float(value)
        
        # Update GPParams (the main parameter storage)
        if self._gp_params_ref is not None:
            if name in self._gp_params_ref.hypers_names:
                idx = self._gp_params_ref.hypers_names.index(name)
                # Use nn.Parameter.data to avoid in-place operation errors
                new_val = torch.log(torch.tensor(float(value)))
                with torch.no_grad():
                    self._gp_params_ref.raw.data[idx] = new_val
    
    def iter_hypers(self) -> Iterator[Tuple[str, float]]:
        """
        Iterate through hyperparameters and their values.
        Used primarily by GPParams during initialization.
        """
        # Ensure _params_dict exists
        if not hasattr(self, '_params_dict'):
            self._params_dict = {name: 1.0 for name in self.hypers}
        
        # Return parameters from _params_dict
        # After initialization, these should match GPParams values
        for name in self.hypers:
            yield name, self._params_dict[name]
    
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