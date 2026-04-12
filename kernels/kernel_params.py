import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Iterator, Tuple, Optional, Union
from .kernel import Kernel


def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: log(exp(x) - 1). Numerically stable for large x."""
    return x + torch.log(-torch.expm1(-x))


# -----------------------------------------------------------------------------
# Parameter container that wraps any kernel and a noise variance
# -----------------------------------------------------------------------------
class GPParams(nn.Module): # inherits from nn.Module
    def __init__(self, kernel: Union[Kernel, object], init_sig2: Union[float, torch.Tensor],
                 transform: str = "softplus"):
        """
        Container for kernel hyperparameters and noise variance.

        Parameters
        ----------
        kernel : Kernel or object with iter_hypers() and set_hyper() methods
            The kernel object whose parameters will be managed
        init_sig2 : float or torch.Tensor
            Initial noise variance
        transform : str
            Bijection from raw to positive space. "softplus" (default) or "exp".
        """
        super().__init__()
        if transform not in ("softplus", "exp"):
            raise ValueError(f"transform must be 'softplus' or 'exp', got '{transform}'")
        self.transform = transform
        self.kernel = kernel

        # Check if kernel has iter_hypers method, ie is it a kernel object or a string
        if hasattr(kernel, 'iter_hypers'):
            # collect kernel hyper‐names & initial values
            hypers = list(kernel.iter_hypers())
            self.hypers_names = [name for name, _ in hypers]
            init_vals = [float(val) for _, val in hypers]
        else:
            # Handle case where kernel is a string or doesn't have iter_hypers
            self.hypers_names = []
            init_vals = []

        # Add noise variance
        init_vals.append(float(init_sig2))

        # raw = inverse-transform of all positives, packed into one vector
        pos_tensor = torch.tensor(init_vals, dtype=torch.get_default_dtype())
        if transform == "exp":
            init_raw = torch.log(pos_tensor)
        else:
            init_raw = _softplus_inv(pos_tensor)
        self.raw = nn.Parameter(init_raw)        # shape=(num_hypers+1,)

        # Establish a reference to this GPParams instance on the kernel if supported
        # This enables direct parameter access in kernel methods like spectral_density
        if hasattr(kernel, '_gp_params_ref'):
            kernel._gp_params_ref = self

    @property
    def pos(self):
        """All positive parameters: kernel‐hypers followed by noise variance"""
        if self.transform == "exp":
            return self.raw.exp()
        return F.softplus(self.raw)              # shape=(num_hypers+1,)

    def raw_jacobian(self) -> torch.Tensor:
        """d(pos)/d(raw) for the current raw values — used for chain-rule gradients."""
        if self.transform == "exp":
            return self.raw.exp()
        return torch.sigmoid(self.raw)
    
    @property
    def sig2(self):
        """Noise variance (last entry)"""
        return self.pos[-1]
    
    # def sync_all_parameters(self):
    #     """
    #     Write current positive hypers back into self.kernel and return noise variance.
    #     Both kernel hyperparameter and noise variance synchronization.
        
    #     Returns:
    #         noise_variance: The current noise variance value
    #     """
    #     pos_vals = self.pos.detach().cpu().tolist()
        
    #     # Update kernel hyperparameters if any exist
    #     if hasattr(self.kernel, 'set_hyper') and self.hypers_names:
    #         for i, name in enumerate(self.hypers_names):
    #             # Convert to float to avoid potential PyTorch tensor issues
    #             self.kernel.set_hyper(name, float(pos_vals[i]))
            
    #     # Return noise variance as a detached tensor
    #     return self.sig2.detach().clone() 