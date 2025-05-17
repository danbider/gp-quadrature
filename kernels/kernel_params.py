import torch
import torch.nn as nn
from typing import List, Iterator, Tuple, Optional, Union
from .kernel import Kernel

# -----------------------------------------------------------------------------
# Parameter container that wraps any kernel and a noise variance
# -----------------------------------------------------------------------------
class GPParams(nn.Module):
    def __init__(self, kernel: Union[Kernel, object], init_sig2: Union[float, torch.Tensor]):
        """
        Container for kernel hyperparameters and noise variance.
        
        Parameters
        ----------
        kernel : Kernel or object with iter_hypers() and set_hyper() methods
            The kernel object whose parameters will be managed
        init_sig2 : float or torch.Tensor
            Initial noise variance
        """
        super().__init__()
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
        
        # raw = log of all positives, packed into one vector
        init_raw = torch.log(torch.tensor(init_vals, dtype=torch.get_default_dtype()))
        self.raw = nn.Parameter(init_raw)        # shape=(D+1,)
    
    @property
    def pos(self):
        """All positive parameters: kernel‐hypers followed by noise variance"""
        return self.raw.exp()                    # shape=(D+1,)
    
    @property
    def sig2(self):
        """Noise variance (last entry)"""
        return self.pos[-1]
    
    def sync_all_parameters(self):
        """
        Write current positive hypers back into self.kernel and return noise variance.
        This consolidates both kernel hyperparameter and noise variance synchronization.
        
        Returns:
            noise_variance: The current noise variance value
        """
        pos_vals = self.pos.detach().cpu().tolist()
        
        # Update kernel hyperparameters if any exist
        if hasattr(self.kernel, 'set_hyper') and self.hypers_names:
            for i, name in enumerate(self.hypers_names):
                # Convert to float to avoid potential PyTorch tensor issues
                self.kernel.set_hyper(name, float(pos_vals[i]))
            
        # Return noise variance as a detached tensor
        return self.sig2.detach().clone() 