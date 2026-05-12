from copy import deepcopy
from typing import List, Tuple, Union, Optional, Sequence
import torch
from kernels.matern import Matern
from kernels.squared_exponential import SquaredExponential

import math
class GetTruncationBound:
    def __init__(self, eps: float, kern: callable, initial_upper_bound: float = 1000.0, initial_lower_bound: float = 0.0, max_iterations: int = 200, dtype: torch.dtype = torch.float64):
        r"""
        Initialize the GetTruncationBound class with hyperparameters.
        We want to find $L$ such that $f(L) \approx \epsilon$, where $f$ is a monotonically decreasing function.

        Args:
        eps (float): Absolute value at which to truncate
        kern (callable): Function handle (e.g., kernel or its Fourier transform)
        initial_upper_bound (float): Initial upper bound for the search
        initial_lower_bound (float): Initial lower bound for the search
        max_iterations (int): Maximum number of iterations for bisection
        dtype (torch.dtype): Precision to use for computations (default: torch.float64)
        """
        self.eps = eps
        self.kern = kern
        self.initial_upper_bound = initial_upper_bound
        self.initial_lower_bound = initial_lower_bound
        self.max_iterations = max_iterations
        self.dtype = dtype
    
    def find_upper_bound_for_bisection(self) -> float:
        """
        Finds an upper bound for the truncation bound.

        Returns:
        float: Upper bound for the bisection
        """
        a = self.initial_lower_bound
        b = self.initial_upper_bound
        nmax = 10

        for _ in range(nmax):
            if self.kern(torch.tensor(b, device='cpu', dtype=self.dtype)) > self.eps: # function value at b is greater than eps
                b *= 2 # double the upper bound
            else:
                break # if function value at b is less than eps, we have found a valid upper bound
        return b

    def find_truncation_bound(self) -> float:
        """
        Finds where a monotonically decreasing function reaches a given value.

        Returns:
        float: Approximate value at which kern(L) = eps
        """
        # Make sure starting upper bound is large enough
        a = self.initial_lower_bound
        b = self.find_upper_bound_for_bisection()

        # Start bisection
        for _ in range(self.max_iterations):
            # compute the input midpoint
            mid = (a + b) / 2
            # compute the function value at the input midpoint
            fmid = self.kern(torch.tensor(mid, device='cpu', dtype=self.dtype))
            # if the function value at the input midpoint is greater than eps, we need to search the right half
            if fmid > self.eps:
                a = mid
            else:
                b = mid

        return mid
    

def get_xis(kernel_obj: Union[Matern, SquaredExponential], eps: float, L: int, use_integral: bool = False, l2scaled: bool = False, dtype: torch.dtype = torch.float64, trunc_eps: Optional[float] = None) -> Tuple[torch.Tensor, float, int]:
    """
    Return 1D equispaced Fourier quadrature nodes for given tolerance.

    Args:
    kernel_obj (Union[Matern, SquaredExponential]): Kernel class instance with kernel and spectral_density methods. TODO: add more kernels
    eps (float): Tolerance parameter, e.g. 1e-6
    L (float): Max size of spatial domain in any coordinate
    use_integral (bool): Whether to use integral method for truncation bound
    l2scaled (bool): Whether to use L2 scaling
    dtype (torch.dtype): Precision to use for computations (default: torch.float64)

    Returns:
    Tuple[torch.Tensor, float, int]: xis, h, mtot
    """
    dim = kernel_obj.dimension  # spatial dimension
    
    # spatial radial ker func (TODO: understand this)
    eps_use = eps
    if trunc_eps is None:
        trunc_eps = eps
    
    if use_integral:
        truncation_bound = GetTruncationBound(eps, kernel_obj.kernel, dtype=dtype)
        
        Ltime = truncation_bound.find_truncation_bound()  # find eps-support
        h_spacing = 1 / (L + Ltime)  # xi node spacing so nearest aliased tail <= eps
        
        # Fourier radial ker func
        khat_modified = lambda r: abs(r**(dim-1)) * kernel_obj.spectral_density(r) / kernel_obj.spectral_density(torch.tensor(0, device='cpu', dtype=dtype))  # polar factor & rel to 0
        truncation_bound_freq = GetTruncationBound(trunc_eps, khat_modified, dtype=dtype)
        Lfreq = truncation_bound_freq.find_truncation_bound()  # find eps-support
        
        hm = math.ceil(Lfreq / h_spacing)  # half number of nodes to cover [-Lfreq,Lfreq]
    else:
        if isinstance(kernel_obj, Matern): # heuristic for matern kernel
            l = kernel_obj.get_hyper('lengthscale')
            nu = kernel_obj.nu
            dim = kernel_obj.dimension
            eps_use = eps / kernel_obj.get_hyper('variance')
            if l2scaled:
                # L2 norm of the kernel k
                rl2sq = ((2*nu/math.pi/l**2)**(dim/2) * kernel_obj.spectral_density(torch.tensor(0, device='cpu', dtype=dtype))**2 / 2 *
                         math.gamma(dim/2+2*nu) / math.gamma(dim+2*nu) * 2**(-dim/2))
                eps_use = eps * math.sqrt(rl2sq)
            
            eps = eps_use
            h_spacing = 1 / (L + 0.85*l/math.sqrt(nu)*math.log(1/eps))  # heuristic \eqref{hheur}
            # hm is the half-number of nodes in the grid [-hm, hm]
            hm = math.ceil((math.pi**(nu+dim/2) * l**(2*nu) * eps/0.15)**(-1/(2*nu+dim/2)) / h_spacing)  # heuristic \eqref{mheur}
        
        elif isinstance(kernel_obj, SquaredExponential): # heuristic for se kernel
            l = kernel_obj.get_hyper('lengthscale')
            dim = kernel_obj.dimension
            var = kernel_obj.get_hyper('variance')
            eps_use = eps / var
            if l2scaled:
                rl2sq = kernel_obj.kernel(torch.tensor(0, device='cpu', dtype=dtype))**2 * (math.sqrt(math.pi)*l**2)**dim
                eps_use = eps * math.sqrt(rl2sq)
            
            eps = eps_use
            h_spacing = 1 / (L + l*math.sqrt(2*math.log(4*dim*3**dim/eps)))
            hm = math.ceil(math.sqrt(math.log(dim*(4**(dim+1))/eps)/2)/math.pi/l/h_spacing)  # again, the paper sometime uses "m"
    
    xis = torch.arange(-hm, hm+1, device='cpu', dtype=dtype) * h_spacing  # use exactly h, so can get bit of spillover

 


    mtot = xis.numel()  # 2m+1

    return xis, h_spacing, mtot


def get_xis_per_dim(
    kernel_obj,
    eps: float,
    L_per_dim: Sequence[float],
    lengthscales: Sequence[float],
    *,
    use_integral: bool = True,
    l2scaled: bool = False,
    dtype: torch.dtype = torch.float64,
    trunc_eps: Optional[float] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
    """Per-dimension wrapper around ``get_xis``.

    For each axis ``k``, build a 1D copy of the kernel with
    ``lengthscale = lengthscales[k]`` and call ``get_xis`` with
    ``L = L_per_dim[k]``. Returns parallel lists of 1D nodes, spacings,
    and node counts — one entry per spatial dimension.

    Parameters
    ----------
    kernel_obj : Kernel
        Reference kernel; only its type/variance are used. We deep-copy it
        and overwrite ``lengthscale`` for each axis.
    eps : float
        Tolerance, passed through to ``get_xis``.
    L_per_dim : sequence of float
        Spatial extent (``x.max - x.min``) per dimension.
    lengthscales : sequence of float
        Per-dimension lengthscale to use for sizing each axis.
    """
    if len(L_per_dim) != len(lengthscales):
        raise ValueError(
            f"L_per_dim and lengthscales must have same length, got "
            f"{len(L_per_dim)} and {len(lengthscales)}"
        )

    # Build a 1D-isotropic stand-in kernel for each axis. We use
    # SquaredExponential or Matern depending on the input type so that
    # get_xis routes to the right heuristic when use_integral=False.
    base_variance = float(kernel_obj.get_hyper('variance'))
    if isinstance(kernel_obj, Matern):
        proxy_factory = lambda l: Matern(
            dimension=1, nu=kernel_obj.nu,
            init_lengthscale=float(l), init_variance=base_variance,
        )
    else:
        # Default to SquaredExponential proxy. ARD-SE is treated as a product
        # of 1D SE factors per axis.
        proxy_factory = lambda l: SquaredExponential(
            dimension=1, init_lengthscale=float(l), init_variance=base_variance,
        )

    xis_1d_list: List[torch.Tensor] = []
    h_list: List[float] = []
    mtot_list: List[int] = []
    for k, (L_k, l_k) in enumerate(zip(L_per_dim, lengthscales)):
        proxy = proxy_factory(l_k)
        xis_k, h_k, mtot_k = get_xis(
            proxy, eps=eps, L=float(L_k),
            use_integral=use_integral, l2scaled=l2scaled,
            dtype=dtype, trunc_eps=trunc_eps,
        )
        xis_1d_list.append(xis_k)
        h_list.append(float(h_k))
        mtot_list.append(int(mtot_k))

    h_per_dim = torch.tensor(h_list, dtype=dtype)
    return xis_1d_list, h_per_dim, mtot_list