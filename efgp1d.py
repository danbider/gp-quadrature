import torch
import math
from typing import Callable, Tuple, Dict, Optional
from cg import ConjugateGradients
from utils.kernels import get_xis
from utils.kernels import GetTruncationBound

def EFGP(x: torch.Tensor, meas: torch.Tensor, sigmasq: float, ker: Dict[str, Callable],
         xtrg: Optional[torch.Tensor] = None, opts: Optional[Dict] = None) -> Tuple[Dict, Dict, Dict]:
    """
    Perform GP regression via equispaced Fourier iterative method in 1D.

    Args:
    x (torch.Tensor): Points where observations are taken, shape (N,)
    meas (torch.Tensor): Observations at the data points, shape (N,)
    sigmasq (float): Noise variance at data points
    ker (Dict[str, Callable]): Covariance kernel info with at least 'khat' field
    xtrg (torch.Tensor, optional): Target points for prediction, shape (n,)
    opts (Dict, optional): Options controlling method params

    Returns:
    Tuple[Dict, Dict, Dict]: y (results at data points), ytrg (results at targets), info (diagnostics)
    """
    if opts is None:
        opts = {}
    
    tol = opts.get('tol', 1e-6)
    
    # Determine problem dimension and sizes
    N = x.shape[0]
    do_trg = xtrg is not None
    n = xtrg.shape[0] if do_trg else 0
    
    # Combine x and xtrg for timing purposes
    xsol = torch.cat([x, xtrg]) if do_trg else x
    
    # Call the core 1D EFGP function
    beta, xis, yhat, iter_info, cpu_time, ws = efgp1d(x, meas, sigmasq, ker, tol, xsol, opts)
    
    # Extract results
    if 'only_trgs' in opts:
        y = {'mean': torch.tensor([])}
        ytrg = {'mean': yhat.mean}
    else:
        y = {'mean': yhat.mean[:N]}
        ytrg = {'mean': yhat.mean[N:]}
    
    # Prepare info dict
    info = {
        'beta': beta,
        'xis': xis,
        'h': xis[1] - xis[0],
        'ximax': torch.max(xis),
        'iter': iter_info,
        'cpu_time': {
            'total': cpu_time[3],
            'precomp': cpu_time[0],
            'cg': cpu_time[1],
            'mean': cpu_time[2]
        }
    }
    
    return y, ytrg, info

def efgp1d(inputs: torch.Tensor, y: torch.Tensor, sigmasq: float, ker: Dict[str, Callable],
           eps: float, xtrgs: torch.Tensor, opts: Optional[Dict] = None) -> Tuple:
    """
    Fast equispaced Fourier NUFFT-based GP regression in 1D.

    Args:
    inputs (torch.Tensor): Location of observations, shape (N,)
    y (torch.Tensor): Observations, shape (N,)
    sigmasq (float): Residual variance for GP regression
    ker (Dict[str, Callable]): Kernel info with 'khat' as Fourier transform
    eps (float): Truncation tolerance for covariance kernel
    xtrgs (torch.Tensor): Locations at which to evaluate posterior mean, shape (n,)
    opts (Dict, optional): Options controlling method params

    Returns:
    Tuple: beta, xis, ytrg, iter_info, time_info, ws
    """
    if opts is None:
        opts = {}
    
    # Timer for precomputation
    t_precomp_start = torch.cuda.Event(enable_timing=True)
    t_precomp_end = torch.cuda.Event(enable_timing=True)
    t_precomp_start.record()
    
    # Determine problem geometry
    N = inputs.shape[0]
    x0, x1 = torch.min(torch.cat([inputs, xtrgs])), torch.max(torch.cat([inputs, xtrgs]))
    L = x1 - x0
    
    # Get Fourier nodes (assuming a function similar to get_xis is implemented)
    xis, h, mtot = get_xis(kernel_obj=ker, eps=eps, L=L, use_integral=opts.get('use_integral', False), l2scaled=opts.get('l2scaled', False))
    
    # Center and scale coordinates
    xcen = (x1 + x0) / 2
    tphx = 2 * math.pi * h * (inputs - xcen)
    tphxtrgs = 2 * math.pi * h * (xtrgs - xcen)
    
    # Compute weights for Fourier basis functions
    khat = ker['khat']
    ws = torch.sqrt(khat(xis) * h)
    
    # Construct first row and column of Toeplitz matrix
    c = torch.ones(N, dtype=torch.complex64)
    XtXcol = torch.fft.fft(c, n=2*mtot-1)
    Gf = torch.fft.fft(XtXcol)
    
    # Construct right-hand side
    rhs = torch.fft.fft(y, n=mtot)
    rhs = ws * rhs
    
    t_precomp_end.record()
    torch.cuda.synchronize()
    t_precomp = t_precomp_start.elapsed_time(t_precomp_end) / 1000  # Convert to seconds
    
    # Define matrix-vector product function for CG
    def Afun(a):
        return ws * Afun2(Gf, ws * a) + sigmasq * a
    
    # Solve linear system with conjugate gradient
    cg_tol = eps / opts.get('cg_tol_fac', 1)
    t_cg_start = torch.cuda.Event(enable_timing=True)
    t_cg_end = torch.cuda.Event(enable_timing=True)
    t_cg_start.record()
    
    cg_solver = ConjugateGradients(Afun, rhs, torch.zeros_like(rhs), tol=cg_tol, max_iter=3*mtot)
    beta = cg_solver.solve()
    iter_info = cg_solver.iters_completed
    
    t_cg_end.record()
    torch.cuda.synchronize()
    t_cg = t_cg_start.elapsed_time(t_cg_end) / 1000  # Convert to seconds
    
    # Evaluate solution at target points
    t_post_start = torch.cuda.Event(enable_timing=True)
    t_post_end = torch.cuda.Event(enable_timing=True)
    t_post_start.record()
    
    tmpvec = ws * beta
    yhat = torch.fft.ifft(tmpvec, n=xtrgs.shape[0])
    ytrg = {'mean': yhat.real}
    
    t_post_end.record()
    torch.cuda.synchronize()
    t_post = t_post_start.elapsed_time(t_post_end) / 1000  # Convert to seconds
    
    # Prepare timing info
    time_info = torch.tensor([t_precomp, t_cg, t_post])
    time_info = torch.cat([time_info, time_info.sum().unsqueeze(0)])
    
    return beta, xis, ytrg, iter_info, time_info, ws

def Afun2(Gf: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Perform fast multiply by Toeplitz matrix for conjugate gradient.

    Args:
    Gf (torch.Tensor): FFT of first column of Toeplitz matrix
    a (torch.Tensor): Input vector

    Returns:
    torch.Tensor: Result of matrix-vector product
    """
    mtot = a.shape[0]
    af = torch.fft.fft(a, n=Gf.shape[0])
    vft = af * Gf
    vft = torch.fft.ifft(vft)
    return vft[:mtot]
