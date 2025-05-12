import torch
from utils.kernels import get_xis
from cg import ConjugateGradients
import time

# TODO: remove the opts dict in favor of args.
def efgp1d_dense(x, y, sigmasq, kernel, eps, x_new, opts=None):
    """
    Dense solve version of EFGP in 1D using PyTorch.
    
    Args:
        x: (N,) tensor of observation locations
        y: (N,) tensor of observations
        sigmasq: float, residual variance for GP regression
        kernel: object with methods k() and khat() for covariance kernel and its Fourier transform
        eps: float, truncation tolerance
        x_new: (Nsol,) tensor of locations for posterior mean evaluation
        opts: optional dict of parameters (only 'get_var' used currently)
    
    Returns:
        dict containing:
            beta: (M,) tensor of Fourier basis weights
            xis: (M,) tensor of Fourier frequencies used
            ytrg: dict with posterior means and optionally variances
            A: (M,M) system matrix (for debugging)
            F: (N,M) design matrix (for debugging)
            ws: (M,) weights for scaling complex exponentials
    """
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    x_new = x_new.to(dtype=torch.float64)
    # Get problem geometry
    x0, x1 = torch.min(torch.cat([x, x_new])), torch.max(torch.cat([x, x_new]))
    L = x1 - x0
    N = x.shape[0]
    
    # Get Fourier frequencies and weights
    xis, h, mtot = get_xis(kernel, eps, L)
    ws = torch.sqrt(kernel.spectral_density(xis) * h)
    D = torch.diag(ws).to(dtype=torch.complex128) # complex dtype

    start_time = time.time()
    # Form design features F 
    F = torch.exp(1j * 2 * torch.pi * torch.outer(x, xis)).to(dtype=torch.complex128)

    # Construct right-hand side
    rhs = D @ torch.conj(F).T @ y.to(dtype=torch.complex128) # y to complex dtype temporarily
    rhs_time = time.time() - start_time
    
    # Form system matrix A
    start_time = time.time()
    A = D @ (torch.conj(F).T @ F) @ D
    construct_system_time = time.time() - start_time
    # Solve the linear system
    start_time = time.time()
    if opts is not None and opts.get('method') == "cholesky":
        # Solve using Cholesky factorization
        chol_factor = torch.linalg.cholesky(A + sigmasq * torch.eye(mtot, dtype=A.dtype))
        beta = torch.cholesky_solve(rhs.unsqueeze(-1), chol_factor).squeeze(-1)
    else:
        # Solve using conjugate gradients
        cg_object = ConjugateGradients(A_apply_function=A + sigmasq * torch.eye(mtot, dtype=A.dtype), b=rhs, x0=torch.zeros_like(rhs), early_stopping=opts.get('early_stopping', False))
        beta = cg_object.solve() 
    solve_time = time.time() - start_time

    # Evaluate posterior mean at target points
    Phi_target = torch.exp(1j * 2 * torch.pi * torch.outer(x_new, xis)) @ D
    yhat = Phi_target @ beta
    
    ytrg = {'mean': torch.real(yhat)}

    timing_results = {
        'rhs_time': rhs_time,
        'construct_system_time': construct_system_time,
        'solve_time': solve_time
    }
    
    # Optionally compute posterior variance
    if opts is not None and opts.get('get_var', False):
        nsol = len(x_new) # number of target points
        c = (A / sigmasq) + torch.eye(mtot, dtype=A.dtype)
        c_inv = torch.linalg.inv(c)
        Phi_target = torch.exp(1j * 2 * torch.pi * torch.outer(x_new, xis)) @ D
        ytrg['var'] = torch.real(torch.diagonal(Phi_target @ c_inv @ Phi_target.T.conj()))
    
    if opts is not None and opts.get('get_log_marginal_likelihood', False):
        logdet = N*torch.log(sigmasq) + torch.logdet((D @ F.adjoint() @ F @ D)/sigmasq + torch.eye(mtot, dtype=torch.float64)).to(dtype=torch.float64)
        alpha = (1/sigmasq) * (y - torch.real(F @ D @ beta))
        log_marg_lik = -0.5 * y.T @ alpha - 0.5 * logdet - 0.5 * N * torch.log(2 * torch.tensor(torch.pi, dtype=torch.float64))
        ytrg['log_marginal_likelihood'] = log_marg_lik
    
    return beta, xis, ytrg, A, F, ws, timing_results
