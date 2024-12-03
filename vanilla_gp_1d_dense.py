import torch
from utils.kernels import get_xis
from cg import ConjugateGradients
import time

# TODO: remove the opts dict in favor of args.
def vanilla_gp_dense(x, y, sigmasq, kernel, x_new, opts=None):
    """
    Dense solve version of vanilla GP in 1D using PyTorch.
    
    Args:
        x: (N,) tensor of observation locations
        y: (N,) tensor of observations
        sigmasq: float, residual variance for GP regression
        kernel: object with methods k() and khat() for covariance kernel and its Fourier transform
        x_new: (Nsol,) tensor of locations for posterior mean evaluation
        opts: optional dict of parameters (only 'get_var' used currently)
    
    Returns:
        dict containing:
            ytrg: dict with posterior means and optionally variances
    """
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    x_new = x_new.to(dtype=torch.float64)
    
    N = x.shape[0]

    # compute K + sigma^2 I
    diff_matrix = x[:, None] - x[None, :] # (N, N) with pairwise differences
    K_plus_sigma_sq = kernel.kernel(diff_matrix) + sigmasq * torch.eye(N, dtype=torch.float64)

    # Solve the linear system
    start_time = time.time()
    if opts is not None and opts.get('method') == "cholesky":
        # Solve using Cholesky factorization
        chol_factor = torch.linalg.cholesky(K_plus_sigma_sq)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), chol_factor).squeeze(-1)
    else:
        # Solve using conjugate gradients
        cg_object = ConjugateGradients(A_apply_function=K_plus_sigma_sq, b=y, x0=torch.zeros_like(y))
        alpha = cg_object.solve()
    solve_time = time.time() - start_time

    # Evaluate posterior mean at target points
    diff_matrix_new = x_new[:, None] - x[None, :] # (Nsol, N) with pairwise differences
    tilde_K = kernel.kernel(diff_matrix_new)
    yhat = tilde_K @ alpha
    
    ytrg = {'mean': torch.real(yhat)}

    timing_results = {
        'rhs_time': None,
        'construct_system_time': None,
        'solve_time': solve_time
    }
    
    # Optionally compute posterior variance
    if opts is not None and opts.get('get_var', False):
        pass
        # nsol = len(x_new) # number of target points
        # c = (K_plus_sigma_sq / sigmasq) + torch.eye(N, dtype=K_plus_sigma_sq.dtype)
        # c_inv = torch.linalg.inv(c)
        # diff_matrix_new = x_new[:, None] - x[None, :] # (Nsol, N) with pairwise differences
        # tilde_K = kernel.kernel(diff_matrix_new)
        # ytrg['var'] = torch.real(torch.diagonal(tilde_K @ c_inv @ tilde_K.T.conj()))
    
    return alpha, ytrg, timing_results
