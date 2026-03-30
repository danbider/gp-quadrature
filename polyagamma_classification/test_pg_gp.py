#!/usr/bin/env python3
"""
Test script for the PolyagammaGP class.
This demonstrates the same functionality as shown in the notebook.
"""

import sys
import torch
import matplotlib.pyplot as plt
import math

# Add paths (adjust as needed for your setup)
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature')
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature/kernels')

from kernels import SquaredExponential
from vanilla_gp_sampling import sample_bernoulli_gp
from PG_GP import PolyagammaGP

def compute_exact_gradients(model, m, verbose=True):
    """
    Compute gradients using exact kernel matrix (for comparison with NUFFT method).
    This follows the approach from the notebook.
    """
    print("Computing exact gradients for comparison...")
    
    # Get data and parameters
    x = model.x
    y = model.y
    omega = model.q.Delta
    kernel = model.kernel
    n = x.shape[0]
    device = x.device
    dtype = model.dtype
    
    # Flatten m if needed
    m_flat = m.flatten()
    
    # Build exact kernel matrix using naive approach
    F_train = torch.exp(2 * math.pi * 1j * torch.matmul(x, model.xis.T)).to(torch.complex128)
    ws2_diag = torch.diag(model.ws2.to(dtype=torch.complex128))
    Kff = F_train @ ws2_diag @ F_train.T.conj()
    Kff = Kff.real.to(dtype=dtype)
    
    # Add jitter for numerical stability
    jitter = 1e-4
    K = Kff + jitter * torch.eye(n, device=device, dtype=dtype)
    
    try:
        L = torch.linalg.cholesky(K, upper=False)
    except RuntimeError:
        # fallback: add more jitter if not positive definite
        K = Kff + (jitter * 10) * torch.eye(n, device=device, dtype=dtype)
        L = torch.linalg.cholesky(K, upper=False)
    
    # K⁻¹ via Cholesky
    I = torch.eye(n, device=device, dtype=dtype)
    K_inv = torch.cholesky_solve(I, L, upper=False)
    
    # S⁻¹ = K⁻¹ + diag(ω) ⇒ S = (S⁻¹)⁻¹
    S_inv = K_inv + torch.diag(omega)
    try:
        LS = torch.linalg.cholesky(S_inv, upper=False)
    except RuntimeError:
        S_inv = S_inv + (jitter * 10) * torch.eye(n, device=device, dtype=dtype)
        LS = torch.linalg.cholesky(S_inv, upper=False)
    S = torch.cholesky_inverse(LS, upper=False)
    
    # Compute kernel derivatives
    # ∂K/∂var and ∂K/∂ls using spectral representation
    dK_dvar = F_train @ torch.diag(model.Dprime[:, 1].to(dtype=torch.complex128)) @ F_train.T.conj()
    dK_dls = F_train @ torch.diag(model.Dprime[:, 0].to(dtype=torch.complex128)) @ F_train.T.conj()
    
    # Convert to real
    dK_dvar_r = dK_dvar.real.to(dtype=dtype)
    dK_dls_r = dK_dls.real.to(dtype=dtype)
    
    # helper: v = K⁻¹ m
    m_col = m_flat.unsqueeze(-1)
    v = torch.cholesky_solve(m_col, L, upper=False).squeeze(-1)
    
    # Gradient w.r.t. variance
    t1var = v @ (dK_dvar_r @ v)
    KinvS = torch.cholesky_solve(S, L, upper=False)
    t2var = torch.sum(KinvS * (K_inv @ dK_dvar_r))
    t3var = torch.sum(K_inv * dK_dvar_r)
    grad_var_exact = 0.5 * (t1var + t2var - t3var)
    
    # Gradient w.r.t. lengthscale
    t1ls = v @ (dK_dls_r @ v)
    t2ls = torch.sum(KinvS * (K_inv @ dK_dls_r))
    t3ls = torch.sum(K_inv * dK_dls_r)
    grad_ls_exact = 0.5 * (t1ls + t2ls - t3ls)
    
    if verbose:
        print(f"Exact method - Variance terms (t1, t2, t3): {t1var:.6f}, {t2var:.6f}, {t3var:.6f}")
        print(f"Exact method - Lengthscale terms (t1, t2, t3): {t1ls:.6f}, {t2ls:.6f}, {t3ls:.6f}")
    
    return torch.tensor([grad_ls_exact, grad_var_exact])

def main():
    # Set default dtype
    torch.set_default_dtype(torch.float64)
    
    # Parameters
    n = 1000  # Number of points
    d = 2     # Dimensionality
    true_length_scale = 0.5
    true_variance = 1.0
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Generate data
    x = torch.rand(n, d, dtype=dtype, device=device) * 2 - 1
    y, f = sample_bernoulli_gp(x, length_scale=true_length_scale, variance=true_variance)
    
    print(f"Generated {n} points with shape {x.shape}")
    
    # Initialize kernel
    kernel = SquaredExponential(dimension=d, init_lengthscale=0.5, init_variance=2.0)
    
    # Initialize model
    model = PolyagammaGP(
        x=x, 
        y=y, 
        kernel=kernel, 
        eps=1e-4, 
        nufft_eps=1e-8,
        cg_tol=1e-6,
        device=device,
        dtype=dtype
    )
    
    print("Model initialized successfully!")
    
    # Fit the model (E-step)
    print("\n" + "="*50)
    print("Running E-step...")
    print("="*50)
    
    m, Sigma_diags, acc, Sz, probes = model.fit(
        max_iters=20, 
        verbose=True,
        J=20
    )
    
    print(f"\nFinal accuracy: {acc:.4f}")
    
    # Run M-step
    print("\n" + "="*50)
    print("Running M-step...")
    print("="*50)
    
    term1, term2, term3 = model.m_step(m, J=20, verbose=True)
    
    # Compute gradients from NUFFT method
    grad_nufft = 0.5 * (term1 + term2 - term3)
    print(f"\nNUFFT-based gradients: {grad_nufft}")
    
    # Compute exact gradients for comparison
    print("\n" + "="*50)
    print("Computing exact gradients for comparison...")
    print("="*50)
    
    grad_exact = compute_exact_gradients(model, m, verbose=True)
    
    # Compare gradients
    print("\n" + "="*50)
    print("GRADIENT COMPARISON")
    print("="*50)
    print(f"NUFFT method:  [lengthscale: {grad_nufft[0]:.6f}, variance: {grad_nufft[1]:.6f}]")
    print(f"Exact method:  [lengthscale: {grad_exact[0]:.6f}, variance: {grad_exact[1]:.6f}]")
    print(f"Absolute diff: [lengthscale: {abs(grad_nufft[0] - grad_exact[0]):.6f}, variance: {abs(grad_nufft[1] - grad_exact[1]):.6f}]")
    print(f"Relative diff: [lengthscale: {abs(grad_nufft[0] - grad_exact[0]) / abs(grad_exact[0]) * 100:.3f}%, variance: {abs(grad_nufft[1] - grad_exact[1]) / abs(grad_exact[1]) * 100:.3f}%]")
    
    # Check if gradients are close
    rtol = 1e-3  # relative tolerance
    atol = 1e-6  # absolute tolerance
    close = torch.allclose(grad_nufft, grad_exact, rtol=rtol, atol=atol)
    print(f"Gradients match within tolerance (rtol={rtol}, atol={atol}): {close}")
    
    # Plot results (if 2D)
    if d == 2:
        plt.figure(figsize=(15, 4))
        
        plt.subplot(1, 3, 1)
        plt.scatter(m.detach().cpu().numpy(), f.detach().cpu().numpy(), alpha=0.5)
        plt.xlabel("Posterior mean (m)")
        plt.ylabel("True latent function (f)")
        plt.title("Posterior mean vs True function")
        
        plt.subplot(1, 3, 2)
        plt.scatter(x[:, 0].detach().cpu().numpy(), x[:, 1].detach().cpu().numpy(), 
                   c=y.detach().cpu().numpy(), cmap='RdYlBu', alpha=0.7)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Data points colored by labels")
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        methods = ['NUFFT', 'Exact']
        ls_grads = [grad_nufft[0].item(), grad_exact[0].item()]
        var_grads = [grad_nufft[1].item(), grad_exact[1].item()]
        
        x_pos = range(len(methods))
        width = 0.35
        
        plt.bar([p - width/2 for p in x_pos], ls_grads, width, label='Lengthscale', alpha=0.7)
        plt.bar([p + width/2 for p in x_pos], var_grads, width, label='Variance', alpha=0.7)
        
        plt.xlabel('Method')
        plt.ylabel('Gradient Value')
        plt.title('Gradient Comparison')
        plt.xticks(x_pos, methods)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pg_gp_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Print hyperparameters
    print(f"\nCurrent hyperparameters: {model.get_hyperparameters()}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 