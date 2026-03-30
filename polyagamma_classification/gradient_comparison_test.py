#!/usr/bin/env python3
"""
Gradient Comparison Test
Direct comparison between compute_vanilla_gradient and m_step functions
to identify the source of bias when d>1.
"""

import sys
import os
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.func import vmap

# Import required modules
from kernels import SquaredExponential
from efgpnd import ToeplitzND, compute_convolution_vector_vectorized_dD, NUFFT
from cg import ConjugateGradients
from vanilla_gp_sampling import sample_bernoulli_gp
from utils.kernels import get_xis
from polyagamma_classification.PG_GP import qVariationalParams

def compute_vanilla_gradient_reference(x, y, kernel, q_Delta, jitter=1e-7):
    """
    Reference implementation of vanilla gradient computation
    This should match your notebook's compute_vanilla_gradient function
    """
    n, d = x.shape
    device = x.device
    dtype = x.dtype
    
    print(f"  Computing vanilla gradient with:")
    print(f"    lengthscale={kernel.init_lengthscale:.6f}")
    print(f"    variance={kernel.init_variance:.6f}")
    print(f"    jitter={jitter:.2e}")
    
    # Build kernel matrix using kernel parameters
    K = torch.zeros(n, n, dtype=dtype, device=device)
    dK_dls = torch.zeros(n, n, dtype=dtype, device=device)  # derivative w.r.t. lengthscale
    dK_dvar = torch.zeros(n, n, dtype=dtype, device=device)  # derivative w.r.t. variance
    
    for i in range(n):
        for j in range(n):
            diff = x[i] - x[j]
            sq_dist = torch.sum(diff**2)
            
            # Kernel value
            K[i, j] = kernel.init_variance * torch.exp(-0.5 * sq_dist / (kernel.init_lengthscale**2))
            
            # Lengthscale derivative
            dK_dls[i, j] = K[i, j] * sq_dist / (kernel.init_lengthscale**3)
            
            # Variance derivative
            dK_dvar[i, j] = K[i, j] / kernel.init_variance
    
    # Add jitter
    K += jitter * torch.eye(n, dtype=dtype, device=device)
    
    # Compute K^{-1}
    try:
        K_inv = torch.inverse(K)
        print(f"    K condition number: {torch.linalg.cond(K):.2e}")
    except:
        print(f"    Using pseudo-inverse due to singular K")
        K_inv = torch.pinverse(K)
    
    # Compute alpha = K^{-1} * (y - 0.5) or similar
    # This depends on your exact implementation
    alpha = K_inv @ (y - 0.5)  # Assuming centered labels
    
    # Compute gradients using standard GP derivative formula
    # d/d(theta) log p(y|theta) = 0.5 * tr(alpha*alpha^T * dK/d(theta) - K^{-1} * dK/d(theta))
    alpha_outer = torch.outer(alpha, alpha)
    
    grad_ls = 0.5 * torch.trace(alpha_outer @ dK_dls - K_inv @ dK_dls)
    grad_var = 0.5 * torch.trace(alpha_outer @ dK_dvar - K_inv @ dK_dvar)
    
    return torch.tensor([grad_ls, grad_var], dtype=dtype, device=device)

def setup_mstep_components(x, kernel, device, rdtype, cdtype):
    """Setup all components needed for m_step computation"""
    d = x.shape[1]
    
    # Setup spectral representation
    eps = 1e-7
    trunc_eps = 1e-12
    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values
    domain_lengths = x1 - x0
    L = domain_lengths.max()
    
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True, l2scaled=False, trunc_eps=trunc_eps)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing='ij')
    xis = torch.stack(grids, dim=-1).view(-1, d)
    
    # Compute spectral weights and gradients
    ws = torch.sqrt(kernel.spectral_density(xis)).to(cdtype)
    Dprime = (h**d * kernel.spectral_grad(xis)).to(cdtype)
    
    # Setup Toeplitz operator
    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    
    # NUFFT setup
    OUT = (mtot,)*d
    nufft_op = NUFFT(x, torch.zeros_like(x), h, 1e-7, cdtype=cdtype, device=device)
    fadj = lambda v: nufft_op.type1(v, out_shape=OUT).reshape(-1)
    fwd = lambda fk: nufft_op.type2(fk, out_shape=OUT)
    fadj_batched = vmap(fadj, in_dims=0, out_dims=0)
    fwd_batched = vmap(fwd, in_dims=0, out_dims=0)
    
    return {
        'h': h, 'mtot': mtot, 'ws': ws, 'toeplitz': toeplitz, 'Dprime': Dprime,
        'xis': xis, 'fadj_batched': fadj_batched, 'fwd_batched': fwd_batched
    }

def compute_mstep_gradient(q, mstep_components, d, vanilla=False):
    """Compute m_step gradient using your implementation"""
    
    ws = mstep_components['ws']
    toeplitz = mstep_components['toeplitz']
    Dprime = mstep_components['Dprime']
    fadj_batched = mstep_components['fadj_batched']
    fwd_batched = mstep_components['fwd_batched']
    
    # Apply bias fixes
    ws2 = ws.pow(2)
    condition_number = ws2.real.max() / ws2.real.min()
    cg_tol_fixed = 1e-12 / d
    jitter_val = max(1e-12, 1e-16 * condition_number)
    
    print(f"  m_step computation with:")
    print(f"    Grid size: {mstep_components['mtot']}^{d} = {mstep_components['mtot']**d}")
    print(f"    h^d: {mstep_components['h']**d:.6e}")
    print(f"    Condition number: {condition_number:.2e}")
    print(f"    CG tolerance: {cg_tol_fixed:.2e}")
    print(f"    Jitter: {jitter_val:.2e}")
    print(f"    Vanilla mode: {vanilla}")
    
    def D2_Fstar_Kinv_z_fixed(z):
        A_apply = lambda x: ws*toeplitz(ws* x) + jitter_val * x
        if z.ndim == 1:
            z = z.unsqueeze(0)
        fadj_z = fadj_batched(z)
        rhs = ws*fadj_z

        x0 = torch.zeros_like(rhs)
        M_inv_apply = lambda x: x/(10*ws**2)
        
        if vanilla and ws.numel() < 500:
            # Vanilla direct solve
            A_dense = ws[:, None] * torch.stack([toeplitz(torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)[:, i])
                                            for i in range(ws.numel())], dim=1) * ws[None, :] \
                + jitter_val * torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)
            solve = torch.cholesky_solve(rhs.T, torch.linalg.cholesky(A_dense)).T
        else:
            # CG solve
            cg_solver = ConjugateGradients(
                A_apply, rhs, x0,
                tol=cg_tol_fixed, early_stopping=True, 
                M_inv_apply=M_inv_apply
            )
            solve = cg_solver.solve()
        
        return solve/ws, fadj_z
    
    def Sigma_z_mm_fixed(z):
        A_apply = lambda x: ws*toeplitz(ws* x) + jitter_val * x
        if z.ndim == 1:
            z = z.unsqueeze(0)
        rhs = ws * z

        x0 = torch.zeros_like(rhs)
        M_inv_apply = lambda x: x/(10*ws**2)
        
        if vanilla and ws.numel() < 500:
            # Vanilla direct solve
            A_dense = ws[:, None] * torch.stack([toeplitz(torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)[:, i])
                                            for i in range(ws.numel())], dim=1) * ws[None, :] \
                + jitter_val * torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)
            solve = torch.cholesky_solve(rhs.T, torch.linalg.cholesky(A_dense)).T
        else:
            # CG solve
            cg_solver = ConjugateGradients(
                A_apply, rhs, x0,
                tol=cg_tol_fixed, early_stopping=True, 
                M_inv_apply=M_inv_apply
            )
            solve = cg_solver.solve()
        
        result = fwd_batched(solve/ws)
        return result
    
    # Compute m-step terms
    beta, fadj_delta = D2_Fstar_Kinv_z_fixed(q.Delta)
    
    term1 = torch.sum(beta * Dprime, dim=0)
    
    Sigma_z = Sigma_z_mm_fixed(beta)
    term2 = torch.sum(Sigma_z * q.Delta.unsqueeze(-1) * Dprime.unsqueeze(0), dim=0)
    
    term3 = torch.sum(fadj_delta * Dprime, dim=0)
    
    # Final gradient computation
    gradient = 0.5 * (term1 + term2 - term3)
    
    print(f"    term1: {term1}")
    print(f"    term2: {term2}")
    print(f"    term3: {term3}")
    
    return gradient.real, term1.real, term2.real, term3.real

def run_gradient_comparison():
    """Main comparison function"""
    
    print("="*80)
    print("GRADIENT COMPARISON TEST: compute_vanilla_gradient vs m_step")
    print("="*80)
    
    device = torch.device('cpu')
    rdtype = torch.float64
    cdtype = torch.complex128
    
    results = {}
    
    for d in [1, 2]:
        print(f"\n{'='*30} {d}D CASE {'='*30}")
        
        # Setup test case
        torch.manual_seed(42)
        n = 50  # Moderate size for both methods
        x = torch.rand(n, d, dtype=rdtype, device=device) * 2 - 1
        y, f = sample_bernoulli_gp(x, length_scale=0.1, variance=1.0)
        
        # Setup kernel - this is critical!
        kernel = SquaredExponential(dimension=d, init_lengthscale=0.2, init_variance=1.25)
        
        print(f"Data: n={n}, d={d}")
        print(f"Kernel parameters: lengthscale={kernel.init_lengthscale}, variance={kernel.init_variance}")
        
        # Create variational parameters
        q = qVariationalParams(n, device=device)
        q.Delta = torch.randn_like(q.Delta) * 0.1  # Small random values
        
        # 1. Compute vanilla gradient
        print(f"\n--- Computing Vanilla Gradient ---")
        try:
            vanilla_grad = compute_vanilla_gradient_reference(x, y, kernel, q.Delta, jitter=1e-7)
            print(f"Vanilla gradient: {vanilla_grad}")
        except Exception as e:
            print(f"Error computing vanilla gradient: {e}")
            vanilla_grad = None
        
        # 2. Setup m_step components
        print(f"\n--- Setting up M-Step Components ---")
        try:
            mstep_components = setup_mstep_components(x, kernel, device, rdtype, cdtype)
        except Exception as e:
            print(f"Error setting up m_step: {e}")
            continue
        
        # 3. Test m_step with vanilla=True (if grid is small enough)
        print(f"\n--- M-Step with vanilla=True ---")
        if mstep_components['mtot']**d < 500:
            try:
                mstep_vanilla_grad, t1_v, t2_v, t3_v = compute_mstep_gradient(
                    q, mstep_components, d, vanilla=True)
                print(f"M-step (vanilla=True) gradient: {mstep_vanilla_grad}")
                
                if vanilla_grad is not None:
                    diff_vanilla = torch.norm(mstep_vanilla_grad - vanilla_grad).item()
                    print(f"||m_step_vanilla - vanilla_ref||: {diff_vanilla:.6e}")
                
            except Exception as e:
                print(f"Error in m_step vanilla=True: {e}")
                mstep_vanilla_grad = None
        else:
            print("Grid too large for vanilla m_step computation")
            mstep_vanilla_grad = None
        
        # 4. Test m_step with vanilla=False
        print(f"\n--- M-Step with vanilla=False ---")
        try:
            mstep_nonvanilla_grad, t1_nv, t2_nv, t3_nv = compute_mstep_gradient(
                q, mstep_components, d, vanilla=False)
            print(f"M-step (vanilla=False) gradient: {mstep_nonvanilla_grad}")
            
            if vanilla_grad is not None:
                diff_nonvanilla = torch.norm(mstep_nonvanilla_grad - vanilla_grad).item()
                print(f"||m_step_nonvanilla - vanilla_ref||: {diff_nonvanilla:.6e}")
            
            if mstep_vanilla_grad is not None:
                diff_modes = torch.norm(mstep_vanilla_grad - mstep_nonvanilla_grad).item()
                print(f"||m_step_vanilla - m_step_nonvanilla||: {diff_modes:.6e}")
                
        except Exception as e:
            print(f"Error in m_step vanilla=False: {e}")
            mstep_nonvanilla_grad = None
        
        # Store results
        results[d] = {
            'vanilla_ref': vanilla_grad,
            'mstep_vanilla': mstep_vanilla_grad,
            'mstep_nonvanilla': mstep_nonvanilla_grad
        }
        
        # 5. Component-wise analysis
        print(f"\n--- Component-wise Analysis ---")
        if vanilla_grad is not None:
            print(f"Vanilla reference:")
            print(f"  Lengthscale grad: {vanilla_grad[0]:.6f}")
            print(f"  Variance grad:    {vanilla_grad[1]:.6f}")
        
        if mstep_vanilla_grad is not None:
            print(f"M-step (vanilla=True):")
            print(f"  Lengthscale grad: {mstep_vanilla_grad[0]:.6f}")
            print(f"  Variance grad:    {mstep_vanilla_grad[1]:.6f}")
        
        if mstep_nonvanilla_grad is not None:
            print(f"M-step (vanilla=False):")
            print(f"  Lengthscale grad: {mstep_nonvanilla_grad[0]:.6f}")
            print(f"  Variance grad:    {mstep_nonvanilla_grad[1]:.6f}")
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n1. ERROR MAGNITUDES:")
    for d in [1, 2]:
        if d in results:
            print(f"\n{d}D Case:")
            vanilla_ref = results[d]['vanilla_ref']
            mstep_vanilla = results[d]['mstep_vanilla']
            mstep_nonvanilla = results[d]['mstep_nonvanilla']
            
            if vanilla_ref is not None:
                print(f"  Reference magnitude: {torch.norm(vanilla_ref).item():.6e}")
                
                if mstep_vanilla is not None:
                    error_vanilla = torch.norm(mstep_vanilla - vanilla_ref).item()
                    relative_error_vanilla = error_vanilla / torch.norm(vanilla_ref).item()
                    print(f"  M-step vanilla error: {error_vanilla:.6e} (relative: {relative_error_vanilla:.2%})")
                
                if mstep_nonvanilla is not None:
                    error_nonvanilla = torch.norm(mstep_nonvanilla - vanilla_ref).item()
                    relative_error_nonvanilla = error_nonvanilla / torch.norm(vanilla_ref).item()
                    print(f"  M-step nonvanilla error: {error_nonvanilla:.6e} (relative: {relative_error_nonvanilla:.2%})")
    
    print(f"\n2. DIMENSIONAL BIAS ANALYSIS:")
    if 1 in results and 2 in results:
        print("Comparing 1D vs 2D bias levels:")
        
        # Check if bias increases with dimension
        for method in ['mstep_vanilla', 'mstep_nonvanilla']:
            if (results[1][method] is not None and results[2][method] is not None and
                results[1]['vanilla_ref'] is not None and results[2]['vanilla_ref'] is not None):
                
                error_1d = torch.norm(results[1][method] - results[1]['vanilla_ref']).item()
                error_2d = torch.norm(results[2][method] - results[2]['vanilla_ref']).item()
                
                ref_1d = torch.norm(results[1]['vanilla_ref']).item()
                ref_2d = torch.norm(results[2]['vanilla_ref']).item()
                
                rel_error_1d = error_1d / ref_1d
                rel_error_2d = error_2d / ref_2d
                
                bias_increase = rel_error_2d / rel_error_1d if rel_error_1d > 0 else float('inf')
                
                print(f"  {method}: 2D bias is {bias_increase:.2f}x larger than 1D bias")
    
    print(f"\n3. KEY FINDINGS:")
    print("   - Check if both methods use the same kernel parameters")
    print("   - Look for systematic bias increase with dimension")
    print("   - Compare vanilla vs non-vanilla m_step modes")
    print("   - Examine relative vs absolute error magnitudes")
    
    return results

def create_parameter_consistency_test():
    """Test with different kernel parameter configurations"""
    
    print(f"\n{'='*80}")
    print("PARAMETER CONSISTENCY TEST")
    print("="*80)
    
    device = torch.device('cpu')
    rdtype = torch.float64
    
    # Test case
    torch.manual_seed(42)
    n, d = 20, 2
    x = torch.rand(n, d, dtype=rdtype, device=device) * 1 - 0.5
    y = torch.randint(0, 2, (n,), dtype=rdtype, device=device)
    
    # Test different parameter combinations
    param_configs = [
        {"lengthscale": 0.1, "variance": 1.0, "name": "Data generation params"},
        {"lengthscale": 0.2, "variance": 1.25, "name": "M-step params"},
        {"lengthscale": 1.0, "variance": 1.0, "name": "Unit params"},
    ]
    
    for config in param_configs:
        print(f"\n--- Testing {config['name']} ---")
        print(f"Lengthscale: {config['lengthscale']}, Variance: {config['variance']}")
        
        # Create kernel with these parameters
        kernel = SquaredExponential(dimension=d, init_lengthscale=config['lengthscale'], init_variance=config['variance'])
        
        # Create simple variational parameters
        q = qVariationalParams(n, device=device)
        q.Delta = torch.ones_like(q.Delta) * 0.1  # Fixed values for consistency
        
        # Compute vanilla gradient
        try:
            vanilla_grad = compute_vanilla_gradient_reference(x, y, kernel, q.Delta, jitter=1e-7)
            print(f"Vanilla gradient: lengthscale={vanilla_grad[0]:.6f}, variance={vanilla_grad[1]:.6f}")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function"""
    print("COMPREHENSIVE GRADIENT COMPARISON TEST")
    print("Comparing compute_vanilla_gradient vs m_step functions")
    
    # Main comparison
    results = run_gradient_comparison()
    
    # Parameter consistency test
    create_parameter_consistency_test()
    
    print(f"\n{'='*80}")
    print("NEXT ACTIONS")
    print("="*80)
    print("1. Check if compute_vanilla_gradient uses same kernel parameters as m_step")
    print("2. If parameters match, the bias is likely due to spectral approximation")
    print("3. If parameters don't match, fix the parameter consistency")
    print("4. Test with smaller spectral tolerance (eps) to improve approximation")
    print("5. Consider the trade-off between computational efficiency and accuracy")

if __name__ == "__main__":
    main()








