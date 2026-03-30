#!/usr/bin/env python3
"""
M-Step vs Vanilla Gradient Diagnosis
Investigates why m_step (both vanilla=True and vanilla=False) introduces bias 
compared to compute_vanilla_gradient function, especially when d>1.
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

def setup_test_case(d, n=200, seed=42):
    """Setup test case for given dimension"""
    torch.manual_seed(seed)
    device = torch.device('cpu')
    rdtype = torch.float64
    cdtype = torch.complex128
    
    # Generate data
    x = torch.rand(n, d, dtype=rdtype, device=device) * 2 - 1
    y, f = sample_bernoulli_gp(x, length_scale=0.1, variance=1.0)
    
    # Setup kernel
    kernel = SquaredExponential(dimension=d, init_lengthscale=0.2, init_variance=1.25)
    
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
        'x': x, 'y': y, 'f': f, 'kernel': kernel, 'h': h, 'mtot': mtot,
        'ws': ws, 'toeplitz': toeplitz, 'Dprime': Dprime, 'xis': xis,
        'fadj_batched': fadj_batched, 'fwd_batched': fwd_batched,
        'device': device, 'rdtype': rdtype, 'cdtype': cdtype, 'd': d, 'n': n
    }

def naive_kernel(x, xis):
    """Compute naive kernel matrix"""
    # Simple squared exponential kernel computation
    # This should match what's used in compute_vanilla_gradient
    diff = x.unsqueeze(1) - xis.unsqueeze(0)  # (n, M, d)
    sq_dist = torch.sum(diff**2, dim=-1)  # (n, M)
    return torch.exp(-0.5 * sq_dist)  # Simple unit variance, unit lengthscale for now

def compute_vanilla_gradient_reference(x, y, kernel, q_Delta):
    """
    Reference implementation of vanilla gradient computation
    This should match your compute_vanilla_gradient function
    """
    n, d = x.shape
    
    # Build kernel matrix
    K = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            diff = x[i] - x[j]
            sq_dist = torch.sum(diff**2)
            K[i, j] = kernel.init_variance * torch.exp(-0.5 * sq_dist / (kernel.init_lengthscale**2))
    
    # Add jitter
    K += 1e-7 * torch.eye(n)
    
    # Compute gradients w.r.t. lengthscale and variance
    K_inv = torch.inverse(K)
    
    # Lengthscale gradient
    dK_dls = torch.zeros_like(K)
    for i in range(n):
        for j in range(n):
            diff = x[i] - x[j]
            sq_dist = torch.sum(diff**2)
            dK_dls[i, j] = K[i, j] * sq_dist / (kernel.init_lengthscale**3)
    
    # Variance gradient  
    dK_dvar = K / kernel.init_variance
    
    # Compute gradient terms (simplified version)
    # This is a simplified version - you should match your exact compute_vanilla_gradient
    alpha = K_inv @ (y - 0.5)  # Assuming y is centered
    
    grad_ls = 0.5 * torch.trace(alpha[:, None] @ alpha[None, :] @ dK_dls - K_inv @ dK_dls)
    grad_var = 0.5 * torch.trace(alpha[:, None] @ alpha[None, :] @ dK_dvar - K_inv @ dK_dvar)
    
    return torch.tensor([grad_ls, grad_var])

def m_step_implementation(q, setup_dict, vanilla=False):
    """
    Implementation of m_step matching your notebook version
    """
    ws = setup_dict['ws']
    toeplitz = setup_dict['toeplitz'] 
    Dprime = setup_dict['Dprime']
    fadj_batched = setup_dict['fadj_batched']
    fwd_batched = setup_dict['fwd_batched']
    d = setup_dict['d']
    
    # Apply bias fixes as implemented in your notebook
    ws2 = ws.pow(2)
    condition_number = ws2.real.max() / ws2.real.min()
    cg_tol_fixed = 1e-12 / d
    jitter_val = max(1e-12, 1e-16 * condition_number)
    
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
    
    return term1, term2, term3

def diagnose_mstep_bias():
    """Main diagnosis function"""
    
    print("="*60)
    print("M-STEP vs VANILLA GRADIENT BIAS DIAGNOSIS")
    print("="*60)
    
    device = torch.device('cpu')
    results = {}
    
    for d in [1, 2]:
        print(f"\n{'='*20} {d}D CASE {'='*20}")
        
        # Setup test case
        setup_dict = setup_test_case(d, n=100)  # Smaller n for faster computation
        
        # Create variational parameters
        q = qVariationalParams(setup_dict['n'], device=device)
        q.Delta = torch.randn_like(q.Delta) * 0.1  # Small random values
        
        print(f"Grid size: {setup_dict['mtot']}^{d} = {setup_dict['mtot']**d}")
        print(f"h^d = {setup_dict['h']**d:.6e}")
        
        # 1. Compute reference vanilla gradient
        print(f"\n--- Computing Reference Vanilla Gradient ---")
        try:
            vanilla_grad_ref = compute_vanilla_gradient_reference(
                setup_dict['x'], setup_dict['y'], setup_dict['kernel'], q.Delta
            )
            print(f"Reference vanilla gradient: {vanilla_grad_ref}")
        except Exception as e:
            print(f"Error computing reference vanilla gradient: {e}")
            vanilla_grad_ref = None
        
        # 2. Test m_step with vanilla=True
        print(f"\n--- Testing m_step with vanilla=True ---")
        if setup_dict['mtot']**d < 500:  # Only if grid is small enough
            try:
                t1_v, t2_v, t3_v = m_step_implementation(q, setup_dict, vanilla=True)
                mstep_vanilla_grad = 0.5 * (t1_v + t2_v - t3_v)
                print(f"m_step (vanilla=True): term1={t1_v}, term2={t2_v}, term3={t3_v}")
                print(f"m_step (vanilla=True) gradient: {mstep_vanilla_grad}")
                
                if vanilla_grad_ref is not None:
                    diff_vanilla = torch.norm(mstep_vanilla_grad - vanilla_grad_ref).item()
                    print(f"||m_step_vanilla - reference_vanilla||: {diff_vanilla:.6e}")
                    
            except Exception as e:
                print(f"Error in m_step vanilla=True: {e}")
                mstep_vanilla_grad = None
        else:
            print("Grid too large for vanilla computation")
            mstep_vanilla_grad = None
        
        # 3. Test m_step with vanilla=False  
        print(f"\n--- Testing m_step with vanilla=False ---")
        try:
            t1_nv, t2_nv, t3_nv = m_step_implementation(q, setup_dict, vanilla=False)
            mstep_nonvanilla_grad = 0.5 * (t1_nv + t2_nv - t3_nv)
            print(f"m_step (vanilla=False): term1={t1_nv}, term2={t2_nv}, term3={t3_nv}")
            print(f"m_step (vanilla=False) gradient: {mstep_nonvanilla_grad}")
            
            if vanilla_grad_ref is not None:
                diff_nonvanilla = torch.norm(mstep_nonvanilla_grad - vanilla_grad_ref).item()
                print(f"||m_step_nonvanilla - reference_vanilla||: {diff_nonvanilla:.6e}")
                
            if mstep_vanilla_grad is not None:
                diff_modes = torch.norm(mstep_vanilla_grad - mstep_nonvanilla_grad).item()
                print(f"||m_step_vanilla - m_step_nonvanilla||: {diff_modes:.6e}")
                
        except Exception as e:
            print(f"Error in m_step vanilla=False: {e}")
            mstep_nonvanilla_grad = None
        
        # 4. Detailed component analysis
        print(f"\n--- Component Analysis ---")
        if setup_dict['mtot']**d < 500:
            # Analyze individual components that might cause bias
            try:
                # Test if the issue is in spectral representation
                print("Analyzing spectral representation...")
                
                # Check h^d scaling
                hd_scaling = setup_dict['h']**d
                print(f"h^d scaling: {hd_scaling:.6e}")
                
                # Check Dprime values
                Dprime_range = (setup_dict['Dprime'].real.min().item(), setup_dict['Dprime'].real.max().item())
                print(f"Dprime range: [{Dprime_range[0]:.6e}, {Dprime_range[1]:.6e}]")
                
                # Check if the issue is in the kernel evaluation
                print("Checking kernel consistency...")
                
                # Compare spectral vs direct kernel evaluation at a few points
                test_points = setup_dict['x'][:5]  # First 5 points
                xis_subset = setup_dict['xis'][:10]  # First 10 spectral points
                
                # Direct kernel evaluation
                K_direct = torch.zeros(5, 10)
                for i in range(5):
                    for j in range(10):
                        diff = test_points[i] - xis_subset[j]
                        K_direct[i, j] = setup_dict['kernel'].init_variance * torch.exp(
                            -0.5 * torch.sum(diff**2) / (setup_dict['kernel'].init_lengthscale**2)
                        )
                
                # Spectral kernel evaluation (simplified)
                F_test = naive_kernel(test_points, xis_subset)
                
                kernel_diff = torch.norm(K_direct - F_test).item()
                print(f"Kernel evaluation difference: {kernel_diff:.6e}")
                
            except Exception as e:
                print(f"Error in component analysis: {e}")
        
        # Store results
        results[d] = {
            'vanilla_ref': vanilla_grad_ref,
            'mstep_vanilla': mstep_vanilla_grad,
            'mstep_nonvanilla': mstep_nonvanilla_grad
        }
    
    # Summary analysis
    print(f"\n{'='*60}")
    print("BIAS DIAGNOSIS SUMMARY")
    print("="*60)
    
    print("\n1. DIMENSIONAL COMPARISON:")
    for d in [1, 2]:
        print(f"\n{d}D Case:")
        if results[d]['vanilla_ref'] is not None:
            print(f"  Reference vanilla: {results[d]['vanilla_ref']}")
        if results[d]['mstep_vanilla'] is not None:
            print(f"  m_step vanilla:    {results[d]['mstep_vanilla']}")
        if results[d]['mstep_nonvanilla'] is not None:
            print(f"  m_step nonvanilla: {results[d]['mstep_nonvanilla']}")
    
    print("\n2. POTENTIAL BIAS SOURCES:")
    print("   a) Spectral approximation error")
    print("   b) NUFFT approximation error") 
    print("   c) Different kernel evaluation methods")
    print("   d) h^d scaling effects in higher dimensions")
    print("   e) Toeplitz vs direct matrix operations")
    print("   f) Different jitter/regularization approaches")
    
    print("\n3. RECOMMENDED INVESTIGATIONS:")
    print("   - Check if compute_vanilla_gradient uses same kernel parameters")
    print("   - Verify spectral approximation accuracy")
    print("   - Compare NUFFT vs direct Fourier transforms")
    print("   - Check for consistent jitter/regularization")
    print("   - Verify gradient computation formulas match")
    
    return results

def main():
    """Main function"""
    print("M-STEP vs VANILLA GRADIENT BIAS DIAGNOSIS")
    print("Investigating bias in m_step compared to compute_vanilla_gradient")
    print("="*60)
    
    results = diagnose_mstep_bias()
    
    print(f"\nDiagnosis completed!")
    print(f"Check the output above for potential sources of bias.")

if __name__ == "__main__":
    main()








