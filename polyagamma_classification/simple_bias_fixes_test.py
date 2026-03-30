#!/usr/bin/env python3
"""
Simple Bias Fixes Test
Tests the proposed CG solver fixes for the m-step bias.
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

def setup_spectral_representation(x, kernel, device=None, rdtype=None, cdtype=None):
    """Setup spectral representation for the kernel"""
    if device is None:
        device = x.device
    if rdtype is None:
        rdtype = x.dtype
    if cdtype is None:
        cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
    
    # Get spectral parameters
    d = x.shape[1]
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
    
    return xis, h, mtot, ws, toeplitz, Dprime

def solve_d2_fstar_kinv_z(z, ws, toeplitz, fadj_batched, cg_tol, jitter_val, vanilla=False):
    """Test D2_Fstar_Kinv_z with configurable parameters"""
    A_apply = lambda x: ws*toeplitz(ws* x) + jitter_val * x
    if z.ndim == 1:
        z = z.unsqueeze(0)
    fadj_z = fadj_batched(z)
    rhs = ws*fadj_z

    x0 = torch.zeros_like(rhs)
    
    M_inv_apply = lambda x: x/(10*ws**2)
    if vanilla and ws.numel() < 500:  # Only for small cases
        A_dense = ws[:, None] * torch.stack([toeplitz(torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)[:, i])
                                        for i in range(ws.numel())], dim=1) * ws[None, :] \
            + jitter_val * torch.eye(ws.numel(), device=ws.device, dtype=ws.dtype)
        solve = torch.cholesky_solve(rhs.T, torch.linalg.cholesky(A_dense)).T
    else:
        cg_solver = ConjugateGradients(
            A_apply, rhs, x0,
            tol=cg_tol, early_stopping=True, 
            M_inv_apply=M_inv_apply
        )
        solve = cg_solver.solve()

    return solve/ws

def run_bias_fixes():
    """Test bias fixes for different dimensions"""
    
    print("="*60)
    print("BIAS FIXES TESTING - CG SOLVER COMPARISON")
    print("="*60)
    
    device = torch.device('cpu')
    rdtype = torch.float64
    cdtype = torch.complex128
    
    results = {}
    
    for d in [1, 2]:
        print(f"\n{'='*20} {d}D CASE {'='*20}")
        
        # Setup test case
        torch.manual_seed(42)
        n = 200
        x = torch.rand(n, d, dtype=rdtype, device=device) * 2 - 1
        y, f = sample_bernoulli_gp(x, length_scale=0.1, variance=1.0)
        
        # Setup kernel
        kernel = SquaredExponential(dimension=d, init_lengthscale=0.2, init_variance=1.25)
        
        # Setup spectral representation
        xis, h, mtot, ws, toeplitz, Dprime = setup_spectral_representation(
            x, kernel, device=device, rdtype=rdtype, cdtype=cdtype)
        
        # NUFFT setup
        OUT = (mtot,)*d
        nufft_op = NUFFT(x, torch.zeros_like(x), h, 1e-7, cdtype=cdtype, device=device)
        fadj = lambda v: nufft_op.type1(v, out_shape=OUT).reshape(-1)
        fadj_batched = vmap(fadj, in_dims=0, out_dims=0)
        
        # Compute condition number
        ws2 = ws.pow(2)
        cond_estimate = ws2.real.max() / ws2.real.min()
        
        print(f"Grid size: {mtot}^{d} = {mtot**d}")
        print(f"h^d = {h**d:.6e}")
        print(f"Condition estimate: {cond_estimate:.2e}")
        
        # Define configurations to test
        configs = {
            'original': {
                'jitter': 1e-7,
                'cg_tol': 1e-10,
                'description': 'Original settings'
            },
            'tight_tol': {
                'jitter': 1e-7,
                'cg_tol': 1e-12 / d,  # Dimension-dependent tolerance
                'description': 'Tighter CG tolerance'
            },
            'adaptive_jitter': {
                'jitter': max(1e-12, 1e-16 * cond_estimate.item()),
                'cg_tol': 1e-10,
                'description': 'Adaptive jitter'
            },
            'both_fixes': {
                'jitter': max(1e-12, 1e-16 * cond_estimate.item()),
                'cg_tol': 1e-12 / d,
                'description': 'Both fixes combined'
            }
        }
        
        # Test vector
        test_vector = torch.randn(n, device=device, dtype=rdtype)
        
        results[d] = {}
        
        # Test each configuration
        for config_name, config in configs.items():
            print(f"\n--- {config['description']} ---")
            print(f"Jitter: {config['jitter']:.2e}, CG tol: {config['cg_tol']:.2e}")
            
            # Test CG solver
            beta_cg = solve_d2_fstar_kinv_z(
                test_vector, ws, toeplitz, fadj_batched,
                config['cg_tol'], config['jitter'], vanilla=False
            )
            
            # Test vanilla if grid is small enough
            if mtot**d < 500:
                beta_vanilla = solve_d2_fstar_kinv_z(
                    test_vector, ws, toeplitz, fadj_batched,
                    config['cg_tol'], config['jitter'], vanilla=True
                )
                diff = torch.norm(beta_cg - beta_vanilla).item()
                print(f"    ||CG - Vanilla||: {diff:.6e}")
                results[d][config_name] = diff
            else:
                print(f"    Grid too large for vanilla comparison")
                results[d][config_name] = None
    
    # Create comparison plots
    create_comparison_plots(results)
    
    return results

def create_comparison_plots(results):
    """Create comparison plots of the bias fixes"""
    
    print(f"\n{'='*60}")
    print("BIAS FIXES COMPARISON SUMMARY")
    print("="*60)
    
    # Create plots for each dimension that has results
    valid_dims = [d for d in results.keys() if any(v is not None for v in results[d].values())]
    
    if not valid_dims:
        print("No valid results for plotting")
        return
    
    fig, axes = plt.subplots(1, len(valid_dims), figsize=(6*len(valid_dims), 6))
    if len(valid_dims) == 1:
        axes = [axes]
    
    for idx, d in enumerate(valid_dims):
        ax = axes[idx]
        results_d = results[d]
        
        # Filter out None results
        valid_configs = [k for k, v in results_d.items() if v is not None]
        valid_errors = [results_d[k] for k in valid_configs]
        
        if valid_errors:
            bars = ax.bar(range(len(valid_configs)), valid_errors, alpha=0.7, 
                         color=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'][:len(valid_configs)])
            ax.set_yscale('log')
            ax.set_xlabel('Configuration')
            ax.set_ylabel('||CG - Vanilla|| (log scale)')
            ax.set_title(f'{d}D Case: CG vs Vanilla Error')
            ax.set_xticks(range(len(valid_configs)))
            ax.set_xticklabels(valid_configs, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, error in zip(bars, valid_errors):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                       f'{error:.1e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('bias_fixes_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print numerical summary
    print("\nNumerical Results Summary:")
    for d in valid_dims:
        print(f"\n{d}D Case:")
        results_d = results[d]
        for config_name, error in results_d.items():
            if error is not None:
                print(f"  {config_name:>15}: {error:.6e}")
            else:
                print(f"  {config_name:>15}: Grid too large for comparison")
        
        # Compute improvements
        if results_d.get('original') is not None:
            original_error = results_d['original']
            print(f"\nImprovements over original:")
            for config_name in ['tight_tol', 'adaptive_jitter', 'both_fixes']:
                if results_d.get(config_name) is not None:
                    improvement = original_error / results_d[config_name]
                    print(f"  {config_name:>15}: {improvement:.2f}x better")

def main():
    """Main function"""
    print("SIMPLE BIAS FIXES TEST")
    print("Testing CG solver configurations for bias reduction")
    print("="*60)
    
    results = run_bias_fixes()
    
    print(f"\n{'='*60}")
    print("FINAL RECOMMENDATIONS")
    print("="*60)
    
    # Find best performing configuration for 2D
    if 2 in results and results[2].get('original') is not None:
        original_2d = results[2]['original']
        best_config = None
        best_improvement = 0
        
        for config_name, error in results[2].items():
            if error is not None and config_name != 'original':
                improvement = original_2d / error
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_config = config_name
        
        if best_config:
            print(f"Best performing configuration for 2D: '{best_config}'")
            print(f"Improvement: {best_improvement:.2f}x reduction in error")
            print(f"\nThis should significantly reduce bias in m-step terms 2 and 3")
        else:
            print("Could not determine best configuration for 2D.")
    else:
        print("2D results not available.")
    
    print(f"\nTest completed! Results saved to 'bias_fixes_comparison.png'")

if __name__ == "__main__":
    main()







