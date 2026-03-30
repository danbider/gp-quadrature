#!/usr/bin/env python3
"""
Focused M-Step Bias Diagnosis
Specifically investigates the differences between m_step and compute_vanilla_gradient
that cause bias when d>1, focusing on the key computational differences.
"""

import sys
import os
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature')

import torch
import numpy as np
import matplotlib.pyplot as plt

# Import required modules
from kernels import SquaredExponential
from vanilla_gp_sampling import sample_bernoulli_gp

def diagnose_mstep_vs_vanilla_bias():
    """
    Focused diagnosis of the bias between m_step and compute_vanilla_gradient
    """
    
    print("="*60)
    print("FOCUSED M-STEP vs VANILLA GRADIENT BIAS DIAGNOSIS")
    print("="*60)
    print("Investigating why m_step introduces bias compared to compute_vanilla_gradient")
    print("Focus: Differences in computation methods when d>1")
    
    device = torch.device('cpu')
    rdtype = torch.float64
    
    for d in [1, 2]:
        print(f"\n{'='*20} {d}D CASE {'='*20}")
        
        # Setup test case
        torch.manual_seed(42)
        n = 100  # Smaller for easier analysis
        x = torch.rand(n, d, dtype=rdtype, device=device) * 2 - 1
        y, f = sample_bernoulli_gp(x, length_scale=0.1, variance=1.0)
        
        # Setup kernel - using same parameters as your notebook
        kernel = SquaredExponential(dimension=d, init_lengthscale=0.2, init_variance=1.25)
        
        print(f"Data: n={n}, d={d}")
        print(f"Kernel: lengthscale={kernel.init_lengthscale}, variance={kernel.init_variance}")
        
        # Key differences to investigate:
        
        print(f"\n--- 1. KERNEL PARAMETER DIFFERENCES ---")
        # Check if compute_vanilla_gradient uses different kernel parameters
        print(f"m_step uses: lengthscale={kernel.init_lengthscale:.6f}, variance={kernel.init_variance:.6f}")
        print("Question: Does compute_vanilla_gradient use the same parameters?")
        print("         Or does it use true_length_scale=0.1, true_variance=1.0?")
        
        print(f"\n--- 2. SPECTRAL vs DIRECT COMPUTATION ---")
        # m_step uses spectral representation, compute_vanilla_gradient uses direct
        
        # Direct kernel matrix (what compute_vanilla_gradient likely uses)
        print("Computing direct kernel matrix...")
        K_direct = torch.zeros(n, n, dtype=rdtype)
        for i in range(n):
            for j in range(n):
                diff = x[i] - x[j]
                sq_dist = torch.sum(diff**2)
                # Using kernel parameters from m_step
                K_direct[i, j] = kernel.init_variance * torch.exp(-0.5 * sq_dist / (kernel.init_lengthscale**2))
        
        # Add jitter
        K_direct += 1e-7 * torch.eye(n)
        
        print(f"Direct kernel matrix: shape={K_direct.shape}, condition_number={torch.linalg.cond(K_direct):.2e}")
        
        print(f"\n--- 3. GRADIENT COMPUTATION FORMULAS ---")
        # Check if the gradient formulas are equivalent
        
        print("Key questions:")
        print("a) Does compute_vanilla_gradient use the same gradient formula as m_step?")
        print("b) m_step computes: 0.5 * (term1 + term2 - term3)")
        print("   where term1, term2, term3 come from spectral representation")
        print("c) compute_vanilla_gradient likely uses direct matrix derivatives")
        
        print(f"\n--- 4. DIMENSIONAL SCALING EFFECTS ---")
        # Check h^d scaling that affects spectral methods
        
        # Approximate spectral grid size (simplified)
        # This is what affects the spectral approximation accuracy
        domain_size = (x.max(dim=0).values - x.min(dim=0).values).max().item()
        approx_h = 0.3  # Rough estimate
        hd_factor = approx_h**d
        
        print(f"Domain size: {domain_size:.3f}")
        print(f"Approximate h^d factor: {hd_factor:.6e}")
        print(f"h^d becomes smaller as d increases, affecting spectral accuracy")
        
        print(f"\n--- 5. JITTER AND REGULARIZATION ---")
        # Different regularization approaches
        
        print("m_step uses:")
        print("  - Adaptive jitter based on condition number")
        print("  - Spectral regularization through Toeplitz operations")
        print("compute_vanilla_gradient likely uses:")
        print("  - Fixed jitter (e.g., 1e-7)")
        print("  - Direct matrix regularization")
        
        print(f"\n--- 6. APPROXIMATION ERRORS ---")
        print("Potential sources of bias in m_step vs compute_vanilla_gradient:")
        print("a) Spectral truncation error (finite grid)")
        print("b) NUFFT approximation error")
        print("c) Toeplitz approximation vs full kernel matrix")
        print("d) Different numerical precision in CG vs direct solve")
        print("e) h^d scaling effects in gradient computation")
        
        if d == 2:
            print(f"\nFor d=2 specifically:")
            print(f"- h^d factor is much smaller ({hd_factor:.6e})")
            print(f"- Spectral grid is much larger (M^2 vs M)")
            print(f"- NUFFT approximation may be less accurate")
            print(f"- Condition numbers are typically much worse")
    
    print(f"\n{'='*60}")
    print("SPECIFIC DIAGNOSTIC RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. PARAMETER CONSISTENCY CHECK:")
    print("   - Verify compute_vanilla_gradient uses same kernel parameters as m_step")
    print("   - Check if it uses init_lengthscale/init_variance vs true_length_scale/true_variance")
    
    print("\n2. FORMULA VERIFICATION:")
    print("   - Compare gradient formulas between m_step and compute_vanilla_gradient")
    print("   - Ensure both compute d/d(lengthscale) and d/d(variance) consistently")
    
    print("\n3. APPROXIMATION ACCURACY TEST:")
    print("   - Test m_step with very fine spectral grid (small eps)")
    print("   - Compare spectral kernel approximation vs direct kernel matrix")
    
    print("\n4. REGULARIZATION ALIGNMENT:")
    print("   - Use same jitter value in both methods")
    print("   - Test with different jitter values to see impact")
    
    print("\n5. SIMPLE TEST CASE:")
    print("   - Use identical simple kernel (e.g., unit variance, unit lengthscale)")
    print("   - Compare on very small problem (n=10) where both can use direct methods")
    
    print("\n6. STEP-BY-STEP COMPARISON:")
    print("   - Implement simplified versions of both methods")
    print("   - Compare intermediate results (kernel matrices, gradients, etc.)")

def create_simple_comparison_test():
    """Create a simple test to isolate the bias source"""
    
    print(f"\n{'='*60}")
    print("SIMPLE COMPARISON TEST")
    print("="*60)
    
    # Very simple test case
    torch.manual_seed(42)
    n = 10  # Very small
    
    for d in [1, 2]:
        print(f"\n--- Simple {d}D Test (n={n}) ---")
        
        x = torch.rand(n, d, dtype=torch.float64) * 0.5  # Small domain
        y = torch.randint(0, 2, (n,), dtype=torch.float64)  # Simple binary labels
        
        # Simple kernel parameters
        lengthscale = 1.0
        variance = 1.0
        
        # Build kernel matrix directly
        K = torch.zeros(n, n, dtype=torch.float64)
        for i in range(n):
            for j in range(n):
                diff = x[i] - x[j]
                sq_dist = torch.sum(diff**2)
                K[i, j] = variance * torch.exp(-0.5 * sq_dist / (lengthscale**2))
        
        # Add jitter
        K += 1e-7 * torch.eye(n)
        
        print(f"Kernel matrix condition number: {torch.linalg.cond(K):.2e}")
        
        # Simple gradient computation (what compute_vanilla_gradient might do)
        K_inv = torch.inverse(K)
        alpha = K_inv @ (y - 0.5)  # Simple centering
        
        # Lengthscale gradient
        dK_dls = torch.zeros_like(K)
        for i in range(n):
            for j in range(n):
                diff = x[i] - x[j]
                sq_dist = torch.sum(diff**2)
                dK_dls[i, j] = K[i, j] * sq_dist / (lengthscale**3)
        
        # Variance gradient
        dK_dvar = K / variance
        
        # Gradient computation
        grad_ls = 0.5 * torch.trace(torch.outer(alpha, alpha) @ dK_dls - K_inv @ dK_dls)
        grad_var = 0.5 * torch.trace(torch.outer(alpha, alpha) @ dK_dvar - K_inv @ dK_dvar)
        
        print(f"Direct gradient: lengthscale={grad_ls:.6f}, variance={grad_var:.6f}")
        
        # This is what you should compare against your m_step output
        print(f"Compare this to your m_step output for the same data!")

def main():
    """Main function"""
    print("FOCUSED M-STEP BIAS DIAGNOSIS")
    print("Investigating specific differences causing bias when d>1")
    
    diagnose_mstep_vs_vanilla_bias()
    create_simple_comparison_test()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print("="*60)
    print("1. Check your compute_vanilla_gradient function for parameter consistency")
    print("2. Compare gradient formulas between the two methods")
    print("3. Test with the simple cases above to isolate the bias source")
    print("4. Consider that the bias might be due to spectral approximation limitations")

if __name__ == "__main__":
    main()








