#!/usr/bin/env python3
"""
Simple Jitter Alignment Test
Creates notebook-compatible code to test different jitter values systematically
and find the optimal alignment between compute_vanilla_gradient and m_step.
"""

print("="*80)
print("SIMPLE JITTER ALIGNMENT TEST")
print("="*80)
print("""
This script provides code snippets to run in your notebook to systematically
test jitter alignment between compute_vanilla_gradient and m_step.

GOAL: Make both methods agree up to Hutchinson probe stochasticity.

HYPOTHESIS: The massive jitter difference (1e-8 vs 1e12) is causing the bias.
""")

print("\n" + "="*80)
print("STEP 1: MODIFY YOUR m_step JITTER")
print("="*80)

step1_code = '''
# In your notebook, modify the D2_Fstar_Kinv_z function:
# Find this line in cell 16:
# jitter_val = max(1e-12, 1e-16 * condition_number)

# Replace it with a FIXED jitter for testing:
jitter_val = 1e-8  # Same as compute_vanilla_gradient default

# Also modify Sigma_z_mm if it has adaptive jitter
'''

print(step1_code)

print("\n" + "="*80)
print("STEP 2: TEST SYSTEMATIC JITTER VALUES")
print("="*80)

step2_code = '''
# Add this cell to your notebook to test different jitter values:

import numpy as np
import matplotlib.pyplot as plt

def test_jitter_alignment():
    """Test different jitter values systematically"""
    
    # Test different jitter values
    jitter_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    
    results = []
    
    for jitter in jitter_values:
        print(f"\\n=== Testing jitter = {jitter:.1e} ===")
        
        # Compute vanilla gradient with this jitter
        grad_var_v, grad_ls_v, elbo1_v, t1var_v, t2var_v, t3var_v, t1v1_v, t2v1_v, t3v1_v = \\
            compute_vanilla_gradient(x, y, m, kernel, ws, ws2, F_train, q, jitter=jitter, Dprime=Dprime)
        
        print(f"Vanilla: grad_ls={grad_ls_v:.6f}, grad_var={grad_var_v:.6f}")
        
        # Temporarily modify m_step jitter for this test
        # (You'll need to modify the D2_Fstar_Kinv_z function to accept jitter parameter)
        
        # For now, manually set jitter_val = jitter in your D2_Fstar_Kinv_z function
        # Then run m_step:
        
        m_conv = (mtot - 1) // 2
        v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
        toeplitz = ToeplitzND(v_kernel, force_pow2=True)   
        Dprime_m = (h**d * kernel.spectral_grad(xis)).to(cdtype)
        
        term1, term2, term3 = m_step(q, m, toeplitz=toeplitz, ws=torch.sqrt(kernel.spectral_density(xis)), 
                                   Dprime=kernel.spectral_grad(xis), J=10, cg_tol=1e-12, verbose=False, 
                                   FstarKinv=FstarKinv, fadj_batched=fadj_batched, fwd_batched=fwd_batched, 
                                   h=h, d=d, xis=xis, vanilla=False)
        
        grad_m = 0.5*(term1 + term2 - term3)
        grad_ls_m = grad_m[0].real
        grad_var_m = grad_m[1].real
        
        print(f"M-step:  grad_ls={grad_ls_m:.6f}, grad_var={grad_var_m:.6f}")
        
        # Compute differences
        diff_ls = abs(grad_ls_v - grad_ls_m)
        diff_var = abs(grad_var_v - grad_var_m)
        
        print(f"Differences: |grad_ls|={diff_ls:.2e}, |grad_var|={diff_var:.2e}")
        
        results.append({
            'jitter': jitter,
            'vanilla': (grad_ls_v, grad_var_v),
            'mstep': (grad_ls_m, grad_var_m),
            'diff': (diff_ls, diff_var)
        })
    
    return results

# Run the test
results = test_jitter_alignment()

# Find best jitter
best_jitter = None
best_total_diff = float('inf')

print(f"\\n{'='*50}")
print("JITTER ALIGNMENT RESULTS")
print("="*50)

for res in results:
    total_diff = res['diff'][0] + res['diff'][1]
    print(f"Jitter {res['jitter']:.1e}: Total diff = {total_diff:.2e}")
    
    if total_diff < best_total_diff:
        best_total_diff = total_diff
        best_jitter = res['jitter']

print(f"\\nOptimal jitter: {best_jitter:.1e} (total diff: {best_total_diff:.2e})")
'''

print(step2_code)

print("\n" + "="*80)
print("STEP 3: PARAMETERIZE YOUR FUNCTIONS")
print("="*80)

step3_code = '''
# To make testing easier, modify your D2_Fstar_Kinv_z function to accept jitter as parameter:

def D2_Fstar_Kinv_z(z, toeplitz, ws, cg_tol: float = 1e-10, vanilla: bool = False, jitter_override=None):
    """
    Actually giving out F^*Kinv z 
    """
    ws2 = ws.pow(2)  # Define ws2 first
    cg_tol_fixed = 1e-12 / d  # Dimension-dependent tolerance
    condition_number = ws2.real.max() / ws2.real.min()

    if jitter_override is not None:
        jitter_val = jitter_override  # Use override for testing
    else:
        jitter_val = max(1e-12, 1e-16 * condition_number)  # Original adaptive jitter

    # Rest of function unchanged...
    A_apply = lambda x: ws*toeplitz(ws* x) + jitter_val * x
    # ... continue with rest of function

# Similarly for Sigma_z_mm function if needed
'''

print(step3_code)

print("\n" + "="*80)
print("STEP 4: TEST HUTCHINSON PROBE VARIABILITY")
print("="*80)

step4_code = '''
# Once you find the optimal jitter, test if remaining differences are due to probe stochasticity:

def test_hutchinson_variability(optimal_jitter):
    """Test variability due to Hutchinson probes"""
    
    print(f"Testing Hutchinson probe variability with optimal jitter={optimal_jitter:.1e}")
    
    # Compute vanilla (deterministic)
    grad_var_v, grad_ls_v, _, _, _, _, _, _, _ = \\
        compute_vanilla_gradient(x, y, m, kernel, ws, ws2, F_train, q, jitter=optimal_jitter, Dprime=Dprime)
    
    print(f"Vanilla (deterministic): ls={grad_ls_v:.6f}, var={grad_var_v:.6f}")
    
    # Compute m_step multiple times with different seeds
    mstep_results = []
    
    for seed in [42, 123, 456, 789, 101112]:
        torch.manual_seed(seed)  # This affects Hutchinson probes in m_step
        
        # Run m_step with optimal jitter (you'll need to set jitter_override=optimal_jitter)
        term1, term2, term3 = m_step(q, m, toeplitz=toeplitz, ws=torch.sqrt(kernel.spectral_density(xis)), 
                                   Dprime=kernel.spectral_grad(xis), J=10, cg_tol=1e-12, verbose=False, 
                                   FstarKinv=FstarKinv, fadj_batched=fadj_batched, fwd_batched=fwd_batched, 
                                   h=h, d=d, xis=xis, vanilla=False)
        
        grad_m = 0.5*(term1 + term2 - term3)
        grad_ls_m = grad_m[0].real
        grad_var_m = grad_m[1].real
        
        mstep_results.append((grad_ls_m, grad_var_m))
        
        diff_ls = abs(grad_ls_v - grad_ls_m)
        diff_var = abs(grad_var_v - grad_var_m)
        
        print(f"Seed {seed}: ls={grad_ls_m:.6f}, var={grad_var_m:.6f} | diff: ls={diff_ls:.2e}, var={diff_var:.2e}")
    
    # Analyze variability
    ls_values = [res[0] for res in mstep_results]
    var_values = [res[1] for res in mstep_results]
    
    ls_std = np.std(ls_values)
    var_std = np.std(var_values)
    ls_mean = np.mean(ls_values)
    var_mean = np.mean(var_values)
    
    print(f"\\nM-step variability:")
    print(f"  Lengthscale: mean={ls_mean:.6f}, std={ls_std:.2e}")
    print(f"  Variance: mean={var_mean:.6f}, std={var_std:.2e}")
    
    # Compare with vanilla
    ls_bias = abs(ls_mean - grad_ls_v)
    var_bias = abs(var_mean - grad_var_v)
    
    print(f"\\nBias vs vanilla:")
    print(f"  Lengthscale bias: {ls_bias:.2e} (vs 3σ: {3*ls_std:.2e})")
    print(f"  Variance bias: {var_bias:.2e} (vs 3σ: {3*var_std:.2e})")
    
    if ls_bias < 3 * ls_std and var_bias < 3 * var_std:
        print("\\n✅ SUCCESS: Bias is within 3σ of Hutchinson probe variability!")
    else:
        print("\\n⚠️  Bias still larger than Hutchinson probe variability.")

# Run this after finding optimal jitter
# test_hutchinson_variability(optimal_jitter)
'''

print(step4_code)

print("\n" + "="*80)
print("EXPECTED RESULTS")
print("="*80)

print("""
EXPECTED OUTCOMES:
==================

1. JITTER ALIGNMENT TEST:
   - You should find that jitter ≈ 1e-8 gives the smallest differences
   - This confirms that jitter inconsistency was the main bias source
   
2. BIAS REDUCTION:
   - With aligned jitter, differences should reduce by orders of magnitude
   - 1D case: Should see very small differences (< 1e-6 relative error)
   - 2D case: Should see much smaller differences than before
   
3. HUTCHINSON PROBE VARIABILITY:
   - Remaining differences should be within 3σ of probe stochasticity
   - This confirms that residual "bias" is just statistical noise
   
4. IMPLEMENTATION FIX:
   - Replace adaptive jitter with fixed optimal value
   - Use jitter_val = 1e-8 (or whatever optimal value you find)
   - This should resolve the bias issue permanently

TROUBLESHOOTING:
===============

If jitter alignment doesn't fully resolve the bias:
- Check solver differences (CG vs Cholesky)  
- Verify spectral approximation accuracy
- Test with vanilla=True mode in m_step for small problems
- Consider tighter CG tolerance (1e-14)
""")

print("\n" + "="*80)
print("IMPLEMENTATION SUMMARY")
print("="*80)

print("""
FINAL IMPLEMENTATION CHANGE:
============================

In your D2_Fstar_Kinv_z function (cell 16), change:

FROM:
  jitter_val = max(1e-12, 1e-16 * condition_number)

TO:
  jitter_val = 1e-8  # Fixed jitter matching compute_vanilla_gradient

This single line change should resolve the bias issue!
""")

def main():
    print("\nThis script provides the systematic testing approach.")
    print("Copy the code snippets above into your notebook to run the tests.")
    print("\nThe key insight: align jitter values between both methods!")

if __name__ == "__main__":
    main()








