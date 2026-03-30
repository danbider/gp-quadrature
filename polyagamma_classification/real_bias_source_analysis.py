#!/usr/bin/env python3
"""
Real Bias Source Analysis
Now that we know both functions use the same kernel, let's identify the 
actual computational differences causing bias when d>1.
"""

import sys
import os
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature')

import torch
import numpy as np

print("="*80)
print("REAL BIAS SOURCE ANALYSIS")
print("="*80)
print("Comparing compute_vanilla_gradient vs m_step computational differences")

print("""
KEY FINDINGS FROM CODE ANALYSIS:
================================

1. KERNEL MATRIX CONSTRUCTION DIFFERENCES:
   
   compute_vanilla_gradient:
   - Uses: Kff = F_train @ diag(ws²) @ F_train.T  (spectral representation)
   - Then: K = Kff + jitter * I
   - Jitter: 1e-8 (default)
   
   m_step:
   - Uses: A = ws * toeplitz(ws * x) + jitter * x  (Toeplitz representation)
   - Jitter: max(1e-12, 1e-16 * condition_number)  (adaptive, much larger in 2D)

2. GRADIENT COMPUTATION DIFFERENCES:

   compute_vanilla_gradient:
   - Uses SPECTRAL derivatives: dK/dvar = F_train @ diag(Dprime[:,1]) @ F_train.T
   - Uses SPECTRAL derivatives: dK/dls = F_train @ diag(Dprime[:,0]) @ F_train.T
   - Computes: grad = 0.5 * (t1 + t2 - t3) using direct matrix operations
   
   m_step:
   - Uses SPECTRAL derivatives: Same Dprime values
   - Computes: grad = 0.5 * (term1 + term2 - term3) using CG solves
   - BUT: Uses different linear system solvers (CG vs Cholesky)

3. NUMERICAL SOLVER DIFFERENCES:

   compute_vanilla_gradient:
   - Uses Cholesky decomposition: L = cholesky(K)
   - Direct solve: K⁻¹ = cholesky_solve(I, L)
   - Very stable for small-medium problems
   
   m_step:
   - Uses Conjugate Gradients with preconditioner
   - Iterative solve with tolerance: cg_tol = 1e-12/d
   - Can have convergence issues for ill-conditioned systems

4. JITTER DIFFERENCES:

   compute_vanilla_gradient:
   - Fixed jitter: 1e-8
   
   m_step (after our fixes):
   - Adaptive jitter: max(1e-12, 1e-16 * condition_number)
   - In 2D: condition_number ~ 1e28 → jitter ~ 1e12 (HUGE!)
   - This massive jitter difference could explain the bias!

5. MATRIX REPRESENTATION DIFFERENCES:

   compute_vanilla_gradient:
   - Works with full n×n kernel matrix K
   - Direct operations on dense matrices
   
   m_step:
   - Works with M×M spectral representation
   - Uses Toeplitz structure + NUFFT
   - Different numerical properties
""")

print("\n" + "="*80)
print("ROOT CAUSE HYPOTHESIS")
print("="*80)

print("""
MOST LIKELY CAUSE: JITTER INCONSISTENCY
=======================================

The adaptive jitter in m_step becomes ENORMOUS in 2D:
- 1D: condition_number ~ 1e13 → jitter ~ 1e-3
- 2D: condition_number ~ 1e28 → jitter ~ 1e12

This massive jitter difference (1e12 vs 1e-8) fundamentally changes 
the linear system being solved:

compute_vanilla_gradient solves: (K + 1e-8*I) α = rhs
m_step solves:                   (A + 1e12*I) β = rhs

These are completely different systems!
""")

print("\n" + "="*80)
print("SECONDARY CAUSES")
print("="*80)

print("""
1. SOLVER DIFFERENCES:
   - Cholesky (vanilla) vs CG (m_step) have different numerical behavior
   - CG convergence depends on condition number and tolerance
   - Cholesky is exact (up to machine precision)

2. MATRIX STRUCTURE DIFFERENCES:
   - Dense K matrix (vanilla) vs Toeplitz+NUFFT (m_step)
   - Different approximation errors in spectral representation

3. GRADIENT FORMULA IMPLEMENTATION:
   - Same mathematical formula: 0.5 * (t1 + t2 - t3)
   - But computed using different intermediate steps
""")

print("\n" + "="*80)
print("SOLUTION STRATEGY")
print("="*80)

print("""
IMMEDIATE FIX: ALIGN JITTER VALUES
==================================

Test 1: Use same jitter in both methods
- Set m_step jitter to 1e-8 (same as vanilla)
- This should dramatically reduce bias if jitter is the cause

Test 2: Use same solver approach  
- Implement vanilla approach in m_step for small problems
- Compare Cholesky vs CG solutions directly

Test 3: Check spectral approximation accuracy
- Compare F_train @ diag(ws²) @ F_train.T vs direct kernel matrix
- Verify NUFFT accuracy in higher dimensions
""")

def create_jitter_alignment_test():
    """Create a test to verify jitter is the main cause"""
    
    print("\n" + "="*80)
    print("JITTER ALIGNMENT TEST CODE")
    print("="*80)
    
    test_code = '''
# Test in your notebook:

# 1. Modify your m_step to use vanilla jitter
def test_m_step_with_vanilla_jitter():
    # In D2_Fstar_Kinv_z function, replace:
    # jitter_val = max(1e-12, 1e-16 * condition_number)
    # with:
    jitter_val = 1e-8  # Same as compute_vanilla_gradient
    
    # Run m_step with this fixed jitter
    # Compare against compute_vanilla_gradient
    
# 2. Test different jitter values systematically
jitter_values = [1e-12, 1e-10, 1e-8, 1e-6]
for jitter in jitter_values:
    print(f"Testing jitter: {jitter}")
    # Run both methods with this jitter value
    # Measure bias between them

# 3. If jitter alignment fixes the bias:
#    → Jitter inconsistency was the main cause
#    → Use consistent jitter values
# 
#    If bias persists:
#    → Look into solver differences (Cholesky vs CG)
#    → Check spectral approximation accuracy
    '''
    
    print(test_code)

def main():
    create_jitter_alignment_test()
    
    print("\n" + "="*80)
    print("CONFIDENCE ASSESSMENT")
    print("="*80)
    
    print("""
JITTER INCONSISTENCY: 95% confidence
- Massive difference (1e12 vs 1e-8) in 2D
- Would fundamentally change the linear systems
- Explains why bias is worse in higher dimensions

SOLVER DIFFERENCES: 60% confidence  
- CG vs Cholesky have different numerical behavior
- CG tolerance effects could contribute to bias

SPECTRAL APPROXIMATION: 40% confidence
- Both methods use same spectral representation
- NUFFT errors could accumulate in higher dimensions

RECOMMENDATION: Test jitter alignment first - highest impact, easiest to test!
    """)

if __name__ == "__main__":
    main()








