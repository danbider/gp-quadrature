#!/usr/bin/env python3
"""
Deeper Bias Diagnosis
Since jitter alignment doesn't fully resolve the bias, let's identify
the other systematic differences between compute_vanilla_gradient and m_step.
"""

print("="*80)
print("DEEPER BIAS DIAGNOSIS")
print("="*80)
print("""
OBSERVATION: Even with aligned jitter (1e-8), the methods still don't agree.
This means there are additional systematic differences beyond jitter.

Let's identify ALL the computational differences...
""")

print("\n" + "="*80)
print("SYSTEMATIC DIFFERENCES ANALYSIS")
print("="*80)

print("""
DIFFERENCE #1: MATRIX CONSTRUCTION METHODS
==========================================

compute_vanilla_gradient:
  1. Builds: Kff = F_train @ diag(ws²) @ F_train.T  (n×n matrix)
  2. Uses: K = Kff + jitter * I
  3. Direct operations on full n×n kernel matrix

m_step:
  1. Uses: A = ws * toeplitz(ws * x) + jitter * x  (M×M matrix)
  2. Never constructs the full n×n kernel matrix
  3. Works in spectral space with Toeplitz structure

POTENTIAL ISSUE: These might not be mathematically equivalent!
The spectral representation K ≈ F @ diag(ws²) @ F.T is an approximation.
""")

print("""
DIFFERENCE #2: LINEAR SYSTEM SOLVERS
====================================

compute_vanilla_gradient:
  - Uses Cholesky decomposition: L = cholesky(K)
  - Direct solve: K⁻¹ = cholesky_solve(I, L)
  - Exact (up to machine precision)

m_step:
  - Uses Conjugate Gradients with preconditioner
  - Iterative solve with tolerance
  - Approximate (depends on CG convergence)

POTENTIAL ISSUE: Even with same jitter, CG vs Cholesky can give different results
due to different numerical properties and convergence behavior.
""")

print("""
DIFFERENCE #3: GRADIENT COMPUTATION PATHS
==========================================

compute_vanilla_gradient:
  1. Computes full K matrix derivatives: dK/dθ
  2. Uses direct matrix operations: tr(α αᵀ dK - K⁻¹ dK)
  3. All operations on n×n matrices

m_step:
  1. Uses same Dprime derivatives but in spectral space
  2. Computes via three terms: 0.5 * (term1 + term2 - term3)
  3. All operations on M×M matrices with NUFFT transforms

POTENTIAL ISSUE: Different numerical paths can accumulate different errors,
especially when M×M ≠ n×n or when NUFFT introduces approximation errors.
""")

print("""
DIFFERENCE #4: SPECTRAL APPROXIMATION ACCURACY
===============================================

Key question: How accurate is the spectral approximation?

The fundamental assumption is:
  K ≈ F_train @ diag(ws²) @ F_train.T

But this is only exact in the limit of infinite spectral grid.
With finite grid (M points), there are approximation errors.

HYPOTHESIS: Approximation errors become larger when d > 1 because:
- Spectral grid grows as M^d (exponentially with dimension)
- NUFFT accuracy decreases in higher dimensions
- h^d scaling effects become more pronounced
""")

print("\n" + "="*80)
print("DIAGNOSTIC TEST STRATEGY")
print("="*80)

diagnostic_code = '''
# Add these diagnostic tests to your notebook:

def test_spectral_approximation_accuracy():
    """Test how accurate the spectral approximation is"""
    
    print("=== SPECTRAL APPROXIMATION ACCURACY TEST ===")
    
    # Method 1: Direct kernel matrix construction
    print("\\n1. Direct kernel matrix:")
    K_direct = torch.zeros(n, n, dtype=rdtype, device=device)
    for i in range(n):
        for j in range(n):
            diff = x[i] - x[j]
            sq_dist = torch.sum(diff**2)
            K_direct[i, j] = kernel.init_variance * torch.exp(-0.5 * sq_dist / (kernel.init_lengthscale**2))
    
    print(f"K_direct condition number: {torch.linalg.cond(K_direct):.2e}")
    print(f"K_direct trace: {torch.trace(K_direct):.6f}")
    
    # Method 2: Spectral approximation (as used in compute_vanilla_gradient)
    print("\\n2. Spectral approximation:")
    Kff_spectral = F_train @ torch.diag(ws2.to(dtype=cdtype)) @ F_train.T.conj()
    Kff_spectral = Kff_spectral.real
    
    print(f"Kff_spectral condition number: {torch.linalg.cond(Kff_spectral):.2e}")
    print(f"Kff_spectral trace: {torch.trace(Kff_spectral):.6f}")
    
    # Compare approximation accuracy
    approx_error = torch.norm(K_direct - Kff_spectral).item()
    relative_error = approx_error / torch.norm(K_direct).item()
    
    print(f"\\nSpectral approximation error:")
    print(f"  Absolute error: {approx_error:.2e}")
    print(f"  Relative error: {relative_error:.2%}")
    
    # Element-wise comparison
    max_error = torch.max(torch.abs(K_direct - Kff_spectral)).item()
    mean_error = torch.mean(torch.abs(K_direct - Kff_spectral)).item()
    
    print(f"  Max element error: {max_error:.2e}")
    print(f"  Mean element error: {mean_error:.2e}")
    
    return K_direct, Kff_spectral, approx_error

def test_solver_differences():
    """Test if Cholesky vs CG solvers give different results for same system"""
    
    print("\\n=== SOLVER COMPARISON TEST ===")
    
    # Use the spectral approximation matrix (same as vanilla method)
    Kff = F_train @ torch.diag(ws2.to(dtype=cdtype)) @ F_train.T.conj()
    Kff = Kff.real
    K = Kff + 1e-8 * torch.eye(n, device=device)  # Same jitter
    
    # Test vector
    test_rhs = torch.randn(n, device=device, dtype=rdtype)
    
    print("\\n1. Cholesky solve (vanilla method):")
    L = torch.linalg.cholesky(K, upper=False)
    sol_cholesky = torch.cholesky_solve(test_rhs.unsqueeze(-1), L, upper=False).squeeze(-1)
    
    residual_chol = torch.norm(K @ sol_cholesky - test_rhs).item()
    print(f"   Cholesky residual: {residual_chol:.2e}")
    
    print("\\n2. Direct inverse (for comparison):")
    K_inv = torch.inverse(K)
    sol_direct = K_inv @ test_rhs
    
    residual_direct = torch.norm(K @ sol_direct - test_rhs).item()
    print(f"   Direct residual: {residual_direct:.2e}")
    
    # Compare solutions
    solver_diff = torch.norm(sol_cholesky - sol_direct).item()
    print(f"\\nSolver difference: {solver_diff:.2e}")
    
    return sol_cholesky, sol_direct, solver_diff

def test_gradient_computation_paths():
    """Compare gradient computation using same matrices but different paths"""
    
    print("\\n=== GRADIENT COMPUTATION PATH TEST ===")
    
    # Use same kernel matrix and jitter for both
    Kff = F_train @ torch.diag(ws2.to(dtype=cdtype)) @ F_train.T.conj()
    Kff = Kff.real
    K = Kff + 1e-8 * torch.eye(n, device=device)
    
    # Same variational parameters
    omega = q.Delta
    
    # Method 1: Vanilla gradient path (simplified)
    print("\\n1. Vanilla gradient computation path:")
    
    # Cholesky decomposition
    L = torch.linalg.cholesky(K, upper=False)
    I = torch.eye(n, device=device, dtype=rdtype)
    K_inv = torch.cholesky_solve(I, L, upper=False)
    
    # S computation
    S_inv = K_inv + torch.diag(omega)
    LS = torch.linalg.cholesky(S_inv, upper=False)
    S = torch.cholesky_inverse(LS, upper=False)
    
    # Posterior mean (assuming zero for simplicity)
    m_test = torch.zeros(n, device=device, dtype=rdtype)
    m_col = m_test.unsqueeze(-1)
    v = torch.cholesky_solve(m_col, L, upper=False).squeeze(-1)
    
    # Derivative matrices
    dK_dvar = F_train @ torch.diag(Dprime[:, 1].to(dtype=cdtype)) @ F_train.T.conj()
    dK_dls = F_train @ torch.diag(Dprime[:, 0].to(dtype=cdtype)) @ F_train.T.conj()
    
    dK_dvar_r = dK_dvar.real
    dK_dls_r = dK_dls.real
    
    # Vanilla gradients
    t1var_v = v @ (dK_dvar_r @ v)
    KinvS = torch.cholesky_solve(S, L, upper=False)
    t2var_v = torch.sum(KinvS * (K_inv @ dK_dvar_r))
    t3var_v = torch.sum(K_inv * dK_dvar_r)
    grad_var_v = 0.5 * (t1var_v + t2var_v - t3var_v)
    
    print(f"   Vanilla var gradient: {grad_var_v:.6f}")
    print(f"   Vanilla terms: t1={t1var_v:.6f}, t2={t2var_v:.6f}, t3={t3var_v:.6f}")
    
    # Method 2: Try to replicate m_step computation using same matrices
    print("\\n2. M-step style computation (if possible):")
    print("   This would require implementing the full m_step logic with same matrices...")
    print("   The key difference is that m_step uses Toeplitz + CG instead of full matrices + Cholesky")
    
    return grad_var_v, (t1var_v, t2var_v, t3var_v)

def comprehensive_bias_diagnosis():
    """Run all diagnostic tests"""
    
    print("\\n" + "="*60)
    print("COMPREHENSIVE BIAS DIAGNOSIS")
    print("="*60)
    
    # Test 1: Spectral approximation accuracy
    K_direct, Kff_spectral, approx_error = test_spectral_approximation_accuracy()
    
    # Test 2: Solver differences
    sol_chol, sol_direct, solver_diff = test_solver_differences()
    
    # Test 3: Gradient computation paths
    grad_var_v, terms_v = test_gradient_computation_paths()
    
    print("\\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    print(f"1. Spectral approximation error: {approx_error:.2e}")
    if approx_error > 1e-6:
        print("   ⚠️  Large spectral approximation error - this could be a major source of bias!")
    else:
        print("   ✅ Spectral approximation seems accurate")
    
    print(f"\\n2. Solver difference: {solver_diff:.2e}")
    if solver_diff > 1e-10:
        print("   ⚠️  Solvers give different results - numerical precision issue!")
    else:
        print("   ✅ Solvers agree well")
    
    print(f"\\n3. Need to compare gradient computation paths directly...")
    
    return {
        'approx_error': approx_error,
        'solver_diff': solver_diff,
        'grad_var_v': grad_var_v
    }

# Run the comprehensive diagnosis
results = comprehensive_bias_diagnosis()
'''

print(diagnostic_code)

print("\n" + "="*80)
print("LIKELY ROOT CAUSES (BEYOND JITTER)")
print("="*80)

print("""
HYPOTHESIS 1: SPECTRAL APPROXIMATION ERROR
===========================================
The spectral representation K ≈ F @ diag(ws²) @ F.T may not be accurate enough.

Test: Compare K_direct vs K_spectral matrices directly.
If approximation error is large (>1e-6), this could be the main bias source.

Fix: Use finer spectral grid (smaller eps in get_xis) or higher-order approximation.

HYPOTHESIS 2: TOEPLITZ vs FULL MATRIX OPERATIONS
=================================================
m_step uses Toeplitz structure + NUFFT, while vanilla uses full matrices.
These may not be numerically equivalent.

Test: Compare results when both methods use the same matrix representation.
Fix: Implement m_step using full matrices for small problems.

HYPOTHESIS 3: CG CONVERGENCE ISSUES
====================================
Even with same jitter, CG may not converge to the same solution as Cholesky.

Test: Compare CG vs Cholesky solutions for the same linear system.
Fix: Use much tighter CG tolerance or switch to direct solver.

HYPOTHESIS 4: NUFFT APPROXIMATION ERRORS
=========================================
NUFFT introduces additional approximation errors, especially in higher dimensions.

Test: Compare NUFFT transforms vs direct DFT.
Fix: Use higher precision NUFFT or direct transforms for small problems.
""")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("""
IMMEDIATE ACTIONS:
==================

1. Run the diagnostic tests above in your notebook
2. Check spectral approximation accuracy first (most likely culprit)
3. If spectral error is large, try smaller eps (e.g., eps=1e-8 instead of 1e-7)
4. Compare solver outputs directly
5. Test with vanilla=True mode in m_step for small problems

EXPECTED FINDINGS:
==================

Most likely: Spectral approximation error increases with dimension
- 1D: Small error, methods mostly agree
- 2D: Larger error, significant bias appears

Secondary: NUFFT accuracy decreases in higher dimensions
Tertiary: CG vs Cholesky numerical differences

The jitter alignment was necessary but not sufficient!
""")

def main():
    print("\nThis diagnostic will help identify the remaining bias sources.")
    print("Copy the test functions into your notebook and run them.")
    print("\nKey insight: Multiple systematic differences beyond jitter!")

if __name__ == "__main__":
    main()








