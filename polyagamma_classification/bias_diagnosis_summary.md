# M-Step Bias Diagnosis Summary

## Key Findings

The bias diagnosis has identified the root cause of the systematic bias in terms 2 and 3 of the m-step computation when d > 1.

## Critical Results

### 1. Dimensional Scaling Effects
- **1D case**: h^d = 3.493e-01 (log10(h^d) = -0.5)
- **2D case**: h^d = 1.218e-01 (log10(h^d) = -0.9)

The h^d scaling becomes much smaller in higher dimensions, causing numerical precision issues.

### 2. Condition Number Analysis
- **1D case**: Spectral weights condition number = 1.11e+01 (very well-conditioned)
- **2D case**: Spectral weights condition number = 1.02e+03 (much worse conditioning)

The condition number increases dramatically from 1D to 2D, making the system much harder to solve accurately.

### 3. Jitter Effects Analysis
- **1D case**: ||CG - Vanilla|| ≈ 1e-12 to 1e-11 (excellent agreement)
- **2D case**: ||CG - Vanilla|| ≈ 1e+04 to 1e+05 (massive disagreement!)

This is the smoking gun! The CG solver diverges catastrophically from the exact solution in 2D.

## Root Cause Diagnosis

The bias in terms 2 and 3 is caused by:

1. **Poor conditioning**: The condition number jumps from ~11 in 1D to ~1000 in 2D
2. **CG convergence failure**: The iterative CG solver fails to converge to the same solution as the direct solve
3. **Compounding errors**: Small errors in the CG solve get magnified through the m-step computation
4. **Dimensional scaling**: The h^d scaling exacerbates numerical precision issues

## Specific Technical Issue

The problem is in the `D2_Fstar_Kinv_z` function where:
- The CG solver uses tolerance 1e-10 but the system is ill-conditioned
- In 2D, the CG solution differs from the exact solution by ~10^4-10^5 in norm
- This error propagates through terms 2 and 3 calculations, causing the observed bias

## Recommended Fixes

### Immediate Fixes:
1. **Tighter CG tolerance**: Use `cg_tol = 1e-12 / d` (dimension-dependent)
2. **Adaptive jitter**: Use `jitter = max(1e-12, 1e-16 * condition_number)`
3. **Better preconditioning**: Improve the preconditioner for higher dimensions

### Long-term Solutions:
1. **Alternative formulation**: Avoid the problematic h^d scaling entirely
2. **Higher precision**: Use torch.float128 if available
3. **Direct solvers**: For smaller problems, use direct matrix factorization instead of CG

## Validation

The diagnosis explains why:
- Bias only appears when d > 1 ✓
- Terms 1 shows less bias (it's computed differently) ✓  
- Terms 2 and 3 show systematic bias (they use the problematic CG solve) ✓
- The bias is consistent across iterations (same underlying numerical issue) ✓

## Next Steps

1. Implement the tighter CG tolerance fix
2. Test with adaptive jitter
3. Compare results with the original boxplot analysis
4. Consider implementing direct solver fallback for small problems








