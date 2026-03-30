#!/usr/bin/env python3
"""
Dimension Scaling Analysis
Since d=1 works perfectly but d>1 shows bias, this points to specific
dimension-dependent issues in the spectral approximation.
"""

print("="*80)
print("DIMENSION SCALING ANALYSIS")
print("="*80)
print("""
KEY OBSERVATION: d=1 works perfectly, d>1 shows bias
This is a SMOKING GUN pointing to specific dimension-dependent issues!
""")

print("\n" + "="*80)
print("DIMENSION-DEPENDENT FACTORS")
print("="*80)

print("""
FACTOR 1: h^d SCALING CATASTROPHE
==================================

The spectral grid spacing scales as h^d where h = L/M.

In 1D: h^1 = h (reasonable scaling)
In 2D: h^2 = h² (quadratic scaling)
In 3D: h^3 = h³ (cubic scaling)

Example with h = 0.1:
- 1D: scaling factor = 0.1
- 2D: scaling factor = 0.01  (100x smaller!)
- 3D: scaling factor = 0.001 (1000x smaller!)

This h^d factor appears in the spectral weights and can cause:
1. Numerical underflow/overflow issues
2. Condition number problems
3. Loss of precision in floating point arithmetic

HYPOTHESIS: The h^d scaling makes the spectral representation
numerically unstable in higher dimensions.
""")

print("""
FACTOR 2: SPECTRAL GRID DENSITY
================================

The spectral approximation accuracy depends on grid density.

In 1D: M points cover 1D frequency space adequately
In 2D: Same M points now cover 2D frequency space → sparser coverage
In 3D: Same M points cover 3D frequency space → very sparse coverage

The approximation K ≈ F @ diag(ws²) @ F.T becomes less accurate
as the effective grid density decreases with dimension.

HYPOTHESIS: Spectral approximation error increases exponentially
with dimension due to curse of dimensionality.
""")

print("""
FACTOR 3: NUFFT ACCURACY DEGRADATION
=====================================

NUFFT (Non-Uniform FFT) accuracy typically decreases with dimension:

1D NUFFT: High accuracy, well-conditioned
2D NUFFT: Moderate accuracy, some conditioning issues  
3D+ NUFFT: Lower accuracy, potential conditioning problems

The NUFFT is used in:
- Forward transforms: fadj_batched(z)
- Toeplitz operations: toeplitz(ws * x)

HYPOTHESIS: NUFFT approximation errors accumulate and become
significant in higher dimensions.
""")

print("""
FACTOR 4: CONDITION NUMBER EXPLOSION
=====================================

The condition number of the spectral system can grow dramatically:

κ = max(ws²) / min(ws²)

In higher dimensions:
- Spectral weights ws can have wider dynamic range
- h^d scaling can exacerbate the condition number
- Small eigenvalues → numerical instability

HYPOTHESIS: The linear systems become ill-conditioned in higher
dimensions, causing CG convergence issues.
""")

print("\n" + "="*80)
print("DIAGNOSTIC TESTS FOR DIMENSION SCALING")
print("="*80)

diagnostic_code = '''
# Add these dimension-specific diagnostic tests to your notebook:

def analyze_dimension_scaling_effects():
    """Analyze how various factors scale with dimension"""
    
    print("=== DIMENSION SCALING EFFECTS ANALYSIS ===")
    
    # Current dimension and grid parameters
    print(f"\\nCurrent setup:")
    print(f"  Dimension d = {d}")
    print(f"  Grid size M = {len(ws)}")
    print(f"  Domain size L = {L}")  # You may need to define L
    print(f"  Grid spacing h = L/M = {L/len(ws) if 'L' in globals() else 'Unknown'}")
    
    # h^d scaling analysis
    if 'L' in globals():
        h = L / len(ws)
        hd_factor = h ** d
        print(f"\\nh^d scaling factor:")
        print(f"  h = {h:.6f}")
        print(f"  h^d = h^{d} = {hd_factor:.2e}")
        
        # Compare to 1D case
        h1_factor = h ** 1
        scaling_ratio = hd_factor / h1_factor
        print(f"  Ratio to 1D: {scaling_ratio:.2e} ({scaling_ratio:.1%})")
        
        if abs(scaling_ratio) < 1e-6:
            print("  ⚠️  EXTREME scaling! This could cause numerical issues.")
        elif abs(scaling_ratio) < 1e-3:
            print("  ⚠️  Large scaling difference from 1D.")
        else:
            print("  ✅ Reasonable scaling factor.")
    
    # Spectral weights analysis
    print(f"\\nSpectral weights analysis:")
    ws_real = ws.real if torch.is_complex(ws) else ws
    print(f"  ws range: [{ws_real.min():.2e}, {ws_real.max():.2e}]")
    print(f"  ws mean: {ws_real.mean():.2e}")
    print(f"  ws std: {ws_real.std():.2e}")
    
    # ws² analysis (used in kernel approximation)
    ws2_real = (ws.pow(2)).real if torch.is_complex(ws) else ws.pow(2)
    print(f"\\nws² analysis:")
    print(f"  ws² range: [{ws2_real.min():.2e}, {ws2_real.max():.2e}]")
    print(f"  ws² condition number: {ws2_real.max() / ws2_real.min():.2e}")
    
    # Check for numerical issues
    if ws2_real.min() < 1e-15:
        print("  ⚠️  Very small ws² values - potential underflow!")
    if ws2_real.max() / ws2_real.min() > 1e12:
        print("  ⚠️  Large condition number - numerical instability likely!")
    
    # Grid density analysis
    n_data = len(x)  # Number of data points
    grid_density = len(ws) / (d * n_data)  # Rough measure
    print(f"\\nGrid density analysis:")
    print(f"  Data points: {n_data}")
    print(f"  Spectral points: {len(ws)}")
    print(f"  Density ratio: {grid_density:.2f} (spectral/data per dimension)")
    
    if grid_density < 1:
        print("  ⚠️  Low spectral grid density - approximation may be poor!")
    
    return {
        'hd_factor': hd_factor if 'L' in globals() else None,
        'ws_condition': ws2_real.max() / ws2_real.min(),
        'grid_density': grid_density
    }

def compare_1d_vs_higher_d_accuracy():
    """Compare spectral approximation accuracy between 1D and current dimension"""
    
    print("\\n=== 1D vs HIGHER-D ACCURACY COMPARISON ===")
    
    print(f"\\nCurrent dimension: {d}D")
    
    # For this test, we'd need to create a 1D version of the same problem
    # This is conceptual - you'd need to adapt your data
    print("""
    To run this test, you would:
    
    1. Create a 1D version of your current problem:
       - Use only first dimension of x: x_1d = x[:, 0:1]  
       - Keep same kernel parameters
       - Use same number of spectral points
    
    2. Compute spectral approximation error for both:
       - 1D case: ||K_1d_direct - K_1d_spectral||
       - 2D case: ||K_2d_direct - K_2d_spectral|| (current)
    
    3. Compare the approximation errors:
       - If 1D error << 2D error: Confirms dimension scaling issue
       - If errors similar: Points to other causes
    """)
    
    # What we can check now: effective spectral resolution
    # In 1D: M points cover 1D frequency space
    # In dD: M points cover dD frequency space (much sparser)
    
    effective_resolution_1d = len(ws)  # All points available for 1D
    effective_resolution_dd = len(ws) / (d ** 0.5)  # Rough estimate for dD
    
    print(f"\\nEffective spectral resolution estimate:")
    print(f"  1D effective resolution: {effective_resolution_1d}")
    print(f"  {d}D effective resolution: {effective_resolution_dd:.1f}")
    print(f"  Resolution ratio: {effective_resolution_dd / effective_resolution_1d:.2%}")
    
    return effective_resolution_1d, effective_resolution_dd

def test_spectral_approximation_vs_dimension():
    """Test how spectral approximation quality depends on dimension"""
    
    print("\\n=== SPECTRAL APPROXIMATION vs DIMENSION TEST ===")
    
    # We can test this by comparing kernel evaluations
    # Take a small subset of points to make direct computation feasible
    n_test = min(20, len(x))  # Small subset for direct kernel computation
    x_test = x[:n_test]
    
    print(f"\\nTesting with {n_test} points...")
    
    # Direct kernel matrix (ground truth)
    print("\\n1. Computing direct kernel matrix...")
    K_direct = torch.zeros(n_test, n_test, dtype=torch.float64, device=device)
    for i in range(n_test):
        for j in range(n_test):
            diff = x_test[i] - x_test[j]
            sq_dist = torch.sum(diff**2)
            K_direct[i, j] = kernel.init_variance * torch.exp(-0.5 * sq_dist / (kernel.init_lengthscale**2))
    
    print(f"   K_direct condition number: {torch.linalg.cond(K_direct):.2e}")
    
    # Spectral approximation
    print("\\n2. Computing spectral approximation...")
    # You'll need to adapt this to use your F_train for the test subset
    print("   [This requires adapting F_train to the test subset]")
    print("   F_test = fadj_batched(x_test)  # NUFFT transform")
    print("   K_spectral = F_test @ diag(ws²) @ F_test.T")
    
    # For now, just show the analysis framework
    print(f"\\n3. Analysis framework:")
    print(f"   approximation_error = ||K_direct - K_spectral||")
    print(f"   relative_error = approximation_error / ||K_direct||")
    print(f"   max_element_error = max|K_direct - K_spectral|")
    
    print(f"\\nExpected results:")
    print(f"   - 1D: Small approximation error (< 1e-10)")
    print(f"   - 2D: Larger approximation error (> 1e-6)")
    print(f"   - 3D: Much larger approximation error (> 1e-4)")
    
    return K_direct

def comprehensive_dimension_analysis():
    """Run all dimension-related diagnostic tests"""
    
    print("\\n" + "="*60)
    print("COMPREHENSIVE DIMENSION ANALYSIS")
    print("="*60)
    
    # Test 1: Dimension scaling effects
    scaling_results = analyze_dimension_scaling_effects()
    
    # Test 2: 1D vs higher-D comparison
    res_1d, res_dd = compare_1d_vs_higher_d_accuracy()
    
    # Test 3: Spectral approximation quality
    K_direct = test_spectral_approximation_vs_dimension()
    
    print("\\n" + "="*60)
    print("DIMENSION ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Current dimension: {d}D")
    
    if scaling_results['hd_factor'] is not None:
        hd_factor = scaling_results['hd_factor']
        print(f"\\n1. h^d scaling factor: {hd_factor:.2e}")
        if abs(hd_factor) < 1e-6:
            print("   ⚠️  CRITICAL: Extreme h^d scaling - major numerical issues expected!")
        elif abs(hd_factor) < 1e-3:
            print("   ⚠️  WARNING: Large h^d scaling - numerical issues likely!")
        else:
            print("   ✅ h^d scaling seems reasonable")
    
    condition_num = scaling_results['ws_condition']
    print(f"\\n2. Spectral condition number: {condition_num:.2e}")
    if condition_num > 1e12:
        print("   ⚠️  CRITICAL: Extremely ill-conditioned - CG will fail!")
    elif condition_num > 1e8:
        print("   ⚠️  WARNING: Ill-conditioned - CG convergence issues likely!")
    else:
        print("   ✅ Reasonable conditioning")
    
    grid_density = scaling_results['grid_density']
    print(f"\\n3. Grid density ratio: {grid_density:.2f}")
    if grid_density < 0.5:
        print("   ⚠️  WARNING: Low grid density - poor spectral approximation!")
    else:
        print("   ✅ Adequate grid density")
    
    print(f"\\n4. Effective resolution ratio: {res_dd/res_1d:.2%}")
    if res_dd/res_1d < 0.5:
        print("   ⚠️  WARNING: Much lower effective resolution than 1D!")
    
    return scaling_results

# Run the comprehensive dimension analysis
results = comprehensive_dimension_analysis()
'''

print(diagnostic_code)

print("\n" + "="*80)
print("DIMENSION-SPECIFIC HYPOTHESES")
print("="*80)

print("""
PRIMARY HYPOTHESIS: h^d SCALING CATASTROPHE
============================================

The most likely culprit is the h^d scaling factor.

In your spectral representation, there's likely a factor of h^d somewhere
that becomes extremely small in higher dimensions:

h = L/M (grid spacing)
h^d scaling factor

Example: If h = 0.1 and d = 2, then h^d = 0.01 (100x smaller than 1D)

This can cause:
1. Numerical underflow in spectral weights
2. Extreme condition numbers  
3. CG solver failure
4. Loss of floating-point precision

SECONDARY HYPOTHESIS: SPECTRAL GRID SPARSITY
=============================================

The same number of spectral points M must cover d-dimensional frequency space.
Effective density decreases exponentially with dimension → poor approximation.

TERTIARY HYPOTHESIS: NUFFT ACCURACY DEGRADATION  
================================================

NUFFT accuracy decreases in higher dimensions, adding approximation errors.
""")

print("\n" + "="*80)
print("SMOKING GUN TEST")
print("="*80)

print("""
DEFINITIVE TEST:
================

Run this in your notebook to confirm the h^d scaling issue:

```python
# Check if h^d scaling is the culprit
print(f"Current dimension: {d}")
print(f"Grid spacing h ≈ {L/len(ws):.6f}")  # You may need to define L
print(f"h^d scaling factor: {(L/len(ws))**d:.2e}")

# Check spectral weights magnitude
print(f"ws magnitude range: [{ws.abs().min():.2e}, {ws.abs().max():.2e}]")
print(f"ws² magnitude range: [{ws.pow(2).abs().min():.2e}, {ws.pow(2).abs().max():.2e}]")

# The smoking gun: if ws² values are extremely small (< 1e-12), 
# this confirms h^d scaling is causing numerical issues
```

EXPECTED RESULT:
================

You should see:
- 1D: h^d factor ≈ 0.01 to 0.1 (reasonable)  
- 2D: h^d factor ≈ 1e-4 to 1e-6 (problematic)
- ws² values become extremely small in 2D

This explains why d=1 works perfectly but d>1 fails!
""")

def main():
    print("\nThe fact that d=1 works but d>1 doesn't is the key clue!")
    print("This points directly to dimension-dependent scaling issues.")
    print("\nMost likely: h^d scaling catastrophe in the spectral representation.")

if __name__ == "__main__":
    main()








