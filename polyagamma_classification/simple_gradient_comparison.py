#!/usr/bin/env python3
"""
Simple Gradient Comparison Test
Direct comparison to identify parameter inconsistencies between 
compute_vanilla_gradient and m_step functions.
"""

import sys
import os
sys.path.append('/Users/colecitrenbaum/Documents/GPs/gp-quadrature')

import torch
import numpy as np

# Import required modules
from kernels import SquaredExponential
from vanilla_gp_sampling import sample_bernoulli_gp

def compute_vanilla_gradient_test(x, y, lengthscale, variance, jitter=1e-7):
    """
    Test implementation of vanilla gradient computation
    This should match your notebook's compute_vanilla_gradient function
    """
    n, d = x.shape
    device = x.device
    dtype = x.dtype
    
    print(f"  Vanilla gradient computation:")
    print(f"    n={n}, d={d}")
    print(f"    lengthscale={lengthscale:.6f}")
    print(f"    variance={variance:.6f}")
    print(f"    jitter={jitter:.2e}")
    
    # Build kernel matrix
    K = torch.zeros(n, n, dtype=dtype, device=device)
    dK_dls = torch.zeros(n, n, dtype=dtype, device=device)  # derivative w.r.t. lengthscale
    dK_dvar = torch.zeros(n, n, dtype=dtype, device=device)  # derivative w.r.t. variance
    
    for i in range(n):
        for j in range(n):
            diff = x[i] - x[j]
            sq_dist = torch.sum(diff**2)
            
            # Kernel value
            K[i, j] = variance * torch.exp(-0.5 * sq_dist / (lengthscale**2))
            
            # Lengthscale derivative
            dK_dls[i, j] = K[i, j] * sq_dist / (lengthscale**3)
            
            # Variance derivative
            dK_dvar[i, j] = K[i, j] / variance
    
    # Add jitter
    K += jitter * torch.eye(n, dtype=dtype, device=device)
    
    # Compute K^{-1}
    try:
        K_inv = torch.inverse(K)
        cond_num = torch.linalg.cond(K).item()
        print(f"    K condition number: {cond_num:.2e}")
    except:
        print(f"    Using pseudo-inverse due to singular K")
        K_inv = torch.pinverse(K)
        cond_num = float('inf')
    
    # Compute alpha = K^{-1} * (y - 0.5)
    # This assumes binary classification with labels centered at 0.5
    y_centered = y - 0.5
    alpha = K_inv @ y_centered
    
    # Compute gradients using standard GP derivative formula
    # d/d(theta) log p(y|theta) = 0.5 * tr(alpha*alpha^T * dK/d(theta) - K^{-1} * dK/d(theta))
    alpha_outer = torch.outer(alpha, alpha)
    
    grad_ls = 0.5 * torch.trace(alpha_outer @ dK_dls - K_inv @ dK_dls)
    grad_var = 0.5 * torch.trace(alpha_outer @ dK_dvar - K_inv @ dK_dvar)
    
    return grad_ls.item(), grad_var.item(), cond_num

def test_parameter_consistency():
    """
    Test different parameter combinations to identify the source of bias
    """
    
    print("="*80)
    print("PARAMETER CONSISTENCY TEST")
    print("="*80)
    print("Testing different parameter combinations to find the source of bias")
    
    device = torch.device('cpu')
    rdtype = torch.float64
    
    # Create test data
    torch.manual_seed(42)
    results = {}
    
    for d in [1, 2]:
        print(f"\n{'='*30} {d}D CASE {'='*30}")
        
        n = 30  # Small size for clear comparison
        x = torch.rand(n, d, dtype=rdtype, device=device) * 2 - 1
        y, f = sample_bernoulli_gp(x, length_scale=0.1, variance=1.0)
        
        print(f"Data generated with: length_scale=0.1, variance=1.0")
        print(f"Data shape: x={x.shape}, y={y.shape}")
        
        # Test different parameter combinations
        param_sets = [
            {"lengthscale": 0.1, "variance": 1.0, "name": "Data generation parameters"},
            {"lengthscale": 0.2, "variance": 1.25, "name": "M-step kernel parameters"},
            {"lengthscale": 0.15, "variance": 1.125, "name": "Intermediate parameters"},
        ]
        
        results[d] = {}
        
        for param_set in param_sets:
            print(f"\n--- {param_set['name']} ---")
            
            try:
                grad_ls, grad_var, cond_num = compute_vanilla_gradient_test(
                    x, y, param_set['lengthscale'], param_set['variance']
                )
                
                print(f"    Results:")
                print(f"      Lengthscale gradient: {grad_ls:.6f}")
                print(f"      Variance gradient:    {grad_var:.6f}")
                print(f"      Gradient magnitude:   {np.sqrt(grad_ls**2 + grad_var**2):.6f}")
                
                results[d][param_set['name']] = {
                    'grad_ls': grad_ls,
                    'grad_var': grad_var,
                    'magnitude': np.sqrt(grad_ls**2 + grad_var**2),
                    'condition': cond_num
                }
                
            except Exception as e:
                print(f"    Error: {e}")
                results[d][param_set['name']] = None
    
    # Analysis
    print(f"\n{'='*80}")
    print("PARAMETER IMPACT ANALYSIS")
    print("="*80)
    
    for d in [1, 2]:
        print(f"\n{d}D Case - Parameter Impact:")
        
        if results[d].get("Data generation parameters") and results[d].get("M-step kernel parameters"):
            data_gen = results[d]["Data generation parameters"]
            mstep_params = results[d]["M-step kernel parameters"]
            
            ls_ratio = mstep_params['grad_ls'] / data_gen['grad_ls'] if data_gen['grad_ls'] != 0 else float('inf')
            var_ratio = mstep_params['grad_var'] / data_gen['grad_var'] if data_gen['grad_var'] != 0 else float('inf')
            mag_ratio = mstep_params['magnitude'] / data_gen['magnitude'] if data_gen['magnitude'] != 0 else float('inf')
            
            print(f"  Lengthscale gradient ratio (m_step/data_gen): {ls_ratio:.2f}")
            print(f"  Variance gradient ratio (m_step/data_gen):    {var_ratio:.2f}")
            print(f"  Magnitude ratio (m_step/data_gen):           {mag_ratio:.2f}")
            
            print(f"  Data gen params:  LS grad={data_gen['grad_ls']:.2e}, Var grad={data_gen['grad_var']:.2e}")
            print(f"  M-step params:    LS grad={mstep_params['grad_ls']:.2e}, Var grad={mstep_params['grad_var']:.2e}")
    
    # Cross-dimensional analysis
    print(f"\n{'='*80}")
    print("DIMENSIONAL BIAS ANALYSIS")
    print("="*80)
    
    if 1 in results and 2 in results:
        print("Comparing 1D vs 2D gradient magnitudes:")
        
        for param_name in ["Data generation parameters", "M-step kernel parameters"]:
            if (results[1].get(param_name) and results[2].get(param_name)):
                mag_1d = results[1][param_name]['magnitude']
                mag_2d = results[2][param_name]['magnitude']
                
                ratio = mag_2d / mag_1d if mag_1d != 0 else float('inf')
                
                print(f"  {param_name}:")
                print(f"    1D magnitude: {mag_1d:.2e}")
                print(f"    2D magnitude: {mag_2d:.2e}")
                print(f"    2D/1D ratio:  {ratio:.2f}")
    
    return results

def test_jitter_effects():
    """Test how different jitter values affect the gradients"""
    
    print(f"\n{'='*80}")
    print("JITTER SENSITIVITY TEST")
    print("="*80)
    
    device = torch.device('cpu')
    rdtype = torch.float64
    
    # Small 2D test case
    torch.manual_seed(42)
    n, d = 20, 2
    x = torch.rand(n, d, dtype=rdtype, device=device) * 1 - 0.5  # Smaller domain
    y = torch.randint(0, 2, (n,), dtype=rdtype, device=device)
    
    # Use m_step parameters
    lengthscale, variance = 0.2, 1.25
    
    jitter_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    
    print(f"Testing jitter sensitivity with lengthscale={lengthscale}, variance={variance}")
    print(f"Data: n={n}, d={d}")
    
    jitter_results = []
    
    for jitter in jitter_values:
        try:
            grad_ls, grad_var, cond_num = compute_vanilla_gradient_test(x, y, lengthscale, variance, jitter)
            magnitude = np.sqrt(grad_ls**2 + grad_var**2)
            
            jitter_results.append({
                'jitter': jitter,
                'grad_ls': grad_ls,
                'grad_var': grad_var,
                'magnitude': magnitude,
                'condition': cond_num
            })
            
            print(f"  Jitter {jitter:.1e}: LS={grad_ls:.2e}, Var={grad_var:.2e}, Mag={magnitude:.2e}, Cond={cond_num:.1e}")
            
        except Exception as e:
            print(f"  Jitter {jitter:.1e}: Error - {e}")
    
    # Analyze jitter stability
    if len(jitter_results) > 1:
        print(f"\nJitter Stability Analysis:")
        base_result = jitter_results[0]  # Use smallest jitter as reference
        
        for result in jitter_results[1:]:
            ls_change = abs(result['grad_ls'] - base_result['grad_ls']) / abs(base_result['grad_ls']) if base_result['grad_ls'] != 0 else 0
            var_change = abs(result['grad_var'] - base_result['grad_var']) / abs(base_result['grad_var']) if base_result['grad_var'] != 0 else 0
            
            print(f"  Jitter {result['jitter']:.1e}: LS change={ls_change:.1%}, Var change={var_change:.1%}")

def main():
    """Main function"""
    print("SIMPLE GRADIENT COMPARISON TEST")
    print("Focus: Parameter consistency and jitter effects")
    
    # Test parameter consistency
    param_results = test_parameter_consistency()
    
    # Test jitter effects
    test_jitter_effects()
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. PARAMETER CONSISTENCY:")
    print("   - Your m_step uses: lengthscale=0.2, variance=1.25")
    print("   - Your data was generated with: length_scale=0.1, variance=1.0")
    print("   - This mismatch is likely the PRIMARY source of bias!")
    
    print("\n2. DIMENSIONAL EFFECTS:")
    print("   - Check if gradient magnitudes scale differently in 1D vs 2D")
    print("   - Higher dimensions may have different numerical stability")
    
    print("\n3. NEXT STEPS:")
    print("   a) Verify which parameters your compute_vanilla_gradient actually uses")
    print("   b) Test m_step with the SAME parameters as compute_vanilla_gradient")
    print("   c) If parameters match and bias persists, it's due to spectral approximation")
    print("   d) Consider using consistent kernel parameters across all methods")
    
    print("\n4. IMMEDIATE TEST:")
    print("   - Run your m_step with a kernel that has lengthscale=0.1, variance=1.0")
    print("   - Compare against compute_vanilla_gradient with the same parameters")
    print("   - This should dramatically reduce the bias if parameter mismatch is the cause")

if __name__ == "__main__":
    main()








